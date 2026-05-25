"""
Variable Importance Evaluator using Gradient-Based Attribution

This module computes variable importance scores by analyzing gradients during
model inference. It uses the gradient of the loss with respect to each input
variable to estimate how much each variable contributes to the model's predictions.

The analysis is performed seasonally to capture different importance patterns
during breakup and freezeup periods.

Methodology:
    1. Load testing data and create samples
    2. For each sample, compute model prediction and loss
    3. Backpropagate to get gradients w.r.t. inputs
    4. Accumulate absolute gradients across samples
    5. Normalize to get importance scores

"""

import pickle
import calendar
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import torch
from tqdm import tqdm

from src.data.data_module import create_dataset
from src.model.model import load_model
from src.utils.constants import (
    BREAKUP_SEASON,
    FREEZEUP_SEASON,
    get_season,
)


class VariableImportanceCalculator:
    """
    Calculate variable importance scores using gradient-based attribution.
    
    This class computes importance scores by analyzing how much each input variable
    affects the model's predictions, as measured by the gradient of the loss function.
    """
    
    def __init__(
        self,
        config: Dict,
        model_dir: str,
        data_dir: str = "data/nc/"
    ):
        """
        Initialize the Variable Importance Calculator.
        
        Args:
            config: Model configuration dictionary
            model_dir: Directory containing model checkpoints and config
            data_dir: Directory containing input NetCDF data
        """
        self.config = config
        self.model_dir = Path(model_dir)
        self.data_dir = data_dir  # Keep as string to preserve trailing slash
        
        # Extract key parameters from config
        self.variables = config['variables']
        self.sequence_length = config.get('sequence_length', 7)
        self.sites = config.get('sites', [])
        
        # Number of input channels: variables + 3 static (lake_water, lake_ice, land)
        self.n_features = len(self.variables) + 3
        
        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model and statistics
        self.model = None
        self.across_site_stats = None

        # Cached outputs
        self.temporal_wide_df = None
        self.temporal_counts_df = None
        self.legacy_results_df = None
        
    def load_model_and_stats(self):
        """Load the trained model and dataset statistics."""
        print("Loading model and dataset statistics...")
        
        # Load model
        self.config['resume'] = True
        self.config['run_dir'] = str(self.model_dir)
        lit_model, ckpt, _ = load_model(self.config)
        self.model = lit_model
        self.model.eval()
        self.model.to(self.device)
        
        print(f"Model loaded from: {ckpt}")
        
        # Load statistics
        stats_path = self.model_dir / "dataset_stats.pkl"
        if stats_path.exists():
            with open(stats_path, "rb") as f:
                self.across_site_stats = pickle.load(f)
            print(f"Dataset statistics loaded from: {stats_path}")
        else:
            raise FileNotFoundError(f"Dataset statistics not found at {stats_path}")
       
    def normalize_importance(
        self,
        importances: torch.Tensor
    ) -> Dict[str, float]:
        """
        Normalize importance scores and create a dictionary mapping variables to scores.
        
        Args:
            importances: Raw importance tensor (n_met_vars, sequence_length)
                        Now only includes meteorological variables
            
        Returns:
            Dictionary mapping variable names to normalized importance scores
        """
        importance_normalized = self._normalize_tensor(importances)
        
        # Convert to numpy
        importance_values = importance_normalized.numpy()
        
        # Create dictionary (only meteorological variables now)
        importance_dict = dict(zip(self.variables, importance_values))
        
        return importance_dict

    def _normalize_tensor(self, importances: torch.Tensor) -> torch.Tensor:
        """Normalize a temporal-range importance tensor to sum to 1 across variables."""
        importance_avg = importances.mean(dim=1)
        denominator = torch.sum(importance_avg)
        if torch.abs(denominator) < 1e-12:
            return torch.zeros_like(importance_avg)
        return importance_avg / denominator

    def _season_months(self, season_config: Dict[str, int]) -> List[int]:
        """Return month numbers covered by a season config."""
        start_month = season_config['start_month']
        end_month = season_config['end_month']
        if start_month <= end_month:
            return list(range(start_month, end_month + 1))
        return list(range(start_month, 13)) + list(range(1, end_month + 1))
    
    def calculate_all_importances(
        self,
        test_start: str,
        test_end: str
    ) -> pd.DataFrame:
        """
        Calculate variable importance for overall, breakup, and freezeup periods.
        
        Processes each sample once and bins gradients by date.
        
        Args:
            test_start: Start of test period
            test_end: End of test period
            
        Returns:
            DataFrame with columns: Variable, Overall, Breakup, Freezeup
            Note: Only includes meteorological variables (excludes lake_water, lake_ice, land)
        """
        # Load model and dataset stats if not already loaded
        if self.model is None or self.across_site_stats is None:
            self.load_model_and_stats()
        
        print("\n" + "="*70)
        print("COMPUTING VARIABLE IMPORTANCE")
        print("="*70)
        print("Processing each sample once and binning by month and season...")
        
        # Create dataset for full test period
        dataset = create_dataset(self.config, test_start, test_end)
        # Apply scaling if stats are available
        if self.across_site_stats is not None:
            print("   Applying scaling to dataset")
            dataset.toggle_scaling(self.across_site_stats)
        
        # Initialize importance accumulators for each period
        # Only track meteorological variables (first len(self.variables) channels)
        n_met_vars = len(self.variables)
        overall_importances = torch.zeros(n_met_vars, self.sequence_length)
        breakup_importances = torch.zeros(n_met_vars, self.sequence_length)
        freezeup_importances = torch.zeros(n_met_vars, self.sequence_length)
        breakup_count = 0
        freezeup_count = 0

        month_names = [calendar.month_abbr[m] for m in range(1, 13)]
        monthly_importances = {
            month_name: torch.zeros(n_met_vars, self.sequence_length)
            for month_name in month_names
        }
        monthly_count = {month_name: 0 for month_name in month_names}
        
        # Define loss function
        from src.model.model import TrainingLoss
        criterion = TrainingLoss()
        
        # Process each sample once
        print(f"\nProcessing {len(dataset)} samples...")
        for i in tqdm(range(len(dataset)), desc="Computing importance"):
            input_seq, mask, target = dataset[i]
            
            # Get the date for this sample
            # From Ice_Cover_Dataset.__getitem__: date = self.target_dates[index % len(self.target_dates)]
            date = dataset.get_date_from_index(i)
            season = get_season(date)
            
            # Enable gradient computation for input
            input_seq.requires_grad = True
            
            # Move to device and add batch dimension
            input_batch = input_seq.unsqueeze(0).to(self.device)
            mask_batch = mask.unsqueeze(0).to(self.device)
            target_batch = target.unsqueeze(0).to(self.device)
            
            # Forward pass
            pred = self.model(input_batch)
            
            # Compute loss (only ice channel, index 1:2)
            loss = criterion(pred, target_batch[:, :, 1:2], mask_batch)
            
            # Backward pass
            loss.backward()
            
            # Extract gradients for meteorological variables only (first n_met_vars channels)
            # input_seq shape: (seq_len, n_channels, H, W)
            # We want gradients for channels 0:n_met_vars
            grads = input_seq.grad[:, :n_met_vars, :, :]
            
            # Average over spatial dimensions
            grad_importance = torch.mean(torch.abs(grads), dim=(2, 3)).T  # Shape: (n_met_vars, seq_len)
            
            # Accumulate into appropriate bins
            overall_importances += grad_importance.cpu()
            month_name = calendar.month_abbr[date.month]
            monthly_importances[month_name] += grad_importance.cpu()
            monthly_count[month_name] += 1

            if season == 'breakup':
                breakup_importances += grad_importance.cpu()
                breakup_count += 1
            elif season == 'freezeup':
                freezeup_importances += grad_importance.cpu()
                freezeup_count += 1
            
            # Clear gradients
            input_seq.grad = None
        
        print(f"\n✓ Processed all {len(dataset)} samples")
        
        # Normalize importance scores for legacy output (only meteorological variables)
        overall_dict = self.normalize_importance(overall_importances)
        breakup_dict = self.normalize_importance(breakup_importances)
        freezeup_dict = self.normalize_importance(freezeup_importances)
        
        # Compile results into DataFrame (only meteorological variables)
        results_df = pd.DataFrame({
            'Variable': self.variables,
            'Overall': [overall_dict[var] for var in self.variables],
            'Breakup': [breakup_dict[var] for var in self.variables],
            'Freezeup': [freezeup_dict[var] for var in self.variables]
        })

        # Temporal-range normalized wide output
        breakup_months = self._season_months(BREAKUP_SEASON)
        freezeup_months = self._season_months(FREEZEUP_SEASON)

        breakup_month_names = [calendar.month_abbr[m] for m in breakup_months]
        freezeup_month_names = [calendar.month_abbr[m] for m in freezeup_months]

        breakup_aggregate = torch.zeros(n_met_vars, self.sequence_length)
        freezeup_aggregate = torch.zeros(n_met_vars, self.sequence_length)
        breakup_aggregate_count = 0
        freezeup_aggregate_count = 0

        for month_name in breakup_month_names:
            breakup_aggregate += monthly_importances[month_name]
            breakup_aggregate_count += monthly_count[month_name]

        for month_name in freezeup_month_names:
            freezeup_aggregate += monthly_importances[month_name]
            freezeup_aggregate_count += monthly_count[month_name]

        temporal_wide_df = pd.DataFrame({'Variable': self.variables})
        for month_name in month_names:
            if monthly_count[month_name] > 0:
                month_avg = monthly_importances[month_name] / monthly_count[month_name]
            else:
                month_avg = torch.zeros_like(monthly_importances[month_name])

            temporal_wide_df[month_name] = self._normalize_tensor(month_avg).numpy()

        if breakup_aggregate_count > 0:
            breakup_avg = breakup_aggregate / breakup_aggregate_count
        else:
            breakup_avg = torch.zeros_like(breakup_aggregate)

        if freezeup_aggregate_count > 0:
            freezeup_avg = freezeup_aggregate / freezeup_aggregate_count
        else:
            freezeup_avg = torch.zeros_like(freezeup_aggregate)

        temporal_wide_df['Breakup'] = self._normalize_tensor(breakup_avg).numpy()
        temporal_wide_df['Freezeup'] = self._normalize_tensor(freezeup_avg).numpy()

        temporal_counts_df = pd.DataFrame(
            {
                'TemporalRange': month_names + ['Breakup', 'Freezeup'],
                'NumSamples': [monthly_count[m] for m in month_names] + [
                    breakup_aggregate_count,
                    freezeup_aggregate_count,
                ]
            }
        )

        self.temporal_wide_df = temporal_wide_df
        self.temporal_counts_df = temporal_counts_df
        self.legacy_results_df = results_df

        print(
            f"Temporal sample counts: breakup={breakup_count}, freezeup={freezeup_count}, "
            f"monthly_total={sum(monthly_count.values())}"
        )
        
        return results_df

    def get_temporal_wide_results(self) -> pd.DataFrame:
        """Return normalized wide-format temporal variable importance results."""
        if self.temporal_wide_df is None:
            raise RuntimeError("Temporal results are not available. Run calculate_all_importances first.")
        return self.temporal_wide_df

    def get_temporal_counts(self) -> pd.DataFrame:
        """Return sample counts by temporal range used for temporal variable importance."""
        if self.temporal_counts_df is None:
            raise RuntimeError("Temporal counts are not available. Run calculate_all_importances first.")
        return self.temporal_counts_df
    
    def save_results(
        self,
        results_df: pd.DataFrame,
        output_path: Optional[str] = None
    ):
        """
        Save variable importance results to CSV.
        
        Args:
            results_df: DataFrame with variable importance results
            output_path: Path to save CSV (default: model_dir/variable_importance.csv)
        """
        if output_path is None:
            output_path = self.model_dir / "variable_importance.csv"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nVariable importance results saved to: {output_path}")
    
    def print_results(self, results_df: pd.DataFrame):
        """Print variable importance results in a formatted table."""
        print("\n" + "="*70)
        print("VARIABLE IMPORTANCE RESULTS")
        print("="*70)
        print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("="*70)


def compute_variable_importance(
    model_name: str,
    test_start: str,
    test_end: str,
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to compute variable importance for a trained model.
    
    Args:
        model_name: Name of the model run (e.g., "LIF_DL_Best")
        output_dir: Optional output directory (default: results/{model_name}/evaluations/)
        test_start: Start of test period
        test_end: End of test period
        
    Returns:
        DataFrame with variable importance results
    """
    import yaml
    
    # Set up paths
    model_dir = Path(f"results/{model_name}")
    config_path = model_dir / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['name'] = model_name
    config['run_dir'] = str(model_dir)
    
    # Create calculator
    calculator = VariableImportanceCalculator(
        config=config,
        model_dir=str(model_dir),
        data_dir="data/nc/"
    )
    
    # Calculate importance
    results_df = calculator.calculate_all_importances(
        test_start=test_start,
        test_end=test_end
    )
    
    # Print results
    calculator.print_results(results_df)

    temporal_wide_df = calculator.get_temporal_wide_results()
    temporal_counts_df = calculator.get_temporal_counts()

    print("\n" + "="*70)
    print("TEMPORAL VARIABLE IMPORTANCE (NORMALIZED WIDE)")
    print("="*70)
    print(temporal_wide_df.head().to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("="*70)
    
    # Save results
    if output_dir is None:
        output_dir = model_dir / "evaluations"
    
    output_path = Path(output_dir) / "variable_importance.csv"
    calculator.save_results(results_df, output_path)

    temporal_output_path = Path(output_dir) / "variable_importances_normalized.csv"
    temporal_wide_df.to_csv(temporal_output_path, index=False)
    print(f"Temporal normalized wide results saved to: {temporal_output_path}")

    return results_df
