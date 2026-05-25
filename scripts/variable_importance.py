"""
Compute Variable Importance for a Trained Model

This script calculates variable importance scores using gradient-based attribution.
It analyzes how much each input variable contributes to the model's predictions
during the testing period, with separate analyses for breakup and freezeup seasons.

Usage:
    python scripts/compute_variable_importance.py --name LIF_DL_Best
"""

from argparse import ArgumentParser
import sys
import os
import yaml
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.variable_importance import compute_variable_importance
from src.utils.constants import EVAL_START, EVAL_END


def parse_arguments():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Compute variable importance scores for a trained model")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the model run (e.g., 'LIF_DL_Best')"
    )    
    return parser.parse_args()


def main(config, args):
    """
    Main execution function.
    """
    
    run_dir = config.get("run_dir", "results/evaluations")
    output_dir = os.path.join(run_dir, f"evaluations")
    os.makedirs(output_dir, exist_ok=True)
      
    try:
        # Compute variable importance
        results_df = compute_variable_importance(
            model_name=args.name,
            test_start=EVAL_START,
            test_end=EVAL_END,
            output_dir=output_dir,
        )

        temporal_csv_path = os.path.join(
            output_dir,
            "variable_importances_normalized.csv"
        )

        if os.path.exists(temporal_csv_path):
            temporal_preview = pd.read_csv(temporal_csv_path).head()
            print("\nTemporal normalized wide importance preview:")
            print(temporal_preview.to_string(index=False))
        else:
            print(
                f"\nTemporal CSV not found at: {temporal_csv_path}. "
                "Variable importance completed, but preview was skipped."
            )

        print("\nVariable importance calculation completed successfully!")

    except Exception as e:
        print(f"\nError during variable importance calculation: {e}")
        raise


if __name__ == "__main__":
    args = parse_arguments()

    # Get run name from arguments
    # Get the run name from arguments
    run_name = args.name

    # Find the config file based on the run name
    run_dir = f"results/{run_name}"
    config_path = os.path.join("results", run_name, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found for run name '{run_name}' at expected path: {config_path}")

    # Load config and set the run directory
    config = yaml.safe_load(open(config_path))
    config["run_dir"] = run_dir

    # Confirm that the config name matches the provided run name
    if config.get("name", None) != run_name:
        raise ValueError(f"Config name '{config.get('name', None)}' does not match provided run name '{run_name}'")

    main(config, args)
