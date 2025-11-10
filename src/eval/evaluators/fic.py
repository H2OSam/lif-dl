"""FIC (Fraction Ice Cover) evaluator for temporal timeseries metrics.

Evaluates continuous FIC predictions (0-1 scale) using temporal accuracy metrics.
Applies to:
- CIS comparisons (native FIC data)
- Spatial sources converted to FIC timeseries

Metrics:
- MAE: Mean Absolute Error
- RMSE: Root Mean Square Error
- MASE: Mean Absolute Scaled Error
"""

from typing import Dict, Any
import numpy as np
import pandas as pd

from . import BaseEvaluator
from ..utils import convert_spatial_to_fic, align_daily


class FICEvaluator(BaseEvaluator):
    """
    FIC Evaluator using new architecture.
    
    Handles two comparison types:
    1. CIS comparisons: Native FIC data, weekly temporal resolution
    2. Spatial comparisons: Convert spatial binary maps to FIC timeseries
    
    Preprocessing:
    - CIS: Align daily model output to weekly CIS dates
    - Spatial: Convert binary spatial maps to mean FIC per timestep
    
    Metrics:
    - Temporal accuracy: MAE, RMSE
    - Relative performance: MASE (vs baseline)
    """
    name = "fic"
    
    def preprocess(self, context, obs_name: str, pred_name: str) -> Dict[str, Any]:
        """
        Preprocess sources to FIC timeseries for comparison.
        
        Handles two scenarios:
        1. CIS comparison: Align daily model to weekly CIS observations
        2. Spatial comparison: Convert spatial binary maps to FIC timeseries
        
        All inputs are xr.DataArray, outputs are numpy arrays for metric calculations.
        
        Args:
            context: EvaluationContext with all sources (all xr.DataArray)
            obs_name: Name of observation source ('cis' or spatial source)
            pred_name: Name of prediction source (spatial source)
            
        Returns:
            Dictionary containing:
            - 'obs': 1D array of observation FIC values
            - 'pred': 1D array of prediction FIC values
            - 'n_samples': Number of valid timesteps
            - 'baseline_mae': MAE of naive baseline (for MASE calculation)
        """
        obs_source = context.get_source(obs_name)
        pred_source = context.get_source(pred_name)
        mask = context.mask
        
        if (obs_name == 'cis'):
            # CIS comparison: Weekly native FIC data (already a 1D DataArray)
            # Convert spatial prediction to daily FIC timeseries DataArray
            pred_fic_da = convert_spatial_to_fic(pred_source, mask)
            
            # For baseline calculation, use obs_source directly (it's a DataArray)
            obs_fic_da = obs_source
            
        else:
            # Spatial comparison: Both sources are spatial binary maps
            # Convert both to FIC timeseries (DataArrays with time dimension)
            obs_fic_da = convert_spatial_to_fic(obs_source, mask)
            pred_fic_da = convert_spatial_to_fic(pred_source, mask)
            
        # Align by date using xarray-native alignment
        obs_aligned, pred_aligned, common_dates = align_daily(
            obs_fic_da, pred_fic_da, obs_name, pred_name
        )
        
        # Create valid mask
        valid_mask = ~np.isnan(obs_aligned) & ~np.isnan(pred_aligned)
        obs_valid = obs_aligned[valid_mask]
        pred_valid = pred_aligned[valid_mask]
            
        # Calculate baseline MAE for MASE
        baseline_fic_da = None
        if hasattr(context, 'metadata') and context.metadata:
            baseline_sources = context.metadata.get('baseline_sources', {})
            
            if obs_name in baseline_sources:
                baseline_source = baseline_sources[obs_name]
                
                # Process baseline source - check if it's already DataArray or needs conversion
                if obs_name.upper() == 'CIS':
                    # Baseline CIS is already a DataArray (from updated load_cis)
                    baseline_fic_da = baseline_source
                else:
                    # Spatial source - convert to FIC DataArray
                    baseline_fic_da = convert_spatial_to_fic(baseline_source, mask)
        
        baseline_mae = self._get_baseline_mae(obs_fic_da, baseline_fic_da)
        
        return {
            'obs': obs_valid,
            'pred': pred_valid,
            'n_samples': len(obs_valid),
            'baseline_mae': baseline_mae
        }
    
    def evaluate(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute FIC temporal accuracy metrics.
        
        Args:
            preprocessed: Dictionary from preprocess() containing:
                - obs: 1D array of observation FIC values
                - pred: 1D array of prediction FIC values
                - n_samples: Number of valid timesteps
                - baseline_mae: MAE of naive baseline
                
        Returns:
            Dictionary with metrics:
            - MAE: Mean absolute error
            - RMSE: Root mean square error
            - MASE: Mean absolute scaled error (vs naive baseline)
            - n_samples: Number of timesteps used
        """
        obs = preprocessed['obs']
        pred = preprocessed['pred']
        n_samples = preprocessed['n_samples']
        baseline_mae = preprocessed.get('baseline_mae', None)
        
        # Calculate all metrics
        results = {
            "MAE": self._mae(obs, pred),
            "RMSE": self._rmse(obs, pred),
            "n_samples": n_samples
        }
        
        # Add MASE if baseline is available
        if baseline_mae is not None and baseline_mae > 0:
            results["MASE"] = self._mase(obs, pred, baseline_mae)
        
        return results
    
    def extract_intermediate_data(self, context, source_name: str) -> Dict[str, Any]:
        """
        Extract FIC timeseries for a single source.
        
        This method extracts the raw FIC timeseries data for a source,
        which can be saved for downstream figure generation.
        
        Args:
            context: EvaluationContext with all sources (all xr.DataArray)
            source_name: Name of source (e.g., 'ims', 'lif_dl', 'cis')
            
        Returns:
            Dictionary with:
            - dates: Array of datetime values
            - fic: Array of FIC values (0-1 scale)
            - site: Site name
            - source: Source name
        """
        source = context.get_source(source_name)
        mask = context.mask
        
        # Extract FIC timeseries based on source type
        if source_name.lower() == 'cis':
            # CIS is already FIC data (1D DataArray)
            dates = pd.DatetimeIndex(source.time.values)
            fic_values = source.values
        else:
            # Spatial source - convert to FIC (returns DataArray)
            fic_da = convert_spatial_to_fic(source, mask)
            dates = pd.DatetimeIndex(fic_da.time.values)
            fic_values = fic_da.values
        
        return {
            'dates': dates,
            'fic': fic_values,
            'site': context.site,
            'source': source_name
        }
    
    @staticmethod
    def _mae(obs, pred):
        """
        Mean Absolute Error.
        
        Formula: mean(|pred - obs|)
        Range: [0, ∞) where 0 is perfect
        
        Args:
            obs: Observed FIC values
            pred: Predicted FIC values
        """
        if len(obs) == 0:
            return float('nan')
        
        mae = np.mean(np.abs(pred - obs))
        return float(mae)
    
    @staticmethod
    def _rmse(obs, pred):
        """
        Root Mean Square Error.
        
        Formula: sqrt(mean((pred - obs)²))
        Range: [0, ∞) where 0 is perfect
        
        Args:
            obs: Observed FIC values
            pred: Predicted FIC values
        """
        if len(obs) == 0:
            return float('nan')
        
        rmse = np.sqrt(np.mean((pred - obs) ** 2))
        return float(rmse)

    @staticmethod
    def _get_baseline_mae(obs_da, baseline_da=None):
        """
        Calculate baseline MAE for naive predictor:
        Predict next observation = last year's observation.
        
        Args:
            obs_da: Observed FIC values in xr.DataArray with time dimension
            baseline_da: Baseline FIC values from previous year (xr.DataArray with time dimension)
                        If None, will try to use obs data shifted by 1 year (less accurate)
        
        Returns:
            float: Baseline MAE value
        """
        # Convert DataArray to Series for reindexing operations
        obs = pd.Series(data=obs_da.values, index=pd.DatetimeIndex(obs_da.time.values))
        
        if baseline_da is not None:
            # Convert baseline to Series
            baseline = pd.Series(data=baseline_da.values, index=pd.DatetimeIndex(baseline_da.time.values))
            # Incorporate the extra baseline data
            combined_obs = pd.concat([baseline, obs])
        else:
            # No baseline data provided, use obs shifted by 1 year
            combined_obs = obs
            
        # Extract all the dates in the obs index and shift by 1 year
        obs_dates = obs.index
        shifted_dates = obs_dates - pd.Timedelta(days=365)
        
        # Get the nearest obs values for the shifted dates
        # Apply a maximum distance of 7 days to avoid poor matches
        prev_season = combined_obs.reindex(shifted_dates, method='nearest', tolerance=pd.Timedelta(days=7))
        
        # Calculate MAE between obs and prev_season, ignoring NaNs
        valid_mask = ~np.isnan(obs.values) & ~np.isnan(prev_season.values)
        if np.sum(valid_mask) == 0:
            return float('nan')
        
        mae = np.mean(np.abs(obs.values[valid_mask] - prev_season.values[valid_mask]))
        return float(mae)        

    @staticmethod
    def _mase(obs, pred, baseline_mae):
        """
        Mean Absolute Scaled Error.
        
        Formula: MAE(pred) / MAE(baseline)
        Where baseline_mae is the MAE of a naive baseline predictor
        (typically: next value = current value).
        
        Range: [0, ∞) where < 1 is better than baseline
        - MASE < 1: Better than baseline
        - MASE = 1: Same as baseline
        - MASE > 1: Worse than baseline
        
        Args:
            obs: Observed FIC values
            pred: Predicted FIC values
            baseline_mae: MAE of baseline predictor
        """
        if baseline_mae == 0 or baseline_mae is None:
            return float('nan')
        
        mae = FICEvaluator._mae(obs, pred)
        
        if np.isnan(mae):
            return float('nan')
        
        mase = mae / baseline_mae
        return float(mase)
