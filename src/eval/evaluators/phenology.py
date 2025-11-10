"""Phenology evaluator for ice transition date metrics.

Extracts phenological event dates (freeze-up, break-up) from FIC timeseries
and computes date errors between observation and prediction.

Events:
- Break-Up Start (First Open Water): First transition from ice-covered to transitional
- Break-Up End (Water Clear of Ice): Last transition from transitional to open water
- Freeze-Up Start (Freeze Onset): First transition from open water to transitional
- Freeze-Up End (Continuous Ice Cover): Last transition from transitional to ice-covered

Metrics:
- MAE (days): Mean Absolute Error in days for each event type
"""

from typing import Dict, Any
import numpy as np
import pandas as pd

from . import BaseEvaluator
from ..utils import convert_spatial_to_fic, calculate_transitions


class PhenologyEvaluator(BaseEvaluator):
    """
    Phenology Evaluator using new architecture.
    
    Handles two comparison types:
    1. CIS comparisons: Native FIC data, weekly temporal resolution
    2. Spatial comparisons: Convert spatial binary maps to FIC timeseries
    
    Preprocessing:
    - Convert spatial to FIC if needed
    - Align daily/weekly data
    - Classify FIC into 3 states (open water, transitional, ice-covered)
    - Extract transition dates for each year
    
    Metrics:
    - MAE (days) for each event: Breakup_Start, Breakup_End, Freezeup_Start, Freezeup_End
    """
    name = "phenology"
    
    def preprocess(self, context, obs_name: str, pred_name: str) -> Dict[str, Any]:
        """
        Extract phenology transition dates from each source independently.
        
        Process:
        1. Convert spatial sources to FIC timeseries (or use native FIC for CIS)
        2. Classify FIC into 3 states (open water, transitional, ice-covered)
        3. Extract transition dates for each year using calculate_transitions()
        4. Return transition dates (no alignment needed - matched by year in evaluate())
        
        Args:
            context: EvaluationContext with all sources
            obs_name: Name of observation source ('cis' or spatial source)
            pred_name: Name of prediction source (spatial source)
            
        Returns:
            Dictionary containing:
            - 'obs_transitions': Dict of obs transition dates by event type
            - 'pred_transitions': Dict of pred transition dates by event type
        """
        obs_source = context.get_source(obs_name)
        pred_source = context.get_source(pred_name)
        mask = context.mask
        
        # Process observation source
        if obs_name == 'cis':
            # CIS is already FIC data
            # Already converted to 0-1 scale by DataLoader
            obs_fic_values = obs_source.values
            obs_dates = pd.DatetimeIndex(obs_source.time.values)
        else:
            # Spatial source - convert to FIC
            obs_fic_values = convert_spatial_to_fic(obs_source, mask)
            obs_dates = pd.DatetimeIndex(obs_source.time.values)
        
        # Process prediction source (always spatial)
        pred_fic_values = convert_spatial_to_fic(pred_source, mask)
        pred_dates = pd.DatetimeIndex(pred_source.time.values)
        
        # Classify FIC into 3 states
        # 0: Open water (FIC < 0.1)
        # 1: Transitional (0.1 ≤ FIC < 0.9)
        # 3: Ice-covered (FIC ≥ 0.9)
        obs_classes = self._classify_fic(obs_fic_values)
        pred_classes = self._classify_fic(pred_fic_values)
        
        # Extract transition dates independently for each source
        obs_transitions = calculate_transitions(obs_classes, obs_dates)
        pred_transitions = calculate_transitions(pred_classes, pred_dates)
        
        return {
            'obs_transitions': obs_transitions,
            'pred_transitions': pred_transitions
        }
    
    def evaluate(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute phenology date error metrics.
        
        Calculates Mean Absolute Error (MAE) in days for each event type:
        - Breakup_Start_MAE: First Open Water (FOW)
        - Breakup_End_MAE: Water Clear of Ice (WCI)
        - Freezeup_Start_MAE: Freeze Onset (FO)
        - Freezeup_End_MAE: Continuous Ice Cover (CIC)
        
        Args:
            preprocessed: Dictionary from preprocess() containing:
                - obs_transitions: Dict of obs transition dates
                - pred_transitions: Dict of pred transition dates
                
        Returns:
            Dictionary with metrics:
            - Breakup_Start_MAE: MAE in days for break-up start
            - Breakup_End_MAE: MAE in days for break-up end
            - Freezeup_Start_MAE: MAE in days for freeze-up start
            - Freezeup_End_MAE: MAE in days for freeze-up end
        """
        obs_transitions = preprocessed['obs_transitions']
        pred_transitions = preprocessed['pred_transitions']
        
        results = {}
        
        # Compute MAE for each event type
        for event_name in ['Breakup_start', 'Breakup_end', 'Freezeup_start', 'Freezeup_end']:
            mae, rmse = self._compute_event_metrics(
                obs_transitions[event_name],
                pred_transitions[event_name]
            )
            
            # Convert event name to output format
            metric_name = event_name.replace('_', '_').replace('start', 'Start').replace('end', 'End')
            results[f'{metric_name}_MAE'] = mae
            results[f'{metric_name}_RMSE'] = rmse
        
        return results
    
    @staticmethod
    def _classify_fic(fic_values: np.ndarray) -> np.ndarray:
        """
        Classify FIC values into 3 states.
        
        Args:
            fic_values: 1D array of FIC values (0-1 scale)
            
        Returns:
            1D array of class labels:
            - 0: Open water (FIC < 0.1)
            - 1: Transitional (0.1 ≤ FIC < 0.9)
            - 3: Ice-covered (FIC ≥ 0.9)
        """
        classes = np.zeros_like(fic_values, dtype=int)
        
        # Open water
        classes[fic_values < 0.1] = 0
        
        # Transitional
        classes[(fic_values >= 0.1) & (fic_values < 0.9)] = 1
        
        # Ice-covered (using class 3 to match calculate_transitions expectations)
        classes[fic_values >= 0.9] = 3
        
        return classes
    
    @staticmethod
    def _compute_event_metrics(obs_dates: pd.DatetimeIndex, pred_dates: pd.DatetimeIndex) -> tuple:
        """
        Compute MAE and RMSE in days between observation and prediction dates.
        
        Matches dates by year and calculates absolute difference in days.
        
        Args:
            obs_dates: DatetimeIndex of observation event dates
            pred_dates: DatetimeIndex of prediction event dates
            
        Returns:
            Tuple of results (mae, rmse)
            - mae: Mean Absolute Error in days
            - rmse: Root Mean Square Error in days
        """
        if len(obs_dates) == 0 or len(pred_dates) == 0:
            return float('nan'), 0
        
        # Match dates by year
        obs_years = {date.year: date for date in obs_dates}
        pred_years = {date.year: date for date in pred_dates}
        
        # Find common years
        common_years = set(obs_years.keys()).intersection(set(pred_years.keys()))
        
        if len(common_years) == 0:
            return float('nan'), 0
        
        # Compute absolute differences in days
        errors = []
        for year in common_years:
            obs_date = obs_years[year]
            pred_date = pred_years[year]
            
            # Calculate difference in days
            diff_days = abs((pred_date - obs_date).days)
            errors.append(diff_days)
        
        # Compute MAE
        mae = np.mean(errors)
        # Compute RMSE
        rmse = np.sqrt(np.mean(np.square(errors)))

        return float(mae), float(rmse)
