"""Overall evaluator for spatial binary ice cover classification.

Evaluates binary classification performance between spatial ice cover datasets
(IMS vs LIF-DL, IMS vs FLake, etc.) where 0=water, 1=ice.

Computes metrics for:
1. Full year (all data)
2. Break-up season only (March 1 - July 31)
3. Freeze-up season only (September 1 - January 31)

Metrics:
- Overall Accuracy: Fraction of correctly classified pixels
- F1-Score: Harmonic mean of precision and recall
- IoU (Jaccard Index): Intersection over Union for ice class
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from src.eval.evaluators import BaseEvaluator
from src.eval.utils import align_daily
from src.utils.constants import BREAKUP_SEASON, FREEZEUP_SEASON


class OverallEvaluator(BaseEvaluator):
    """
    Evaluator for binary classification metrics on spatial ice cover.
    
    Works with daily-resolution spatial data (IMS, LIF-DL, FLake).
    Applies lake mask to exclude land pixels and computes pixel-wise
    classification accuracy.
    
    Note: Not compatible with CIS data (non-spatial FIC timeseries).
    """
    
    name = "overall"
    requires_spatial = True  # CIS data not supported
    
    def preprocess(
        self,
        context,  # EvaluationContext
        obs_name: str,
        pred_name: str
    ) -> Dict[str, Any]:
        """
        Preprocess spatial sources for binary classification.
        
        Steps:
        1. Get observation and prediction sources from context
        2. Align them by date intersection (daily alignment)
        3. Apply lake mask to exclude land pixels
        4. Return flattened arrays ready for classification metrics
        
        Args:
            context: EvaluationContext with all sources
            obs_name: Observation source name ('ims')
            pred_name: Prediction source name ('lif_dl' or 'flake')
        
        Returns:
            Dictionary with:
            - 'obs': Observation data (2D: timesteps × valid_pixels)
            - 'pred': Prediction data (2D: timesteps × valid_pixels)
            - 'dates': DatetimeIndex of aligned dates
            - 'n_timesteps': Number of aligned timesteps
            - 'n_pixels': Number of valid lake pixels
        """
        # Get sources
        obs_source = context.get_source(obs_name)
        pred_source = context.get_source(pred_name)
        mask = context.mask
        
        # Align by date intersection
        obs_aligned, pred_aligned, common_dates = align_daily(
            obs_source, pred_source, obs_name, pred_name
        )
        
        # Apply lake mask to exclude land pixels
        valid_mask = (mask == 1)
        
        # Flatten spatial dimensions, keeping time dimension
        mask_flat = valid_mask.flatten()
        obs_flat = obs_aligned.reshape(obs_aligned.shape[0], -1)
        pred_flat = pred_aligned.reshape(pred_aligned.shape[0], -1)
        
        # Select only valid lake pixels
        obs_valid = obs_flat[:, mask_flat]
        pred_valid = pred_flat[:, mask_flat]
        
        return {
            'obs': obs_valid,
            'pred': pred_valid,
            'dates': common_dates,  # Add dates for seasonal filtering
            'n_timesteps': len(common_dates),
            'n_pixels': mask_flat.sum()
        }
    
    @staticmethod
    def _overall_accuracy(obs, pred):
        """
        Overall Accuracy: Fraction of correctly classified pixels.
        
        Formula: (TP + TN) / (TP + TN + FP + FN)
        Range: [0, 1] where 1 is perfect
        """
        
        if obs.size == 0:
            return float('nan')
        
        # Sum correct predictions across all timesteps and pixels
        correct = (obs == pred).sum()
        # Total number of pixels across all timesteps
        total = obs.size
        
        return float(correct / total)
    
    @staticmethod
    def _f1_score(obs, pred):
        """
        F1-Score for ice class (class=1).
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Where:
        - Precision = TP / (TP + FP) - Of predicted ice, how much is correct?
        - Recall = TP / (TP + FN) - Of actual ice, how much was detected?
        
        Range: [0, 1] where 1 is perfect
        """
        
        if len(obs) == 0:
            return float('nan')
        
        # Calculate confusion matrix components for ice class (1)
        tp = ((obs == 1) & (pred == 1)).sum()  # True Positive
        fp = ((obs == 0) & (pred == 1)).sum()  # False Positive
        fn = ((obs == 1) & (pred == 0)).sum()  # False Negative
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate F1
        if (precision + recall) == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return float(f1)
    
    @staticmethod
    def _iou(obs, pred):
        """
        IoU (Intersection over Union) / Jaccard Index for ice class.
        
        Formula: TP / (TP + FP + FN)
        
        Measures overlap between predicted and actual ice.
        Range: [0, 1] where 1 is perfect overlap
        """
        
        if len(obs) == 0:
            return float('nan')
        
        # Calculate for ice class (1)
        tp = ((obs == 1) & (pred == 1)).sum()  # Intersection
        fp = ((obs == 0) & (pred == 1)).sum()
        fn = ((obs == 1) & (pred == 0)).sum()
        
        union = tp + fp + fn
        
        if union == 0:
            # No ice in either observation or prediction
            return 1.0 if tp == 0 else 0.0
        
        iou = tp / union
        return float(iou)
    
    @staticmethod
    def _filter_by_season(obs, pred, dates, season_def):
        """
        Filter data to include only dates within a specified season.
        
        Args:
            obs: Observation data (2D: timesteps × pixels)
            pred: Prediction data (2D: timesteps × pixels)
            dates: DatetimeIndex of dates
            season_def: Season definition dict with keys:
                       start_month, start_day, end_month, end_day
        
        Returns:
            Tuple of (obs_filtered, pred_filtered) or (empty, empty) if no data in season
        """
        dates_pd = pd.DatetimeIndex(dates)
        
        # Create boolean mask for dates in season
        # Handle seasons that cross year boundary (e.g., Sep-Jan)
        start_month = season_def['start_month']
        start_day = season_def['start_day']
        end_month = season_def['end_month']
        end_day = season_def['end_day']
        
        if start_month <= end_month:
            # Season within same year (e.g., Mar-Jul)
            season_mask = (
                ((dates_pd.month > start_month) | 
                 ((dates_pd.month == start_month) & (dates_pd.day >= start_day))) &
                ((dates_pd.month < end_month) | 
                 ((dates_pd.month == end_month) & (dates_pd.day <= end_day)))
            )
        else:
            # Season crosses year boundary (e.g., Sep-Jan)
            season_mask = (
                ((dates_pd.month > start_month) | 
                 ((dates_pd.month == start_month) & (dates_pd.day >= start_day))) |
                ((dates_pd.month < end_month) | 
                 ((dates_pd.month == end_month) & (dates_pd.day <= end_day)))
            )
        
        # Filter data
        if season_mask.sum() == 0:
            # No dates in this season
            return np.array([]), np.array([])
        
        obs_filtered = obs[season_mask]
        pred_filtered = pred[season_mask]
        
        return obs_filtered, pred_filtered

    def evaluate(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute binary classification metrics on preprocessed data.
        
        Computes metrics for:
        1. Full year (all data) - prefix "All_"
        2. Break-up season only (Mar 1 - Jul 31) - prefix "Breakup_"
        3. Freeze-up season only (Sep 1 - Jan 31) - prefix "Freezeup_"
        
        Args:
            preprocessed: Dictionary from preprocess() with:
                - 'obs': Observation data (2D: timesteps × pixels)
                - 'pred': Prediction data (2D: timesteps × pixels)
                - 'dates': DatetimeIndex of dates
                - 'n_timesteps': Number of timesteps
                - 'n_pixels': Number of valid pixels
            
        Returns:
            Dictionary with classification metrics for each period:
            - All_Accuracy, All_F1, All_IoU
            - Breakup_Accuracy, Breakup_F1, Breakup_IoU
            - Freezeup_Accuracy, Freezeup_F1, Freezeup_IoU
            - n_timesteps, n_pixels
        """
        obs = preprocessed['obs']
        pred = preprocessed['pred']
        dates = preprocessed['dates']
        
        results = {}
        
        # 1. Full year metrics (all data)
        results['All_Accuracy'] = self._overall_accuracy(obs, pred)
        results['All_F1'] = self._f1_score(obs, pred)
        results['All_IoU'] = self._iou(obs, pred)
        
        # 2. Break-up season metrics
        obs_breakup, pred_breakup = self._filter_by_season(obs, pred, dates, BREAKUP_SEASON)
        if obs_breakup.size > 0:
            results['Breakup_Accuracy'] = self._overall_accuracy(obs_breakup, pred_breakup)
            results['Breakup_F1'] = self._f1_score(obs_breakup, pred_breakup)
            results['Breakup_IoU'] = self._iou(obs_breakup, pred_breakup)
        else:
            results['Breakup_Accuracy'] = float('nan')
            results['Breakup_F1'] = float('nan')
            results['Breakup_IoU'] = float('nan')
        
        # 3. Freeze-up season metrics
        obs_freezeup, pred_freezeup = self._filter_by_season(obs, pred, dates, FREEZEUP_SEASON)
        if obs_freezeup.size > 0:
            results['Freezeup_Accuracy'] = self._overall_accuracy(obs_freezeup, pred_freezeup)
            results['Freezeup_F1'] = self._f1_score(obs_freezeup, pred_freezeup)
            results['Freezeup_IoU'] = self._iou(obs_freezeup, pred_freezeup)
        else:
            results['Freezeup_Accuracy'] = float('nan')
            results['Freezeup_F1'] = float('nan')
            results['Freezeup_IoU'] = float('nan')
        
        # Add metadata
        results['n_timesteps'] = preprocessed['n_timesteps']
        results['n_pixels'] = preprocessed['n_pixels']
        
        return results
