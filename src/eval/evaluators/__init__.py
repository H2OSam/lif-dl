"""
Base evaluator class and shared evaluation interfaces.

Provides abstract base class that all evaluators should inherit from,
ensuring consistent API and workflow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
import xarray as xr


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    
    Defines the standard two-phase evaluation workflow:
    1. preprocess(): Transform raw sources into evaluation-ready format
    2. evaluate(): Compute metrics on preprocessed data
    
    Subclasses must implement both methods.
    
    Attributes:
        name: Evaluator identifier (used in results dict)
        requires_spatial: If True, evaluator requires spatial data (not compatible with CIS)
    """
    
    name: str = "base"  # Override in subclasses
    requires_spatial: bool = False  # Override in subclasses if spatial data required
    
    @abstractmethod
    def preprocess(
        self,
        context: 'EvaluationContext',
        obs_name: str,
        pred_name: str
    ) -> Dict[str, Any]:
        """
        Preprocess raw data sources for evaluation.
        
        This method has access to ALL sources in the context, allowing
        evaluators to use multiple sources for preprocessing (e.g.,
        SpatialEvaluator can access IMS, LIF-DL, and FLake for unified
        data range computation).
        
        Args:
            context: EvaluationContext with all sources, mask, and metadata
            obs_name: Name of observation source (e.g., 'ims', 'cis')
            pred_name: Name of prediction source (e.g., 'lif_dl', 'flake')
        
        Returns:
            Dictionary with preprocessed data needed for evaluate().
            Structure is evaluator-specific, but commonly includes:
            - 'obs': Processed observation data
            - 'pred': Processed prediction data
            - 'mask': Relevant mask (optional)
            - Any other evaluator-specific preprocessed data
        
        Raises:
            ValueError: If required sources are not available
            KeyError: If source names are invalid
        """
        raise NotImplementedError("Subclasses must implement preprocess()")
    
    @abstractmethod
    def evaluate(
        self,
        preprocessed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute evaluation metrics on preprocessed data.
        
        Args:
            preprocessed: Dictionary from preprocess() method
        
        Returns:
            Dictionary of metrics. Keys should be metric names,
            values should be numeric (float/int) or simple types.
            
            Example:
            {
                "RMSE": 0.15,
                "R2": 0.82,
                "n_samples": 365
            }
        
        Raises:
            ValueError: If preprocessed data is invalid
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
    
    def run(
        self,
        context: 'EvaluationContext',
        obs_name: str,
        pred_name: str
    ) -> Dict[str, Any]:
        """
        Convenience method to run full evaluation workflow.
        
        Calls preprocess() then evaluate() in sequence.
        
        Args:
            context: EvaluationContext with sources
            obs_name: Observation source name
            pred_name: Prediction source name
        
        Returns:
            Dictionary of evaluation metrics from evaluate()
        """
        preprocessed = self.preprocess(context, obs_name, pred_name)
        metrics = self.evaluate(preprocessed)
        return metrics
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
