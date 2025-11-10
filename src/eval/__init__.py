"""
Model Evaluation Module

This module provides tools for evaluating ice cover prediction models against
observational datasets (IMS, CIS) and baseline models (FLake).

Key components:
- utils: Phenology extraction, season classification, and data preprocessing
- engine: Data loading and orchestration
- evaluators: Individual evaluation metrics
"""

# Import constants from centralized location
from src.utils.constants import (
    BREAKUP_SEASON,
    FREEZEUP_SEASON,
    EVAL_START,
    EVAL_END,
    FLAKE_ICE_THRESHOLD,
)

# Import evaluation utilities (phenology and preprocessing)
from .utils import (
    classify_date,
    filter_dates_by_year,
    calculate_transitions,
    spatial_break_up,
    spatial_freeze_up,
    convert_to_timestamp,
    align_daily,
    convert_spatial_to_fic,
    apply_lake_mask,
)

__all__ = [
    # Constants (re-exported for convenience)
    'BREAKUP_SEASON',
    'FREEZEUP_SEASON',
    'EVAL_START',
    'EVAL_END',
    'FLAKE_ICE_THRESHOLD',
    # Phenology functions
    'classify_date',
    'filter_dates_by_year',
    'calculate_transitions',
    'spatial_break_up',
    'spatial_freeze_up',
    'convert_to_timestamp',
    # Preprocessing functions
    'align_daily',
    'convert_spatial_to_fic',
    'apply_lake_mask',
]
