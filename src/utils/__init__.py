"""
Project Utilities Module

Provides cross-cutting utilities for the entire project including:
- Logging configuration (training and experiment tracking)
- Project-wide constants (seasons, evaluation periods, thresholds)
- Plotting functions (standardized visualization utilities)
"""

from .logging import (
    setup_training_logging,
    init_system_logger,
    init_experiment_logger,
    init_wandb_logger,
)

from .constants import (
    BREAKUP_SEASON,
    FREEZEUP_SEASON,
    EVAL_START,
    EVAL_END,
    FLAKE_ICE_THRESHOLD,
)

from .plotting import (
    plot_variable_importance_grouped,
    plot_spatial_timing_maps,
    plot_local_morans_i,
    plot_fic_temporal_evolution,
)

__all__ = [
    # Logging
    'setup_training_logging',
    'init_system_logger',
    'init_experiment_logger',
    'init_wandb_logger',
    # Constants
    'BREAKUP_SEASON',
    'FREEZEUP_SEASON',
    'EVAL_START',
    'EVAL_END',
    'FLAKE_ICE_THRESHOLD',
    # Plotting
    'plot_variable_importance_grouped',
    'plot_spatial_timing_maps',
    'plot_local_morans_i',
    'plot_fic_temporal_evolution',
]
