"""
Project-Wide Constants

This module contains all standardized constants used throughout the project,
including season definitions, evaluation periods, and thresholds.
"""

DATA_DIR = "data/"
NC_DATA_DIR = DATA_DIR + "nc/"
CSV_DATA_DIR = DATA_DIR + "csv/"

BREAKUP_SEASON = {
    'start_month': 4,   # April 1
    'start_day': 1,
    'end_month': 7,     # July 31
    'end_day': 31
}

FREEZEUP_SEASON = {
    'start_month': 9,    # September 1
    'start_day': 1,
    'end_month': 12,     # December 31
    'end_day': 31
}

# Global date range for evaluation (test period)
EVAL_START = "2018-01-01"
EVAL_END = "2021-12-31"

# FLake ice thickness threshold (meters) - below this is considered water
FLAKE_ICE_THRESHOLD = 0.001  # 1mm

def get_season(date):
    """Determine the season (breakup or freezeup) for a given date."""
    month = date.month

    if BREAKUP_SEASON['start_month'] <= month <= BREAKUP_SEASON['end_month']:
        return 'breakup'

    if FREEZEUP_SEASON['end_month'] < FREEZEUP_SEASON['start_month']:
        # Season spans year-end
        if month >= FREEZEUP_SEASON['start_month'] or month <= FREEZEUP_SEASON['end_month']:
            return 'freezeup'
    else:
        if FREEZEUP_SEASON['start_month'] <= month <= FREEZEUP_SEASON['end_month']:
            return 'freezeup'

    return 'unknown'
