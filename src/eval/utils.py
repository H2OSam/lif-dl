"""
Utility Functions for Model Evaluation

This module contains functions for:
- Data alignment and preprocessing
- Ice phenology date extraction
- Spatial phenology analysis
- Season classification
"""

from typing import Tuple
import numpy as np
import numpy.ma as ma
import pandas as pd
from pandas import date_range
import xarray as xr

# Import constants from centralized location
from src.utils.constants import (
    BREAKUP_SEASON,
    FREEZEUP_SEASON,
    EVAL_START,
    EVAL_END,
    FLAKE_ICE_THRESHOLD,
)

##################################################################################
# Function to classify each sample of dataset as break-up, freeze-up or neither
# Updated to use standardized season definitions
##################################################################################

def classify_date(date):
    """
    Classify each date as one of:
        0: Not freeze-up and not break-up
        1: Freeze-up
        2: Break-up
    
    Classification uses standardized season definitions:
    - Freeze-up: September 1 to January 31 (next year)
    - Break-up: March 1 to July 31
    
    Parameters:
    - date: pandas Timestamp or datetime object
    
    Returns:
    - int: 0 (neither), 1 (freeze-up), 2 (break-up)
    """
    # Freeze-up season (Sep 1 to Jan 31)
    freeze_start = (FREEZEUP_SEASON['start_month'], FREEZEUP_SEASON['start_day'])
    # Month 13 used for January of next year (makes comparison work properly)
    freeze_end = (FREEZEUP_SEASON['end_month'] + 12, FREEZEUP_SEASON['end_day'])
    
    # Break-up season (Mar 1 to Jul 31)
    break_start = (BREAKUP_SEASON['start_month'], BREAKUP_SEASON['start_day'])
    break_end = (BREAKUP_SEASON['end_month'], BREAKUP_SEASON['end_day'])
    
    # Convert date to (month, day), adjusting for cross-year freeze-up
    m, d = date.month, date.day
    # For freeze-up season comparison, adjust months after December
    m_adjusted = m if m >= freeze_start[0] else m + 12
    
    if freeze_start <= (m_adjusted, d) and (m_adjusted, d) <= freeze_end:
        return 1  # Freeze-up class
    elif break_start <= (m, d) and (m, d) <= break_end:
        return 2  # Break-up class
    else:
        return 0  # Neither


def filter_dates_by_year(dates, season, keep='earliest'):
    """
    Filter dates to keep only one instance per year within the specified season.
    
    Uses standardized season definitions:
    - Break-up: March 1 to July 31
    - Freeze-up: September 1 to January 31 (next year)
    
    Parameters:
    - dates: pandas DatetimeIndex of dates
    - season: 'breakup' or 'freezeup'
    - keep: 'earliest' or 'latest', whether to keep the earliest or latest date per year
    
    Returns:
    - filtered_dates: pandas DatetimeIndex of filtered dates
    """
    # Extract year range
    years = dates.year
    years = range(min(years), max(years) + 1)
    filtered_dates = []
    
    for year in years:
        # Define season date range using standardized definitions
        if season == 'breakup':
            # Break-up: March 1 to July 31 of same year
            _start = pd.to_datetime(f"{year}-{BREAKUP_SEASON['start_month']:02d}-{BREAKUP_SEASON['start_day']:02d}")
            _end = pd.to_datetime(f"{year}-{BREAKUP_SEASON['end_month']:02d}-{BREAKUP_SEASON['end_day']:02d}")
            
        elif season == 'freezeup':
            # Freeze-up: September 1 (year) to January 31 (year+1)
            _start = pd.to_datetime(f"{year}-{FREEZEUP_SEASON['start_month']:02d}-{FREEZEUP_SEASON['start_day']:02d}")
            _end = pd.to_datetime(f"{year+1}-{FREEZEUP_SEASON['end_month']:02d}-{FREEZEUP_SEASON['end_day']:02d}")
            
        else:
            raise ValueError("Parameter 'season' must be either 'breakup' or 'freezeup'.")
        
        # Filter dates within the season range
        in_range = dates[(dates >= _start) & (dates <= _end)]
        if len(in_range) == 0:
            continue
        
        # Keep earliest or latest date for this year
        if keep == 'earliest':
            filtered_dates.append(in_range[np.argmin(in_range)])
        elif keep == 'latest':
            filtered_dates.append(in_range[np.argmax(in_range)])
        else:
            raise ValueError("Parameter 'keep' must be either 'earliest' or 'latest'.")
    
    return pd.DatetimeIndex(filtered_dates)


##################################################################################
# Functions for calculating ice phenology dates from series of ice cover data
##################################################################################

def convert_to_timestamp(dates):
    """Convert dates to pandas Timestamp if they are not already."""
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(pd.to_datetime(dates))
    return dates


def calculate_transitions(pred_classes, dates):
    """
    Calculate transition start and end points for breakup and freezeup events.
    
    This function uses a multi-step process:
    1. Each date's ice cover is classified into classes based on fraction:
       - Class 0: [0.0-0.1] (mostly water)
       - Class 1: [0.1-0.9] (partial ice)
       - Class 2: [0.9-1.0] (mostly ice)
    2. Compute np.diff() on the class timeseries
    3. Classify each date as a transition type based on the diff value
    4. Extract earliest date for each transition type of interest
    
    Parameters:
    - pred_classes: numpy array of predicted classes (0, 1, 2)
    - dates: list or numpy array of dates corresponding to pred_classes
    
    Returns:
    - dict with keys:
        - "Breakup_start": DatetimeIndex of First Open Water (FOW) dates
        - "Breakup_end": DatetimeIndex of Water Clear of Ice (WCI) dates
        - "Freezeup_start": DatetimeIndex of Freeze Onset (FO) dates
        - "Freezeup_end": DatetimeIndex of Continuous Ice Cover (CIC) dates
    """
    # Convert dates to Timestamp
    dates = convert_to_timestamp(dates)
    
    # Calculate transitions using np.diff
    # Transitions indicate changes between ice cover classes
    diff = np.diff(pred_classes)
    
    # Identify transition types:
    # Breakup transitions (ice → water):
    #   - diff == -2: class 2→0 (full ice to open water)
    #   - diff == -3: This shouldn't occur normally (would be class 3→0)
    # Freeze-up transitions (water → ice):
    #   - diff == 1: class 0→1 or 1→2 (water to partial, or partial to full)
    #   - diff == 2: class 0→2 (open water to full ice)
    #   - diff == 3: This shouldn't occur normally
    
    breakup_start = list(dates[np.where((diff == -2) | (diff == -3))[0] + 1])  # FOW
    breakup_end = list(dates[np.where((diff == -1) | (diff == -3))[0] + 1])    # WCI
    freezeup_start = list(dates[np.where((diff == 1) | (diff == 3))[0] + 1])   # FO
    freezeup_end = list(dates[np.where((diff == 2) | (diff == 3))[0] + 1])     # CIC
    
    # Convert transition dates to Timestamp
    breakup_start = convert_to_timestamp(breakup_start)
    breakup_end = convert_to_timestamp(breakup_end)
    freezeup_start = convert_to_timestamp(freezeup_start)
    freezeup_end = convert_to_timestamp(freezeup_end)
    
    # Filter to get one date per year using standardized seasons
    breakup_start = filter_dates_by_year(breakup_start, season='breakup', keep='earliest')
    breakup_end = filter_dates_by_year(breakup_end, season='breakup', keep='latest')
    freezeup_start = filter_dates_by_year(freezeup_start, season='freezeup', keep='earliest')
    freezeup_end = filter_dates_by_year(freezeup_end, season='freezeup', keep='latest')
    
    # Sort in ascending order
    breakup_start = breakup_start.sort_values()
    breakup_end = breakup_end.sort_values()
    freezeup_start = freezeup_start.sort_values()
    freezeup_end = freezeup_end.sort_values()
    
    return {
        "Breakup_start": breakup_start,   # First Open Water (FOW)
        "Breakup_end": breakup_end,       # Water Clear of Ice (WCI)
        "Freezeup_start": freezeup_start, # Freeze Onset (FO)
        "Freezeup_end": freezeup_end      # Continuous Ice Cover (CIC)
    }


##################################################################################
# Functions for calculating spatial phenology maps
# Used for spatial analysis and metrics (SSIM, Kendall, Moran's I)
##################################################################################

def spatial_break_up(ice_cover, mask, offset=0):
    """
    Calculate spatial maps of break-up phenology dates.
    
    For each pixel, identifies:
    - First Open Water (FOW): First date when ice_cover == 0 (any water)
    - Water Clear of Ice (WCI): Last date when ice_cover transitions from ice to water
    
    Parameters:
    - ice_cover: numpy array of shape (time, height, width) with binary ice cover
                 (1 = ice, 0 = water)
    - mask: numpy array of shape (height, width) indicating valid lake pixels
            (1 = valid, 0 = land/invalid)
    - offset: integer offset to add to resulting phenology day-of-year values
    
    Returns:
    - first_open_water: 2D array (height, width) of FOW day-of-year
    - water_clear_of_ice: 2D array (height, width) of WCI day-of-year
    
    Pixels with no transitions are set to NaN.
    """
    # Create masked array to handle land pixels
    ice_cover = ma.array(ice_cover, mask=1 - np.tile(mask, (ice_cover.shape[0], 1, 1)))
    N = ice_cover.shape[0]
    
    # Convert to binary: 1 where water is present (ice_cover == 0)
    any_water = ice_cover == 0
    
    # First Open Water: first date with any water (argmax finds first True)
    first_open_water = ma.array(np.argmax(any_water, axis=0), mask=1 - mask)
    
    # Water Clear of Ice: last date with water (search from end)
    water_clear_of_ice = ma.array(np.argmax(1 - any_water[::-1], axis=0), mask=1 - mask)
    water_clear_of_ice = N - water_clear_of_ice
    
    # Handle pixels with no water during the period (set to NaN)
    any_water_mask = np.any(any_water, axis=0)
    first_open_water = np.where(any_water_mask, first_open_water, np.nan) + offset
    water_clear_of_ice = np.where(any_water_mask, water_clear_of_ice, np.nan) + offset
    
    return first_open_water, water_clear_of_ice


def spatial_freeze_up(ice_cover, mask, offset=0):
    """
    Calculate spatial maps of freeze-up phenology dates.
    
    For each pixel, identifies:
    - Freeze Onset (FO): First date when ice_cover == 1 (any ice)
    - Continuous Ice Cover (CIC): Last date when ice_cover transitions from water to ice
    
    Parameters:
    - ice_cover: numpy array of shape (time, height, width) with binary ice cover
                 (1 = ice, 0 = water)
    - mask: numpy array of shape (height, width) indicating valid lake pixels
            (1 = valid, 0 = land/invalid)
    - offset: integer offset to add to resulting phenology day-of-year values
    
    Returns:
    - freeze_onset: 2D array (height, width) of FO day-of-year
    - continuous_ice_cover: 2D array (height, width) of CIC day-of-year
    
    Pixels with no transitions are set to NaN.
    """
    # Create masked array to handle land pixels
    ice_cover = ma.array(ice_cover, mask=1 - np.tile(mask, (ice_cover.shape[0], 1, 1)))
    N = ice_cover.shape[0]
    
    # Convert to binary: 1 where ice is present (ice_cover == 1)
    any_ice = ice_cover == 1
    
    # Freeze Onset: first date with any ice (argmax finds first True)
    freeze_onset = ma.array(np.argmax(any_ice, axis=0), mask=1 - mask)
    
    # Continuous Ice Cover: last date with ice (search from end)
    continuous_ice_cover = ma.array(np.argmax(1 - any_ice[::-1], axis=0), mask=1 - mask)
    continuous_ice_cover = N - continuous_ice_cover
    
    # Handle pixels with no ice during the period (set to NaN)
    any_ice_mask = np.any(any_ice, axis=0)
    freeze_onset = np.where(any_ice_mask, freeze_onset, np.nan) + offset
    continuous_ice_cover = np.where(any_ice_mask, continuous_ice_cover, np.nan) + offset
    
    return freeze_onset, continuous_ice_cover


##################################################################################
# Data Alignment and Preprocessing Functions
##################################################################################

def align_daily(
    source1: xr.DataArray,
    source2: xr.DataArray,
    source1_name: str = "source1",
    source2_name: str = "source2"
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Align two data sources by intersecting dates.
    
    Args:
        source1: First xarray DataArray with 'time' dimension
        source2: Second xarray DataArray with 'time' dimension
        source1_name: Name for logging (default "source1")
        source2_name: Name for logging (default "source2")
    
    Returns:
        Tuple of (source1_aligned, source2_aligned, common_dates):
        - source1_aligned: numpy array aligned to common dates
        - source2_aligned: numpy array aligned to common dates
        - common_dates: DatetimeIndex of intersection
    
    Raises:
        ValueError: If no overlapping dates found
    """
    # Get time coordinates
    time1 = pd.DatetimeIndex(source1.time.values)
    time2 = pd.DatetimeIndex(source2.time.values)
    
    # Find intersection
    common_dates = time1.intersection(time2)
    
    if len(common_dates) == 0:
        raise ValueError(
            f"No overlapping dates between {source1_name} and {source2_name}\n"
            f"{source1_name} range: {time1.min()} to {time1.max()}\n"
            f"{source2_name} range: {time2.min()} to {time2.max()}"
        )
    
    # Select common dates
    source1_aligned = source1.sel(time=common_dates).values
    source2_aligned = source2.sel(time=common_dates).values
    
    return source1_aligned, source2_aligned, common_dates

def convert_spatial_to_fic(
    spatial_data: xr.DataArray,
    mask: np.ndarray
) -> xr.DataArray:
    """
    Convert spatial binary ice cover to Fraction Ice Cover (FIC) timeseries.
    
    Args:
        spatial_data: 3D DataArray (time, height, width) with binary ice cover
                     (0=water, 1=ice, other values masked as land)
        mask: 2D array (height, width) with lake mask (1=lake, 0=land)
    
    Returns:
        1D DataArray of FIC values (fraction of lake covered by ice) for each timestep
        with time dimension preserved from input
    """
    # Get spatial values
    spatial_values = spatial_data.values
    n_timesteps = spatial_values.shape[0]
    fic_timeseries = np.zeros(n_timesteps)
    
    for t in range(n_timesteps):
        # Get lake pixels only
        lake_pixels = (mask == 1)
        
        if lake_pixels.sum() == 0:
            fic_timeseries[t] = np.nan
            continue
        
        # Calculate mean ice cover over lake pixels
        ice_data = spatial_values[t][lake_pixels]
        fic_timeseries[t] = ice_data.mean()
    
    # Create DataArray with time dimension
    fic_da = xr.DataArray(
        data=fic_timeseries,
        dims=['time'],
        coords={'time': spatial_data.time},
        attrs={
            'source': spatial_data.attrs.get('source', 'unknown'),
            'site': spatial_data.attrs.get('site', 'unknown'),
            'units': 'fraction',
            'long_name': 'Fraction Ice Cover',
            'derived_from': 'spatial binary ice cover'
        }
    )
    
    return fic_da


def apply_lake_mask(
    spatial_data: np.ndarray,
    mask: np.ndarray,
    exclude_values: list = None
) -> np.ndarray:
    """
    Apply lake mask to spatial data and optionally exclude specific values.
    
    Args:
        spatial_data: 3D array (time, height, width) or 2D array (height, width)
        mask: 2D array (height, width) with lake mask (1=lake, 0=land)
        exclude_values: List of values to treat as invalid (e.g., land encoding)
                       For binary ice: exclude_values=[2, -1] to remove land
    
    Returns:
        2D array (time, valid_pixels) if input is 3D, or
        1D array (valid_pixels) if input is 2D
        
        Only includes pixels where mask==1 and value not in exclude_values
    """
    is_3d = spatial_data.ndim == 3
    
    # Create valid mask (lake pixels)
    valid_mask = (mask == 1)
    
    # Exclude specific values if requested
    if exclude_values is not None:
        if is_3d:
            # For 3D data, check across time dimension
            for val in exclude_values:
                valid_mask = valid_mask & ~np.any(spatial_data == val, axis=0)
        else:
            # For 2D data, check directly
            for val in exclude_values:
                valid_mask = valid_mask & (spatial_data != val)
    
    # Apply mask
    if is_3d:
        # Flatten spatial dimensions, keep time
        mask_flat = valid_mask.flatten()
        data_flat = spatial_data.reshape(spatial_data.shape[0], -1)
        data_masked = data_flat[:, mask_flat]
    else:
        # Just mask the 2D array
        data_masked = spatial_data[valid_mask]
    
    return data_masked
