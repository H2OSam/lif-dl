"""
Evaluation Engine and DataLoader for Ice Cover Model Evaluation

This module provides the infrastructure for loading and aligning multiple data sources
(Model predictions, IMS observations, FLake baseline, CIS observations) and orchestrating
evaluation metrics across these sources.

Key Components:
- EvaluationContext: Standardized container for evaluation inputs
- BaseDataLoader: Handles loading of raw data sources with minimal preprocessing
- ValidationEngine: Orchestrates evaluators and collects results
"""

import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import xarray as xr
from src.utils.constants import EVAL_START, EVAL_END, FLAKE_ICE_THRESHOLD


class EvaluationContext:
    """
    Standardized container for evaluation inputs.
    
    Provides a clean interface for passing data to evaluators, ensuring
    all evaluators receive consistent inputs.
    
    Attributes:
        sources: Dict mapping source names to xarray Datasets
                 Keys: 'ims', 'lif_dl', 'flake', 'cis'
        site: Site name (e.g., 'great_slave_lake')
        mask: 2D numpy array - lake mask (1=lake, 0=land)
        metadata: Optional dict with additional context
    """
    
    def __init__(
        self,
        sources: Dict[str, xr.Dataset],
        site: str,
        mask: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize EvaluationContext.
        
        Args:
            sources: Dict of xarray Datasets for each data source
            site: Site name
            mask: Lake mask (2D array)
            metadata: Optional metadata dict
        """
        self.sources = sources
        self.site = site
        self.mask = mask
        self.metadata = metadata or {}
    
    def get_source(self, name: str) -> xr.Dataset:
        """Get a specific data source by name."""
        if name not in self.sources:
            raise KeyError(f"Source '{name}' not found. Available: {list(self.sources.keys())}")
        return self.sources[name]
    
    def has_source(self, name: str) -> bool:
        """Check if a source is available."""
        return name in self.sources
    
    def __repr__(self):
        return (f"EvaluationContext(site='{self.site}', "
                f"sources={list(self.sources.keys())}, "
                f"mask_shape={self.mask.shape})")


class DataLoader:
    """
    Loads the various data sources required for evaluation.
    Applies minimal preprocessing to prepare raw data for evaluators, which 
    handle detailed alignment and processing as needed.
    """
    
    def __init__(
        self,
        model_dir: str,
        data_dir: str = "data/nc",
        cis_dir: str = "data/cis"
    ):
        """
        Initialize DataLoader with paths to data sources.
        
        Args:
            model_dir: Directory containing model forecast NetCDF files
                      (e.g., "results/LIF_DL_Best/forecasts/")
            data_dir: Directory containing IMS and FLake NetCDF files
                     (default: "data/nc")
            cis_dir: Directory containing CIS CSV files
                    (default: "data/cis")
        """
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.cis_dir = Path(cis_dir)
        
        # Separate caches for each data source
        self._lif_dl_cache = {}
        self._flake_cache = {}
        self._ims_cache = {}
        self._cis_cache = {}
        self._mask_cache = {}
        
    def load_lif_dl(self, site: str) -> xr.DataArray:
        """
        Load LIF-DL model forecast for a site.
        
        Preprocessing:
        - Crops to global evaluation date range
        - Extracts ice cover prediction variable
        
        Args:
            site: Site name (e.g., "great_slave_lake")
            
        Returns:
            xarray DataArray with ice cover predictions (time, height, width)
            
        Raises:
            FileNotFoundError: If forecast file doesn't exist
        """
        if site in self._lif_dl_cache:
            return self._lif_dl_cache[site]
        
        # Try both locations: model_dir/forecasts/ and model_dir/
        forecast_path = self.model_dir / "forecasts" / f"{site}_forecast.nc"
        if not forecast_path.exists():
            forecast_path = self.model_dir / f"{site}_forecast.nc"
        
        if not forecast_path.exists():
            raise FileNotFoundError(
                f"Model forecast not found. Tried:\n"
                f"  - {self.model_dir / 'forecasts' / f'{site}_forecast.nc'}\n"
                f"  - {self.model_dir / f'{site}_forecast.nc'}"
            )
        
        ds = xr.open_dataset(forecast_path)
        
        # Crop to global date range
        ds = ds.sel(time=slice(EVAL_START, EVAL_END))
        
        # Extract ice cover prediction variable
        if "ice_cover_prediction" in ds.variables:
            data = ds["ice_cover_prediction"]
        elif "prediction" in ds.variables:
            data = ds["prediction"]
        elif "LIF_DL" in ds.variables:
            data = ds["LIF_DL"]
        else:
            raise ValueError(f"No prediction variable found in model forecast for {site}")
        
        self._lif_dl_cache[site] = data
        return data
    
    def load_flake(self, site: str) -> xr.DataArray:
        """
        Load FLake baseline ice cover for a site.
        
        Preprocessing:
        - Crops to global evaluation date range
        - Converts ice thickness to binary ice cover using threshold
        
        Args:
            site: Site name (e.g., "great_slave_lake")
            
        Returns:
            xarray DataArray with binary ice cover (time, height, width)
            Values: 1=ice, 0=water
            
        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        if site in self._flake_cache:
            return self._flake_cache[site]
        
        data_path = self.data_dir / f"{site}.nc"
        if not data_path.exists():
            raise FileNotFoundError(
                f"FLake data not found: {data_path}\n"
                f"Expected format: {self.data_dir}/<site>.nc"
            )
        
        ds = xr.open_dataset(data_path)
        
        # Crop to global date range
        ds = ds.sel(time=slice(EVAL_START, EVAL_END))
        
        # Extract FLake ice thickness
        if "flake_ice_depth" not in ds.variables:
            raise ValueError(
                f"Required variable 'flake_ice_depth' not found in {site}.nc\n"
                f"Available variables: {list(ds.variables)}"
            )
        
        flake_thickness = ds["flake_ice_depth"]
        
        # Convert thickness to binary ice cover (preprocessing)
        flake_binary = (flake_thickness >= FLAKE_ICE_THRESHOLD).astype(np.float32)
        
        self._flake_cache[site] = flake_binary
        return flake_binary
    
    def load_ims(self, site: str) -> xr.DataArray:
        """
        Load IMS observations for a site.
        
        Preprocessing:
        - Crops to global evaluation date range
        
        Args:
            site: Site name (e.g., "great_slave_lake")
            
        Returns:
            xarray DataArray with ice cover observations (time, height, width)
            
        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        if site in self._ims_cache:
            return self._ims_cache[site]
        
        data_path = self.data_dir / f"{site}.nc"
        if not data_path.exists():
            raise FileNotFoundError(
                f"IMS data not found: {data_path}\n"
                f"Expected format: {self.data_dir}/<site>.nc"
            )
        
        ds = xr.open_dataset(data_path)
        
        # Crop to global date range
        ds = ds.sel(time=slice(EVAL_START, EVAL_END))
        
        # Extract IMS ice cover
        if "IMS_Surface_Values" not in ds.variables:
            raise ValueError(
                f"Required variable 'IMS_Surface_Values' not found in {site}.nc\n"
                f"Available variables: {list(ds.variables)}"
            )
        
        ims_data = ds["IMS_Surface_Values"]
        
        self._ims_cache[site] = ims_data
        return ims_data
    
    def load_cis(self, site: str) -> xr.DataArray:
        """
        Load CIS observations from CSV file and convert to DataArray.
        
        Preprocessing:
        - Converts ice concentration from 0-10 scale to 0-1 scale
        - Crops to global evaluation date range
        - Converts to xarray DataArray with time dimension
        
        CIS data format:
        - Columns: Date, Ice-covered
        - Ice-covered: 0-10 scale (tenths of ice coverage)
        - Temporal resolution: Weekly
        
        Args:
            site: Site name (e.g., "great_slave_lake")
            
        Returns:
            xarray DataArray with:
            - Dimension: time
            - Values: FIC in 0-1 scale (fraction)
            - Attributes: source, site, units
            
        Raises:
            FileNotFoundError: If CIS file doesn't exist
        """
        if site in self._cis_cache:
            return self._cis_cache[site]
        
        cis_path = self.cis_dir / f"{site}.csv"
        if not cis_path.exists():
            raise FileNotFoundError(
                f"CIS data not found: {cis_path}\n"
                f"Expected format: {self.cis_dir}/<site>.csv"
            )
        
        df = pd.read_csv(cis_path, parse_dates=['Date'])
        
        # Preprocessing: Convert from 0-10 scale to 0-1 scale (fraction ice cover)
        df['Ice-covered'] = df['Ice-covered'] / 10.0
        
        # Crop to global date range
        eval_start_pd = pd.to_datetime(EVAL_START)
        eval_end_pd = pd.to_datetime(EVAL_END)
        df = df[(df['Date'] >= eval_start_pd) & (df['Date'] <= eval_end_pd)].copy()
        
        # Convert to xarray DataArray with time dimension
        da = xr.DataArray(
            data=df['Ice-covered'].values,
            dims=['time'],
            coords={'time': pd.DatetimeIndex(df['Date'].values)},
            attrs={
                'source': 'CIS',
                'site': site,
                'units': 'fraction',
                'long_name': 'Fraction Ice Cover',
                'temporal_resolution': 'weekly'
            }
        )
        
        self._cis_cache[site] = da
        return da
    
    def load_lake_mask(self, site: str) -> np.ndarray:
        """
        Load lake mask for a site.
        
        Args:
            site: Site name
            
        Returns:
            2D numpy array (height, width) with 1=lake, 0=land
            
        Raises:
            ValueError: If lake_mask not found in data file
        """
        if site in self._mask_cache:
            return self._mask_cache[site]
        
        data_path = self.data_dir / f"{site}.nc"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_path}\n"
                f"Expected format: {self.data_dir}/<site>.nc"
            )
        
        ds = xr.open_dataset(data_path)
        
        if 'lake_mask' not in ds.variables:
            raise ValueError(
                f"No 'lake_mask' variable found in {site}.nc\n"
                f"Available variables: {list(ds.variables)}"
            )
        
        mask = ds['lake_mask'].values
        self._mask_cache[site] = mask
        return mask
    
    def load_all_sources(
        self,
        site: str,
    ) -> Dict[str, xr.DataArray]:
        """
        Load all available data sources for a site.
        
        Loads raw sources with minimal preprocessing, returning xarray objects for evaluators to
        process as needed.
        
        Args:
            site: Site name (e.g., "great_slave_lake")
        
        Returns:
            Dictionary with keys:
            - 'ims': IMS observations (xarray DataArray)
            - 'lif_dl': LIF-DL predictions (xarray DataArray)
            - 'flake': FLake predictions (xarray DataArray)
            - 'cis': CIS observations (xarray DataArray)
            
        Note:
            All sources are cropped to evaluation period and have minimal
            preprocessing applied (threshold for FLake, scale conversion for CIS).
            No temporal alignment is performed - evaluators handle that.
        """
        sources = {}
        
        # Load sources
        sources['ims'] = self.load_ims(site)
        sources['lif_dl'] = self.load_lif_dl(site)
        sources['flake'] = self.load_flake(site)
        try:
            sources['cis'] = self.load_cis(site)
        except FileNotFoundError:
            sources['cis'] = None
        
        return sources
    
    def load_baseline_year(
        self,
        site: str,
        baseline_year: str = "2017"
    ) -> Dict[str, Any]:
        """
        Load baseline year data for calculating naive predictors.
        
        Loads one year of data before evaluation period to compute baselines like:
        - Phenology: "Next year's event = this year's event"
        - FIC MASE: "Next year's FIC = this year's FIC"
        
        Args:
            site: Site name (e.g., "great_slave_lake")
            baseline_year: Year to load (default "2017" for 2018-2021 eval period)
        
        Returns:
            Dictionary with keys:
            - 'ims': IMS data for baseline year (if available)
            - 'cis': CIS data for baseline year (if available)
            
        Note:
            Only loads observation sources (IMS, CIS, FLake) for baseline.
            Model predictions not needed for baseline calculation.
        """
        baseline_sources = {}
        
        baseline_start = f"{baseline_year}-01-01"
        baseline_end = f"{baseline_year}-12-31"
        
        try:
            # Load IMS for baseline year
            ims_ds = xr.open_dataset(self.data_dir / f"{site}.nc")
            ims_baseline = ims_ds['IMS_Surface_Values'].sel(time=slice(baseline_start, baseline_end))
            if len(ims_baseline.time) > 0:
                baseline_sources['ims'] = ims_baseline
        except (FileNotFoundError, KeyError):
            pass
        
        try:
            # Load CIS for baseline year
            cis_path = self.cis_dir / f"{site}.csv"
            if cis_path.exists():
                df = pd.read_csv(cis_path, parse_dates=['Date'])
                df['Ice-covered'] = df['Ice-covered'] / 10.0  # Convert to 0-1 scale
                
                baseline_start_pd = pd.to_datetime(baseline_start)
                baseline_end_pd = pd.to_datetime(baseline_end)
                df_baseline = df[(df['Date'] >= baseline_start_pd) & (df['Date'] <= baseline_end_pd)].copy()
                
                if len(df_baseline) > 0:
                    
                    # Convert to xarray DataArray with time dimension
                    da_baseline = xr.DataArray(
                        data=df_baseline['Ice-covered'].values,
                        dims=['time'],
                        coords={'time': pd.DatetimeIndex(df_baseline['Date'].values)},
                        attrs={
                            'source': 'CIS',
                            'site': site,
                            'units': 'fraction',
                            'long_name': 'Fraction Ice Cover',
                            'temporal_resolution': 'weekly'
                        }
                    )
                    
                    baseline_sources['cis'] = da_baseline
        except (FileNotFoundError, KeyError):
            pass
        
        return baseline_sources


class ValidationEngine:
    """
    Orchestrates evaluation across multiple data sources and evaluators.
    
    New Architecture (Evaluator-Driven Preprocessing):
    1. Use DataLoader.load_all_sources() to load all sources once per site
    2. Create EvaluationContext with all sources
    3. Each evaluator accesses needed sources and performs own preprocessing
    4. Collect results and handle errors gracefully
    5. Return structured results dictionary
    
    Workflow:
    - load_all_sources() → EvaluationContext → evaluator.run() → results
    
    Benefits:
    - Load sources once, use for all comparisons
    - Evaluators control their own preprocessing
    - SpatialEvaluator can access all 3 sources for unified scaling
    """
    
    def __init__(self, evaluators: List[object]):
        """
        Initialize ValidationEngine with evaluators.
        
        Args:
            evaluators: List of evaluator objects. Each must inherit from
                       BaseEvaluator and implement:
                - name: str attribute
                - preprocess(context, obs_name, pred_name): method
                - evaluate(preprocessed): method
                - run(context, obs_name, pred_name): convenience method
        """
        self.evaluators = evaluators
    
    def run(
        self,
        context: 'EvaluationContext',
        comparisons: List[Tuple[str, str]],
        evaluator_filter: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run evaluators on provided context for specified comparisons.
        
        Uses new architecture where evaluators access all sources via context
        and perform their own preprocessing.
        
        Args:
            context: EvaluationContext with all sources loaded
            comparisons: List of (obs_name, pred_name) tuples to evaluate.
                        Examples: [("ims", "lif_dl"), ("cis", "lif_dl")]
            evaluator_filter: Optional list of evaluator names to run.
                             If None, runs all evaluators.
        
        Returns:
            Dictionary mapping comparison -> evaluator_name -> result dict:
            {
                "ims_vs_lif_dl": {
                    "overall": {
                        "status": "ok",
                        "metrics": {...}
                    },
                    "fic": {...}
                }
            }
        """
        all_results = {}
        
        # Run each comparison
        for obs_name, pred_name in comparisons:
            comp_label = f"{obs_name}_vs_{pred_name}"
            comp_results = {}
            
            # Check if sources exist in context
            if not context.has_source(obs_name):
                comp_results["error"] = {
                    "status": "error",
                    "error": f"Observation source '{obs_name}' not available",
                    "metrics": None
                }
                all_results[comp_label] = comp_results
                continue
            
            if not context.has_source(pred_name):
                comp_results["error"] = {
                    "status": "error",
                    "error": f"Prediction source '{pred_name}' not available",
                    "metrics": None
                }
                all_results[comp_label] = comp_results
                continue
            
            # Run each evaluator
            for evaluator in self.evaluators:
                # Check filter
                if evaluator_filter is not None:
                    if evaluator.name not in evaluator_filter:
                        continue
                
                # Check if evaluator requires spatial data but obs is non-spatial (CIS)
                if evaluator.requires_spatial and obs_name == 'cis':
                    comp_results[evaluator.name] = {
                        "status": "skipped",
                        "metrics": None,
                        "error": "Not applicable for non-spatial CIS data"
                    }
                    continue
                
                try:
                    # Call evaluator's run method (preprocess + evaluate)
                    metrics = evaluator.run(context, obs_name, pred_name)
                    
                    comp_results[evaluator.name] = {
                        "status": "ok",
                        "metrics": metrics,
                        "error": None
                    }
                    
                except Exception as e:
                    # Catch evaluator errors gracefully
                    comp_results[evaluator.name] = {
                        "status": "error",
                        "metrics": None,
                        "error": str(e)
                    }
                    warnings.warn(
                        f"Evaluator '{evaluator.name}' failed for {comp_label} "
                        f"at site {context.site}: {e}"
                    )
            
            all_results[comp_label] = comp_results
        
        return all_results
    
    def run_full_evaluation(
        self,
        model_dir: str,
        sites: List[str],
        comparisons: Optional[List[Tuple[str, str]]] = None,
        data_dir: str = "data/nc/",
        cis_dir: str = "data/cis",
        evaluator_filter: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run complete evaluation across multiple sites and comparison types.
        
        Uses new architecture where all sources are loaded once per site,
        then evaluators access what they need via EvaluationContext.
        
        Args:
            model_dir: Directory containing model forecasts
            sites: List of site names to evaluate
            comparisons: List of (obs_name, pred_name) tuples to compare.
                        If None, runs all standard comparisons.
                        Examples: [("ims", "lif_dl"), ("cis", "lif_dl")]
            data_dir: Directory with IMS/FLake NetCDF files
            cis_dir: Directory with CIS CSV files
            evaluator_filter: Optional list of evaluator names to run
            
        Returns:
            Nested dictionary:
            {
                "site_name": {
                    "ims_vs_lif_dl": {
                        "overall": {"status": "ok", "metrics": {...}},
                        "fic": {"status": "ok", "metrics": {...}},
                        ...
                    },
                    "cis_vs_lif_dl": {...}
                }
            }
        """
        if comparisons is None:
            # Standard comparisons
            comparisons = [
                ("ims", "lif_dl"),
                ("ims", "flake"),
                ("cis", "lif_dl"),
                ("cis", "flake")
            ]
        
        # Initialize DataLoader
        loader = DataLoader(model_dir, data_dir, cis_dir)
        
        # Results structure
        all_results = {}
        
        # Iterate over sites
        for site in sites:
            try:
                # Load all sources for this site (NEW ARCHITECTURE)
                sources = loader.load_all_sources(site)
                
                # Load baseline year data for phenology/FIC baselines
                baseline_sources = loader.load_baseline_year(site)
                
                # Load lake mask
                mask = loader.load_lake_mask(site)
                
                # Create EvaluationContext with baseline data in metadata
                context = EvaluationContext(
                    sources=sources,
                    site=site,
                    mask=mask,
                    metadata={'baseline_sources': baseline_sources}
                )
                
                # Run evaluators on all comparisons
                site_results = self.run(
                    context=context,
                    comparisons=comparisons,
                    evaluator_filter=evaluator_filter
                )
                
                all_results[site] = site_results
                
            except FileNotFoundError as e:
                # Handle missing data files
                all_results[site] = {
                    "error": {
                        "status": "error",
                        "error": f"Missing data: {e}",
                        "metrics": None
                    }
                }
                warnings.warn(f"Skipping {site}: {e}")
            
            except Exception as e:
                # Handle other errors
                all_results[site] = {
                    "error": {
                        "status": "error",
                        "error": str(e),
                        "metrics": None
                    }
                }
                warnings.warn(f"Error in {site}: {e}")
        
        return all_results
