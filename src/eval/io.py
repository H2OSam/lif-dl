"""
Input/Output utilities for evaluation results and intermediate data.

This module handles saving evaluation results and intermediate data to various formats:
- CSV files for evaluation metrics (structured by lake, observation source, and model)
- FIC timeseries data (CSV format for plotting)
- Spatial maps (NetCDF format for timing anomalies and LMI clusters)
"""

from pathlib import Path
import pandas as pd
import xarray as xr


def save_fic_timeseries(fic_data, output_path):
    """
    Save FIC timeseries data to CSV.
    
    Parameters
    ----------
    fic_data : dict
        Dict with keys: dates, fic, site, source
    output_path : Path or str
        Path to save CSV file
    """
    df = pd.DataFrame({
        'Date': fic_data['dates'],
        'FIC': fic_data['fic']
    })
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_spatial_maps(spatial_data, output_path):
    """
    Save spatial timing and LMI maps to NetCDF.
    
    Parameters
    ----------
    spatial_data : dict
        Dict with timing maps, LMI maps, mask, etc.
    output_path : Path or str
        Path to save NetCDF file
    """
    # Create xarray Dataset
    ds = xr.Dataset({
        'bus_timing': (['y', 'x'], spatial_data['bus_timing']),
        'fus_timing': (['y', 'x'], spatial_data['fus_timing']),
        'bus_lmi_clusters': (['y', 'x'], spatial_data['bus_lmi_clusters']),
        'fus_lmi_clusters': (['y', 'x'], spatial_data['fus_lmi_clusters']),
        'mask': (['y', 'x'], spatial_data['mask'])
    })
    
    # Add metadata
    ds.attrs['site'] = spatial_data['site']
    ds.attrs['source'] = spatial_data['source']
    ds.attrs['clusters'] = "Cluster labels for LMI analysis: 0=non-significant, 1=High-High, 2=Low-High, 3=Low-Low, 4=High-Low"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)


def save_intermediate_data(context, evaluators_dict, output_dir):
    """
    Save intermediate data per source (not per comparison).
    
    For each source (ims, lif_dl, flake, cis) in the context, this function
    extracts and saves intermediate data for visualization purposes.
    
    Parameters
    ----------
    context : EvaluationContext
        EvaluationContext for current site
    evaluators_dict : dict
        Dict mapping evaluator names to evaluator objects
    output_dir : Path or str
        Base output directory
        
    Returns
    -------
    list
        List of Path objects for saved files
    """
    output_dir = Path(output_dir)
    site = context.site
    
    # Get all unique sources in this context
    sources = list(context.sources.keys())
    saved_files = []
    
    for source_name in sources:
        # FIC timeseries (works for all sources)
        if 'fic' in evaluators_dict:
            try:
                fic_data = evaluators_dict['fic'].extract_intermediate_data(context, source_name)
                fic_path = output_dir / 'fic_timeseries' / f'{site}_{source_name}.csv'
                save_fic_timeseries(fic_data, fic_path)
                saved_files.append(fic_path)
            except Exception as e:
                print(f"    Warning: Could not save FIC data for {source_name}: {e}")
        
        # Spatial maps (only for spatial sources, not CIS)
        if 'spatial' in evaluators_dict and source_name.lower() != 'cis':
            try:
                spatial_data = evaluators_dict['spatial'].extract_intermediate_data(context, source_name)
                spatial_path = output_dir / 'spatial_maps' / f'{site}_{source_name}.nc'
                save_spatial_maps(spatial_data, spatial_path)
                saved_files.append(spatial_path)
            except Exception as e:
                print(f"    Warning: Could not save spatial data for {source_name}: {e}")
    
    return saved_files


def save_results_as_csv(all_results, output_dir):
    """
    Save evaluation results as structured CSV files.
    
    Creates one CSV per evaluator with columns:
    - Lake (site name)
    - Obs (observation source: IMS or CIS)
    - Model (prediction source: LIF-DL or FLake)
    - [Metric columns specific to each evaluator]
    
    Parameters
    ----------
    all_results : dict
        Nested dict of evaluation results
    output_dir : Path or str
        Directory to save CSV files (will be created if needed)
        
    Returns
    -------
    list
        List of Path objects for saved CSV files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data collectors for each evaluator
    overall_data = []
    fic_data = []
    spatial_data = []
    phenology_data = []
    
    # Iterate through all results
    for site, site_results in all_results.items():
        if 'error' in site_results:
            continue
            
        for comp_label, comp_results in site_results.items():
            # Parse comparison label (e.g., "ims_vs_lif_dl" -> obs="ims", model="lif_dl")
            parts = comp_label.split('_vs_')
            if len(parts) != 2:
                continue
            obs, model = parts
            obs = obs.upper()  # IMS or CIS
            model = model.upper().replace('_', '-')  # LIF-DL or FLAKE
            
            # Process each evaluator's results
            for eval_name, eval_result in comp_results.items():
                if eval_result['status'] != 'ok':
                    continue
                    
                metrics = eval_result['metrics']
                
                # Build row with Lake, Obs, Model + all metrics
                row = {
                    'Lake': site,
                    'Obs': obs,
                    'Model': model,
                    **metrics  # Unpack all metrics as columns
                }
                
                # Add to appropriate collector
                if eval_name == 'overall':
                    overall_data.append(row)
                elif eval_name == 'fic':
                    fic_data.append(row)
                elif eval_name == 'spatial':
                    spatial_data.append(row)
                elif eval_name == 'phenology':
                    phenology_data.append(row)
    
    # Save each evaluator's results to CSV
    saved_files = []
    
    if overall_data:
        df = pd.DataFrame(overall_data)
        metric_cols = [c for c in df.columns if c not in ['Lake', 'Obs', 'Model']]
        df = df[['Lake', 'Obs', 'Model'] + metric_cols]
        df = df.sort_values(['Lake', 'Obs', 'Model']).reset_index(drop=True)
        
        output_path = output_dir / 'overall.csv'
        df.to_csv(output_path, index=False)
        saved_files.append(output_path)
    
    if fic_data:
        df = pd.DataFrame(fic_data)
        metric_cols = [c for c in df.columns if c not in ['Lake', 'Obs', 'Model']]
        df = df[['Lake', 'Obs', 'Model'] + metric_cols]
        df = df.sort_values(['Lake', 'Obs', 'Model']).reset_index(drop=True)
        
        output_path = output_dir / 'fic.csv'
        df.to_csv(output_path, index=False)
        saved_files.append(output_path)
    
    if spatial_data:
        df = pd.DataFrame(spatial_data)
        metric_cols = [c for c in df.columns if c not in ['Lake', 'Obs', 'Model']]
        df = df[['Lake', 'Obs', 'Model'] + metric_cols]
        df = df.sort_values(['Lake', 'Obs', 'Model']).reset_index(drop=True)
        
        output_path = output_dir / 'spatial.csv'
        df.to_csv(output_path, index=False)
        saved_files.append(output_path)
    
    if phenology_data:
        df = pd.DataFrame(phenology_data)
        metric_cols = [c for c in df.columns if c not in ['Lake', 'Obs', 'Model']]
        df = df[['Lake', 'Obs', 'Model'] + metric_cols]
        df = df.sort_values(['Lake', 'Obs', 'Model']).reset_index(drop=True)
        
        output_path = output_dir / 'phenology.csv'
        df.to_csv(output_path, index=False)
        saved_files.append(output_path)
    
    return saved_files
