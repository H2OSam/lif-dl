"""
This script runs the model predictor to create a forecast using a trained model.
It loads the configuration, initializes the Predictor, loads the model,
and runs predictions on the provided input data, in an autoregressive manner.
"""

import yaml
import os
import pandas as pd
from argparse import ArgumentParser
import numpy as np
import xarray as xr

from src.model.predictor import Predictor
from src.data.data_module import create_dataset
from src.utils.constants import EVAL_START, EVAL_END

def parse_arguments():
    parser = ArgumentParser(description="Run lake ice forecast using trained model.")
    parser.add_argument("--name", type=str, required=True,
                        help="Name of the forecasting run. Must match the run name of a trained model.")
    parser.add_argument("--site", type=str, default=None,
                        help="If provided, run forecast for a specific site only.")
    return parser.parse_args()

def main():
    """
    Main forecasting pipeline, to produce forecasts using a trained model.
    """
    
    # Get the run name from arguments
    args = parse_arguments()
    run_name = args.name
    specific_site = args.site
    
    # Find the config file based on the run name
    run_dir = run_dir = f"results/{run_name}"
    config_path = os.path.join("results", run_name, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found for run name '{run_name}' at expected path: {config_path}")
    
    # Load config and set the run directory
    config = yaml.safe_load(open(config_path))
    config["run_dir"] = run_dir
    
    # Confirm that the config name matches the provided run name
    if config.get("name", None) != run_name:
        raise ValueError(f"Config name '{config.get('name', None)}' does not match provided run name '{run_name}'")
    
    run_dir = config.get("run_dir", "results/forecasts")
    # Create output directory if it doesn't exist
    output_dir = os.path.join(run_dir, f"forecasts")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create Predictor instance
    print("Initializing Predictor...")
    predictor = Predictor(config)
    
    # Load input dataset for forecasting
    print("Creating dataset for forecasting...")
    dataset = create_dataset(config,
                             start_date=EVAL_START,
                             end_date=EVAL_END)

    # If the training run saved dataset statistics, apply them so inputs are scaled
    # the same way as during training (not applying scaling often yields predictions
    # biased towards 0 which explains outputs being only 0 after rounding).
    dataset_stats_path = os.path.join(config.get('run_dir', ''), 'dataset_stats.pkl')
    if os.path.exists(dataset_stats_path):
        import pickle
        print(f"Loading dataset stats from {dataset_stats_path} and applying scaling...")
        with open(dataset_stats_path, 'rb') as f:
            try:
                across_site_stats = pickle.load(f)
            except Exception:
                across_site_stats = None
        if across_site_stats is not None:
            try:
                dataset.toggle_scaling(across_site_stats)
                print("Dataset scaling applied.")
            except Exception as e:
                print(f"Failed to apply dataset scaling: {e}")
        else:
            print("No dataset stats found in the pickle file.")
    else:
        print("No dataset_stats.pkl found for this run; proceeding without scaling.")

    # Check if specific_site is provided
    if specific_site is not None:
        print(f"Running forecast for specific site: {specific_site}")
        sites = [specific_site]
    else:
        print("Running forecast for all sites specified in the config.")
        sites = config.get("sites", [])
    
    # Use the evaluation date range
    time_index = pd.date_range(start=EVAL_START, end=EVAL_END)
    
    # Forecast for each site
    for site in sites:
        print(f"Running forecast for site: {site}")
        model_predictions, _ = predictor.forecast(dataset, site, EVAL_START, EVAL_END)
        
        # the predictions have shape (time, height, width) and are in numpy format
        print(f"Forecast completed for site: {site}, predictions shape: {model_predictions.shape}")
        
        # Create a xarray DataArray from the prediction with dims and coords
        pred_xarray = xr.DataArray(
            model_predictions,
            dims=["time", "y", "x"],
            coords={"time": time_index}
        )
        # Set the variable name
        pred_xarray.name = "ice_cover_prediction"
        # Add attributes
        run_name = config.get("name", "default_run")
        pred_xarray.attrs["description"] = f"Lake ice cover prediction from trained model {run_name}"
        pred_xarray.attrs["class_values"] = "-1: Land, 0: Water, 1: Ice"
        
        # Save predictions to disk, compressing with zlib
        output_path = os.path.join(output_dir, f"{site}_forecast.nc")
        pred_xarray.to_netcdf(output_path)       
        print(f"Saved forecast for site: {site} to {output_path}")
    print("All forecasts completed.")

if __name__ == "__main__":
    main()
