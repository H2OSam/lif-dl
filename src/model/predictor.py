"""
Predictor module for wrapping a trained model and running inference.
"""

from typing import Tuple, Any, Dict
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from src.model.model import load_model
from src.data.data_module import Ice_Cover_Dataset

class Predictor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lit_model = None
        self.checkpoint = None
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set resume to True in config to load from checkpoint if available
        self.config['resume'] = True
        
        # Load the model upon initialization
        self.load()

    def load(self):
        """Load or create a LitModel and keep it for predictions.
        Returns (lit_model, checkpoint_path, completed_epochs)
        """
        lit, ckpt, _ = load_model(self.config)
        self.lit_model = lit
        self.lit_model.eval()
        self.lit_model.to(self.device)
        self.checkpoint = ckpt
        return lit, ckpt

    def forecast(self, input_dataset: Ice_Cover_Dataset, 
                 site: str, start_date: str, end_date: str) -> np.ndarray:
        """Run forecast on the input dataset for the specified site and date range.
        
        Args:
            input_dataset (Ice_Cover_Dataset): Dataset to supply input forcing data.
            site (str): Site identifier.
            start_date (str): Start date for the forecast.
            end_date (str): End date for the forecast.
        """
        
        # Verify that site and dates are valid
        self.verify_forecast_parameters(site, start_date, end_date,
                                        input_dataset.sites,
                                        pd.date_range(start=input_dataset.start_date,
                                                      end=input_dataset.end_date))
        site_predictions = []
        site_targets = []
        prev_ice_cover = None
        seq_len = self.config.get("sequence_length", 7)
        date_index = pd.date_range(
            start=start_date,
            end=pd.to_datetime(end_date) - pd.Timedelta(days=seq_len - 1)
        )
        
        with torch.no_grad():
            for date in tqdm(date_index):
                
                # Get index of this site/date in the dataset
                idx = input_dataset.get_lake_date_index(date, site)
                input_seq, lake_mask, target = input_dataset[idx]
                lake_mask = lake_mask.squeeze().long()
                
                # Check if its the first prediction
                if len(site_predictions) == 0:
                    # prev_ice_cover is the initial ice cover from input sequence
                    prev_ice_cover = input_seq[:, -3:-1, :, :].clone()
                else:
                    # Update the previous ice cover by shifting and then adding the last prediction
                    
                    # Shift prev_ice_cover left by one time step
                    prev_ice_cover = torch.roll(prev_ice_cover, shifts=-1, dims=0)
                    
                    # Get the last prediction and add it to the end
                    last_pred = site_predictions[-1][0]  # Shape (H, W)
                    
                    # Convert last prediction from raw prediction to binary, one-hot format
                    last_pred_one_hot = self._to_one_hot(last_pred, lake_mask) # Shape (2, H, W)
                    
                    # Replace the last time step's ice cover channels with the last prediction
                    prev_ice_cover[-1, :, :, :] = last_pred_one_hot
                
                # Prepare model input by replacing ice cover channels in input_seq with prev_ice_cover
                model_input = input_seq.clone()
                model_input[:, -3:-1, :, :] = prev_ice_cover
                
                # Run model prediction
                model_input = model_input.unsqueeze(0).to(self.device)  # Add batch dimension and move to device
                pred = self.lit_model(model_input).squeeze()  # Remove singleton dimensions
                pred = pred.cpu() # Force prediction back to CPU
                
                if date != date_index[-1]:
                    # If this is not the last date, take only the first time step prediction
                    pred = pred[0:1, :, :]  # Shape (1, H, W)
                    target = target[0:1]       # Shape (1, H, W)
                
                # Convert target to single channel
                target = torch.argmax(target, dim=1)  # Convert target to single channel

                # Store prediction and target
                site_predictions.append(pred)
                site_targets.append(target)
        
        # Concatenate predictions and targets along time axis
        site_predictions = torch.concat(site_predictions, axis=0).numpy()  # Shape (time, H, W)
        site_targets = torch.concat(site_targets, axis=0).numpy()          # Shape (time, H, W)      
        
        # Round the model predictions to get binary ice/water
        site_predictions = np.round(site_predictions).astype(int)
        
        # Apply the lake mask to make land have class -1
        lake_mask_np = lake_mask.numpy()
        site_predictions = np.where(lake_mask_np, site_predictions, -1)
        site_targets = np.where(lake_mask_np, site_targets, -1)
        
        return site_predictions, site_targets
                    
    def verify_forecast_parameters(self, site: str, start_date: str, end_date: str,
                                   available_sites: pd.Index, available_dates: pd.DatetimeIndex):
        """Verify that the forecast parameters are valid.
        
        Args:
            site (str): Site identifier.
            start_date (str): Start date for the forecast.
            end_date (str): End date for the forecast.
            available_sites (pd.Index): Available site identifiers in the dataset.
            available_dates (pd.DatetimeIndex): Available dates in the dataset.
        """
        if site not in available_sites:
            raise ValueError(f"Site '{site}' not found in dataset. Available sites: {available_sites.tolist()}")
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Pad the start and end dates by the model's sequence length
        # This ensures that the model has enough context for the first and last predictions
        seq_len = self.config.get("sequence_length", 7)
        
        if start not in available_dates:
            raise ValueError(f"Start date '{start_date}' not found in dataset. Available dates range from {available_dates.min().date()} to {available_dates.max().date()}.")
        
        if end not in available_dates:
            raise ValueError(f"End date '{end_date}' not found in dataset. Available dates range from {available_dates.min().date()} to {available_dates.max().date()}.")
        
        if start >= end:
            raise ValueError(f"Start date '{start_date}' must be before end date '{end_date}'.")

    def _to_one_hot(self, pred_array: np.ndarray, lake_mask: np.ndarray) -> torch.Tensor:
        """Convert raw model prediction (one-channel, 0-1 float) to binary one-hot encoding (two-channel).
        Where the classes are 0 water, 1 ice, and all the land pixels are set to 0 in both channels.
        Args:
            pred_array (torch.Tensor): Raw model prediction of shape (H, W), values in [0, 1].
            lake_mask (torch.Tensor): Lake mask of shape (H, W), values in {0, 1}.
        Returns:
            torch.Tensor: One-hot encoded prediction of shape (2, H, W).
        """
        
        # First, round the prediction to get binary ice/water
        binary_pred = (pred_array >= 0.5).long()

        # Get the one-hot encoding
        one_hot = torch.nn.functional.one_hot(binary_pred, num_classes=2).permute(2, 0, 1)  # Shape (2, H, W)
        
        # Mask out land pixels (set to 0 in both channels)
        one_hot *= lake_mask.unsqueeze(0)
        
        return one_hot
        
        