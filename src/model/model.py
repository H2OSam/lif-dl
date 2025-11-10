"""
Model definitions and utilities for ice cover modelling.
Includes PyTorch Lightning wrapper, model factory, and checkpoint management.
"""

import os
import glob
import pickle
import yaml
import wandb
from torch import nn
import torch.nn.functional as F
from torch import optim
from pytorch_lightning import LightningModule

from src.model.lif_dl import LIF_DL


class TrainingLoss(nn.Module):
    """
    mask: a tensor with spatial dimension matching the prediction, used to mask out loss across certain pixels
    temporal_weights: a tensor of size (sequence_length), which applies a weighting to the loss of each prediction in the sequence

    """
    def __init__(self, temporal_weights=None):
        super(TrainingLoss, self).__init__()

        self.time_weights = temporal_weights
    
    def __call__(self, pred, target, mask=None):
        # Mask out loss on certain pixels
        if mask is not None:
            pred = pred*mask
            target = target*mask
        
        # Calculate BCE between output and target, but don't reduce any dimensions (do pixel wise across entire input/output)
        L = F.binary_cross_entropy(pred, target, reduction='sum') / (mask.sum()*pred.size(1))
        return L


class LitModel(LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.loss = TrainingLoss()
        self.config = config
        self.lr = config["lr"]
        self.optim = config["optim"]
    
    def forward(self, x):
        pred = self.model(x)
        return pred
    
    def training_step(self, batch, batch_idx):
        inputs, mask, targets = batch
        pred = self.model(inputs)
        # Extract just the ice class from targets. Preserve the dimension though!
        loss = self.loss(pred, targets[:,:,1:2], mask)
        # Logging to tensorboard
        self.log("train_loss", loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, mask, targets = batch
        pred = self.model(inputs)
        # Extract just the ice class from targets. Preserve the dimension though!
        loss = self.loss(pred, targets[:,:,1:2], mask)
        
        self.log("val_loss", loss, sync_dist=True)
        self.log("learning_rate", self.optimizer.param_groups[0]["lr"], sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        if self.reduce_lr_on_plateau is not None:
            self.reduce_lr_on_plateau.step(self.trainer.callback_metrics["val_loss"])

    def configure_optimizers(self):
        if self.optim == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.config["weight_decay"])
        elif self.optim == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.config["weight_decay"], momentum=self.config["momentum"])
        self.reduce_lr_on_plateau = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        # return {"optimizer": self.optimizer, "lr_scheduler": self.reduce_lr_on_plateau, "monitor":"validation loss"}
        return self.optimizer


#===============================================================
# Model Factory and Checkpoint Utilities
#===============================================================

def create_model(config):
    """
    Factory function to create a model from configuration.
    
    Currently supports LIF_DL architecture. Can be extended to support
    additional architectures in the future.
    
    Args:
        config (dict): Configuration dictionary containing:
            - model (str): Model architecture name (default: "LIF_DL")
            - num_transformers (int): Number of transformer blocks
            - hidden (int): Hidden dimension size
            - dropout (float): Dropout rate
            - variables (list): List of input variables
            
    Returns:
        nn.Module: Instantiated model
        
    Example:
        >>> config = {"model": "LIF_DL", "num_transformers": 2, ...}
        >>> model = create_model(config)
    """
    model_name = config.get("model", "LIF_DL")
    
    if model_name == "LIF_DL":
        model = LIF_DL(
            stack_num=config["num_transformers"],
            channel=config["hidden"],
            dropout=config["dropout"],
            in_ch=len(config["variables"]) + 3,  # variables + 3 ice channels
            out_ch=1
        )
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
    
    return model


def get_checkpoint_path(config):
    """
    Get the checkpoint directory path based on configuration.
    
    Uses the run directory from config if available, otherwise constructs
    the path from the run name.
    
    Args:
        config (dict): Configuration dictionary containing:
            - run_dir (str, optional): Pre-determined run directory path
            - name (str): Human-readable run name (used if run_dir not set)
            
    Returns:
        str: Path to checkpoint directory
        
    Example:
        >>> config = {"name": "baseline_experiment"}
        >>> path = get_checkpoint_path(config)
        >>> # Returns: "results/baseline_experiment"
    """
    # Use pre-determined run_dir if available (set in train.py)
    if "run_dir" in config:
        return config["run_dir"]
    
    # Fallback: calculate from name (for backward compatibility)
    run_name = config["name"]
    path_to_ckpt = f"results/{run_name}"
    
    return path_to_ckpt


def save_training_metadata(checkpoint_path, config, metadata):
    """
    Save training metadata to checkpoint directory for reproducibility.
    
    Saves:
    - Configuration YAML (includes run ID and name)
    - Dataset statistics (for scaling)
    - Train/validation indices (for reproducibility)
    
    Args:
        checkpoint_path (str): Path to checkpoint directory
        config (dict): Configuration dictionary (includes name and id)
        metadata (dict): Metadata dictionary containing:
            - across_site_stats: Statistics across all sites
            - site_stats: Per-site statistics
            - train_indices: Training set indices
            - val_indices: Validation set indices
            
    Example:
        >>> save_training_metadata(path, config, metadata)
    """
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Save configuration (includes both name and id)
    with open(os.path.join(checkpoint_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    # Save run info separately for easy reference
    run_info = {
        "run_name": config.get("name"),
        "run_id": config.get("id"),
        "group": config.get("group"),
        "created_at": config.get("timestamp", "unknown")
    }
    with open(os.path.join(checkpoint_path, "run_info.yaml"), "w") as f:
        yaml.dump(run_info, f)
    
    # Save dataset statistics
    with open(os.path.join(checkpoint_path, "dataset_stats.pkl"), "wb") as f:
        pickle.dump(metadata["across_site_stats"], f)
        pickle.dump(metadata["site_stats"], f)
    
    # Save dataset indices
    with open(os.path.join(checkpoint_path, "dataset_indices.pkl"), "wb") as f:
        pickle.dump(metadata["train_indices"], f)
        pickle.dump(metadata["val_indices"], f)


def load_model(config, checkpoint_path=None):
    """
    Load model from checkpoint if resuming, otherwise create new LitModel.
    
    This function handles all checkpoint logic:
    - If resuming and checkpoints exist: returns (lit_model, checkpoint_file_path, completed_epochs)
    - If resuming but no checkpoints: returns (new_lit_model, None, 0) with warning
    - If not resuming: returns (new_lit_model, None, 0)
    
    Args:
        config (dict): Configuration dictionary with 'resume' flag
        checkpoint_path (str): Path to checkpoint directory (optional)
        
    Returns:
        tuple: (LitModel, checkpoint_file_path or None, completed_epochs)
            - LitModel: PyTorch Lightning model wrapper
            - checkpoint_file_path: Path to resume from, or None for fresh start
            - completed_epochs: Number of epochs already completed (for adjusting max_epochs)
        
    Example:
        >>> config = {"resume": True, ...}
        >>> lit_model, ckpt, epochs = load_or_create_litmodel(config, ckpt_path)
        >>> # Adjust max_epochs to run additional epochs
        >>> config["max_epochs"] = epochs + config["epochs"]
        >>> trainer.fit(lit_model, train_loader, val_loader, ckpt_path=ckpt)
    """
    # Create the base model
    model = create_model(config)
    
    # Check if we should resume
    should_resume = config.get("resume", False)
    
    # Look for existing checkpoints
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(config)
    ckpts = sorted(glob.glob(os.path.join(checkpoint_path, "*.ckpt")))
    
    if should_resume and len(ckpts) > 0:
        # Resume from most recent checkpoint
        checkpoint_file = ckpts[-1]
        
        # Extract epoch number from filename (e.g., "run-epoch=005-val_loss=0.123.ckpt")
        import re
        match = re.search(r'epoch=(\d+)', checkpoint_file)
        if match:
            completed_epochs = int(match.group(1)) + 1  # +1 because epoch is 0-indexed but we count completed epochs
        else:
            completed_epochs = 0
            print(f"Warning: Could not parse epoch from checkpoint filename: {checkpoint_file}")
        
        print(f"Will resume from checkpoint: {os.path.basename(checkpoint_file)}")
        print(f"Completed epochs: {completed_epochs}")
        
        # Try to load the LightningModule from checkpoint so the returned
        # lit_model already contains the saved weights and optimizer state.
        try:
            # Use Lightning's loader which will construct the object and load state
            lit_model = LitModel.load_from_checkpoint(checkpoint_file, model=model, config=config)
            print(f"Loaded checkpoint into LitModel: {os.path.basename(checkpoint_file)}")
        except Exception as e:
            # Fall back to creating a fresh instance and load state dict manually
            print(f"Warning: automatic load_from_checkpoint failed: {e}. Falling back to manual state load.")
            lit_model = LitModel(model, config)
            try:
                import torch
                ckpt_data = torch.load(checkpoint_file, map_location='cpu')
                # Lightning stores parameters in 'state_dict'
                state_dict = ckpt_data.get('state_dict', None)
                if state_dict is not None:
                    lit_model.load_state_dict(state_dict, strict=False)
                    print("State dict loaded into LitModel (partial load allowed).")
                else:
                    print("Warning: checkpoint did not contain 'state_dict'; starting from scratch.")
            except Exception as e2:
                print(f"Failed to load checkpoint state dict: {e2}. Starting from scratch.")

        return lit_model, checkpoint_file, completed_epochs
    
    elif should_resume and len(ckpts) == 0:
        # User wants to resume but no checkpoints found
        print(f"Warning: --resume specified but no checkpoints found in {checkpoint_path}")
        print("Starting new training run instead...")
        lit_model = LitModel(model, config)
        return lit_model, None, 0

    else:
        # Create new model
        lit_model = LitModel(model, config)
        return lit_model, None, 0
