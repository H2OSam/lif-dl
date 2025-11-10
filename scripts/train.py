"""
Training script for ice cover prediction models using ERA5 and IMS datasets.

This script provides a clean entry point for training deep learning models to predict
lake ice cover. It handles data loading, model initialization, checkpoint management,
and training orchestration using PyTorch Lightning and Weights & Biases.

Key Features:
    - Automatic data loading with class balancing
    - WandB integration for experiment tracking
    - Checkpoint resumption support
    - Configurable via YAML files

General Flow:
    1. Parse command line arguments and load configuration
    2. Initialize logging (experiment tracking + system logging)
    3. Load and preprocess data with automatic scaling and balancing
    4. Create model from configuration
    5. Setup checkpointing and training callbacks
    6. Train model with PyTorch Lightning

Usage:
    # Start a new training run
    python train.py --config configs/default.yaml --name "my_experiment"
    
    # Resume an existing run
    python train.py --config configs/default.yaml --name "my_experiment" --resume
    
    # Organize runs into groups
    python train.py --config configs/default.yaml --name "baseline_v1" --group "baselines"
"""

import os
import warnings
import datetime
import logging
import wandb
import yaml
from argparse import ArgumentParser, RawDescriptionHelpFormatter

# Suppress pydantic warnings from dependencies
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")
# Suppress pkg_resources deprecation warning from PyTorch Lightning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

from src.data.data_module import create_dataloaders
from src.model.model import (
    LitModel,
    create_model,
    get_checkpoint_path,
    save_training_metadata,
    load_model
)
from src.utils.logging import setup_training_logging

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


def parse_arguments():
    """Parse command line arguments for training script."""
    parser = ArgumentParser(
        description="Train an Ice Cover Modelling neural network",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Start a new training run
            python train.py --config configs/default.yaml --name "my_experiment"
            
            # Resume an existing run (continues from last checkpoint)
            python train.py --name "my_experiment" --resume
            
            # Start a debug run
            python train.py --config configs/debug.yaml --name "debug_test"
            
            # Organize related runs into groups
            python train.py --name "baseline_v1" --group "paper_results"
    """
    )

    parser.add_argument(
        '--config', 
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)"
    )
    parser.add_argument(
        '--name', 
        default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Name for this training run (default: current timestamp)"
    )
    parser.add_argument(
        '--id', 
        default=wandb.util.generate_id(),
        help="Unique ID for this run (default: auto-generated)"
    )
    parser.add_argument(
        '--group', 
        default=None,
        help="Group name for organizing related runs (default: None)"
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help="Resume training from existing run with the specified name"
    )
    parser.add_argument(
        '--log-level',
        default="INFO",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        '--log-file',
        default=None,
        help="Path to log file. If not specified, only console logging is used."
    )
    
    return parser.parse_args()


def main(config, logger, experiment_logger):
    """
    Main training pipeline for ice cover prediction models.
    
    Orchestrates the complete training workflow including data loading, model creation,
    checkpoint management, and training execution. Supports both new training runs and
    resuming from checkpoints.
    
    Args:
        config (dict): Configuration dictionary containing all training parameters:
            - data_dir: Path to data directory
            - sites: List of lake sites to train on
            - start_date/end_date: Date range for training data
            - sequence_length: Length of input sequences
            - variables: List of meteorological variables
            - batch_size: Training batch size
            - epochs: Maximum number of training epochs
            - lr: Learning rate
            - And other model/training hyperparameters
        logger (logging.Logger): System logger for progress and error messages.
        experiment_logger (Logger): Experiment logger (WandB/CSV) for tracking metrics.
            
    Returns:
        None. Trained model is saved to checkpoint directory.
    """
    # 1. Load data and create DataLoaders
    train_loader, valid_loader, metadata = create_dataloaders(config, logger)

    # 2. Create model
    logger.info("Creating model...")
    checkpoint_path = get_checkpoint_path(config)
    lit_model, resume_ckpt, completed_epochs = load_model(config, checkpoint_path)
    
    # Calculate max_epochs based on resumption state
    # Set a very high limit (1000) to allow unlimited resumption
    # Early stopping will terminate training when appropriate
    if config.get("resume", False) and completed_epochs > 0:
        epochs_to_run = config.get("epochs", 10)
        total_max_epochs = completed_epochs + epochs_to_run
        logger.info(f"Resuming: will train for {epochs_to_run} more epochs ({completed_epochs} -> {total_max_epochs})")
    else:
        # Fresh start: run for specified epochs (up to limit of 1000 total)
        total_max_epochs = min(config.get("epochs", 10), 1000)
    
    # Save metadata for reproducibility (only if not resuming)
    if not config.get("resume", False):
        save_training_metadata(checkpoint_path, config, metadata)
    
    # 4. Create PyTorch Lightning trainer
    logger.info("Creating PyTorch Lightning trainer...")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=checkpoint_path,
        filename=config["name"] + '-{epoch:03d}-{val_loss:.3f}'
    )
    early_stopper = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
        verbose=True
    )

    trainer = Trainer(
        logger=experiment_logger,
        accelerator="gpu",
        callbacks=[early_stopper, checkpoint_callback],
        max_epochs=total_max_epochs,
        default_root_dir="outputs/",
        profiler="simple"
    )

    # 5. Training!
    logger.info("Starting model training...")
    # If a checkpoint file was returned, pass it to Trainer so it restores
    # training state (epoch, optimizer, schedulers). We still allow the
    # LitModel to have been loaded with weights for notebook convenience.
    if resume_ckpt:
        trainer.fit(lit_model, train_loader, valid_loader, ckpt_path=resume_ckpt)
    else:
        trainer.fit(lit_model, train_loader, valid_loader)
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Determine run directory and check for conflicts
    run_name = str(args.name)
    run_dir = f"results/{run_name}"
    run_exists = os.path.exists(run_dir)
    
    # Handle run name conflicts
    if run_exists and not args.resume:
        # Warn user about overwriting existing run
        print(f"\n{'='*70}")
        print(f"WARNING: Run directory '{run_dir}' already exists!")
        print(f"{'='*70}")
        print(f"\nThis will DELETE the existing run and start fresh.")
        print(f"If you want to resume the existing run instead, use: --resume")
        print(f"\nExisting run contents:")
        try:
            for item in os.listdir(run_dir):
                print(f"  - {item}")
        except Exception:
            pass
        print(f"\n{'='*70}")
        
        response = input("\nDo you want to DELETE this run and start fresh? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted. Use --resume to continue the existing run, or choose a different --name.")
            exit(0)
        
        # Delete the existing run directory
        import shutil
        print(f"Deleting existing run directory: {run_dir}")
        shutil.rmtree(run_dir)
        print("Deleted successfully. Starting fresh run...\n")
    
    elif args.resume and not run_exists:
        # User wants to resume but run doesn't exist
        print(f"\n{'='*70}")
        print(f"ERROR: Cannot resume - run '{run_name}' does not exist!")
        print(f"{'='*70}")
        print(f"\nExpected directory: {run_dir}")
        print(f"\nAvailable runs in 'results/':")
        try:
            runs = [d for d in os.listdir("results/") 
                   if os.path.isdir(os.path.join("results/", d)) 
                   and not d.startswith('.')]
            if runs:
                for run in sorted(runs):
                    print(f"  - {run}")
            else:
                print("  (no runs found)")
        except Exception:
            print("  (could not list runs)")
        print(f"\n{'='*70}\n")
        exit(1)
    
    # Determine log file path
    log_file = args.log_file
    if log_file is None:
        log_file = f"{run_dir}/training.log"
    
    # Extract config path
    config_path = str(args.config)
    
    # Load configuration from YAML file (we need it to setup logging)
    try:
        with open(config_path) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        # Use a basic logger for error messages before setup_training_logging
        print(f"ERROR: Configuration file '{config_path}' not found.")
        print("Available configs in 'configs/' directory:")
        if os.path.exists("configs"):
            for config_file in os.listdir("configs"):
                if config_file.endswith(('.yaml', '.yml')):
                    print(f"  - configs/{config_file}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"ERROR: Error parsing YAML file '{config_path}': {e}")
        exit(1)
    
    # Update config with command line arguments
    group = args.group
    
    # For resuming runs, try to load the saved ID and config
    if args.resume and run_exists:
        saved_config_path = os.path.join(run_dir, "config.yaml")
        if os.path.exists(saved_config_path):
            with open(saved_config_path, 'r') as f:
                saved_config = yaml.load(f, Loader=yaml.FullLoader)
                # Use saved ID for WandB continuity
                wid = saved_config.get("id", str(args.id))
                print(f"Resuming with saved run ID: {wid}")
        else:
            wid = str(args.id)
    else:
        wid = str(args.id)
    
    config["id"] = wid
    config["name"] = run_name
    config["group"] = group
    config["timestamp"] = datetime.datetime.now().isoformat()
    config["run_dir"] = run_dir
    config["resume"] = args.resume
    
    # Create run directory
    os.makedirs(run_dir, exist_ok=True)
    
    # Setup comprehensive logging (experiment logger + system logger)
    experiment_logger, logger = setup_training_logging(
        config=config,
        log_level=args.log_level,
        log_file=log_file
    )
    
    logger.info("="*60)
    if args.resume:
        logger.info("Resuming Ice Cover Modelling Training")
        logger.info(f"Resuming run: {run_name}")
    else:
        logger.info("Starting Ice Cover Modelling Training")
        logger.info(f"Training run: {run_name} (ID: {wid})")
    logger.info("="*60)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Configuration file: {config_path}")
    if group:
        logger.info(f"Run group: {group}")

    # Set random seed for reproducibility
    logger.info("Setting random seed for reproducibility")
    seed_everything(72, workers=True)

    # Start training
    try:
        # Pass both loggers to main
        main(config, logger, experiment_logger)
        logger.info("Training pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
