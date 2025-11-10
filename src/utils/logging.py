"""
Logging utilities for ice cover modelling training.
Handles both experiment tracking (WandB/CSV) and system logging (console/file).
"""

import os
import logging
import wandb
from pytorch_lightning.loggers import WandbLogger, CSVLogger

# ==============================================
# TRAINING AND EXPERIMENT LOGGING UTILITIES
# ==============================================

def setup_training_logging(config, log_level="INFO", log_file=None):
    """
    Setup comprehensive logging for training: experiment tracking + system logging.
    
    This is the main entry point for initializing all logging for a training run.
    It creates both:
    1. Experiment logger (WandbLogger or CSVLogger) for metrics/artifacts
    2. System logger (Python logging) for progress updates and error handling
    
    Args:
        config (dict): Configuration dictionary containing:
            - name (str): Run name (used for logger naming)
            - id (str): Run ID
            - logging.use_wandb (bool): Whether to use WandB
            - logging.wandb_project (str): WandB project name
            - And other config parameters
        log_level (str): Logging level for system logger (DEBUG, INFO, WARNING, ERROR)
        log_file (str, optional): Path to log file for system logger. If None, console only.
        
    Returns:
        tuple: (experiment_logger, system_logger)
            - experiment_logger: WandbLogger or CSVLogger for PyTorch Lightning
            - system_logger: Python Logger for progress/error messages
            
    Example:
        >>> config = {"name": "my_run", "logging": {"use_wandb": True}}
        >>> exp_logger, sys_logger = setup_training_logging(config, log_file="run.log")
        >>> sys_logger.info("Starting training...")
        >>> trainer = Trainer(logger=exp_logger, ...)
    """
    # 1. Create experiment logger (for metrics/artifacts)
    experiment_logger = init_experiment_logger(config)
    
    # 2. Create system logger (for progress/errors)
    system_logger = init_system_logger(
        name=f"training.{config.get('name', 'default')}",
        log_level=log_level,
        log_file=log_file
    )
    
    return experiment_logger, system_logger


def init_system_logger(name="ice_cover_training", log_level="INFO", log_file=None):
    """
    Initialize system logger for progress updates and error handling.
    
    Creates a Python logger with console output and optional file logging.
    This is separate from the experiment logger and is used for general
    script execution messages, debugging, and error tracking.
    
    Args:
        name (str): Logger name (default: "ice_cover_training")
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file (str, optional): Path to log file. If None, console only.
        
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        >>> logger = init_system_logger("training.run1", "INFO", "logs/run1.log")
        >>> logger.info("Loading data...")
        >>> logger.error("Failed to load file", exc_info=True)
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (always enabled)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"System logging to file: {log_file}")
    
    return logger


def init_experiment_logger(config):
    """
    Initialize experiment logger based on configuration.
    
    Creates either a WandB logger (if enabled) or falls back to CSV logger.
    Supports configurable WandB project and entity settings.
    
    Args:
        config (dict): Configuration dictionary containing:
            - logging.use_wandb (bool): Whether to use WandB (default: False)
            - logging.wandb_project (str): WandB project name
            - logging.wandb_entity (str): WandB username/team (optional)
            - logging.offline (bool): Run WandB in offline mode
            - id (str): Unique run ID
            - name (str): Human-readable run name
            - group (str, optional): Group name for organizing related runs
            
    Returns:
        Logger: WandbLogger or CSVLogger instance for PyTorch Lightning Trainer
        
    Example:
        >>> config = {
        ...     "logging": {"use_wandb": True, "wandb_project": "my_project"},
        ...     "name": "experiment_1"
        ... }
        >>> logger = init_logger(config)
        >>> trainer = Trainer(logger=logger, ...)
    """
    # Get logging config (with defaults)
    logging_config = config.get("logging", {})
    use_wandb = logging_config.get("use_wandb", False)
    
    if use_wandb:
        return init_wandb_logger(config)
    else:
        # Fallback to CSV logger
        return CSVLogger(
            save_dir="results/",
            name=config.get("name", "experiment")
        )


def init_wandb_logger(config):
    """
    Initialize Weights & Biases logger for experiment tracking.
    
    Creates a WandbLogger instance for PyTorch Lightning integration.
    Uses configurable project name and entity from config.
    
    Args:
        config (dict): Configuration dictionary containing:
            - logging.wandb_project (str): WandB project name
            - logging.wandb_entity (str, optional): WandB username/team
            - logging.offline (bool): Run in offline mode
            - id (str): Unique run ID
            - name (str): Human-readable run name
            - group (str, optional): Group name for organizing related runs
            
    Returns:
        WandbLogger: Configured logger instance for PyTorch Lightning Trainer
        
    Example:
        >>> config = {
        ...     "logging": {
        ...         "wandb_project": "ice-cover",
        ...         "wandb_entity": "my-team",
        ...         "offline": False
        ...     },
        ...     "id": "abc123",
        ...     "name": "experiment_1"
        ... }
        >>> wandb_logger = init_wandb_logger(config)
        >>> trainer = Trainer(logger=wandb_logger, ...)
    """
    # Get logging configuration
    logging_config = config.get("logging", {})
    wandb_project = logging_config.get("wandb_project", "Ice_Cover_Modelling")
    wandb_entity = logging_config.get("wandb_entity", None)
    offline_mode = logging_config.get("offline", False)
    save_local = logging_config.get("save_local_wandb", False)
    
    # Set offline mode if requested
    if offline_mode:
        os.environ["WANDB_MODE"] = "offline"
    
    # Configure wandb directory location
    if not save_local:
        # Store in results/.wandb (hidden, consolidated)
        os.environ["WANDB_DIR"] = "results/.wandb"
        os.environ["WANDB_CACHE_DIR"] = "results/.wandb/cache"
    # else: use default wandb/ directory in project root
    
    # Initialize WandB run
    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=config["name"],
        id=config["id"],
        resume="allow"
    )
    
    # Create PyTorch Lightning logger
    wandb_logger = WandbLogger(
        project=wandb_project,
        entity=wandb_entity,
        id=config["id"],
        name=config["name"],
        resume="allow",
        config=config,
        allow_val_change=True,
        save_dir="results/",
        save_code=False,
        group=config.get("group", None)
    )
    
    return wandb_logger
