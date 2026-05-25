"""
This module covers the data loading and preprocessing
"""

from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import xarray as xr
import pandas as pd
import numpy as np
import datetime
from torch.nn.functional import one_hot
import torch
import bisect
import logging


# Canonical variable ordering - this ensures consistent ordering regardless of config
CANONICAL_VARIABLE_ORDER = [
    'temperature_2m',
    'wind_speed_10m',
    'surface_solar_radiation_downwards_sum',
    'total_precipitation_sum',
    'relative_humidity',
    'total_cloud_cover',
    'accumulated_freezing_dd',
    'accumulated_thawing_dd',
    'lake_depth',
]


def create_dataset(config, start_date=None, end_date=None):
    """
    Create an Ice_Cover_Dataset from configuration.
    
    DATE RANGE SEMANTICS:
    - start_date: First date for which predictions are desired
    - end_date: Last date for which predictions are desired
    
    The dataset will provide predictions covering every day in [start_date, end_date].
    Internally, the dataset creates samples (target_dates) such that their outputs
    fully cover this range.
    
    Data requirements:
    - Historical data needed from (start_date - sequence_length) onwards
    - Future data needed through end_date
    
    Example with sequence_length=7:
    - Request: start='2021-01-01', end='2021-01-10'
    - Loads data: 2020-12-25 to 2021-01-10
    - Creates 4 samples (target_dates) with outputs:
      * 2021-01-01: predicts [2021-01-01 to 2021-01-07]
      * 2021-01-02: predicts [2021-01-02 to 2021-01-08]
      * 2021-01-03: predicts [2021-01-03 to 2021-01-09]
      * 2021-01-04: predicts [2021-01-04 to 2021-01-10]
    
    Args:
        config: Configuration dictionary containing:
            - data_dir: Path to data files
            - sites: List of lake sites
            - start_date: First date to predict (default if not overridden)
            - end_date: Last date to predict (default if not overridden)
            - sequence_length: Model sequence length
            - variables: List of meteorological variables
        start_date: Optional override for coverage start date
        end_date: Optional override for coverage end date
        
    Returns:
        Ice_Cover_Dataset instance
        
    Example:
        >>> config = {
        ...     'data_dir': 'data/nc/',
        ...     'sites': ['great_slave_lake'],
        ...     'sequence_length': 7,
        ...     'variables': ['temperature_2m', ...]
        ... }
        >>> # Get predictions covering 2021
        >>> dataset = create_dataset(config, start_date='2021-01-01', end_date='2021-12-31')
    """
    # Extract configuration parameters
    data_dir = config["data_dir"]
    sites = config["sites"]
    
    if start_date is None:
        start_date = config["start_date"]
    if end_date is None:
        end_date = config["end_date"]
    
    sequence_length = config["sequence_length"]
    variables = config["variables"]
    
    return Ice_Cover_Dataset(
        data_dir, sites, start_date, end_date, sequence_length, variables
    )

def create_dataloaders(config, logger=None):
    """
    Create training and validation DataLoaders from configuration.
    
    Handles the complete data pipeline:
    - Dataset loading and scaling
    - Class balancing via freeze/thaw classification
    - Train/validation splitting (with optional chunked validation)
    - Weighted sampling for balanced training
    
    Args:
        config (dict): Configuration dictionary containing:
            - data_dir: Path to data directory
            - sites: List of lake site names
            - start_date: Training start date
            - end_date: Training end date
            - sequence_length: Length of input sequences
            - variables: List of meteorological variables to use
            - batch_size: Batch size for DataLoaders
            - val_chunk (optional): Chunk size for validation split
            - ft_weight (optional): Weight for freeze/thaw class
            - else_weight (optional): Weight for other class
        logger (logging.Logger, optional): Logger for progress messages
        
    Returns:
        tuple: (train_loader, val_loader, metadata_dict)
            - train_loader: DataLoader for training with weighted sampling
            - val_loader: DataLoader for validation
            - metadata_dict: Dictionary containing:
                - across_site_stats: Statistics across all sites
                - site_stats: Per-site statistics
                - train_indices: Indices used for training
                - val_indices: Indices used for validation
                
    Example:
        >>> config = {...}
        >>> train_loader, val_loader, metadata = create_dataloaders(config)
    """
    if logger is None:
        logger = logging.getLogger('ice_cover_training')
    
    # Extract configuration parameters
    batch_size = config["batch_size"]
    
    # 1. Load the dataset
    logger.info("Loading the dataset...")
    dataset = create_dataset(config)
    logger.info(f"Dataset using canonical variable order: {dataset.vars}")
    
    # 2. Calculate statistics and apply scaling
    logger.info("Scaling the dataset...")
    across_site_stats, site_stats = dataset.statistics()
    dataset.toggle_scaling(across_site_stats)
    
    # 3. Class Balancing - Calculate freeze/thaw classes and weights
    logger.info("Performing class balancing by calculating freeze/thaw classes and weights...")
    fr_th_classes = dataset.classify_dates()
    fr_th_weights = np.zeros(len(dataset))
    
    # Count samples in each class
    ft_count = (fr_th_classes == 1).sum()
    else_count = (fr_th_classes == 0).sum()
    logger.info(
        f"Class distribution - Class IDs: [0,1], Class Counts: [{else_count}, {ft_count}]"
    )
    
    # Map classes to sample weights
    if 'ft_weight' not in config.keys():
        ft_weight = 1/ft_count if ft_count != 0 else 0
    else:
        ft_weight = config['ft_weight']
    
    if 'else_weight' not in config.keys():
        else_weight = 1/else_count if else_count != 0 else 0
    else:
        else_weight = config['else_weight']
    
    logger.info(f"Class weights - FT Weight: {ft_weight}, Else Weight: {else_weight}")
    fr_th_weights[fr_th_classes == 0] = else_weight
    fr_th_weights[fr_th_classes == 1] = ft_weight
    
    assert fr_th_weights.shape[0] == len(dataset), \
        "Number of weights does not match number of indices in training dataset."
    
    # 4. Train/Val split
    logger.info("Performing train/validation split...")
    logger.info(f"Full dataset size: {len(dataset)}")
    
    if "val_chunk" not in config.keys() or config["val_chunk"] is None:
        # Simple random validation split (20%)
        num_samples = int(0.2 * len(dataset))
        validation_indices = np.random.choice(
            list(range(0, len(dataset))), num_samples, replace=False
        )
    else:
        # Chunked validation split - takes multiple chunks from each site
        validation_indices = []
        num_chunks = int(np.floor((0.2 / config["val_chunk"])))
        logger.debug(f"Number of validation chunks: {num_chunks}")
        
        for s_index, e_index in dataset.index_map.values():
            indices = list(range(s_index, e_index))
            # Size of each chunk (such that combined chunks = 20% of site data)
            split = int(np.floor((config["val_chunk"]) * len(indices)))
            
            # Extract multiple chunks
            for i in range(num_chunks):
                N = len(indices)
                # Select random start and extract chunk
                start = np.random.choice(list(range(0, N - split)))
                val_indices = indices[start:start + split]
                assert len(val_indices) == split
                
                # Add to validation set
                validation_indices.append(val_indices)
                # Remove from available indices
                indices = indices[:start] + indices[start + split:]
        
        # Concatenate all validation chunks
        validation_indices = np.concatenate(validation_indices)
    
    logger.info(
        f"Validation set size: {len(validation_indices)} "
        f"({len(validation_indices)/len(dataset):.2%} of total)"
    )
    
    # Create train indices (all indices not in validation)
    train_indices = np.array(list(set(range(0, len(dataset))) - set(validation_indices)))
    assert (set(validation_indices) & set(train_indices)) == set(), \
        "Train/Val indices overlap!!"
    
    # Create dataset subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, validation_indices)
    
    # 5. Create samplers and DataLoaders
    logger.info("Creating weighted samplers and data loaders...")
    train_weights = fr_th_weights[train_indices]
    
    # Weighted sampler for balanced training
    train_sampler = WeightedRandomSampler(
        train_weights, num_samples=len(train_dataset), replacement=True
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4
    )
    valid_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    # 6. Package metadata for checkpointing
    metadata = {
        "across_site_stats": across_site_stats,
        "site_stats": site_stats,
        "train_indices": train_indices,
        "val_indices": validation_indices
    }
    
    return train_loader, valid_loader, metadata


class Ice_Cover_Dataset(Dataset):
    """
    PyTorch Dataset for loading + aligning multiple IMS & ERA5 NetCDF datasets.
    
    DATE SEMANTICS:
    - start_date/end_date define the COVERAGE range (dates you want predictions for)
    - Internally creates target_dates (sample dates) to cover this range
    - Each sample predicts sequence_length days starting from its target_date
    
    Example with sequence_length=7:
    - Request coverage: start='2021-01-01', end='2021-01-10'
    - Creates 4 samples (target_dates=[2021-01-01, 02, 03, 04]):
      * Sample 2021-01-01: input=[2020-12-25 to 2020-12-31], output=[2021-01-01 to 2021-01-07]
      * Sample 2021-01-02: input=[2020-12-26 to 2021-01-01], output=[2021-01-02 to 2021-01-08]
      * Sample 2021-01-03: input=[2020-12-27 to 2021-01-02], output=[2021-01-03 to 2021-01-09]
      * Sample 2021-01-04: input=[2020-12-28 to 2021-01-03], output=[2021-01-04 to 2021-01-10]
    - Full coverage achieved: every day from 2021-01-01 to 2021-01-10 is predicted
    
    The dataset enforces a canonical variable ordering to prevent misuse when loading
    trained models. Variables will always be returned in the order defined by
    CANONICAL_VARIABLE_ORDER, regardless of the order specified in the config.
    """
    def __init__(self,
                data_dir = "../Datasets/Combined/",
                sites = ["Great_Slave_Lake"],
                start_date = "2018-01-01",
                end_date = "2021-12-31", 
                sequence_length  = 7,
                variables = [
                    'temperature_2m',
                    'wind_speed_10m',
                    'surface_solar_radiation_downwards_sum',
                    'total_precipitation_sum',
                    'relative_humidity',
                    'total_cloud_cover',
                    'accumulated_freezing_dd',
                    'accumulated_thawing_dd',
                    'lake_depth',
                ]
                ):
        """
        Initialize Ice_Cover_Dataset with coverage-based date semantics.
        
        Args:
            data_dir: Path to directory containing NetCDF files
            sites: List of lake site names
            start_date: First date to predict (coverage start)
            end_date: Last date to predict (coverage end)
            sequence_length: Length of input/output sequences
            variables: List of meteorological variables to load
        """
        # Dataset Attributes
        self.directory = data_dir
        self.sites = sites
        self.seq_length = sequence_length
        self.window_size = 2 * sequence_length

        # Dictionaries mapping site names to data
        self.dataset = {}
        self.lake_masks = {}
        self.bathymetry = {}
        
        
        # Store requested coverage
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        # verify that these dates can be offered
        self._verify_coverage_dates(self.start_date, self.end_date)
        
        # Calculate target_dates (dates used for __getitem__ indexing)
        # Last sample's output must end at end_date
        # So: last_target + seq_length - 1 = end_date
        # Therefore: last_target = end_date - seq_length + 1
        first_target_date = self.start_date
        last_target_date = self.end_date - pd.Timedelta(days=sequence_length - 1)
        
        if last_target_date < first_target_date:
            raise ValueError(
                f"Date range too small for sequence_length={sequence_length}.\n"
                f"Need at least {sequence_length} days between start and end dates.\n"
                f"Got: start={self.start_date.date()}, end={self.end_date.date()}"
            )
        
        self.target_dates = pd.date_range(first_target_date, last_target_date)
        
        # Reorder variables according to canonical ordering
        # This ensures consistent ordering regardless of config order
        self.vars = self._canonicalize_variables(variables)

        # Scaling toggle
        self.scale_data = False
        self.scaling_dict = None
       
        # Call subroutine to handle initial data loading
        self._load_data()
    
    def _canonicalize_variables(self, variables):
        """
        Reorder variables according to CANONICAL_VARIABLE_ORDER.
        
        This ensures that the dataset always returns variables in a consistent order,
        preventing errors when loading trained models with different config orderings.
        
        Args:
            variables (list): List of variable names from config
            
        Returns:
            list: Variables reordered according to canonical ordering
            
        Raises:
            ValueError: If a variable is not in CANONICAL_VARIABLE_ORDER
        """
        # Check that all requested variables are in the canonical list
        unknown_vars = set(variables) - set(CANONICAL_VARIABLE_ORDER)
        if unknown_vars:
            raise ValueError(
                f"Unknown variables not in CANONICAL_VARIABLE_ORDER: {unknown_vars}\n"
                f"Please add them to CANONICAL_VARIABLE_ORDER in data_module.py"
            )
        
        # Reorder according to canonical ordering
        canonical_vars = [var for var in CANONICAL_VARIABLE_ORDER if var in variables]
        
        # Log if reordering occurred
        if canonical_vars != variables:
            import logging
            logger = logging.getLogger('ice_cover_dataset')
            logger.info(
                f"Variables reordered from config order {variables} "
                f"to canonical order {canonical_vars}"
            )
        
        return canonical_vars
        
    def __len__(self):
        # return 1 + self.inputs.shape[0] - (2*self.seq_length - self.overlap)
        return self.N

    def __getitem__(self, index):
        """
        Get a single sample from the dataset.
        
        Returns inputs, lake_mask, and target outputs for the given index.
        
        Window structure (with seq_length=7):
        - target_date = 2021-01-01
        - Input sequence: [2020-12-25 to 2020-12-31] (7 days historical ice + weather)
        - Output sequence: [2021-01-01 to 2021-01-07] (7 days target ice)
        - Total window: 14 days
        """
        # 1. Map index to site and target date
        assert (index >= 0 and index < self.N), "index out of dataset range."
        site_no = bisect.bisect_right([rang[1] for rang in self.index_map.values()], index)
        site_name = self.sites[site_no]
        
        # Get the target date (first prediction date for this sample)
        target_date = self.target_dates[index % len(self.target_dates)]
        
        # Calculate window boundaries
        # Input: [target_date - seq_length, target_date - 1]
        # Output: [target_date, target_date + seq_length - 1]
        window_start = target_date - pd.Timedelta(days=self.seq_length)
        window_end = target_date + pd.Timedelta(days=self.seq_length - 1)
        
        # Load the 2*seq_length day window
        data_sequence = self.dataset[site_name].sel(
            time=pd.date_range(window_start, window_end, freq='D')
        )
        
        # Verify we got the expected window size
        assert len(data_sequence.time) == self.window_size, \
            f"Expected {self.window_size} timesteps, got {len(data_sequence.time)}"
        
        # 2. Seperate IMS, ERA5 and Constants - convert to numpy arrays
        era5_sequence = data_sequence[self.vars].to_array().values
        ims_sequence  = data_sequence['IMS_Surface_Values'].values
        lake_mask     = data_sequence['lake_mask'].values
        dim_x, dim_y  = lake_mask.shape
        lake_mask = torch.from_numpy(lake_mask).float().reshape(-1,dim_x,dim_y)
        
        assert era5_sequence.shape[1] == self.window_size
        assert ims_sequence.shape[0] == self.window_size
                
        # 3. Process IMS samples into usable format (one-hot encoding)
        missing_ims = np.isnan(ims_sequence).any()
        if (missing_ims):
            ims_sequence = np.nan_to_num(ims_sequence, nan=2) # set all values to 'out of bounds'
        
        ims_sequence = torch.from_numpy(ims_sequence).long()
        ims_sequence = one_hot(ims_sequence, num_classes=3).float()
        ims_sequence = torch.moveaxis(ims_sequence, -1, 1)

        # Assert that class 2 is equal to the lake mask
        if not missing_ims:
            assert torch.all(ims_sequence[:,2] == 1-lake_mask), "Lake mask and IMS class 2 are not equal."
            assert torch.all(ims_sequence[:,1] + ims_sequence[:,0] == lake_mask), "Lake mask is same as IMS class 1 and 0."
        
        # 4. Process ERA5 data for use
        # Remove any nan / inf values (missing data) - 
            # This should only be the case for Great Bear Lake, where a small 'piece' of ocean is included
        era5_sequence = np.nan_to_num(era5_sequence, nan=0.0, posinf=0.0, neginf=0.0)
        # Scale data if toggle on
        if self.scale_data is True:
            means, stds = np.array(list(self.scaling_dict.values())).T
            means = means.reshape(-1, 1, 1, 1)
            stds = stds.reshape(-1, 1, 1, 1)
            era5_sequence = (era5_sequence - means) / stds
        
        lake_mask = lake_mask.reshape(1,1,dim_x,dim_y)

        # Convert to torch tensor and swap the time/channel axes
        era5_sequence = torch.from_numpy(era5_sequence)
        era5_sequence = torch.moveaxis(era5_sequence, 0, 1)
        
        # 5. Seperate window into desired input/output sequences
        # Input: first seq_length days of ice + last seq_length days of weather
        # Output: last seq_length days of ice (ground truth targets)
        initial_ims = ims_sequence[0:self.seq_length]  # Historical ice
        future_era5 = era5_sequence[self.seq_length:2*self.seq_length]  # Future weather
        
        # The future ims sample is our 'ground truth'. Remove the land class so it has 2 classes present, that way it is in the same
        # format as the model prediction (which is a binary classification). Otherwise will likely be an error in the loss calculation.
        future_ims = ims_sequence[self.seq_length:2*self.seq_length, 0:2]  # Future ice (targets)
        # The future ims sample is our 'ground truth'. Remove the land class so it has 2 classes present, that way it is in the same
        # format as the model prediction (which is a binary classification). Otherwise will likely be an error in the loss calculation.
        future_ims = ims_sequence[self.seq_length:2*self.seq_length, 0:2]  # Future ice (targets)
        
        assert initial_ims.size(0) == future_era5.size(0), "Not getting equivalent sequence sizes..."
        assert initial_ims.size(0) == future_ims.size(0), "Not getting equivalent sequence sizes..."

        # Stack the initial ims with the future era5
        inputs = torch.concat([future_era5, initial_ims], dim=1)
        return inputs, lake_mask, future_ims

    def _verify_coverage_dates(self, start_date, end_date):
        """
        Verify that the requested target dates can be covered by the dataset.
        This assumes the underlying dataset spans 2004-02-25 to 2021-12-31
        """
        
        # The start date must be >= 2004-02-25 + seq_length
        min_target_date = pd.to_datetime("2004-02-25") + pd.Timedelta(days=self.seq_length)
        if start_date < min_target_date:
            raise ValueError(
                f"Requested start_date {start_date} is too early for dataset coverage.\n"
                f"Earliest possible start_date is {min_target_date.date()} "
                f"to allow for sufficient historical data."
            )
            
        # The end date must be <= 2021-12-31
        max_target_date = pd.to_datetime("2021-12-31")
        if end_date > max_target_date:
            raise ValueError(
                f"Requested end_date {end_date} is too late for dataset coverage.\n"
                f"Latest possible end_date is {max_target_date.date()}."
            )        

    def classify_dates(self):
        """
        Classify each sample as freeze/thaw or stable.
        
        For each target_date, check if the OUTPUT window (seq_length days starting
        from target_date) has freeze/thaw activity (FT_Class == True).
        """
        classes = []
        
        for data in self.dataset.values():
            # For each target date, check the prediction window for FT activity
            site_classes = self.target_dates.map(
                lambda d: data.sel(
                    time=pd.date_range(d, periods=self.seq_length, freq='D')
                ).FT_Class.any().values
            )
            classes.append(site_classes)
        
        # Stack the classes
        classes = np.concatenate(classes, axis=0)
        return classes

    def toggle_scaling(self, scaling_dict):
        # Toggle on scaling, and store dictionary provided
        
        # Check that all variables in the scaling dict are in the dataset
        for var in scaling_dict.keys():
            assert var in self.vars, f"Provided variable {var} not in dataset."

        self.scaling_dict = scaling_dict
        self.scale_data = True

    def statistics(self):
        # Return some statistics about the dataset, for scaling purposes
        site_stats = {}

        # Calculate the mean and std's for each site
        for site in self.sites:
            # Calculate the mean and std of each variable
            site_data = self.dataset[site][self.vars]
            mask = self.dataset[site]["lake_mask"]
            # site_means = site_data.where(mask == 1).mean(dim=['time', 'x', 'y']).to_array().values.astype('float32')
            # site_stds  = site_data.where(mask == 1).std(dim=['time', 'x', 'y']).to_array().values.astype('float32')   
            site_means = site_data.mean(dim=['time', 'x', 'y']).to_array().values.astype('float32')
            site_stds  = site_data.std(dim=['time', 'x', 'y']).to_array().values.astype('float32')
            site_stats[site] = dict(zip(self.vars, list(zip(site_means, site_stds))))

        # Combine the means / std
        cur_mean, cur_std = None, None
        N_cur = 0 # the number of elements in the growing mean / std
        for site in self.sites:
            if cur_mean is None and cur_std is None:
                cur_mean, cur_std = np.array(list(site_stats[site].values())).T
                N_cur = self.index_map[site][1] - self.index_map[site][0]
            else:
                # Combine to get the multi-site mean and std
                N_site = self.index_map[site][1] - self.index_map[site][0]
                # Retrieve this sites mean/std
                site_mean, site_std = np.array(list(site_stats[site].values())).T
                # Calculate the new mean from current combining with this site
                new_mean = ((cur_mean * N_cur) + (site_mean * N_site)) / (N_cur + N_site)
                cur_std = np.sqrt(((cur_std**2 * N_cur) + (site_std**2 * N_site) + \
                                   ((cur_mean - new_mean)**2 * N_cur) + ((site_mean - new_mean)**2 * N_site))\
                                      / (N_cur + N_site))
                cur_mean = new_mean
                # update N_total
                N_cur = N_cur + N_site

        across_site_stats = dict(zip(self.vars, list(zip(cur_mean, cur_std))))   
        return across_site_stats, site_stats

    def get_mask(self, site_name):
        # Return the lake mask for the site
        try:
            # Returned in NUMPY format...
            return self.dataset[site_name].lake_mask.values
        except KeyError:
            print("Trouble finding provided site_name")

    def _load_data(self):
        """
        Load NetCDF data for all sites based on coverage dates.
        
        Loads data from (start_date - seq_length) to end_date to ensure
        all samples have sufficient historical context and prediction targets.
        """
        total_length = 0
        self.index_map = {}
        
        # Calculate required data range
        earliest_needed = self.start_date - pd.Timedelta(days=self.seq_length)
        latest_needed = self.end_date
        
        for site in self.sites:
            # Number of samples for this site
            num_samples = len(self.target_dates)
            
            # Load source dataset
            source_ds = xr.open_dataset(
                f"{self.directory}{site}.nc",
                engine="h5netcdf"
            )

            # Select required date range (coverage check)
            try:
                site_data = source_ds.sel(
                    time=pd.date_range(earliest_needed, latest_needed, freq='D')
                )
            except KeyError as e:
                # Provide helpful error message
                min_available = pd.Timestamp(source_ds.time.values.min())
                max_available = pd.Timestamp(source_ds.time.values.max())
                
                raise ValueError(
                    f"Insufficient data for site '{site}' to satisfy requested coverage.\n"
                    f"\n"
                    f"Available data: {min_available.date()} to {max_available.date()}\n"
                    f"Required data:  {earliest_needed.date()} to {latest_needed.date()}\n"
                    f"\n"
                    f"Requested coverage: {self.start_date.date()} to {self.end_date.date()}\n"
                    f"Sequence length: {self.seq_length} days\n"
                    f"\n"
                    f"Requirements:\n"
                    f"  - Need {self.seq_length} days of historical data before start_date\n"
                    f"  - Need data through end_date for predictions\n"
                    f"\n"
                    f"Suggestions:\n"
                    f"  - Move start_date forward to at least {min_available + pd.Timedelta(days=self.seq_length)}\n"
                    f"  - Move end_date back to at most {max_available}\n"
                ) from e

            # Validate that all requested variables are present in the dataset
            required_vars = self.vars + ['lake_mask', 'IMS_Surface_Values', 'flake_ice_depth']
            missing_vars = [var for var in required_vars if var not in source_ds.data_vars]
            if missing_vars:
                available_vars = sorted(list(source_ds.data_vars))
                raise ValueError(
                    f"Missing required variables for site '{site}': {missing_vars}\n"
                    f"Available variables: {available_vars}"
                )

            site_data = site_data[self.vars + ['lake_mask', 'IMS_Surface_Values', 'flake_ice_depth']]
            
            # Calculate additional variables
            site_data = site_data.assign(
                IMS_FIC=lambda x: x.IMS_Surface_Values.where(
                    x.IMS_Surface_Values == 1
                ).where(
                    x.lake_mask == 1
                ).sum(dim=('x','y')) / (x.lake_mask.sum())
            )
            site_data = site_data.assign(
                FT_Class=lambda x: ((x.IMS_FIC > 0.05) & (x.IMS_FIC < 0.95))
            )

            # Update mapping
            self.index_map[site] = [total_length, total_length + num_samples]
            total_length += num_samples
            
            # Extract constants
            self.lake_masks[site] = site_data['lake_mask'].values
            
            # Load into memory
            self.dataset[site] = site_data.sortby('time').load()
        
        # Store total number of samples
        self.N = total_length

    def get_date_from_index(self, index):
        """
        Returns the target_date for a given dataset index.
        
        Args:
            index: Dataset index
        Returns:
            pd.Timestamp: The target_date corresponding to the index
        """
        assert (index >= 0 and index < self.N), "index out of dataset range."
        site_no = bisect.bisect_right([rang[1] for rang in self.index_map.values()], index)
        site_name = self.sites[site_no]
        
        date_index = index % len(self.target_dates)
        target_date = self.target_dates[date_index]
        return target_date


    def get_lake_date_index(self, date, lake):
        """
        Returns the dataset index for a given target_date and lake site.
        
        Args:
            date: Target date (first prediction date for that sample)
            lake: Lake site name
            
        Returns:
            int: The dataset index for the given date and lake.
        """
        assert lake in self.sites, "Provided lake site not in dataset."
        assert date in self.target_dates, f"Provided date {date} not in target_dates. Available: {self.target_dates[0]} to {self.target_dates[-1]}"
        
        site_offset = self.index_map[lake][0]
        date_index = self.target_dates.get_loc(date)
        
        dataset_index = site_offset + date_index
        return dataset_index
