# Lake Ice Forecasting with Deep Learning (LIF-DL)

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.2.4](https://img.shields.io/badge/PyTorch-2.2.4-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code and data used for the research study.

## Overview

LIF-DL (Lake Ice Forecasting with Deep Learning) is a PyTorch-based model that can predict lake ice cover across entire lake surfaces in a single-shot approach.
- **Meteorological inputs**: Temperature, solar radiation, precipitation, wind speed, humidity, cloud cover, accumulated degree days
- **Static features**: Lake bathymetry
- **Previous ice state**: Ice cover conditions leading up to the forecast date(s)

The model has been trained and evaluated on five major North American lakes:
- Great Bear Lake
- Great Slave Lake
- Lake Athabasca
- Lake Winnipeg
- Reindeer Lake

## Key Features

- **Model Configuration and Training**: Train your own version of the LIF-DL.
- **Forecasting Autoregressively**: Deploy a trained model to forecast autoregressively.
- **Multi-Source Evaluation**: Compares the LIF-DL forecasts against IMS (Interactive Multisensor Snow and Ice Mapping System), CIS (Canadian Ice Service), and FLake model.
- **Variable Importance**: Gradient-based attribution to understand which meteorological variables driv predictions.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/h2o-geomatics/Ice_Cover_Modelling.git
cd Ice_Cover_Modelling
```

### 2. Create the conda environment
```bash
conda env create -f environment.yaml
conda activate lif_dl
```

### 3. Install the package in development mode
```bash
pip install -e .
```

## Project Structure

```
Ice_Cover_Modelling/
├── configs/                   # Configuration files
│   ├── default.yaml             # Default training configuration
│   └── debug.yaml               # Debug configuration for testing
├── data/                      # Data directory
│   ├── nc/                      # NetCDF files for each lake
│   └── cis/                     # Canadian Ice Service CSV observations for each lake
├── src/                       # Source code
│   ├── data/                    # Data loading and preprocessing
│   ├── model/                   # Model architecture and training
│   ├── eval/                    # Evaluation framework
│   └── utils/                   # Logging and utilities
├── scripts/                   # Executable scripts
│   ├── train.py                 # Model training
│   ├── forecast.py              # Generate predictions
│   ├── evaluate.py              # Run evaluation metrics
│   └── variable_importance.py   # Variable importance analysis
├── notebooks/                 # Jupyter notebooks
│   ├── 00_load_model.ipynb      # Model loading examples
│   └── 01_create_plots.ipynb    # Results visualization
├── results/                   # Model outputs and evaluations
```

## Usage

### Training a Model

Train the model using the default configuration:

```bash
python scripts/train.py --config configs/default.yaml --name [YOUR-MODEL-NAME]
```

For a quick test run with reduced data:

```bash
python scripts/train.py --config configs/debug.yaml --name [YOUR-MODEL-NAME]
```

Model run data are saved to `./results/[YOUR-MODEL-NAME]/` by default.

#### Weights & Biases Integration (Optional)

To track experiments with [Weights & Biases](https://wandb.ai/):

1. Install wandb (if not already installed):
```bash
pip install wandb
```

2. Login to your wandb account:
```bash
wandb login
```

3. Enable wandb logging in your config file:
```yaml
use_wandb: true
wandb_project: "lake-ice-forecasting"
wandb_entity: "your-username"  # Optional
```

Training metrics, losses, and model checkpoints will be automatically logged to your wandb dashboard.

#### Configuration

Edit `configs/default.yaml` to customize run parameters

**Data**:
```yaml
data_dir: "data/nc/"
sites: ["great_slave_lake", "great_bear_lake", ...]
start_date: "2010-01-01"
end_date: "2012-12-31"
```

**Model**:
```yaml
sequence_length: 7
hidden_channels: 64
kernel_size: 3
num_layers: 3
```

**Training**:
```yaml
batch_size: 4
max_epochs: 100
learning_rate: 0.001
dropout: 0.1
```

### Generating Forecasts

Generate predictions for a trained model using all sites in its training config.

```bash
python scripts/forecast.py --name [YOUR-MODEL-NAME]
```

Options:
- `--site`: Forecast only for a single site (e.g., `great_slave_lake`)

### Running Evaluation

Evaluate model performance against observations:

```bash
python scripts/evaluate.py --name [YOUR-MODEL-NAME]
```

**Options**:
- `--save-intermediate`: Save the intermediate data from evaluators. Used for figure generation in result_figures.ipynb
- `--variable-importance`: Runs variable importance estimation after evaluation.

**Example with all features**:
```bash
python scripts/evaluate.py \
  --name LIF_DL_Best \
  --save-intermediate \
  --variable-importance
```

This will generate:
- `results/LIF_DL_Best/evaluations/*.csv` - Metric tables
- `results/LIF_DL_Best/evaluations/intermediate/` - Data for figures
- `results/LIF_DL_Best/evaluations/variable_importance.csv` - Importance scores

### Variable Importance Analysis

Compute gradient-based variable importance separately:

```bash
python scripts/compute_variable_importance.py --name LIF_DL_Best
```

This analyzes how much each meteorological variable contributes to predictions, with separate scores for overall, break-up, and freeze-up periods.

### Visualization

### Input Data Structure

The model expects NetCDF files in `data/nc/` with the following variables:
- `temperature_2m`: 2-meter air temperature (K)
- `surface_solar_radiation_downwards_sum`: Solar radiation (J/m²)
- `total_precipitation_sum`: Precipitation (m)
- `wind_speed_10m`: Wind speed (m/s)
- `relative_humidity`: Relative humidity (0-1)
- `total_cloud_cover`: Cloud cover fraction (0-1)
- `accumulated_freezing_dd`: Accumulated freezing degree days (K)
- `accumulated_thawing_dd`: Accumulated thawing degree days (K)
- `lake_depth`: Lake depth (m)
- `IMS_Surface_Values`: Ice cover observations (0=water, 1=ice, 2=land)
- `lake_mask`: Lake boundary mask (0=land, 1=lake)
- `flake_ice_depth`: FLake ice thickness (m)

Dimensions: `(time, y, x)` where spatial dimensions are 128×128 pixels.

### Observation Data

- **IMS**: Daily gridded ice observations from NSIDC
- **CIS**: Weekly ice charts (CSV format in `data/cis/`)

<!-- 
## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{johnston2025lif,
  title={A Deep Learning Approach for Lake Ice Cover Forecasting},
  author={Johnston, Samuel and Murfitt, Justin and Duguay, Claude},
  journal={[Journal Name]},
  year={2025}
}
``` -->

## Data Availability

- **IMS Data**: Retrieved from the National Snow and Ice Data Center  
  https://doi.org/10.7265/N52R3PMC (4km IMS data, 2004-2021)
  
- **ERA5 Reanalysis**: Downloaded from the Copernicus Climate Data Store (CDS)
  - ERA5: https://doi.org/10.24381/cds.adbb2d47
  - ERA5-Land: https://doi.org/10.24381/cds.e2161bac
  
- **GLDB Bathymetry**: Global Lake Database  
  http://www.flake.igb-berlin.de/old/ep-data.shtml
  
- **Canadian Ice Service**: Weekly ice charts available from the Canadian Ice Service

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

The authors would like to thank the Digital Research Alliance of Canada (DRAC), who supported this work through a high-performance computing resource allocation.
<!-- NSERC GOES HERE -->

## Contact

For questions or issues:
- **Repository**: https://github.com/h2o-geomatics/Ice_Cover_Modelling
- **Issues**: https://github.com/h2o-geomatics/Ice_Cover_Modelling/issues
