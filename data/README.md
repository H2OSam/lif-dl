# Data Directory

This directory contains the preprocessed datasets used for training and evaluating the LIF-DL model.

## Directory Structure

```
data/
├── nc/                        # NetCDF files - one per lake
│   ├── great_bear_lake.nc
│   ├── great_slave_lake.nc
│   ├── lake_athabasca.nc
│   ├── lake_winnipeg.nc
│   └── reindeer_lake.nc
└── cis/                       # Canadian Ice Service observations
    ├── great_bear_lake.csv
    ├── great_slave_lake.csv
    ├── lake_athabasca.csv
    ├── lake_winnipeg.csv
    └── reindeer_lake.csv
```

## NetCDF Files (`nc/`)

Each NetCDF file contains all input variables and observations for a single lake, aligned spatially and temporally.

### Data Variables

**Meteorological forcing** (time, y, x):
- `temperature_2m` - 2-meter air temperature (K)
- `surface_solar_radiation_downwards_sum` - Downward solar radiation (J/m²)
- `total_precipitation_sum` - Total precipitation (m)
- `relative_humidity` - Relative humidity (0-1)
- `total_cloud_cover` - Cloud cover fraction (0-1)
- `wind_speed_10m` - 10-meter wind speed (m/s)
- `accumulated_freezing_dd` - Accumulated freezing degree days (K·day)
- `accumulated_thawing_dd` - Accumulated thawing degree days (K·day)

**Static features** (y, x):
- `lake_depth` - Bathymetry from Global Lake Database (m)
- `lake_mask` - Lake boundary mask (1=lake, 0=land)

**Observations** (time, y, x):
- `IMS_Surface_Values` - Ice cover from IMS (0=water, 1=ice, 2=land)
- `flake_ice_depth` - FLake model ice thickness (m)

### Dataset Specifications

- **Spatial projection**: EPSG:3411 (NSIDC Polar Stereographic North)
- **Grid resolution**: 4 km
- **Grid dimensions**: 128 × 128 pixels (cropped to lake extent)
- **Temporal range**: 2004-02-25 to 2021-12-31
- **Temporal resolution**: Daily

### Data Processing Done

1. **ERA5/ERA5-Land meteorological data**: 
   - Variables sourced from ERA5-Land where available, otherwise from ERA5
   - Derived variables (relative humidity, accumulated freezing/thawing degree days) computed from native ERA5/ERA5-Land fields
   - Hourly fields aggregated to daily (mean or sum depending on variable)
   - Reprojected from native grid to EPSG:3411 using nearest-neighbor interpolation
   - Cropped to lake extent

2. **IMS ice observations**:
   - Downloaded from NSIDC (4 km resolution)
   - Native EPSG:3411 projection
   - Cropped to lake extent

3. **GLDB bathymetry**:
   - Downloaded from Global Lake DataBase
   - Reprojected and interpolated to match IMS grid
   - Static field (no time dimension)

4. **FLake model output**:
   - Downloaded from Copernicus Climate Data Store (lake depth)
   - Aggregated from hourly to daily average
   - Reprojected to match IMS grid and cropped to each lake's extent

All variables are aligned spatially and temporally within each NetCDF file.

## CIS Files (`cis/`)

Weekly ice concentration observations from the Canadian Ice Service.

### Format

CSV files with two columns:
- `Date` - Observation date (YYYY-MM-DD)
- `Ice-covered` - Ice cover concentration (0-10 scale, tenths)

## Data Availability

These preprocessed datasets are derived from publicly available sources:

- **IMS**: https://doi.org/10.7265/N52R3PMC
- **ERA5**: https://doi.org/10.24381/cds.adbb2d47
- **ERA5-Land**: https://doi.org/10.24381/cds.e2161bac
- **GLDB**: http://www.flake.igb-berlin.de/old/ep-data.shtml
- **CIS**: Available through contacting the Canadian Ice Service (https://www.canada.ca/en/environment-climate-change/services/ice-forecasts-observations/about-ice-service.html)

For data access or questions about preprocessing, please open an issue on the repository.