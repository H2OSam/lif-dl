"""Setup for Ice Cover Modelling package."""

from setuptools import setup, find_packages

setup(
    name="lif_dl",
    version="1.0.0",
    author="Sam Johnston",
    description="Lake Ice Forecasting using Deep Learning",
    url="https://github.com/h2o-geomatics/Ice_Cover_Modelling",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.25.0",
        "pandas>=2.2.0",
        "scipy>=1.15.0",
        "scikit-learn>=1.5.0",
        "matplotlib>=3.10.0",
        "seaborn>=0.13.0",
        "xarray>=2024.3.0",
        "netCDF4",
        "h5netcdf>=1.3.0",
        "h5py>=3.10.0",
        "dask>=2024.1.0",
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "torchmetrics>=1.3.0",
        "pytorch-lightning>=2.2.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0",
        "wandb",
    ],
)