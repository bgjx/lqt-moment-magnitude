"""
lqtmoment: A Python package for calculating moment magnitude using full P, SV, and SH energy components.

Version: 0.2.1

This package computes moment magnitude using full P, SV, and SH energy components, with support for
seismic data processing, ray tracing in a 1-D velocity model, and rapid spectral fitting via advanced
stochastic methods. It is designed for seismologists and researchers analyzing earthquake data.

Dependencies:
    - See `pyproject.toml` or `pip install lqtmoment` for required packages.

Key Features:
- Vectorized computation for incidence angles (LQT rotation)
- Spectral fitting with Brune or Boatwright models
- Instrument response removal and waveform processing

For programmatic use, import as `lqtmoment` (alias for `lqt_moment_magnitude`):
    >>> from lqtmoment import magnitude_estimator
    >>> import pandas as pd
    >>> catalog_df = pd.DataFrame({'event_id': [1], 'time': ['2023-01-01T00:00:00']})
    >>> merged_catalog_df, result_df, fitting_df = magnitude_estimator(
    ...                                             wave_dir="data/waveforms",
    ...                                             cal_dir="data/calibration",
    ...                                             fig_dir="figures",
    ...                                             catalog_df=catalog_df,
    ...                                             config_file="config.ini"
    ...                                             )

For CLI use, run:
    $ lqtmoment --help

See the full documentation at https://github.com/bgjx/lqt-moment-magnitude.
"""

from .api import magnitude_estimator, reload_configuration
from .processing import instrument_remove
from .utils import read_waveforms
from .refraction import calculate_inc_angle
from .fitting_spectral import fit_spectrum_qmc
from .catalog_builder import build_catalog
from .main import main

__all__ = [
    "build_catalog",
    "calculate_inc_angle",
    "fit_spectrum_qmc",
    "instrument_remove",
    "magnitude_estimator",
    "main",
    "read_waveforms",
    "reload_configuration",
    ]

# Package metadata
try:
    from importlib.metadata import version
    __version__ = version("lqtmoment")
except ImportError:
    __version__ = "0.1.0"

__author__ = "Arham Zakki Edelo"
__email__ = "edelo.arham@gmail.com"