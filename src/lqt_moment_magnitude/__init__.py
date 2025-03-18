"""
LQTMomentMag: A Python package for calculating moment magnitude using full P, SV, and SH energy components.

This package provides tools for seismic data processing, ray tracing, and spectral fitting.
For coder use, import 'magnitude_estimator' or 'reload-configuration' from this module.
For command-liner use, run 'LQTMwCalc'. 

Example:
    >>> from LQTMomentMag import magnitude_estimator
    >>> result_df, fitting_df = magnitude_estimator(
    ...         wave_dir="user_dir/data/waveforms",
    ...         cal_dir="user_dir/data/calibration",
    ...         fig_dir="user_dir/figures",      
    ...         catalog_df=catalog_df, 
    ...         config_file="user_dir/config.ini"
    ...         )
"""

from .main import main
from .api import magnitude_estimator, reload_configuration
from .processing import instrument_remove
from .utils import read_waveforms
from .refraction import calculate_inc_angle
from .fitting_spectral import fit_spectrum_qmc


__version__ = "1.0.0"
__author__ = "Arham Zakki Edelo"
__email__ = "edelo.arham@gmail.com"
__all__ = [
    "main",
    "magnitude_estimator",
    "reload_configuration",
    "calculate_inc_angle",
    "read_waveforms",
    "fit_spectrum_qmc",
    "instrument_remove"
    ]