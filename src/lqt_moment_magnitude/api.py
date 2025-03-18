#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A public API for the LQTMomentMag package.

This module provides functions to calculate seismic moment magnitude in the LQT
component system. Users can import and use these functions in their own Python scripts.

Example:
    >>> from LQTMomentMag import magnitude_estimator
    >>> result_df, fitting_df, magnitude_estimator(
    ...     wave_dir = "user_dir/data/waveforms",  
    ...     cal_dir = "user_dir/data/calibration"
    ...     fig_dir = "user_dir/figures",
    ...     catalog_df = catalog_df,
    ...     config_file = "user_dir/data/config.ini"
    ... )
"""

import warnings
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
from .config import CONFIG
from .processing import start_calculate


# Set up logging handler
warnings.filterwarnings("ignore")
logging.basicConfig(
    filename = 'lqt_runtime.log',
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("LQTMomentMag")


def magnitude_estimator(
    wave_dir: str,
    cal_dir: str,
    fig_dir: str,
    catalog_df: pd.DataFrame,
    output_dir: str = "results",
    config_file: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    Calculate seismic moment magnitude in the LQT component system.

    This function processes waveform data using the LQT component system
    to compute moment magnitudes, saving results and figures as specified.

    Args:
        wave_dir (str): Path to the waveform directory.
        cal_dir (str): Path to the calibration directory.
        fig_dir (str): path to save figures.
        catalog_df (pd.DataFrame): DataFrame containing the seismic catalog.
        output_dir (str, optional): Output directory for results. Defaults to "results".
        config_file (str, optional): Path to a custom config.ini file to reload. Defaults to None.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the result DataFrame and fitting DataFrame
    
    Raises:
        FileNotFoundError: If input directories or config file do not exist.
        ValueError: If the catalog is empty or calculation fails.
        PermissionError: If output directories cannot be created.
    """

    # Convert string paths to Path objects
    wave_dir = Path(wave_dir)
    cal_dir = Path(cal_dir)
    fig_dir = Path(fig_dir)
    output_dir = Path(output_dir)

    # Reload configuration file if specified
    if config_file:
        config_path = Path(config_file)
        if config_path.exists():
            try:
                CONFIG.reload(config_path)
            except (FileNotFoundError, ValueError) as e:
                raise ValueError(f"Failed to reload configuration form {config_file}: {e}")
            
        else:
            raise FileNotFoundError(f"Config file {config_file} not found")
    
    # Validate input paths
    for path in [wave_dir, cal_dir]:
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
    
    # Create output directories
    try:
        fig_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Permission denied creating directories: {e}")
    
    # Validate catalog
    if catalog_df.empty:
        raise ValueError("Catalog DataFrame is empty")

    # Call the processing function
    mw_result_df, mw_fitting_df, output_name = start_calculate(wave_dir, cal_dir, fig_dir, catalog_df)

    # Validate calculation output:
    if mw_result_df is None or mw_fitting_df is None:
        raise ValueError("Calculation returned invalid results (None).")
    
    # Saving the results
    try:
        mw_result_df.to_excel(output_dir/f"{output_name}_result.xlsx", index=False)
        mw_fitting_df.to_excel(output_dir/f"{output_name}_fitting_result.xlsx", index=False)
    except Exception as e:
        raise RuntimeError(f"Failed to save results: {e}")
    
    return mw_fitting_df, mw_fitting_df

# Optional: Expose a function to reload config without running calculation

def reload_configuration(config_file: str) -> None:
    """
    Reload the configuration from a specified config.ini file.

    Args:
        config_file (str): Path to the custom config.ini file.
    
    Raises:
        FileNotFoundError: If the config file is not found.
        ValueError: If the configuration is invalid.   
    """
    CONFIG.reload(Path(config_file))