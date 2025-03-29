"""
A public API for the lqt-moment-magnitude package.

Version: 0.1.0

This module provides functions to calculate seismic moment magnitude in the LQT
component system. Users can import and use these functions in their own Python scripts.

Example:
    >>> from lqtmoment import magnitude_estimator
    >>> result_df, fitting_df = magnitude_estimator(
    ...     wave_dir = "data/waveforms",  
    ...     cal_dir = "data/calibration"
    ...     catalog_dir = "data/catalog/catalog.xlsx",
    ...     config_file = "data/new_config.ini"
    ...     fig_dir = "results/figures",
    ... )

Notes:
    - `catalog_df` should contain columns like 'source_id', 'source_lat' and so on... (see documentation for full schema).
    - See https://github.com/bgjx/lqt-moment-magnitude for detailed usage and configuration options.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

from .utils import load_data, setup_logging, REQUIRED_CATALOG_COLUMNS
from .config import CONFIG

try:
    from .processing import start_calculate
except ImportError as e:
    raise ImportError("Failed to import processing module. Ensure lqtmoment is installed correctly.") from e


# Set up logging handler
logger = setup_logging()


def magnitude_estimator(
    wave_dir: str,
    cal_dir: str,
    catalog_file: str,
    config_file: Optional[str] = None,
    fig_dir: str = "figures",
    output_dir: str = "results/calculation",
    id_start: Optional[int] = None,
    id_end: Optional[int] = None,
    generate_figure : bool = None,
    lqt_mode: Optional[bool] = None,
    output_format: str = "excel",
    result_file_prefix: str = "lqt_magnitude"  
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate seismic moment magnitude in the LQT component system.

    This function processes waveform data using the LQT component system
    to compute moment magnitudes, saving results and figures as specified.

    Args:
        wave_dir (str): Path to the waveform directory.
        cal_dir (str): Path to the calibration directory.
        config_file (str, optional): Path to a custom config.ini file to reload. Defaults to None.
        catalog_file (str): Path to the seismic catalog.
        fig_dir (str): path to save figures.
        output_dir (str, optional): Output directory for results. Defaults to "results".
        id_start (Optional[int]): Starting earthquake ID. Defaults to min ID or interactive input.
        id_end (Optional[int]): Ending earthquake ID. Defaults to max ID or interactive input.
        generate_figure (Optional[bool]): Generate and save figures if True. Defaults to False or interactive input.
        lqt_mode (Optional[bool]): Use LQT rotation if True, ZRT otherwise. Defaults to True or interactive input.
        output_format (str): Format for saving results ("excel" or "csv"). Default to "excel".
        result_file_prefix (str): Prefix for result file names. Defaults to "lqt_magnitude"
        
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
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Validate input paths
    for path in [wave_dir, cal_dir]:
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")
    
    # Create output directories
    try:
        fig_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Permission denied creating directories: {e}")
    
    # Load and validate catalog
    catalog_df = load_data(catalog_file)
    missing_columns = [col for col in REQUIRED_CATALOG_COLUMNS if col not in catalog_df.columns]
    if missing_columns:
        raise ValueError(f"Catalog missing required columns: {missing_columns}")

    # Call the processing function
    logger.info(f"Starting magnitude calculation for catalog: {catalog_file}")
    try:
        mw_result_df, mw_fitting_df = start_calculate(
                                        wave_dir, cal_dir, fig_dir,
                                        catalog_df, id_start=id_start,
                                        id_end=id_end, lqt_mode=lqt_mode,
                                        generate_figure=generate_figure,
                                        )
    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        raise ValueError(f"Failed to calculate moment magnitude: {e}") from e

    # Validate calculation output:
    if mw_result_df is None or mw_fitting_df is None:
        raise ValueError("Calculation returned invalid results (None).")
    
    # Saving the results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{result_file_prefix}_result_{timestamp}"
    fitting_file = f"{result_file_prefix}_fitting_result_{timestamp}"
    logger.info(f"Saving results to {output_dir}")
    try:
        if output_format.lower() == "excel":
            mw_result_df.to_excel(output_dir/f"{result_file}.xlsx", index=False)
            mw_fitting_df.to_excel(output_dir/f"{fitting_file}.xlsx", index=False)
        elif output_format.lower() == "csv":
            mw_result_df.to_csv(output_dir/f"{result_file}.csv", index=False)
            mw_fitting_df.to_csv(output_dir/f"{fitting_file}.csv", index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_format}. Use 'excel' or 'csv'. ")
    except Exception as e:
        raise RuntimeError(f"Failed to save results: {e}")
    
    return mw_result_df, mw_fitting_df


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
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        CONFIG.reload(config_path)
    except ValueError as e:
        raise ValueError(f"Failed to reload configuration form {config_path}: {e}")