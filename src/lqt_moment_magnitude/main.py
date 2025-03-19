#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for the lqt-moment-magnitude package.

This module provides complete automatic calculation for seismic moment magnitude
in the LQT component system.

Dependencies:
    - pandas: For catalog data handling.
    - numpy: For numerical calculation.
    - scipy: For mathematical optimization.
    - obspy: For waveform processing.
    - matplotlib: For generating figures.
    - tqdm: For progress feedback

Usage:
    $ LQTMagnitude --help
    $ LQTMagnitude --wave-dir data/waveforms --catalog-file data/catalog/lqt_catalog.xlsx
    $ LQTMagnitude --wave-dir data/waveforms --catalog-file data/catalog/lqt_catalog.xlsx --config path/to/new_config.ini
"""

import sys
import argparse
import logging
import warnings
from pathlib import Path
import pandas as pd 
from typing import Optional, List

from .config import CONFIG

try:
    from .processing import start_calculate
except ImportError as e:
    raise ImportError("Failed to import processing module. Ensure lqtmoment is installed correctly.") from e


# Set up logging handler
warnings.filterwarnings("ignore")
logging.basicConfig(
    filename = 'lqt_runtime.log',
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("lqtmoment")


def main(args: Optional[List[str]] = None) -> None:
    """ 
    Calculate moment magnitude in the LQT component system.

    This function serves as the entry point for the LQTMwCalc command-line tool.
    It parses arguments, loads the seismic catalog, and initiates the moment magnitude
    calculation process.

    Args:
        args (List[str], Optional): Command-line arguments. Defaults to sys.argv[1:] if None.

    Returns:
        None: This function saves results to Excel files and logs the process.
    
    Raises:
        FileNotFoundError: If required input paths do not exists.
        PermissionError: If directories cannot be created.
        ValueError: If calculation output is invalid.
    
    Example:
        $ LQTMagnitude --wave-dir data/waveforms --catalog-file data/catalog/lqt_catalog.xlsx
        $ LQTMagnitude --wave-dir data/waveforms --catalog-file data/catalog/lqt_catalog.xlsx --config path/to/new_config.ini

    """

    parser = argparse.ArgumentParser(description="Calculate moment magnitude in full LQT component.")
    parser.add_argument(
        "--wave-dir",
        type=Path,
        default=Path("data/waveforms"),
        help="Path to waveform directory")
    parser.add_argument(
        "--cal-dir",
        type=Path,
        default=Path("data/calibration"),
        help="Path to the calibration directory")
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("figures"),
        help="Path to save figures")
    parser.add_argument(
        "--catalog-file",
        type=Path,
        default=Path("data/catalog/lqt_catalog.xlsx"),
        help="LQT formatted catalog file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to custom config.ini file to reload")
    parser.add_argument(
        "--version",
        action='version',
        version=f"%(prog)s {__import__('lqt_moment_magnitude').__version__}",
        help = "Show the version and exit"
    )
    args = parser.parse_args(args if args is not None else sys.argv[1:])

    # Reload configuration if specified
    if args.config and args.config.exists():
        try:
            CONFIG.reload(args.config)
        except (FileNotFoundError, ValueError) as e:
            raise RuntimeError (f"Failed to reload configuration: {e}")
    elif args.config and not args.config.exists():
        raise FileNotFoundError(f"Config file {args.config} not found, using default configuration")
    else:
        pass

    # Validate input paths
    for path in [args.wave_dir, args.cal_dir]:
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory:{path}")
    
    if not args.catalog_file.exists():
        raise FileNotFoundError(f"Catalog file not found: {args.catalog_file}")
    if not args.catalog_file.is_file():
        raise ValueError(f"Catalog file must be a file, not a directory")

    # Create output directories
    try:
        args.fig_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Permission denied creating directories: {e}")
            
    # Load catalog with error handling
    try:
        catalog_df = pd.read_excel(args.catalog_file, index_col=None)
    except Exception as e:
        raise ValueError(f"Failed to load catalog file: {e}")

    # Validate catalog
    if catalog_df.empty:
        raise ValueError("Catalog Dataframe is empty.")

    # Call the function to start calculating moment magnitude
    mw_result_df, mw_fitting_df, output_name = start_calculate(args.wave_dir, args.cal_dir, args.fig_dir, catalog_df)
    
    # Validate calculation output
    if mw_result_df is None or mw_fitting_df is None:
        raise ValueError("Calculation return invalid results (None).")

    # save and set dataframe index
    try:
        mw_result_df.to_excel(args.output_dir / f"{output_name}_result.xlsx", index = False)
        mw_fitting_df.to_excel(args.output_dir/ f"{output_name}_fitting_result.xlsx", index = False)
    except Exception as e:
        raise RuntimeError(f"Failed to save results: {e}")

    return None

if __name__ == "__main__" :
    main()