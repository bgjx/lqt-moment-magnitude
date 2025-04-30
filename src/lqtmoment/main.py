"""
Main entry point for the lqt-moment-magnitude package.

Version: 0.2.1

This module provides complete automatic calculation for seismic moment magnitude
in the LQT component system.

Dependencies:
    - See `pyproject.toml` or `pip install lqtmoment` for required packages.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd 
from datetime import datetime
from typing import Optional, List

from .utils import load_data, setup_logging, REQUIRED_CATALOG_COLUMNS
from .config import CONFIG

try:
    from .processing import start_calculate
except ImportError as e:
    raise ImportError("Failed to import processing module. Ensure lqtmoment is installed correctly.") from e


logger = setup_logging()

def main(args: Optional[List[str]] = None) -> None:
    """ 
    Calculate moment magnitude in the LQT component system.

    This function serves as the entry point for the LQTMwCalc command-line tool.
    It parses arguments, loads the seismic catalog, and initiates the moment magnitude
    calculation process.

    Args:
        args (Optional[List[str]]): Command-line arguments. Defaults to sys.argv[1:] if None.

    Returns:
        None: This function saves results to Excel files and logs the process.
    
    Raises:
        FileNotFoundError: If required input paths do not exists.
        PermissionError: If directories cannot be created.
        ValueError: If calculation output is invalid.
    
    Examples:
    ``` bash
        $ lqtmoment --help
        $ lqtmoment --wave-dir data/waveforms --catalog-file data/catalog/lqt_catalog.xlsx
        $ lqtmoment --wave-dir data/waveforms --catalog-file data/catalog/lqt_catalog.xlsx --config data/new_config.ini
    ```
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
        "--catalog-file",
        type=Path,
        default=Path("data/catalog/lqt_catalog.xlsx"),
        help="LQT formatted catalog file")
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("data/new_config.ini"),
        help="Path to custom config.ini file to reload")
    parser.add_argument(
        "--id-start",
        type=int,
        help = "Starting earthquake ID."
    )
    parser.add_argument(
        "--id-end",
        type=int,
        help="Ending earthquake ID."
    )
    parser.add_argument(
        "--non-lqt",
        action="store_false",
        dest="lqt_mode",
        help="Use ZRT rotation instead of LQT for very local earthquake."
    )
    parser.add_argument(
        "--create-figure",
        action="store_true",
        help="Generate and save spectral fitting figures."
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("results/figures"),
        help="Path to save figures")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/calculation"),
        help="Output directory for results")
    parser.add_argument(
        "--output-format",
        type = str,
        default='excel',
        help = "Set output format for saving results ('Excel' or 'csv'). Defaults to excel."
    )
    parser.add_argument(
        "--result-file-prefix",
        type = str,
        default="lqt_magnitude",
        help = "Set prefix for result file names. Defaults to 'lqt_magnitude'"
    ),
    parser.add_argument(
        "--version",
        action='version',
        version=f"%(prog)s {__import__('lqtmoment').__version__}",
        help = "Show the version and exit"
    )
    args = parser.parse_args(args if args is not None else sys.argv[1:])

    # Reload configuration if specified
    if args.config_file and args.config_file.exists():
        try:
            CONFIG.reload(args.config_file)
        except (FileNotFoundError, ValueError) as e:
            raise RuntimeError (f"Failed to reload configuration: {e}")
    elif args.config_file and not args.config_file.exists():
        raise FileNotFoundError(f"Config file {args.config_file} not found, using default configuration")

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
            
    # Load and validate catalog
    catalog_df = load_data(args.catalog_file)
    missing_columns = [col for col in REQUIRED_CATALOG_COLUMNS if col not in catalog_df.columns]
    if missing_columns:
        raise ValueError(f"Catalog missing required columns: {missing_columns}")

    # Call the function to start calculating moment magnitude
    logger.info(f"Starting magnitude calculation for catalog: {args.catalog_file}")
    try:
        merged_catalog_df, mw_result_df, mw_fitting_df = start_calculate(
                                                wave_path= args.wave_dir,
                                                calibration_path= args.cal_dir,
                                                catalog_data= catalog_df,
                                                id_start= args.id_start,
                                                id_end= args.id_end,
                                                lqt_mode= args.lqt_mode,
                                                generate_figure= args.create_figure,
                                                figure_path= args.fig_dir,        
                                        )
    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        raise ValueError(f"Failed to calculate moment magnitude: {e}") from e
    
    # Validate calculation output
    if mw_result_df is None or mw_fitting_df is None:
        raise ValueError("Calculation return invalid results (None).")

    # Saving the results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_file = f"{args.result_file_prefix}_merged_catalog_{timestamp}"
    result_file  = f"{args.result_file_prefix}_result_{timestamp}"
    fitting_file = f"{args.result_file_prefix}_fitting_result_{timestamp}"
    logger.info(f"Saving results to {args.output_dir}")
    try:
        if args.output_format.lower() == 'excel':
            merged_catalog_df.to_excel(args.output_dir / f"{merged_file}.xlsx", index = False)
            mw_result_df.to_excel(args.output_dir / f"{result_file}.xlsx", index = False)
            mw_fitting_df.to_excel(args.output_dir/ f"{fitting_file}.xlsx", index = False)
        elif args.output_format.lower() == 'csv':
            merged_catalog_df.to_csv(args.output_dir / f"{merged_file}.csv", index = False)
            mw_result_df.to_csv(args.output_dir / f"{result_file}.csv", index = False)
            mw_fitting_df.to_csv(args.output_dir/ f"{fitting_file}.csv", index = False)
        else:
            raise ValueError(f"Unsupported output format: {args.output_format}. Use 'excel' or 'csv'. ")
    except Exception as e:
        raise RuntimeError(f"Failed to save results: {e}")

    return None

if __name__ == "__main__" :
    main()