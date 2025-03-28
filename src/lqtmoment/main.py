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
    $ LQTMagnitude --wave-dir data/waveforms --catalog-file data/catalog/lqt_catalog.xlsx --config data/new_config.ini
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
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas")
logging.basicConfig(
    filename = 'lqt_runtime.log',
    level = CONFIG.performance.LOGGING_LEVEL.upper(),
    format = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("lqtmoment")


def load_catalog(catalog_dir: str) -> pd.DataFrame:
    """"
    Load catalog from given catalog dir, this function will handle
    catalog suffix/format (.xlsx / .csv) for more dynamic inputs.

    Args:
        catalog_dir (str): Directory of the catalog file.

    Returns:
        pd.DataFrame: DataFrame of earthquake catalog.
    
    Raises:
        FileNotFoundError: If catalog files do not exist.
        ValueError: If catalog files fail to load or unsupported format.
    """

    catalog_path = Path(catalog_dir)
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Catalog path is not a file: {catalog_path}")
    if catalog_path.suffix == ".xlsx":
        return pd.read_excel(catalog_path, index_col=None)
    elif catalog_path.suffix == ".csv":
        return pd.read_csv(catalog_path, index_col=None)
    else:
        raise ValueError(f"Unsupported catalog file format: {catalog_path.suffix}. Supported formats: .csv, .xlsx")
    

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
        $ LQTMagnitude --wave-dir data/waveforms --catalog-file data/catalog/lqt_catalog.xlsx --config data/new_config.ini

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
        default=Path("results/figures"),
        help="Path to save figures")
    parser.add_argument(
        "--catalog-file",
        type=Path,
        default=Path("data/catalog/lqt_catalog.xlsx"),
        help="LQT formatted catalog file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/calculation"),
        help="Output directory for results")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data/new_config.ini"),
        help="Path to custom config.ini file to reload")
    parser.add_argument(
        "--id-start",
        type=int,
        help = "Starting earthquake ID (overrides interactive input)"
    )
    parser.add_argument(
        "--id-end",
        type=int,
        help="Ending earthquake ID (overrides interactive input)"
    )
    parser.add_argument(
        "--create-figure",
        action="store_true",
        help="Generate and save spectral fitting figures (overrides interactive input)"
    )
    parser.add_argument(
        "--zrt-mode",
        action="store_false",
        dest="lqt_mode",
        help="Use ZRT rotation instead of LQT for very local earthquake (overrides interactive input, default is LQT)"
    )
    parser.add_argument(
        "--version",
        action='version',
        version=f"%(prog)s {__import__('lqtmoment').__version__}",
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
            
    # Load and validate catalog
    catalog_df = load_catalog(args.catalog_file)
    required_columns = [
        "network", "source_id", "source_lat", "source_lon", "source_depth_m",
        "source_origin_time", "station_code", "station_lat", "station_lon",
        "station_elev_m", "p_arr_time", "p_travel_time_sec", "s_arr_time",
        "s_travel_time_sec", "s_p_lag_time_sec", "earthquake_type"
    ]
    missing_columns = [col for col in required_columns if col not in catalog_df.columns]
    if missing_columns:
        raise ValueError(f"Catalog missing required columns: {missing_columns}")

    # Call the function to start calculating moment magnitude
    mw_result_df, mw_fitting_df = start_calculate(args.wave_dir, args.cal_dir, args.fig_dir,
                                                catalog_df, id_start=args.id_start, id_end=args.id_end,
                                                lqt_mode=args.lqt_mode,
                                                figure_statement=args.create_figure,
                                                )
    
    # Validate calculation output
    if mw_result_df is None or mw_fitting_df is None:
        raise ValueError("Calculation return invalid results (None).")

    # save and set dataframe index
    try:
        mw_result_df.to_excel(args.output_dir / "lqt_magnitude_result.xlsx", index = False)
        mw_fitting_df.to_excel(args.output_dir/ "lqt_magnitude_fitting_result.xlsx", index = False)
    except Exception as e:
        raise RuntimeError(f"Failed to save results: {e}")

    return None

if __name__ == "__main__" :
    main()