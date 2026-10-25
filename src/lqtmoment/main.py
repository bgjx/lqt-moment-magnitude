"""
Main entry point for the lqt-moment-magnitude package.

This module provides complete automatic calculation for seismic moment magnitude
in the LQT component system.

Dependencies:
    - See `pyproject.toml` or `pip install lqtmoment` for required packages.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .config import CONFIG
from .utils import REQUIRED_CATALOG_COLUMNS, load_data, setup_logging

try:
    from .processing import start_calculate
except ImportError as e:
    raise ImportError(
        "Failed to import processing module. Ensure lqtmoment is installed correctly."
    ) from e


logger = setup_logging()


def _reload_configuration(config_file: Path) -> None:
    """Reload configuration from file if specified and exists."""
    if config_file and config_file.exists():
        try:
            CONFIG.reload(config_file)
        except (FileNotFoundError, ValueError) as e:
            raise RuntimeError(f"Failed to reload configuration: {e}") from e
    elif config_file and not config_file.exists():
        raise FileNotFoundError(
            f"Config file {config_file} not found, using default configuration"
        )


def _validate_input_paths(wave_dir: Path, cal_dir: Path, catalog_file: Path) -> None:
    """Validate that all input paths exist and are correct types."""
    for path in [wave_dir, cal_dir]:
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory:{path}")

    if not catalog_file.exists():
        raise FileNotFoundError(f"Catalog file not found: {catalog_file}")
    if not catalog_file.is_file():
        raise ValueError("Catalog file must be a file, not a directory")


def _create_output_directories(fig_dir: Path, output_dir: Path) -> None:
    """Create output directories with proper error handling."""
    try:
        fig_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Permission denied creating directories: {e}") from e


def _validate_catalog(catalog_df) -> None:
    """Validate that catalog has all required columns."""
    missing_columns = [
        col for col in REQUIRED_CATALOG_COLUMNS if col not in catalog_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Catalog missing required columns: {missing_columns}")


def _run_calculation(
    wave_dir: Path,
    cal_dir: Path,
    catalog_df,
    id_start: int | None,
    id_end: int | None,
    lqt_mode: bool,
    create_figure: bool,
    fig_dir: Path,
):
    """Run the moment magnitude calculation."""
    logger.info("Starting magnitude calculation for catalog")
    try:
        merged_catalog_df, mw_result_df, mw_fitting_df = start_calculate(
            wave_path=wave_dir,
            calibration_path=cal_dir,
            catalog_data=catalog_df,
            id_start=id_start,
            id_end=id_end,
            lqt_mode=lqt_mode,
            generate_figure=create_figure,
            figure_path=fig_dir,
        )
    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        raise ValueError(f"Failed to calculate moment magnitude: {e}") from e

    if mw_result_df is None or mw_fitting_df is None:
        raise ValueError("Calculation return invalid results (None).")

    return merged_catalog_df, mw_result_df, mw_fitting_df


def _save_results(
    merged_catalog_df,
    mw_result_df,
    mw_fitting_df,
    output_dir: Path,
    output_format: str,
    result_file_prefix: str,
) -> None:
    """Save calculation results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_file = f"{result_file_prefix}_merged_catalog_{timestamp}"
    result_file = f"{result_file_prefix}_result_{timestamp}"
    fitting_file = f"{result_file_prefix}_fitting_result_{timestamp}"

    logger.info(f"Saving results to {output_dir}")
    try:
        if output_format.lower() == "excel":
            merged_catalog_df.to_excel(output_dir / f"{merged_file}.xlsx", index=False)
            mw_result_df.to_excel(output_dir / f"{result_file}.xlsx", index=False)
            mw_fitting_df.to_excel(output_dir / f"{fitting_file}.xlsx", index=False)
        elif output_format.lower() == "csv":
            merged_catalog_df.to_csv(output_dir / f"{merged_file}.csv", index=False)
            mw_result_df.to_csv(output_dir / f"{result_file}.csv", index=False)
            mw_fitting_df.to_csv(output_dir / f"{fitting_file}.csv", index=False)
        else:
            raise ValueError(
                f"Unsupported output format: {output_format}. Use 'excel' or 'csv'."
            )
    except Exception as e:
        raise RuntimeError(f"Failed to save results: {e}") from e


def main(args: list[str] | None) -> None:
    """
    Calculate moment magnitude in the LQT component system.

    This function serves as the entry point for the lqtmoment command-line tool.
    It parses arguments, loads the seismic catalog, and initiates the moment magnitude
    calculation process.

    Args:
        args (list[str] | None): CLI arguments. Defaults to sys.argv[1:] if None.

    Returns:
        None: This function saves results to Excel files and logs the process.

    Raises:
        FileNotFoundError: If required input paths do not exists.
        PermissionError: If directories cannot be created.
        ValueError: If calculation output is invalid.

    Examples:
    ``` bash
        $ lqtmoment --help
        $ lqtmoment --wave-dir data/waveforms --catalog-file
            data/catalog/lqt_catalog.xlsx
        $ lqtmoment --wave-dir data/waveforms --catalog-file
            data/catalog/lqt_catalog.xlsx --config data/new_config.ini
    ```
    """
    parser = argparse.ArgumentParser(
        description="Calculate moment magnitude in full LQT component."
    )
    parser.add_argument(
        "--wave-dir",
        type=Path,
        default=Path("data/waveforms"),
        help="Path to waveform directory",
    )
    parser.add_argument(
        "--cal-dir",
        type=Path,
        default=Path("data/calibration"),
        help="Path to the calibration directory",
    )
    parser.add_argument(
        "--catalog-file",
        type=Path,
        default=Path("data/catalog/lqt_catalog.xlsx"),
        help="LQT formatted catalog file",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("data/new_config.ini"),
        help="Path to custom config.ini file to reload",
    )
    parser.add_argument("--id-start", type=int, help="Starting earthquake ID.")
    parser.add_argument("--id-end", type=int, help="Ending earthquake ID.")
    parser.add_argument(
        "--non-lqt",
        action="store_false",
        dest="lqt_mode",
        help="Use ZRT rotation instead of LQT for very local earthquake.",
    )
    parser.add_argument(
        "--create-figure",
        action="store_true",
        help="Generate and save spectral fitting figures.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("results/figures"),
        help="Path to save figures",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/calculation"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="excel",
        help="Set output for saving results ('Excel' or 'csv'). Defaults to excel.",
    )
    parser.add_argument(
        "--result-file-prefix",
        type=str,
        default="lqt_magnitude",
        help="Set prefix for result file names. Defaults to 'lqt_magnitude'",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__import__('lqtmoment').__version__}",
        help="Show the version and exit",
    )
    args = parser.parse_args(args if args is not None else sys.argv[1:])

    # Reload configuration if specified
    _reload_configuration(args.config_file)

    # Validate input paths
    _validate_input_paths(args.wave_dir, args.cal_dir, args.catalog_file)

    # Create output directories
    _create_output_directories(args.fig_dir, args.output_dir)

    # Load and validate catalog
    catalog_df = load_data(args.catalog_file)
    _validate_catalog(catalog_df)

    # Run calculation
    merged_catalog_df, mw_result_df, mw_fitting_df = _run_calculation(
        args.wave_dir,
        args.cal_dir,
        catalog_df,
        args.id_start,
        args.id_end,
        args.lqt_mode,
        args.create_figure,
        args.fig_dir,
    )

    # Save results
    _save_results(
        merged_catalog_df,
        mw_result_df,
        mw_fitting_df,
        args.output_dir,
        args.output_format,
        args.result_file_prefix,
    )


if __name__ == "__main__":
    main()
