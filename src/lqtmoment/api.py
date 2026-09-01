"""
A public API for the lqt-moment-magnitude package.

This module provides functions to calculate seismic moment magnitude in the LQT
component system. Users can import and use these functions in their own Python scripts.

Dependencies:
    - See `pyproject.toml` or `pip install lqtmoment` for required packages.

Notes:
    - `catalog_df` should contain columns like 'source_id', 'source_lat' and so on...
        (see documentation for full schema).
    - See https://github.com/bgjx/lqt-moment-magnitude for detailed usage and
        configuration options.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import CONFIG
from .utils import REQUIRED_CATALOG_COLUMNS, load_data, setup_logging

try:
    from .processing import start_calculate
except ImportError as e:
    raise ImportError(
        "Failed to import processing module. Ensure lqtmoment is installed correctly."
    ) from e


logger = setup_logging()


def _validate_and_prepare_paths(
    wave_dir: str, cal_dir: str, fig_dir: str, output_dir: str
) -> tuple[Path, Path, Path, Path]:
    """
    Validate input paths and create output directories.

    Raises:
        FileNotFoundError: If input directories don't exist.
        NotADirectoryError: If paths are not directories.
        PermissionError: If output directories cannot be created.
    """
    wave_dir = Path(wave_dir)
    cal_dir = Path(cal_dir)
    fig_dir = Path(fig_dir)
    output_dir = Path(output_dir)

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
        raise PermissionError(f"Permission denied creating directories: {e}") from e

    return wave_dir, cal_dir, fig_dir, output_dir


def _reload_config_if_needed(config_file: str | None) -> None:
    """
    Reload configuration if a custom config file is provided.

    Raises:
        FileNotFoundError: If config file is not found.
        ValueError: If configuration is invalid.
    """
    if not config_file:
        return

    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        CONFIG.reload(config_path)
    except (FileNotFoundError, ValueError) as e:
        raise ValueError(
            f"Failed to reload configuration from {config_file}: {e}"
        ) from e


def _save_results(
    merged_catalog_df: pd.DataFrame,
    mw_result_df: pd.DataFrame,
    mw_fitting_df: pd.DataFrame,
    output_dir: Path,
    output_format: str,
    result_file_prefix: str,
) -> None:
    """
    Save calculation results to files.

    Raises:
        ValueError: If output format is unsupported.
        RuntimeError: If file operations fail.
    """
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
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to save results: {e}") from e


def magnitude_estimator(
    wave_dir: str,
    cal_dir: str,
    catalog_file: str,
    config_file: str | None,
    id_start: int | None = None,
    id_end: int | None = None,
    lqt_mode: bool = True,
    generate_figure: bool = False,
    fig_dir: str = "figures",
    save_output_file: bool = False,
    output_dir: str = "results/calculation",
    output_format: str = "excel",
    result_file_prefix: str = "lqt_magnitude",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate seismic moment magnitude in the LQT component system.

    This function processes waveform data using the LQT component system
    to compute moment magnitudes, saving results and figures as specified.

    Args:
        wave_dir (str): Path to the waveform directory.
        cal_dir (str): Path to the calibration directory.
        catalog_file (str): Path to the seismic catalog.
        config_file (str | None): Path to a custom config.ini file to reload. Defaults
            to None.
        id_start (int | None): Starting earthquake ID. Defaults to min ID.
        id_end (int | None ): Ending earthquake ID. Defaults to max ID.
        lqt_mode (bool): Use LQT rotation if True, ZRT otherwise. Defaults to True.
        generate_figure (bool): Generate and save figures if True. Defaults to False.
        fig_dir (str): path to save figures.
        save_output_file (bool): If True, save the result DataFrames to files. Default
                                            to False.
        output_dir (str): Output directory for results. Defaults to "results".
        output_format (str): Format for saving results ("excel" or "csv"). Default to
            "excel".
        result_file_prefix (str): Prefix for result file names. Defaults to
            "lqt_magnitude"

    Returns:
        tuple[pd.DataFrame, ...]: A tuple containing the merged lqt catalog DataFrame,
            result DataFrame and fitting DataFrame.

    Raises:
        FileNotFoundError: If input directories or config file do not exist.
        ValueError: If the catalog is empty or calculation fails.
        PermissionError: If output directories cannot be created.

    Examples:
    ``` python
        >>> from lqtmoment import magnitude_estimator
        >>> merged_cat_df, result_df, fitting_df = magnitude_estimator(
        ...                                   wave_dir = "data/waveforms",
        ...                                   cal_dir = "data/calibration",
        ...                                   catalog_dir = "data/catalog/catalog.xlsx",
        ...                                   config_file = "data/new_config.ini",
        ...                                   generate_figure = True,
        ...                                   fig_dir = "results/figures",
        ...                                   )
    ```
    """
    # Reload config if provided
    _reload_config_if_needed(config_file)

    # Validate and prepare paths
    wave_dir, cal_dir, fig_dir, output_dir = _validate_and_prepare_paths(
        wave_dir, cal_dir, fig_dir, output_dir
    )

    # Load and validate catalog
    catalog_df = load_data(catalog_file)
    missing_columns = [
        col for col in REQUIRED_CATALOG_COLUMNS if col not in catalog_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Catalog missing required columns: {missing_columns}")

    # Call the processing function
    logger.info(f"Starting magnitude calculation for catalog: {catalog_file}")
    try:
        merged_catalog_df, mw_result_df, mw_fitting_df = start_calculate(
            wave_path=wave_dir,
            calibration_path=cal_dir,
            catalog_data=catalog_df,
            id_start=id_start,
            id_end=id_end,
            lqt_mode=lqt_mode,
            generate_figure=generate_figure,
            figure_path=fig_dir,
        )
    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        raise ValueError(f"Failed to calculate moment magnitude: {e}") from e

    # Validate calculation output
    if merged_catalog_df is None or mw_result_df is None or mw_fitting_df is None:
        raise ValueError("Calculation returned invalid results (None).")

    # Save results if requested
    if save_output_file:
        _save_results(
            merged_catalog_df,
            mw_result_df,
            mw_fitting_df,
            output_dir,
            output_format,
            result_file_prefix,
        )

    return merged_catalog_df, mw_result_df, mw_fitting_df


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
        raise ValueError(
            f"Failed to reload configuration from {config_path}: {e}"
        ) from e
