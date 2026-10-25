"""
Lqt catalog builder module for lqt-moment-magnitude package.

This module helps user to build the LQT catalog format. The special excel format
required by lqt-moment-magnitude to be able to calculate the moment magnitude.

Dependencies:
    - See `pyproject.toml` or `pip install lqtmoment` for required packages.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from obspy.geodetics import gps2dist_azimuth

from .utils import (
    COMPLETE_CATALOG_ORDER_COLUMNS,
    OPTIONAL_HYPO_COLUMNS,
    OPTIONAL_PICKING_COLUMNS,
    REQUIRED_HYPO_COLUMNS,
    REQUIRED_PICKING_COLUMNS,
    REQUIRED_STATION_COLUMNS,
    load_data,
)


def _validate_input_data(
    hypo_dir: Path, picks_dir: Path, station_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and validate input data files.

    Raises:
        ValueError: If required columns are missing or dataframe is empty.
        ValueError: If duplicate IDs are found.
    """
    hypo_df = load_data(hypo_dir)
    picking_df = load_data(picks_dir)
    station_df = load_data(station_dir)

    # Validate the required columns
    for df, name, required_cols in [
        (hypo_df, "hypo_df", REQUIRED_HYPO_COLUMNS),
        (picking_df, "picking_df", REQUIRED_PICKING_COLUMNS),
        (station_df, "station_df", REQUIRED_STATION_COLUMNS),
    ]:
        missing_columns = [col for col in required_cols if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in {name}: {missing_columns}")
        if df.empty:
            raise ValueError(f"{name} is empty")

    # Check for duplicates
    if hypo_df["source_id"].duplicated().any():
        raise ValueError("Duplicate 'source_id' found in hypocenter catalog")
    if station_df["station_code"].duplicated().any():
        raise ValueError("Duplicate 'station_code' found in station file")

    return hypo_df, picking_df, station_df


def _determine_earthquake_type(
    epicentral_distance: float, source_depth_m: float
) -> str:
    """
    Determine earthquake type based on epicentral distance and depth.

    Args:
        epicentral_distance (float): Distance in km
        source_depth_m (float): Depth in meters

    Returns:
        str: Earthquake type classification
    """
    threshold_30_2d = max(30, (2 * source_depth_m / 1e3))

    if epicentral_distance < threshold_30_2d:
        return "very_local_earthquake"
    if epicentral_distance < 100:
        return "local_earthquake"
    if epicentral_distance < 1110:
        return "regional_earthquake"
    if epicentral_distance < 2220:
        return "far_regional_earthquake"
    return "teleseismic_earthquake"


def _parse_arrival_times(
    pick_info: pd.Series,
    source_origin_time: datetime,
) -> tuple[datetime, datetime, datetime, float, float, float]:
    """
    Parse P, S, and coda arrival times from pick data.

    Returns:
        Tuple of (p_arr_time, s_arr_time, coda_time, p_travel_time,
        s_travel_time, s_p_lag_time)

    Raises:
        ValueError: If datetime conversion fails.
    """
    year = pick_info.year
    month = pick_info.month
    day = pick_info.day
    hour_p = pick_info.hour_p
    minute_p = pick_info.minute_p
    second_p = pick_info.p_arr_sec
    hour_s = pick_info.hour_s
    minute_s = pick_info.minute_s
    second_s = pick_info.s_arr_sec
    hour_coda = pick_info.hour_coda
    minute_coda = pick_info.minute_coda
    second_coda = pick_info.sec_coda

    try:
        int_p_second = int(second_p)
        microsecond_p = int((second_p - int_p_second) * 1e6)
        int_s_second = int(second_s)
        microsecond_s = int((second_s - int_s_second) * 1e6)
        p_arr_time = datetime(
            year, month, day, hour_p, minute_p, int_p_second, microsecond_p
        )
        s_arr_time = datetime(
            year, month, day, hour_s, minute_s, int_s_second, microsecond_s
        )
    except ValueError as e:
        raise ValueError(
            "Cannot convert P and S arrival time data to datetime object, "
            "check your catalog data format."
        ) from e

    if not pd.isna(hour_coda) and not pd.isna(minute_coda) and not pd.isna(second_coda):
        int_coda_second = int(second_coda)
        microsecond_coda = int((second_coda - int_coda_second) * 1e6)
        coda_time = datetime(
            year,
            month,
            day,
            hour_coda,
            minute_coda,
            int_coda_second,
            microsecond_coda,
        )
    else:
        coda_time = np.nan

    p_travel_time = p_arr_time - source_origin_time
    p_travel_time = p_travel_time.seconds + (p_travel_time.microseconds * 1e-6)
    s_travel_time = s_arr_time - source_origin_time
    s_travel_time = s_travel_time.seconds + (s_travel_time.microseconds * 1e-6)
    s_p_lag_time = s_arr_time - p_arr_time
    s_p_lag_time = s_p_lag_time.seconds + (s_p_lag_time.microseconds * 1e-6)

    return p_arr_time, s_arr_time, coda_time, p_travel_time, s_travel_time, s_p_lag_time


def _build_catalog_row(
    source_id: int,
    source_lat: float,
    source_lon: float,
    source_depth_m: float,
    source_origin_time: datetime,
    network_code: str,
    station_code: str,
    station_lat: float,
    station_lon: float,
    station_elev: float,
    pick_info: pd.Series,
    hypo_info: pd.Series,
) -> dict:
    """Build a single catalog row from source and station data."""
    epicentral_distance, _, _ = gps2dist_azimuth(
        source_lat, source_lon, station_lat, station_lon
    )
    epicentral_distance = epicentral_distance / 1e3
    earthquake_type = _determine_earthquake_type(epicentral_distance, source_depth_m)

    p_arr_time, s_arr_time, coda_time, p_travel_time, s_travel_time, s_p_lag_time = (
        _parse_arrival_times(pick_info, source_origin_time)
    )

    row = {
        "source_id": source_id,
        "source_lat": source_lat,
        "source_lon": source_lon,
        "source_depth_m": source_depth_m,
        "network_code": network_code,
        "station_code": station_code,
        "station_lat": station_lat,
        "station_lon": station_lon,
        "station_elev_m": station_elev,
        "source_origin_time": source_origin_time.isoformat(),
        "p_arr_time": p_arr_time.isoformat(),
        "p_travel_time_sec": p_travel_time,
        "s_arr_time": s_arr_time.isoformat(),
        "s_travel_time_sec": s_travel_time,
        "s_p_lag_time_sec": s_p_lag_time,
        "coda_time": coda_time.isoformat(),
        "earthquake_type": earthquake_type,
    }

    row.update({key: pick_info.get(key, np.nan) for key in OPTIONAL_PICKING_COLUMNS})
    row.update({key: hypo_info.get(key, np.nan) for key in OPTIONAL_HYPO_COLUMNS})

    return row


def build_catalog(hypo_dir: str, picks_dir: str, station_dir: str) -> pd.DataFrame:
    r"""
    Build a combined catalog from separate hypocenter, pick, and station file.

    Args:
        hypo_dir (str): Path to the hypocenter catalog file.
        picks_dir (str): Path to the picking catalog file.
        station_dir (str): Path to the station file.

    Returns:
        pd.DataFrame : Dataframe object of combined catalog.

    Examples:
    ``` python
        >>> catalog_df = build_catalog(
        ...                 hypo_dir = r"data\catalog\hypo_catalog.xlsx",
        ...                 picks_dir = r"data\catalog\picking_catalog.xlsx",
        ...                 station_dir = r"data\station\station.xlsx"
        ...                 )
    ```
    """
    # Convert string paths to Path objects and load data
    hypo_df, picking_df, station_df = _validate_input_data(
        Path(hypo_dir), Path(picks_dir), Path(station_dir)
    )

    # Build catalog
    rows = []
    for source_id in hypo_df.get("source_id"):
        pick_data = picking_df[picking_df["source_id"] == source_id]
        if pick_data.empty:
            continue

        hypo_info = hypo_df[hypo_df.source_id == source_id].iloc[0]
        source_lat, source_lon, source_depth_m = (
            hypo_info.source_lat,
            hypo_info.source_lon,
            hypo_info.source_depth_m,
        )
        year, month, day, hour, minute, t0 = (
            hypo_info.year,
            hypo_info.month,
            hypo_info.day,
            hypo_info.hour,
            hypo_info.minute,
            hypo_info.t_0,
        )

        # Create datetime object for source_origin_time
        int_t0 = int(t0)
        microsecond = int((t0 - int_t0) * 1e6)
        source_origin_time = datetime(
            int(year), int(month), int(day), int(hour), int(minute), int_t0, microsecond
        )

        for station in pick_data.get("station_code"):
            station_data = station_df[station_df.station_code == station]
            if station_data.empty:
                continue

            station_info = station_data.iloc[0]
            network_code, station_code = (
                station_info.network_code,
                station_info.station_code,
            )
            station_lat, station_lon, station_elev = (
                station_info.station_lat,
                station_info.station_lon,
                station_info.station_elev_m,
            )

            pick_data_subset = pick_data[pick_data.station_code == station]
            if pick_data_subset.empty:
                continue

            pick_info = pick_data_subset.iloc[0]

            row = _build_catalog_row(
                source_id,
                source_lat,
                source_lon,
                source_depth_m,
                source_origin_time,
                network_code,
                station_code,
                station_lat,
                station_lon,
                station_elev,
                pick_info,
                hypo_info,
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    return df[COMPLETE_CATALOG_ORDER_COLUMNS]


def main(args: list[str] | None):
    """
    Runs the catalog builder from command line.

    Args:
        args (list[str] | None): CLI arguments. Defaults to sys.argv[1:] if None.

    Returns:
        None: This function saves results to Excel files and logs the process.

    Raises:
        FileNotFoundError: If required input paths do not exists.

    Example:
        $ lqtcatalog --hypo-file data/catalog/hypo_catalog.xlsx
            --pick-file data/catalog/picking_catalog.xlsx
            --station-file data/station/station.xlsx --output-format csv

    """
    parser = argparse.ArgumentParser(
        description="Auto build lqt-moment-magnitude acceptable catalog format."
    )
    parser.add_argument(
        "--hypo-file",
        type=Path,
        default="data/catalog/hypo_catalog.xlsx",
        help="Hypocenter data file",
    )
    parser.add_argument(
        "--pick-file",
        type=Path,
        default="data/catalog/picking_catalog.xlsx",
        help="Arrival picking data file",
    )
    parser.add_argument(
        "--station-file",
        type=Path,
        default="data/station/station.xlsx",
        help="Station data file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="results/lqt_catalog",
        help="Output directory for results. Defaults to 'results/lqt_catalog'.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="excel",
        help="Set output format results ('excel' or 'csv'). Defaults to 'excel'.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="combined_catalog",
        help="Set base name for the output file. Defaults to 'combined_catalog'.",
    )
    args = parser.parse_args(args if args is not None else sys.argv[1:])

    for path in [args.hypo_file, args.pick_file, args.station_file]:
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Permission denied creating directory: {e}") from e

    combined_dataframe = build_catalog(
        args.hypo_file, args.pick_file, args.station_file
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{args.output_file}_{timestamp}"
    if args.output_format.strip().lower() == "excel":
        combined_dataframe.to_excel(
            args.output_dir / f"{output_file}.xlsx", index=False
        )
    elif args.output_format.strip().lower() == "csv":
        combined_dataframe.to_csv(args.output_dir / f"{output_file}.csv", index=False)
    else:
        raise ValueError(
            f"Unsupported output format: {args.output_format}. Use 'excel' or 'csv'."
        )


if __name__ == "__main__":
    main()
