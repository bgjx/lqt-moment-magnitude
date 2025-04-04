"""
Functionality module for lqt-moment-magnitude package.

Version: 0.1.0

This module helps user to build the LQT catalog format. The special excel format
required by lqt-moment-magnitude to be able to calculate the moment magnitude.

Dependencies:
    - See `pyproject.toml` or `pip install lqtmoment` for required packages.
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

from obspy.geodetics import gps2dist_azimuth

from .utils import load_data, REQUIRED_HYPO_COLUMNS, REQUIRED_PICKING_COLUMNS, REQUIRED_STATION_COLUMNS


def build_catalog(
        hypo_dir: str,
        picks_dir: str,
        station_dir: str
        ) -> pd.DataFrame:
    """
    Build a combined catalog from separate hypocenter, pick, and station file.

    Args:
        hypo_dir (str): Path to the hypocenter catalog file.
        picks_dir (str): Path to the picking catalog file.
        station_dir (str): Path to the station file.

    Returns:
        pd.DataFrame : Dataframe object of combined catalog.

    """
    # Convert string paths to Path objects
    hypo_dir = Path(hypo_dir)
    picks_dir = Path(picks_dir)
    station_dir = Path(station_dir)

    # Load and validate tabular data input (catalog, picks, and station)
    hypo_df = load_data(hypo_dir)
    picking_df = load_data(picks_dir)
    station_df = load_data(station_dir)

    # Validate the required columns
    for (df, name, required_cols) in [
        (hypo_df, 'hypo_df', REQUIRED_HYPO_COLUMNS),
        (picking_df, 'picking_df', REQUIRED_PICKING_COLUMNS),
        (station_df, 'station_df', REQUIRED_STATION_COLUMNS)
        ] :
        missing_columns = [col for col in required_cols if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in {name}: {missing_columns}")
        if df.empty:
            raise ValueError(f"{name} is empty")
    
    # Check for duplicates
    if hypo_df['id'].duplicated().any():
        raise ValueError("Duplicate 'id' found in hypocenter catalog")
    if station_df['station_code'].duplicated().any():
        raise ValueError("Duplicate 'station_code' found in station file")
        
    rows = []
    for id in hypo_df.get("id"):
        pick_data = picking_df[picking_df["id"] == id]
        if pick_data.empty:
            continue
        hypo_info = hypo_df[hypo_df.id == id].iloc[0]
        source_lat, source_lon, source_depth_m, year, month, day, hour, minute, t0, hypo_remarks = hypo_info.lat, hypo_info.lon, hypo_info.depth_m, hypo_info.year, hypo_info.month, hypo_info.day, hypo_info.hour, hypo_info.minute, hypo_info.t_0, hypo_info.remarks
        int_t0 = int(t0)
        microsecond = int((t0 - int_t0)*1e6)
        source_origin_time =datetime(int(year), int(month), int(day), int(hour), int(minute), int_t0, microsecond)
        for station in pick_data.get("station_code"):
            station_data = station_df[station_df.station_code == station]
            if station_data.empty:
                continue
            station_info = station_data.iloc[0]
            network_code, station_code, station_lat, station_lon, station_elev = station_info.network_code, station_info.station_code, station_info.lat, station_info.lon, station_info.elev_m
            
            # cek earthquake distance to determine earthquake type
            epicentral_distance, _, _ = gps2dist_azimuth(source_lat, source_lon, station_lat, station_lon)
            epicentral_distance = epicentral_distance/1e3
            earthquake_type = "very_local_earthquake" if epicentral_distance < max(30, (2*source_depth_m/1e3)) else \
                            "local_earthquake" if max(30, (2*source_depth_m/1e3)) <= epicentral_distance <100 else \
                            "regional_earthquake" if 100 <= epicentral_distance < 1110 else \
                            "far_regional_earthquake" if 1110<= epicentral_distance < 2220 else \
                            "teleseismic_earthquake"

            pick_data_subset= pick_data[pick_data.station_code == station]
            if pick_data_subset.empty:
                continue
            pick_info = pick_data_subset.iloc[0]
            year, month, day, hour, minute_p, second_p, p_polarity, p_onset, minute_s, second_s = pick_info.year, pick_info.month, pick_info.day, pick_info.hour, pick_info.minute_p, pick_info.p_arr_sec, pick_info.p_polarity, pick_info.p_onset, pick_info.minute_s, pick_info.s_arr_sec
            int_p_second = int(second_p)
            microsecond_p = int((second_p - int_p_second)*1e6)
            int_s_second = int(second_s)
            microsecond_s = int((second_s - int_s_second)*1e6)
            p_arr_time = datetime(year, month, day, hour, minute_p, int_p_second, microsecond_p)
            p_travel_time = p_arr_time - source_origin_time
            p_travel_time = p_travel_time.seconds + (p_travel_time.microseconds * 1e-6)
            s_arr_time = datetime(year, month, day, hour, minute_s, int_s_second, microsecond_s)
            s_travel_time = s_arr_time - source_origin_time
            s_travel_time = s_travel_time.seconds + (s_travel_time.microseconds * 1e-6)
            s_p_lag_time = s_arr_time - p_arr_time
            s_p_lag_time = s_p_lag_time.seconds + (s_p_lag_time.microseconds * 1e-6)
            row = {
                "source_id": id,
                "source_lat": source_lat, 
                "source_lon": source_lon,
                "source_depth_m": source_depth_m,
                "source_origin_time": source_origin_time,
                "network_code": network_code,
                "station_code": station_code,
                "station_lat": station_lat,
                "station_lon": station_lon, 
                "station_elev_m": station_elev,
                "p_arr_time": p_arr_time,
                "p_travel_time_sec": p_travel_time,
                "p_polarity": p_polarity,
                "p_onset": p_onset,
                "s_arr_time": s_arr_time,
                "s_travel_time_sec": s_travel_time,
                "s_p_lag_time_sec": s_p_lag_time,
                "earthquake_type": earthquake_type,
                "remarks": hypo_remarks
            }
            rows.append(row)
    return pd.DataFrame(rows)


def main(args=None):
    """
    Runs the catalog builder from command line.

    Args:
        args (List[str], Optional): Command-line arguments. Defaults to sys.argv[1:] if None.

    Returns:
        None: This function saves results to Excel files and logs the process.
    
    Raises:
        FileNotFoundError: If required input paths do not exists.
    
    Example:
        $ lqtcatalog --hypo-file data/catalog/hypo_catalog.xlsx --pick-file data/catalog/picking_catalog.xlsx
                        --station-file data/station/station.xlsx --output-format csv
    
    """
    parser = argparse.ArgumentParser(description="Build lqt-moment-magnitude acceptable catalog format automatically.")
    parser.add_argument(
        "--hypo-file",
        type=Path,
        default="data/catalog/hypo_catalog.xlsx",
        help="Hypocenter data file"
        )
    parser.add_argument(
        "--pick-file",
        type=Path, default="data/catalog/picking_catalog.xlsx",
        help="Arrival picking data file"
        )
    parser.add_argument(
        "--station-file",
        type=Path,
        default="data/station/station.xlsx",
        help="Station data file"
        )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="results/lqt_catalog",
        help="Output directory for results. Defaults to results/lqt_catalog if not specified"
        )
    parser.add_argument(
        "--output-format",
        type=str,
        default="excel",
        help = "Set output format for saving results ('excel' or 'csv'). Defaults to 'excel'."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="combined_catalog",
        help="Set base name for the output file. Defaults to 'combined_catalog' if not specified."
    )
    args = parser.parse_args(args if args is not None else sys.argv[1:])

    for path in [args.hypo_file, args.pick_file, args.station_file]:
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Permission denied creating directory: {e}")
    
    combined_dataframe = build_catalog(args.hypo_file, args.pick_file, args.station_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{args.output_file}_{timestamp}"
    if args.output_format.lower() == 'excel':
        combined_dataframe.to_excel(args.output_dir/f"{output_file}.xlsx", index=False)
    elif args.output_format.lower() == 'csv':
        combined_dataframe.to_csv(args.output_dir/f"{output_file}.csv", index=False)
    else:
        raise ValueError(f"Unsupported output format: {args.output_format}. Use 'excel' or 'csv'.")
    return None

if __name__ == "__main__":
    main()