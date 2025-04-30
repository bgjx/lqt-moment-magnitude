"""
Utilities module for lqt-moment-magnitude package.

Version: 0.2.1

This module provides useful functionalities such as user input validation,
waveform reader, instrument response removal and Signal-to-Noise ratio calculation.

Dependencies:
    - See `pyproject.toml` or `pip install lqtmoment` for required packages.
"""

import logging
import warnings
import os
import glob
import sys
from typing import Tuple, Callable, Optional, List
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import Stream, read, read_inventory, UTCDateTime

from .config import CONFIG


logger = logging.getLogger("lqtmoment")

COMPLETE_CATALOG_ORDER_COLUMNS = [
    "source_id", "source_lat", "source_lon", "source_depth_m",
    "network_code", "station_code", "station_lat", "station_lon", "station_elev_m",
    "source_origin_time", "p_arr_time", "p_travel_time_sec", "p_polarity", "p_onset",
    "s_arr_time", "s_travel_time_sec", "s_p_lag_time_sec", "coda_time",
    "source_err_rms_s", "n_phases", "gap_degree",
    "x_horizontal_err_m", "y_horizontal_err_m", "z_depth_err_m",
    "earthquake_type", "remarks"
    ]

REQUIRED_CATALOG_COLUMNS = [
    "source_id", "source_lat", "source_lon", "source_depth_m", 
    "network_code", "station_code", "station_lat", "station_lon", "station_elev_m",
    "source_origin_time", "p_arr_time", "s_arr_time",
    "s_p_lag_time_sec","coda_time", "earthquake_type"
    ]

REQUIRED_HYPO_COLUMNS = [
    "source_id", "lat", "lon", "depth_m",
    "year", "month", "day", "hour", "minute","t_0"
    ]

OPTIONAL_HYPO_COLUMNS = [
    "source_err_rms_s", "n_phases", "gap_degree",
    "x_horizontal_err_m", "y_horizontal_err_m", "z_depth_err_m",
    "remarks"
    ]

REQUIRED_PICKING_COLUMNS = [
    "source_id", "station_code", "year", "month", "day",
    "hour_p", "minute_p", "p_arr_sec", "p_polarity", "p_onset",
    "hour_s", "minute_s", "s_arr_sec",
    "hour_coda", "minute_coda", "sec_coda"
    ]

OPTIONAL_PICKING_COLUMNS = [
    "p_polarity", "p_onset", 
]

REQUIRED_STATION_COLUMNS = ["network_code", "station_code", "lat", "lon", "elev_m"]

REQUIRED_CONFIG = {
    "wave": ["SNR_THRESHOLD", "WATER_LEVEL", "PRE_FILTER",
            "APPLY_POST_INSTRUMENT_REMOVAL_FILTER",
            "POST_FILTER_F_MIN", "POST_FILTER_F_MAX",
            "TRIM_MODE", "SEC_BF_P_ARR_TRIM", "SEC_AF_P_ARR_TRIM", 
            "PADDING_BEFORE_ARRIVAL",
            "MIN_P_WINDOW", "MAX_P_WINDOW", "MIN_P_WINDOW", "MAX_P_WINDOW", 
            "NOISE_DURATION", "NOISE_PADDING"],
    "magnitude": ["R_PATTERN_P", "R_PATTERN_S", "FREE_SURFACE_FACTOR",
                  "K_P", "K_S", "TAUP_MODEL", "VELOCITY_MODEL_FILE", "MW_CONSTANT"],
    "spectral": ["SMOOTH_WINDOW_SIZE", "F_MIN", "F_MAX", "OMEGA_0_RANGE_MIN", "OMEGA_0_RANGE_MAX",
                 "Q_RANGE_MIN", "Q_RANGE_MAX", "FC_RANGE_BUFFER", "DEFAULT_N_SAMPLES",
                 "N_FACTOR", "Y_FACTOR"],
    "performance": ["USE_PARALLEL", "LOGGING_LEVEL"]
}


def load_data(data_dir: str) -> pd.DataFrame:
    """
    Load tabular data from given data dir, this function will handle
    data suffix/format (.xlsx / .csv) for more dynamic inputs.

    Args:
        data_dir (str): Directory of the data file.

    Returns:
        pd.DataFrame: DataFrame of tabular data.
    
    Raises:
        FileNotFoundError: If data files do not exist.
        ValueError: If data files fail to load or unsupported format.
    """
    data_path = Path(data_dir)
    if not data_path.is_file():
        raise FileNotFoundError(f"Given data path is not a file: {data_path}")
    if data_path.suffix == ".xlsx":
        return pd.read_excel(data_path, index_col=None)
    elif data_path.suffix == ".csv":
        return pd.read_csv(data_path, index_col=None)
    else:
        raise ValueError(f"Unsupported data file format: {data_path.suffix}. Supported formats: .csv, .xlsx")
    

def setup_logging(log_file: str = "lqt_runtime.log") -> logging.Logger:
    """
    Set up logging for lqtmoment package.

    Args:
        log_file (str): The name of the log file. Defaults to 'lqt_runtime.log'
    
    Returns:
        logging.logger: A logger to be used in entire package.
    
    Raises:
        PermissionError: If the log file cannot be written.
    
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning, module='pandas')
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    try:
        logging.basicConfig(
            filename=log_file,
            level = log_level_map[CONFIG.performance.LOGGING_LEVEL.upper()],
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    except PermissionError as e:
        raise PermissionError(f"Cannot write log file {log_file}: {e}")    
    
    return logging.getLogger("lqtmoment")


def get_valid_input(prompt: str, validate_func: Callable, error_msg: str) -> int:
    """
    Function to get valid user input.
    
    Args:
        prompt(str): Prompt to be shown in the terminal.
        validate_func(callable) : A function to validate the input value.
        error_msg(str): Error message to display if the input is invalid.
    
    Returns:
        int: Returns an integer, earthquake ID”.

    Raises:
        KeyboardInterrupt: If the user interrupts the input(Ctrl+C).
    """
    while True:
        value = input(prompt).strip()
        try:
            value = int(value)
            if validate_func(value):
                return int(value)
            print(error_msg)
        except ValueError:
            print(error_msg)
        except KeyboardInterrupt:
            sys.exit("Interrupted by user")


# def get_user_input() -> Tuple[int, int, bool, bool]:
#     """
#     Get user inputs for processing parameters interactively.
    
#     Returns:
#         Tuple[int, int, bool, bool]: Start ID, End ID, LQT mode, and generate figure flag.
#     """
#     id_start = get_valid_input("Earthquake ID to start: ", lambda x: int(x) >= 0, "Please input non-negative integer")
#     id_end   = get_valid_input("Earthquake ID to end: ", lambda x: int(x) >= id_start, f"Please input an integer >= {id_start}")
    
#     while True:
#         try:
#             lqt_mode = input("Do you want to calculate all earthquakes in LQT mode regardless the source distance? [yes/no, default: yes], if [no] let this program decide:").strip().lower()
#             if lqt_mode == "":
#                 lqt_mode = True
#                 break
#             if lqt_mode in ['yes', 'no']:
#                 lqt_mode = (lqt_mode == "yes")
#                 break
#             print("Please enter 'yes' or 'no'")
#         except KeyboardInterrupt:
#             sys.exit("\nOperation cancelled by user")

#     while True:
#         try:
#             generate_figure = input("Do you want to produce the spectral fitting figures [yes/no, default: no]?: ").strip().lower()
#             if generate_figure == "":
#                 generate_figure = False
#                 break
#             if generate_figure in ['yes', 'no']:
#                 generate_figure = (generate_figure == 'yes')
#                 break
#             print("Please enter 'yes' or 'no'")
#         except KeyboardInterrupt:
#             sys.exit("\nOperation cancelled by user")
        
#     return id_start, id_end, lqt_mode, generate_figure 


def read_waveforms(path: Path, source_id: int, station:str) -> Stream:
    """
    Read waveforms file (.mseed) from the specified path and earthquake ID.

    Args:
        path (Path): Parent path of separated by id waveforms directory.
        source_id (int): Unique identifier for the earthquake.
        station (str): Station name.
    Returns:
        Stream: A Stream object containing all the waveforms from specific earthquake id.
    
    Notes:
        Expects waveform file to be in a subdirectory named after the earthquake id
        (e.g., path/earthquake_id/*{station}*.mseed). Currently, the program
        only accept .mseed format. 
    """
    stream = Stream()
    pattern = os.path.join(path/f"{source_id}", f"*{station}*.mseed")
    for w in glob.glob(pattern, recursive = True):
        try:
            stread = read(w)
            stream += stread
        except Exception as e:
            continue
            
    return stream


def wave_trim(
    stream: Stream,
    p_arrival: UTCDateTime,
    coda_time: Optional[UTCDateTime] = None,
    padding_bf_p: float = 10.0,
    padding_af_p: float = 50.0
    ) -> Stream:
    """
    Trim the seismogram to specific time window. 

    Args:
        stream (Stream): A Stream object containing all the waveforms from specific earthquake id.
        p_arrival (UTCDateTime): P arrival time as initial reference for trimming start point.
        coda_time (Optional[UTCDateTime]): Coda time as primary reference for trimming end point.
                                    Defaults to None, use static padding after P arrival 
                                    instead.
        padding_bf_p (float): Time in seconds before P arrival for exact trimming start point.
        padding_af_p (float): Time in seconds after P arrival for exact trimming end point.

    Returns:
        Stream: A stream object containing all trimmed traces.

    Raises:
        ValueError: If trace data is empty or trimming window is invalid.
    """
    for trace in stream:
        if trace.data.size == 0:
            raise ValueError("Trace data cannot be empty")
        trace_starttime = trace.stats.starttime
        trace_endtime = trace.stats.endtime

        # Calculate start time for trimming
        start_trim = max(p_arrival - padding_bf_p, trace_starttime)

        # Calculate end time for trimming
        if coda_time:
            end_trim = min(coda_time, trace_endtime)
        else:
            end_trim = min(p_arrival + padding_af_p, trace_endtime)

        # Validate trimming window
        if start_trim >= end_trim:
            raise ValueError(
                f"Invalid trimming window for trace {trace.id}: "
                f"Start time {start_trim} is not before end time {end_trim}"
            )
        
        # Perform trimming
        trace.trim(start_trim, end_trim)
    
    return stream


def instrument_remove (
    stream: Stream, 
    calibration_path: Path, 
    figure_path: Path,
    network_code: Optional[str] = None,
    pre_filter: Optional[List] = None,
    water_level: Optional[float] = None,
    generate_figure : bool = False,
    ) -> Stream:
    """
    Removes instrument response from a Stream of seismic traces using calibration files.

    Args:
        stream (Stream): A Stream object containing seismic traces with instrument responses to be removed.
        calibration_path (Path): Path to the directory containing the calibration files in RESP format.
        figure_path (Path): Directory path where response removal plots will be saved. If None, plots are not saved.
        network_code (Optional[str]): Network code to use in the calibration file name. If None, attempts to use trace.stats.network.
        pre_filter (Optional[str]): Bandpass filter corners [f1,f2,f3,f4] in Hz. Defaults to None.
        water_level (Optional[str]): Water level for deconvolution stabilization. Defaults to None.
        generate_figure (bool): If True, saves plots of the response removal process. Defaults to False.
        
    Returns:
        Stream: A Stream object containing traces with instrument responses removed.
    Note:
        The naming convention of the calibration or the RESP is RESP.{NETWORK}.{STATION}.{LOCATION}.{CHANNEL}
        (e.g., LQID.LQ.LQT01.00.BHZ) in the calibration directory.
    """
    displacement_stream = Stream()
    for trace in stream:
        try:
            # Construct the calibration filex
            station = trace.stats.station
            channel = trace.stats.channel
            trace_network = trace.stats.network if trace.stats.network else network_code
            if not trace_network:
                raise ValueError(f"Network code not found in trace {trace.id} and not provided as parameter.")
            location = trace.stats.location if trace.stats.location else ""
            inventory_path = calibration_path / f"RESP.{trace_network}.{station}.{location}.{channel}.resp"
            if not inventory_path.exists():
                raise FileNotFoundError(f"Calibration file not found: {inventory_path}")
            
            # Read the calibration file
            inventory = read_inventory(inventory_path, format='RESP')
  
            # Prepare plot path if fig_statement is True
            plot_path = None
            if generate_figure and figure_path:
                plot_path = figure_path.joinpath(f"fig_{station}_{channel}")
            
            # Remove instrument response
            displacement_trace = trace.remove_response(
                                    inventory = inventory,
                                    pre_filt = pre_filter,
                                    water_level = water_level,
                                    output = 'DISP',
                                    zero_mean = True,
                                    taper = True,
                                    taper_fraction = 0.05,
                                    plot = plot_path
                                    )
            # Re-detrend the trace
            displacement_trace.detrend("demean")
            
            # Add the processed trace to the Stream
            displacement_stream+=displacement_trace
            
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"Failed to perform instrument removal: {e}.", exc_info=True)
            continue
            
    return displacement_stream


def trace_snr(data: np.ndarray, noise: np.ndarray) -> float:
    """
    Computes the Signal-to-Noise Ratio (SNR) using the RMS (Root Mean Square) of the signal and noise.

    Args:
        data (np.ndarray): Array of signal data.
        noise (np.ndarray): Array of noise data.

    Returns:
        float: The Signal-to-Noise Ratio (SNR), calculated as the ratio of the RMS of the signal to the RMS of the noise.
    """
    if not data.size or not noise.size:
        raise ValueError("Data and noise arrays must be non-empty.")
    # Compute RMS of the signal
    data_rms = np.sqrt(np.mean(np.square(data)))
    
    # Compute RMS of the noise
    noise_rms = np.sqrt(np.mean(np.square(noise)))
    
    return data_rms / noise_rms