"""
Functionality module for lqt-moment-magnitude package.

This module calculates provides useful functionalities such as user input validation,
waveform reader, instrument response removal and Signal-to-Noise ratio calculator.

Dependencies:
    - See `requirements.txt` or `pip install lqt-moment-magnitude` for required packages.
"""

import os, glob, sys, logging
from typing import Tuple, Callable, Optional
from pathlib import Path

import numpy as np
from obspy import Stream, read, read_inventory

from .config import CONFIG

logger = logging.getLogger("lqtmoment")


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


def get_user_input() -> Tuple[int, int, bool, bool]:
    """
    Get user inputs for processing parameters interactively.
    
    Returns:
        Tuple[int, int, bool, bool]: Start ID, End ID, LQT mode statement 
                                     and crate figure statement.
    """
    
    id_start = get_valid_input("Earthquake ID to start: ", lambda x: int(x) >= 0, "Please input non-negative integer")
    id_end   = get_valid_input("Earthquake ID to end: ", lambda x: int(x) >= id_start, f"Please input an integer >= {id_start}")
    
    while True:
        try:
            lqt_mode = input("Do you want to calculate all earthquakes in LQT mode regardless the source distance? [yes/no, default: yes], if [no] let this program decide:").strip().lower()
            if lqt_mode == "":
                lqt_mode = True
                break
            if lqt_mode in ['yes', 'no']:
                lqt_mode = (lqt_mode == "yes")
                break
            print("Please enter 'yes' or 'no'")
        except KeyboardInterrupt:
            sys.exit("\nOperation cancelled by user")

    while True:
        try:
            figure_statement = input("Do you want to produce the spectral fitting figures [yes/no, default: no]?: ").strip().lower()
            if figure_statement == "":
                figure_statement = False
                break
            if figure_statement in ['yes', 'no']:
                figure_statement = (figure_statement == 'yes')
                break
            print("Please enter 'yes' or 'no'")
        except KeyboardInterrupt:
            sys.exit("\nOperation cancelled by user")
        
    return id_start, id_end, lqt_mode, figure_statement 


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
        (e.g., path/earthquake_id/*{station}*.mseed). For current version the program
        only accept .mseed format. 
    """
    
    stream = Stream()
    pattern = os.path.join(path/f"{source_id}", f"*{station}*.mseed")
    for w in glob.glob(pattern, recursive = True):
        try:
            stread = read(w)
            stream += stread
        except Exception as e:
            logger.warning(f"Skip reading waveform {w} for earthquake {source_id}: {e}.", exc_info=True)
            continue
            
    return stream


def instrument_remove (
    stream: Stream, 
    calibration_path: Path, 
    figure_path: Optional[str] = None, 
    figure_statement: bool = False
    ) -> Stream:
    """
    Removes instrument response from a Stream of seismic traces using calibration files.

    Args:
        stream (Stream): A Stream object containing seismic traces with instrument responses to be removed.
        calibration_path (str): Path to the directory containing the calibration files in RESP format.
        figure_path (Optional[str]): Directory path where response removal plots will be saved. If None, plots are not saved.
        figure_statement (bool): If True, saves plots of the response removal process. Defaults to False.

    Returns:
        Stream: A Stream object containing traces with instrument responses removed.
    Note:
        The naming convention of the calibration or the RESP is RESP.NETWORK.STATIO.LOCATION.COMPONENT
        (e.g., LQID.LQ.LQT1..BHZ) in the calibration directory.
    """
    
    displacement_stream = Stream()
    for trace in stream:
        try:
            # Construct the calibration file
            station, component = trace.stats.station, trace.stats.component
            inventory_path = calibration_path / f"RESP.RD.{station}..BH{component}"
            
            # Read the calibration file
            inventory = read_inventory(inventory_path, format='RESP')
  
            # Prepare plot path if fig_statement is True
            plot_path = None
            if figure_statement and figure_path:
                plot_path = figure_path.joinpath(f"fig_{station}_BH{component}")
            
            # Remove instrument response
            displacement_trace = trace.remove_response(
                                    inventory = inventory,
                                    pre_filt = CONFIG.magnitude.PRE_FILTER,
                                    water_level = CONFIG.magnitude.WATER_LEVEL,
                                    output = 'DISP',
                                    zero_mean = True,
                                    taper = True,
                                    taper_fraction = 0.05,
                                    plot = plot_path
                                    )
            # Re-detrend the trace
            displacement_trace.detrend("linear")
            
            # Add the processed trace to the Stream
            displacement_stream+=displacement_trace
            
        except Exception as e:
            logger.warning(f"Error process instrument removal in trace {trace}: {e}.", exc_info=True)
            continue
            
    return displacement_stream


def trace_snr(data: np.ndarray, noise: np.ndarray) -> float:
    """
    Computes the Signal-to-Noise Ratio (SNR) based on the RMS (Root Mean Square) of the signal and noise.

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