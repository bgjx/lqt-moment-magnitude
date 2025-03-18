#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 19:32:03 2022.
Python code to calculate moment magnitude.

Developed by arham zakki edelo.

contact: 
- edelo.arham@gmail.com
- https://github.com/bgjx

Pre-requisite modules:
->[pathlib, numpy, obpsy] 

"""

import os, glob, sys, warnings, logging
from typing import Tuple, Callable, Optional
from pathlib import Path

import numpy as np
from obspy import UTCDateTime, Stream, Trace, read, read_inventory

from .config import CONFIG

logger = logging.getLogger("mw_calculator")


def get_valid_input(prompt: str, validate_func: callable, error_msg: str) -> int:
    """
    Get valid user input.
    
    Args:
        prompt(str): Prompt to be shown in the terminal.
        validate_func(callable) : A function to validate the input value.
        error_msg(str): Error messages if get wrong input value.
    
    Returns:
        int: Returns an integer event ID”.
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


def get_user_input() -> Tuple[int, int, str, bool]:
    """
    Get user inputs for processing parameters.
    
    Returns:
        Tuple[int, int, str, bool]: Start ID, end ID, output name, and whether to generate figures.
    """
    
    id_start    = get_valid_input("Event ID to start: ", lambda x: int(x) >= 0, "Please input non-negative integer")
    id_end      = get_valid_input("Event ID to end: ", lambda x: int(x) >= id_start, f"Please input an integer >= {id_start}")
    
    while True:
        mw_output   = input("Result file name? (ex. mw_out, press ENTER for default): ").strip() or "mw_output"
        if not any(c in mw_output for c in r'<>:"/\\|?*'):
            break
        print("Enter a valid filename without special characters")

    while True:
        lqt_mode = input("Do you want to calculate all earthquakes in LQT mode regardless the source distance? [yes/no], if [no] let this program decide:").strip().lower()
        if lqt_mode in ['yes', 'no']:
            lqt_mode = (lqt_mode == "yes")
            break
        print("Please enter 'yes' or 'no'")

    while True:
        figure_statement = input("Do you want to produce the spectral fitting figures [yes/no]?: ").strip().lower()
        if figure_statement in ['yes', 'no']:
            figure_statement = (figure_statement == 'yes')
            break
        print("Please enter 'yes' or 'no'")
        
    return id_start, id_end, mw_output, figure_statement, lqt_mode


def read_waveforms(path: Path, event_id: int, station:str) -> Stream:
    """
    Read waveforms file (.mseed) from the specified path and event id.

    Args:
        path (Path): Parent path of separated by id waveforms directory.
        event_id (int): Unique identifier for the earthquake event.
        station (str): Station name.
    Returns:
        Stream: A Stream object containing all the waveforms from specific event id.
    """
    
    stream = Stream()
    pattern = os.path.join(path/f"{event_id}", f"*{station}*.mseed")
    for w in glob.glob(pattern, recursive = True):
        try:
            stread = read(w)
            stream += stread
        except Exception as e:
            logger.warning(f"Skip reading waveform {w} for event {event_id}: {e}.", exc_info=True)
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
            logger.warning(f"Error process instrument removal in trace {trace.id}: {e}.", exc_info=True)
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
    
    # Compute RMS of the signal
    data_rms = np.sqrt(np.mean(np.square(data)))
    
    # Compute RMS of the noise
    noise_rms = np.sqrt(np.mean(np.square(noise)))
    
    return data_rms / noise_rms