"""
Core processing module for the lqt-moment-magnitude package.

Version: 0.2.1

This module implements the seismic moment magnitude calculation using the LQT component system.
It handles instrument response removal, waveform rotation (ZNE to LQT or ZRT), spectral fitting
with quasi-Monte Carlo optimization, and moment magnitude estimation based on user-configured
parameters from `config.ini`. The module processes waveforms, calibrates data, and generates
spectral fitting figures, aggregating results into DataFrames.

Dependencies:
    - See `pyproject.toml` or `pip install lqtmoment` for required packages.

Note: 
    This module is intended for internal use by the `lqt-moment-magnitude` API and CLI.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd 
from obspy import Stream, UTCDateTime
from obspy.signal.detrend import simple as detrend_simple
from obspy.signal.util import _npts2nfft
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy.taup import TauPyModel
from scipy.signal import windows
from tqdm import tqdm

from .config import CONFIG
from .fitting_spectral import fit_spectrum_qmc
from .refraction import calculate_inc_angle
from .plotting import plot_spectral_fitting
from .utils import instrument_remove, read_waveforms, trace_snr, wave_trim, REQUIRED_CONFIG


logger = logging.getLogger("lqtmoment")

def calculate_seismic_spectra(
    trace_data: np.ndarray,
    sampling_rate: float,
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    smooth_window_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the single-sided displacement amplitude spectrum of a seismogram using FFT.
    Always applies a Hann Window to reduce spectral leakage.

    Args:
        trace_data (np.ndarray): Array of displacement signal (in meters).
        sampling_rate (float): Sampling rate of the signal in Hz.
        freq_min (Optional[float]): Minimum frequency to include in output (Hz). Default to None.
        freq_max (Optional[float]): Maximum frequency to include in output (Hz). Default to None.
        smooth_window_size (Optional[int]): Size of the moving average window for smoothing.
                                          If None, no smoothing is applied. Must be odd positive.
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - frequencies: Array of sample frequencies in Hz.
            - amplitudes: Array of displacement amplitudes in nm (nanometers).
    Raises:
        ValueError: If trace_data is empty or invalid sampling rate
    """
    if not trace_data.size or sampling_rate <= 0:
        raise ValueError("Trace data cannot be empty and sampling rate must be positive")
    
    # Apply Hann window
    window = windows.hann(len(trace_data))
    trace_data_processed = trace_data * window

    # Hann window correction 
    window_correction = np.sqrt(8.0/3.0)

    # Zero pad to next power of 2 for FFT efficiency and resolution
    n_samples = len(trace_data_processed)
    nfft = _npts2nfft(n_samples)
    padded_data = np.pad(trace_data_processed, (0, nfft - n_samples), mode='constant')

    # Compute the FFT and single-sided spectrum
    fft_data = np.fft.rfft(padded_data)
    frequencies = np.fft.rfftfreq(nfft, d=1.0/sampling_rate)

    # Scale amplitudes: 2.0 for negative frequencies, 1/nfft for FFT normalization, window_correction for Hann window
    amplitudes = np.abs(fft_data) * (2.0 / nfft) * window_correction
    
    # Convert to nm (nanometers)
    amplitudes *= 1e9

    # Filters to specific frequency range
    if freq_min is not None or freq_max is not None:
        mask = np.ones_like(frequencies, dtype=bool)
        if freq_min is not None:
            mask &= frequencies >= freq_min
        if freq_max is not None:
            mask &= frequencies <= freq_max
        frequencies = frequencies[mask]
        amplitudes = amplitudes[mask]
    
    # Apply moving average smoothing if specified
    if smooth_window_size is not None:
        smoothing_window = np.ones(smooth_window_size)/smooth_window_size
        amplitudes = np.convolve(amplitudes, smoothing_window, mode='same')

    return frequencies, amplitudes


def window_trace(
    streams: Stream,
    p_arr_time: UTCDateTime,
    s_arr_time: UTCDateTime,
    lqt_mode: bool = True
    ) -> Tuple[np.ndarray, ...]:
    """
    Windows seismic trace data around P, SV, and SH phase and extracts noise data.

    Args:
    
        streams (Stream): A stream object containing the seismic data.
        p_arr_time (UTCDateTime): Arrival time in UTCDateTime of the P phase.
        s_arr_time (UTCDateTime): Arrival time in UTCDateTime of the S phase.
        lqt_mode (bool): Use LQT components if True, ZRT if false. Default to True. 

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - P_data: The data windowed around the P phase in the L/Z component (displacement in meters).
            - SV_data: The data windowed around the S phase in the Q/R component (displacement in meters).
            - SH_data: The data windowed around the S phase in the T component (displacement in meters).
            - P_noise: The noise data before the P phase in the L/Z component (displacement in meters).
            - SV_noise: The noise data before the P phase in the Q/R component (displacement in meters).
            - SH_noise: The noise data before the P phase in the T component (displacement in meters).

    Raises:
        ValueError: If component traces are missing.
    
    Notes:
        Window size are dynamically calculated based on S-P time with configurable padding.
    """
    components = ['L', 'Q', 'T'] if lqt_mode else ['Z', 'R', 'T']
    try:
        trace_P, trace_SV, trace_SH = [streams.select(component = comp)[0] for comp in components]
    except IndexError as e:
        raise ValueError(f"Missing {components} components in stream")
    
    # Verify trace starttime consistency
    ref_starttime = trace_P.stats.starttime
    for tr in [trace_SV, trace_SH]:
        if tr.stats.starttime != ref_starttime:
            logger.warning(f"Traces {tr.id} have inconsistent starttime")
    
    # Dynamic window parameters
    s_p_time = s_arr_time - p_arr_time
    p_factor = 0.75 if s_p_time > 1.0 else 1.0
    s_factor = 2.25 if s_p_time > 1.0 else 3.0
    time_after_pick_p = min(max(p_factor * s_p_time, CONFIG.wave.MIN_P_WINDOW), CONFIG.wave.MAX_P_WINDOW)
    time_after_pick_s = min(max(s_factor * s_p_time, CONFIG.wave.MIN_S_WINDOW), CONFIG.wave.MAX_S_WINDOW)
    
    # Prevent P/S window overlap
    p_phase_end_time = p_arr_time + time_after_pick_p
    s_phase_start_time = s_arr_time - CONFIG.wave.PADDING_BEFORE_ARRIVAL
    if p_phase_end_time > s_phase_start_time:
        mid_time = p_arr_time + (s_p_time / 2)
        time_after_pick_p = mid_time - p_arr_time
        s_phase_start_time = mid_time
        logger.info(f"Windows adjusted to prevent P/S overlap: P ends at {mid_time}, S starts at {mid_time}")

    # Find the data index for phase windowing
    p_phase_start_index = int((p_arr_time - trace_P.stats.starttime - CONFIG.wave.PADDING_BEFORE_ARRIVAL)/trace_P.stats.delta)
    p_phase_end_index = int((p_arr_time - trace_P.stats.starttime + time_after_pick_p )/trace_P.stats.delta)
    s_phase_start_index = int((s_phase_start_time - trace_SV.stats.starttime)/trace_SV.stats.delta)
    s_phase_end_index = int((s_arr_time - trace_SV.stats.starttime + time_after_pick_s )/trace_SV.stats.delta)
    noise_start_index = int((p_arr_time - trace_P.stats.starttime - CONFIG.wave.NOISE_DURATION)/trace_P.stats.delta)                             
    noise_end_index  = int((p_arr_time - trace_P.stats.starttime - CONFIG.wave.NOISE_PADDING )/trace_P.stats.delta)

    # Slicing method
    def _slice(data, start_idx, end_idx):
        """ Helper function to do data slicing. """
        start_idx = max(0, min(start_idx, len(data) - 1))
        end_idx = max(start_idx, min(end_idx + 1, len(data)))
        return data[start_idx:end_idx]

    # Window the data by the index
    P_data  = _slice(trace_P.data, p_phase_start_index, p_phase_end_index) 
    SV_data = _slice(trace_SV.data, s_phase_start_index, s_phase_end_index)
    SH_data = _slice(trace_SH.data, s_phase_start_index, s_phase_end_index)
    P_noise = _slice(trace_P.data, noise_start_index, noise_end_index)
    SV_noise = _slice(trace_SV.data, noise_start_index, noise_end_index)
    SH_noise = _slice(trace_SH.data, noise_start_index, noise_end_index)

    # Preprocess data, apply detrending, demean, and taper.
    def _preprocess_data(data: np.ndarray) -> np.ndarray:
        """ Helper function to preprocess data, apply detrending, demean, and taper. """
        if len(data) == 0:
            return data
        data = detrend_simple(data)
        data = data - np.mean(data)
        return data
    
    P_data = _preprocess_data(P_data)
    SV_data = _preprocess_data(SV_data)
    SH_data = _preprocess_data(SH_data)
    P_noise = _preprocess_data(P_noise)
    SV_noise = _preprocess_data(SV_noise)
    SH_noise = _preprocess_data(SH_noise)

    return P_data, SV_data, SH_data, P_noise, SV_noise, SH_noise


def _rotate_stream(
    stream: Stream,
    source_type: str,
    source_coordinate: List[float],
    station_coordinate: List[float],
    azimuth: float,
    s_p_lag_time_sec: float,
    p_arr_time: UTCDateTime,
    s_arr_time: UTCDateTime,
    lqt_mode: bool
    ) -> Stream:
    """
    Rotate the stream from ZNE to LQT or ZRT based on earthquake type and lqt_mode flag.

    Args:
        stream (Stream): Input stream in ZNE components.
        source_type (str): Type of the earthquake (e.g, 'very_local_earthquake').
        source_coordinate (List[float]): Source coordinate [lat, lon, depth].
        station_coordinate (List[float]): Station coordinate [lat, lon, elev].
        azimuth (float): Azimuth from source to station in degrees.
        s_p_lag_time_sec (float): S-P lag time in seconds.
        p_arr_time (UTCDateTime): P arrival time.
        s_arr_time (UTCDateTime): S arrival time.
        lqt_mode (bool): Use LQT rotation if True, ZRT if False.
    
    Returns:
        Stream: Rotated stream in LQT or ZRT components.
    
    Raises:
        ValueError: If rotation fails.  
    """
    if source_type == 'very_local_earthquake' and not lqt_mode:
        stream_zrt = stream.copy()
        stream_zrt.rotate(method="NE->RT", back_azimuth=azimuth)
        p_trace, sv_trace, sh_trace = stream_zrt.traces # Z, R, T components
    elif source_type == 'very_local_earthquake' and lqt_mode:
        trace_Z = stream.select(component='Z')[0]
        _, _, incidence_angle_p, _, _, incidence_angle_s = calculate_inc_angle(
                                                            source_coordinate,
                                                            station_coordinate,
                                                            CONFIG.magnitude.LAYER_BOUNDARIES,
                                                            CONFIG.magnitude.VELOCITY_VP,
                                                            CONFIG.magnitude.VELOCITY_VS,
                                                            source_type,
                                                            trace_Z,
                                                            s_p_lag_time_sec,
                                                            p_arr_time,
                                                            s_arr_time                                                                    
                                                            )
        stream_lqt_p = stream.copy()
        stream_lqt_s = stream.copy()
        stream_lqt_p.rotate(method="ZNE->LQT", back_azimuth=azimuth, inclination=incidence_angle_p)
        stream_lqt_s.rotate(method="ZNE->LQT", back_azimuth=azimuth, inclination=incidence_angle_s)
        p_trace, _, _ = stream_lqt_p.traces # L, Q, T components
        _, sv_trace, sh_trace = stream_lqt_s.traces
    elif source_type == 'local_earthquake':
        trace_Z = stream.select(component='Z')[0]
        _, _, incidence_angle_p, _, _, incidence_angle_s = calculate_inc_angle(
                                                            source_coordinate,
                                                            station_coordinate,
                                                            CONFIG.magnitude.LAYER_BOUNDARIES,
                                                            CONFIG.magnitude.VELOCITY_VP,
                                                            CONFIG.magnitude.VELOCITY_VS,
                                                            source_type,
                                                            trace_Z,
                                                            s_p_lag_time_sec,
                                                            p_arr_time,
                                                            s_arr_time                                                                    
                                                            )
        stream_lqt_p = stream.copy()
        stream_lqt_s = stream.copy()
        stream_lqt_p.rotate(method="ZNE->LQT", back_azimuth=azimuth, inclination=incidence_angle_p)
        stream_lqt_s.rotate(method="ZNE->LQT", back_azimuth=azimuth, inclination=incidence_angle_s)
        p_trace, _, _ = stream_lqt_p.traces # L, Q, T components
        _, sv_trace, sh_trace = stream_lqt_s.traces

    else:
        model = TauPyModel(model=CONFIG.magnitude.TAUP_MODEL)
        arrivals = model.get_travel_times(
            source_depth_in_km=(source_coordinate[2]*-1e-3),
            distance_in_degree=locations2degrees(
                source_coordinate[0], source_coordinate[1],
                station_coordinate[0], station_coordinate[1]
            ),
            phase_list=["P", "S"]
        )
        incidence_angle_p = arrivals[0].incident_angle
        incidence_angle_s = arrivals[1].incident_angle
        stream_lqt_p = stream.copy()
        stream_lqt_s = stream.copy()
        stream_lqt_p.rotate(method="ZNE->LQT", back_azimuth=azimuth, inclination=incidence_angle_p)
        stream_lqt_s.rotate(method="ZNE->LQT", back_azimuth=azimuth, inclination=incidence_angle_s)
        p_trace, _, _ = stream_lqt_p.traces # L, Q, T components
        _, sv_trace, sh_trace = stream_lqt_s.traces
    
    return Stream(traces=[p_trace, sv_trace, sh_trace])


def calculate_moment_magnitude(
    wave_path: Path, 
    calibration_path: Path,
    source_df: pd.DataFrame, 
    pick_df: pd.DataFrame,
    source_id: int, 
    lqt_mode: bool = True,
    generate_figure: bool = False,
    figure_path: Optional[Path] = None, 
    ) -> Tuple[Dict[str, str], Dict[str,List]]:
    
    """
    Processes moment magnitude calculation for an earthquake from given
    hypocenter dataframe and picking dataframe. This function handle the waveform instrument 
    response removal, seismogram rotation, spectral fitting, moment magnitude calculation, and
    figure creation. It return two dictionary objects, magnitude and fitting result.      

    Args:
        wave_path (Path): Path to the directory containing waveform files.
        calibration_path (Path): Path to the calibration files for instrument response removal.
        source_df (pd.DataFrame): DataFrame containing hypocenter information (latitude, longitude, depth).
        pick_df (pd.DataFrame): DataFrame containing pick information (arrival times).
        source_id (int): Unique identifier for the earthquake.
        lqt_mode (bool): If True, perform LQT rotation; otherwise, use ZRT for very local earthquakes
                            default to True.
        generate_figure (bool): Boolean statement to generate and save figures (default is False).                    
        figure_path (Optional[Path]): Path to save the generated figures. Defaults to None, 
                                        then generate folder at current directory.

    Returns:
        Tuple[Dict[str, str], Dict[str, List]]:
            - results (Dict[str, str]): A Dictionary containing calculated moment magnitude and related metrics.
            - fitting_result (Dict[str, List]): A dictionary of detailed fitting results for each station.
    
    Raises:
        ValueError: if source_df or pick_df are empty or wrong format.
        IOError: If waveform or calibration files cannot be read.
    
    Notes:
        The whole process in this function following these steps:
            1. Remove instrument response using calibration file.
            2. Rotate waveforms from ZNE to LQT (or ZRT for non-LQT mode) based on earthquake type.
                (Incidence in LQT rotation is still based on 1-D velocity model for now.)
            3. Window P and S waves, compute their spectra, and fit spectral parameters (
                corner_frequency, omega_0, q_factor) using optimized algorithm (default: QMC).
            4. Calculate seismic moment (M_0) using the formula:
                M_0 = (4 * pi * rho * v^3 * r * Omega_0) / (R * F),
                where rho is density, v is wave velocity, r is distance, 
                R is radiation pattern, and F is free surface factor.
            5. Compute moment magnitude (Mw) using.
                Mw = (2/3) * (log10(M_0) - CONFIG.magnitude.MW_CONSTANT), where M_0 is in Nm.
    """ 
    # Validate all config parameter before doing calculation
    missing_config = []
    for attr in REQUIRED_CONFIG['wave']:
        if not hasattr(CONFIG.wave, attr):
            missing_config.append(f"{attr} (missing in wave config)")
    for attr in REQUIRED_CONFIG['magnitude']:
        if not hasattr(CONFIG.magnitude, attr):
            missing_config.append(f"{attr} (missing in magnitude config)")
    for attr in REQUIRED_CONFIG['spectral']:
        if not hasattr(CONFIG.spectral, attr):
            missing_config.append(f"{attr} (missing in spectral config)")
    for attr in REQUIRED_CONFIG['performance']:
        if not hasattr(CONFIG.performance, attr):
            missing_config.append(f"{attr} (missing in performance config)")
    if missing_config:
        logger.error(f"Earthquake_{source_id}: Missing config attributes: {missing_config}")
        raise ValueError(f"Missing config attributes: {missing_config}")
    
    # Create object collector for fitting result
    fitting_result = {
        "source_id":[],
        "station":[],
        "f_corner_p":[],
        "f_corner_sv":[],
        "f_corner_sh":[],
        "q_factor_p":[],
        "q_factor_sv":[],
        "q_factor_sh":[],
        "omega_0_p_nms":[],
        "omega_0_sv_nms":[],
        "omega_0_sh_nms":[],
        "rms_e_p_nms":[],
        "rms_e_sv_nms":[],
        "rms_e_sh_nms":[],
        "moment_p_Nm":[],
        "moment_s_Nm":[]
    }

    # Create object collector for plotting
    if generate_figure:
        all_streams, all_p_times, all_s_times = [], [], []
        all_freqs = {
            "P": [],
            "SV":[],
            "SH":[],
            "N_P":[],
            "N_SV":[],
            "N_SH":[] 
        }
        all_specs = {
            "P": [], 
            "SV":[],
            "SH":[],
            "N_P":[],
            "N_SV":[],
            "N_SH":[] 
        }
        all_fits = {
            "P":[],
            "SV":[],
            "SH":[]
        }
        station_names = []

    # Get hypocenter details
    source_info = source_df.iloc[0]
    source_origin_time = UTCDateTime(source_info.source_origin_time)
    source_lat, source_lon , source_depth_m =  source_info.source_lat, source_info.source_lon, source_info.source_depth_m
    source_type = source_info.earthquake_type

    # Find the correct velocity and DENSITY value for the specific layer depth
    velocity_P, velocity_S, density_value = None, None, None
    for layer, (top, bottom) in enumerate(CONFIG.magnitude.LAYER_BOUNDARIES):
        top_m, bottom_m = top * 1000, bottom * 1000
        if top_m   <= source_depth_m <= bottom_m:
            velocity_P = CONFIG.magnitude.VELOCITY_VP[layer]*1000
            velocity_S = CONFIG.magnitude.VELOCITY_VS[layer]*1000
            density_value = CONFIG.magnitude.DENSITY[layer]
            break
    if velocity_P is None:
        logger.warning(f"Earthquake_{source_id}: Hypocenter depth not within the defined layers.")
        return {}, fitting_result
    
    # Start spectrum fitting and magnitude estimation
    moments, corner_frequencies, source_radius = [],[],[]
    for station in pick_df.get("station_code").unique():
        
        # Get the station coordinate
        station_info = pick_df[pick_df.station_code == station].iloc[0]
        network_code = station_info.network_code
        station_lat, station_lon, station_elev_m = station_info.station_lat, station_info.station_lon, station_info.station_elev_m
        p_arr_time = UTCDateTime(station_info.p_arr_time)
        s_arr_time = UTCDateTime(station_info.s_arr_time)
        if not np.isnan(station_info.coda_time):
            coda_time = UTCDateTime(station_info.coda_time)
        else:
            coda_time = None

        s_p_lag_time_sec = station_info.s_p_lag_time_sec
        
        # Calculate the source distance and the azimuth (hypo to station azimuth)
        epicentral_distance, azimuth, _ = gps2dist_azimuth(source_lat, source_lon, station_lat, station_lon)
        epicentral_distance_degrees = locations2degrees(source_lat, source_lon, station_lat, station_lon)
        source_distance_m = np.sqrt(epicentral_distance**2 + ((source_depth_m + station_elev_m)**2))
            
        # Read the waveform 
        stream = read_waveforms(wave_path, source_id, station)
        stream_copy = stream.copy()
        if len(stream_copy) < 3:
            logger.warning(f"Earthquake_{source_id}: Not all components available for station {station}"
                           f"to calculate earthquake {source_id} moment magnitude")
            continue

        # Trimming the waveform prior to processing
        try:
            if CONFIG.wave.TRIM_MODE == 'dynamic':
                trimmed_stream = wave_trim(stream_copy, 
                                           p_arr_time, 
                                           coda_time,
                                           CONFIG.wave.SEC_BF_P_ARR_TRIM,
                                           CONFIG.wave.SEC_AF_P_ARR_TRIM)
            else:
                trimmed_stream = wave_trim(stream_copy,
                                           p_arr_time,
                                           CONFIG.wave.SEC_BF_P_ARR_TRIM,
                                           CONFIG.wave.SEC_AF_P_ARR_TRIM)
                
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Earthquake_{source_id}: Failed to trim wave data from {station}: {e}.")
            continue
        
        # Performing resampling if specified
        if CONFIG.wave.RESAMPLE_DATA is not None:
            trimmed_stream.resample(CONFIG.wave.RESAMPLE_DATA)

        # Perform the instrument removal
        try:
            stream_displacement = instrument_remove(
                                        trimmed_stream,
                                        calibration_path,
                                        figure_path,
                                        network_code,
                                        pre_filter=CONFIG.wave.PRE_FILTER,
                                        water_level=CONFIG.wave.WATER_LEVEL,
                                        generate_figure=False)
        except Exception as e:
            logger.warning(f"Earthquake_{source_id}: An error occurred when correcting instrument for station {station}: {e}", exc_info=True)
            continue
        
        # Perform post instrument removal if specified by the User
        if CONFIG.wave.APPLY_POST_INSTRUMENT_REMOVAL_FILTER:
            stream_displacement.filter("bandpass", freqmin=CONFIG.wave.POST_FILTER_F_MIN, freqmax=CONFIG.wave.POST_FILTER_F_MAX, corners=4, zerophase=True)
        
        # Perform station rotation form ZNE to LQT in earthquake type dependent
        source_coordinate = [source_lat, source_lon , -1*source_depth_m]  # depth must be in negative notation
        station_coordinate = [station_lat, station_lon, station_elev_m]

        logger.info(f"Rotating station {station} for {source_type}: {source_id} and LQT mode is {'ON' if lqt_mode else 'OFF'}.")
        try:
            rotated_stream = _rotate_stream(
                            stream_displacement, source_type,
                            source_coordinate, station_coordinate, azimuth,
                            s_p_lag_time_sec, p_arr_time,s_arr_time,
                            lqt_mode)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Earthquake_{source_id}: Failed to rotate components for station {station}: {e}.", exc_info=True)
            continue
        
        # Window the trace
        p_window_data, sv_window_data, sh_window_data, p_noise_data, sv_noise_data, sh_noise_data = window_trace(
                                                                                                    rotated_stream, 
                                                                                                    p_arr_time,
                                                                                                    s_arr_time,
                                                                                                    lqt_mode=lqt_mode)
        
        # Check the data quality (SNR must be above or equal to 1)
        if any(trace_snr(data, noise) <= CONFIG.wave.SNR_THRESHOLD for data, noise in zip ([p_window_data, sv_window_data, sh_window_data], [p_noise_data, sv_noise_data, sh_noise_data])):
            logger.warning(f"Earthquake_{source_id}: SNR below threshold for station {station} to calculate moment magnitude")
            continue

        # check sampling rate
        fs = 1 / rotated_stream[0].stats.delta

        try:
            # Calculate source spectra
            freq_P , spec_P  = calculate_seismic_spectra(p_window_data,
                                                         fs, 
                                                         freq_min=CONFIG.spectral.F_MIN,
                                                         freq_max=CONFIG.spectral.F_MAX,
                                                         smooth_window_size=CONFIG.spectral.SMOOTH_WINDOW_SIZE
                                                         )
            freq_SV, spec_SV = calculate_seismic_spectra(sv_window_data,
                                                         fs,
                                                         freq_min=CONFIG.spectral.F_MIN,
                                                         freq_max=CONFIG.spectral.F_MAX,
                                                         smooth_window_size=CONFIG.spectral.SMOOTH_WINDOW_SIZE
                                                         )
            freq_SH, spec_SH = calculate_seismic_spectra(sh_window_data,
                                                         fs,
                                                         freq_min=CONFIG.spectral.F_MIN,
                                                         freq_max=CONFIG.spectral.F_MAX,
                                                         smooth_window_size=CONFIG.spectral.SMOOTH_WINDOW_SIZE
                                                         )
            
            # Calculate the noise spectra
            freq_N_P,  spec_N_P  = calculate_seismic_spectra(p_noise_data,
                                                            fs,
                                                            freq_min=CONFIG.spectral.F_MIN,
                                                            freq_max=CONFIG.spectral.F_MAX,
                                                            smooth_window_size=CONFIG.spectral.SMOOTH_WINDOW_SIZE
                                                            )
            freq_N_SV, spec_N_SV = calculate_seismic_spectra(sv_noise_data,
                                                            fs,
                                                            freq_min=CONFIG.spectral.F_MIN,
                                                            freq_max=CONFIG.spectral.F_MAX,
                                                            smooth_window_size=CONFIG.spectral.SMOOTH_WINDOW_SIZE
                                                            )
            freq_N_SH, spec_N_SH = calculate_seismic_spectra(sh_noise_data,
                                                            fs,
                                                            freq_min=CONFIG.spectral.F_MIN,
                                                            freq_max=CONFIG.spectral.F_MAX,
                                                            smooth_window_size=CONFIG.spectral.SMOOTH_WINDOW_SIZE
                                                            )
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Earthquake_{source_id}: An error occurred during spectra calculation for station {station}, {e}.", exc_info=True)
            continue

        # Fitting the spectrum, find the optimal value of Omega_O, corner frequency and Q using systematic/stochastic algorithm available
        try:
            fit_P  = fit_spectrum_qmc(freq_P,  spec_P,  abs(float(p_arr_time - source_origin_time)), CONFIG.spectral.F_MIN, CONFIG.spectral.F_MAX, CONFIG.spectral.DEFAULT_N_SAMPLES)
            fit_SV = fit_spectrum_qmc(freq_SV, spec_SV, abs(float(s_arr_time - source_origin_time)), CONFIG.spectral.F_MIN, CONFIG.spectral.F_MAX, CONFIG.spectral.DEFAULT_N_SAMPLES)
            fit_SH = fit_spectrum_qmc(freq_SH, spec_SH, abs(float(s_arr_time - source_origin_time)), CONFIG.spectral.F_MIN, CONFIG.spectral.F_MAX, CONFIG.spectral.DEFAULT_N_SAMPLES)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Earthquake_{source_id}: Error during spectral fitting for event {source_id}, {e}.", exc_info=True)
            continue
        if any(f is None for f in [fit_P, fit_SV, fit_SH]):
            continue

        # Extract fitting spectrum output
        Omega_0_P,  Q_factor_p,  f_c_P,  err_P,  x_fit_P,  y_fit_P  = fit_P
        Omega_0_SV, Q_factor_SV, f_c_SV, err_SV, x_fit_SV, y_fit_SV = fit_SV
        Omega_0_SH, Q_factor_SH, f_c_SH, err_SH, x_fit_SH, y_fit_SH = fit_SH

        # Updating the fitting object collector 
        fitting_result["source_id"].append(source_id)
        fitting_result["station"].append(station)
        fitting_result["f_corner_p"].append(f_c_P)
        fitting_result["f_corner_sv"].append(f_c_SV)
        fitting_result["f_corner_sh"].append(f_c_SH)
        fitting_result["q_factor_p"].append(Q_factor_p)
        fitting_result["q_factor_sv"].append(Q_factor_SV)
        fitting_result["q_factor_sh"].append(Q_factor_SH)
        fitting_result["omega_0_p_nms"].append((Omega_0_P))
        fitting_result["omega_0_sv_nms"].append((Omega_0_SV))
        fitting_result["omega_0_sh_nms"].append((Omega_0_SH))
        fitting_result["rms_e_p_nms"].append((err_P))
        fitting_result["rms_e_sv_nms"].append((err_SV))
        fitting_result["rms_e_sh_nms"].append((err_SH))

        # Calculate the moment magnitude
        try:
            # Calculate the  resultant omega
            omega_P = Omega_0_P*1e-9
            omega_S = (np.sqrt(Omega_0_SV**2 + Omega_0_SH**2))*1e-9
         
            # Calculate seismic moment
            M_0_P = (4.0 * np.pi * density_value * (velocity_P ** 3) * source_distance_m * omega_P) / (CONFIG.magnitude.R_PATTERN_P * CONFIG.magnitude.FREE_SURFACE_FACTOR)
            M_0_S = (4.0 * np.pi * density_value * (velocity_S ** 3) * source_distance_m * omega_S) / (CONFIG.magnitude.R_PATTERN_S * CONFIG.magnitude.FREE_SURFACE_FACTOR)
            fitting_result["moment_p_Nm"].append(M_0_P)
            fitting_result["moment_s_Nm"].append(M_0_S)
            
            # Calculate average seismic moment at station
            moments.append((M_0_P + M_0_S)/2)
            
            # Calculate source radius
            r_P = (CONFIG.magnitude.K_P * velocity_P)/f_c_P
            r_S = (2 * CONFIG.magnitude.K_S * velocity_S)/(f_c_SV + f_c_SH)
            source_radius.append((r_P + r_S)/2)
            
            # Calculate corner frequency mean
            corner_freq_S = (f_c_SV + f_c_SH)/2
            corner_frequencies.append((f_c_P + corner_freq_S)/2)

        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f" Earthquake_{source_id}: Failed to calculate seismic moment for earthquake {source_id}, {e}.", exc_info=True)
            continue
        
        # Update fitting spectral object collector for plotting
        if generate_figure:
            all_streams.append(rotated_stream)
            all_p_times.append(p_arr_time)
            all_s_times.append(s_arr_time)
            all_freqs["P"].append(freq_P)
            all_freqs["SV"].append(freq_SV)
            all_freqs["SH"].append(freq_SH)
            all_freqs["N_P"].append(freq_N_P)
            all_freqs["N_SV"].append(freq_N_SV)
            all_freqs["N_SH"].append(freq_N_SH)
            all_specs["P"].append(spec_P)
            all_specs["SV"].append(spec_SV)
            all_specs["SH"].append(spec_SH)
            all_specs["N_P"].append(spec_N_P)
            all_specs["N_SV"].append(spec_N_SV)
            all_specs["N_SH"].append(spec_N_SH)
            all_fits["P"].append(fit_P)
            all_fits["SV"].append(fit_SV)
            all_fits["SH"].append(fit_SH)
            station_names.append(station)

    if not moments or not corner_frequencies or not source_radius:
        return {}, fitting_result
    
    # Calculate average and std of moment magnitude
    moment_average = np.mean(moments)
    moment_std = np.std(moments)
    mw = ((2.0 / 3.0) * np.log10(moment_average)) - CONFIG.magnitude.MW_CONSTANT
    mw_std = (2.0 /3.0) * moment_std/(moment_average * np.log(10))
 
    results = {"source_id": [source_id], 
                "fc_avg": [np.mean(corner_frequencies)],
                "fc_std": [np.std(corner_frequencies)],
                "src_rad_avg_m": [np.mean(source_radius)],
                "src_rad_std_m": [np.std(source_radius)],
                "stress_drop_bar": [(7 * moment_average) / (16 * np.mean(source_radius)** 3) *1e-5],
                "mw_average": [mw],
                "mw_std": [mw_std]
                }
    
    # Create fitting spectral plot
    if generate_figure and all_streams:
        try:
            plot_spectral_fitting(
                source_id, all_streams, all_p_times, all_s_times, 
                all_freqs, all_specs, all_fits, station_names, figure_path)
        except (ValueError, IOError) as e:
            logger.warning(f"Earthquake_{source_id}: Failed to create spectral fitting plot for event {source_id}, {e}.", exc_info=True)
    
    return results, fitting_result


def start_calculate(
    wave_path: Path,
    calibration_path: Path,
    catalog_data: pd.DataFrame,
    id_start:Optional[int] = None,
    id_end: Optional[int] = None,
    lqt_mode: bool = True,
    generate_figure : bool = False,
    figure_path: Optional[Path] = None,
    ) -> Tuple [pd.DataFrame, pd.DataFrame]:
    """
    This function processes moment magnitude calculation by iterating over a user-specified range
    of earthquake IDs. For each event of earthquake, it extracts source and station data, and 
    computes moment magnitudes using waveform and response file, and aggregates results into
    two DataFrames: magnitude results and spectral fitting parameters.
    
    Args:
        wave_path (Path): Path to the directory containing waveforms file (.miniSEED format).
        calibration_path (Path) : Path to the directory containing calibration file (.RESP format).
        catalog_data (pd.DataFrame): Catalog DataFrame in LQTMomentMag format.
        id_start (Optional[int]): Starting earthquake ID. If not provided, id_start set to min ID in catalog.
        id_end (Optional[int]): Ending earthquake ID. If not provided, id_end set to max ID in catalog.
        lqt_mode (bool): Use LQT rotation if True, ZRT otherwise. Defaults to True.
        generate_figure (bool): Generate and save figures if True. Defaults to False.
        figure_path (Optional[Path]) : Path to the directory where spectral fitting figures will be saved.
                                        Defaults to None, then generate folder at current directory.      

    Returns:
        Tuple [pd.Dataframe, pd.DataFrame]:
            - First DataFrame: Magnitude results with columns ['source_id', 'fc_avg', 'fc_std', ...].
            - Second DataFrame: Fitting results with columns ['source_id', 'station', 'f_corner_p', ...].
    
    Raises:
        ValueError: If catalog_data is empty or missing required columns.
    
    Example:
        >>> catalog = pd.read_excel("lqt_catalog.xlsx")
        >>> result_df, fitting_df = start_calculate(
        ...     Path("data/waveforms"), Path("data/calibration"),
        ...     Path("figures"), catalog)
    """
    
    # Set defaults ID range if not provided through API or CLI use.
    default_id_start = int(catalog_data["source_id"].min())
    default_id_end = int(catalog_data["source_id"].max())

    # Use defaults ID range if id_start and id_end are not provided by user.
    id_start = id_start if id_start is not None else default_id_start
    id_end = id_end if id_end is not None else default_id_end


    # Validate ID range
    if not (isinstance(id_start, int) and isinstance(id_end, int) and id_start <= id_end):
        raise ValueError(f"Invalid ID range: id_start={id_start}, id_end={id_end}")
    if not (id_start in catalog_data["source_id"].values and id_end in catalog_data["source_id"].values):
        raise ValueError(f"ID range {id_start} - {id_end} not found in catalog")
    
    # Initiate dataframe for magnitude calculation results
    df_result = pd.DataFrame(
            columns=["source_id", "fc_avg", "fc_std", "Src_rad_avg_m",
                    "Src_rad_std_m", "Stress_drop_bar",
                    "mw_average", "mw_std"] 
                        )
    df_fitting = pd.DataFrame(
            columns=["source_id", "station", "f_corner_p", "f_corner_sv",
                    "f_corner_sh", "q_factor_p", "q_factor_sv", "q_factor_sh",
                    "omega_0_P_nms", "omega_0_sv_nms", "omega_0_sh_nms",
                    "rms_e_p_nms", "rms_e_sv_nms", "rms_e_sh_nms",
                    "moment_p_Nm", "moment_s_Nm"] 
                        )

    failed_events=0
    result_list = []
    fitting_list = []

    # Pre-grouping catalog by the id for efficiency
    catalog_data_copy = catalog_data.copy()
    grouped_data = catalog_data_copy.groupby("source_id")
    total_earthquakes = id_end - id_start + 1
    with tqdm(
        total = total_earthquakes,
        file=sys.stderr,
        position=0,
        leave=True,
        desc="Processing earthquakes",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ncols=80,
        smoothing=0.1
    ) as pbar:
        for source_id in range (id_start, id_end + 1):
            logger.info("\n\n")
            logger.info(f"**========** Earthquake ID: {source_id} **========**")
            # Extract data for the current event
            try:
                catalog_data = grouped_data.get_group(source_id)
            except KeyError as e:
                logger.warning(f"Earthquake_{source_id}: No data for earthquake ID {source_id}: {e}.", exc_info=True)
                failed_events += 1
                pbar.set_postfix({"Failed": failed_events})
                pbar.update(1)
                continue

            source_data = catalog_data[["source_lat", "source_lon", "source_depth_m",
                                        "source_origin_time", "earthquake_type"]].drop_duplicates()
            
            pick_data = catalog_data[["network_code", "station_code", "station_lat",
                                    "station_lon", "station_elev_m", "p_arr_time",
                                    "s_arr_time", "s_p_lag_time_sec", "coda_time"]].drop_duplicates()
            
            # Check for  empty data frame
            if source_data.empty or pick_data.empty:
                logger.warning(f"Earthquake_{source_id}: No data for earthquake {source_id}")
                failed_events += 1
                pbar.set_postfix({"Failed": failed_events})
                pbar.update(1)
                continue

            # Calculate the moment magnitude
            try:
                mw_results, fitting_result = calculate_moment_magnitude(
                                            wave_path= wave_path,
                                            calibration_path= calibration_path,
                                            source_df= source_data,
                                            pick_df= pick_data,
                                            source_id= source_id,
                                            lqt_mode= lqt_mode,
                                            generate_figure= generate_figure,
                                            figure_path= figure_path
                                            )
                result_list.append(pd.DataFrame.from_dict(mw_results))
                fitting_list.append(pd.DataFrame.from_dict(fitting_result))
            except (ValueError, IOError) as e:
                logger.error(
                    f"Earthquake_{source_id}: Calculation failed for earthquake id {source_id}: {e}",
                    exc_info=True
                )
                failed_events += 1
                pbar.set_postfix({"Failed": failed_events})
                pbar.update(1)
                continue

            pbar.set_postfix({"Failed": failed_events})
            pbar.update(1)
                
    # Concatenate the dataframe
    df_result = pd.concat(result_list, ignore_index = True) if result_list else df_result
    df_fitting = pd.concat(fitting_list, ignore_index = True) if fitting_list else df_fitting

    # Merge the lqt catalog with the magnitude result
    merged_catalog = pd.merge(
        catalog_data_copy,
        df_result[['source_id', 'mw_average']],
        on='source_id',
        how='left'
    )

    # Rename column name 'mw_average' to just 'magnitude' for generalization.
    merged_catalog = merged_catalog.rename(columns={'mw_average':'magnitude'})

    # Re-order the columns by index
    columns_name = merged_catalog.columns.to_list()
    columns_name.remove('magnitude')
    columns_name.insert(5, 'magnitude')
    merged_catalog = merged_catalog[columns_name]

    # Remove unnecessary index in merged catalog
    merged_catalog = merged_catalog.reset_index(drop=True)

    # Summary message
    sys.stdout.write(
        f"Finished. Proceed {total_earthquakes - failed_events} earthquakes successfully,"
        f"{failed_events} failed. Check lqt_runtime.log for details. \n"
    )
    return merged_catalog, df_result, df_fitting