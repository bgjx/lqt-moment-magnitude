"""
Core processing module for the lqt-moment-magnitude package.

This module implements the seismic moment magnitude calculation using the LQT component system.
It handles instrument response removal, waveform rotation (ZNE to LQT or ZRT), spectral fitting
with quasi-Monte Carlo optimization, and moment magnitude estimation based on user-configured
parameters from `config.ini`. The module processes waveforms, calibrates data, and generates
spectral fitting figures, aggregating results into DataFrames.

Dependencies:
    - See `requirements.txt` or `pip install lqt-moment-magnitude` for required packages.

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
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy.taup import TauPyModel
from scipy.signal import windows
from tqdm import tqdm

from .config import CONFIG
from .fitting_spectral import fit_spectrum_bayesian, fit_spectrum_grid_search, fit_spectrum_qmc
from .refraction import calculate_inc_angle
from .plotting import plot_spectral_fitting
from .utils import instrument_remove, read_waveforms, trace_snr


logger = logging.getLogger("lqtmoment")

def calculate_seismic_spectra(
    trace_data: np.ndarray,
    sampling_rate: float,
    apply_window: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the displacement amplitude spectrum of a seismogram using FFT.

    Args:
        trace_data (np.ndarray): Array of displacement signal ( in meters).
        sampling_rate (float): Sampling rate of the signal in Hz.
        apply_window (bool, optional): Apply a Hann window to reduce spectral leakage.
            Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - frequencies: Array of sample frequencies in Hz.
            - amplitudes: Array of displacement amplitudes in nm (nanometers).
    Raises:
        ValueError: If trace_data is empty or invalid sampling rate
    """

    if not trace_data.size or sampling_rate <= 0:
        raise ValueError("Trace data cannot be empty and sampling rate must be positive")
    
    n_samples = len(trace_data)
    if apply_window:
        window = windows.hann(n_samples)
        trace_data_processed = trace_data * window
    else:
        trace_data_processed = trace_data
    
    # Compute the FFT and single-sided spectrum
    fft_data = np.fft.fft(trace_data_processed)
    frequencies = np.fft.fftfreq(n_samples, 1 / sampling_rate)
    amplitudes = np.abs(fft_data) * (2.0/n_samples)
    positive_mask = frequencies >= 0
    frequencies = frequencies[positive_mask]
    amplitudes = amplitudes[positive_mask]

    if apply_window: 
        amplitudes *= 2.0      # Correct for Hann window gain (since average reduction gain is 0.5)
    
    amplitudes *= 1e9           # Convert to nm (nanometers)

    return frequencies, amplitudes


def window_trace(
    streams: Stream,
    p_arr_time: float,
    s_arr_time: float,
    lqt_mode: Optional[bool] = True
    ) -> Tuple[np.ndarray, ...]:
    """
    Windows seismic trace data around P, SV, and SH phase and extracts noise data.

    Args:
    
        streams (Stream): A stream object containing the seismic data.
        p_arr_time (float): Arrival time of the P phase (in seconds from the trace start).
        s_arr_time (float): Arrival time of the S phase (in seconds from the trace start).
        lqt_mode (Optional[bool]): Use LQT components if True, ZRT if false. Default to True. 

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - P_data: The data windowed around the P phase in the L/Z component.
            - SV_data: The data windowed around the S phase in the Q/R component.
            - SH_data: The data windowed around the S phase in the T component.
            - P_noise: The data windowed around the noise period before the P phase in the L/Z component.
            - SV_noise: The data windowed around the noise period before the P phase in the Q/R component.
            - SH_noise: The data windowed around the noise period before the P phase in the T component.

    Raises:
        ValueError: If component traces are missing.
    
    Notes:
        Window size are dynamically calculated based on S-P time with configurable padding.
    """
    
    components = ['L', 'Q', 'T'] if lqt_mode else ['Z', 'R', 'T']
    try:
        [trace_P, trace_SV, trace_SH] = [streams.select(component = comp)[0] for comp in components]
    except IndexError as e:
        raise ValueError(f"Missing {components} components in stream")
    
    # Dynamic window parameters
    s_p_time = s_arr_time - p_arr_time    
    time_after_pick_p = 0.75 * s_p_time
    time_after_pick_s = 1.75 * s_p_time
    
    # Find the data index for phase windowing
    p_phase_start_index = int((p_arr_time - trace_P.stats.starttime - CONFIG.magnitude.PADDING_BEFORE_ARRIVAL)/trace_P.stats.delta)
    p_phase_end_index = int((p_arr_time - trace_P.stats.starttime + time_after_pick_p )/trace_P.stats.delta)
    s_phase_start_index = int((p_arr_time - trace_SV.stats.starttime - CONFIG.magnitude.PADDING_BEFORE_ARRIVAL)/trace_SV.stats.delta)
    s_phase_end_index = int((p_arr_time - trace_SV.stats.starttime + time_after_pick_s )/trace_SV.stats.delta)
    noise_start_index = int((p_arr_time - trace_P.stats.starttime - CONFIG.magnitude.NOISE_DURATION)/trace_P.stats.delta)                             
    noise_end_index  = int((p_arr_time - trace_P.stats.starttime - CONFIG.magnitude.NOISE_PADDING )/trace_P.stats.delta)

    # Window the data by the index
    P_data  = trace_P.data[p_phase_start_index : p_phase_end_index + 1]
    SV_data = trace_SV.data[s_phase_start_index : s_phase_end_index + 1]
    SH_data = trace_SH.data[s_phase_start_index : s_phase_end_index + 1]
    P_noise = trace_P.data[noise_start_index : noise_end_index + 1]
    SV_noise = trace_SV.data[noise_start_index : noise_end_index + 1]
    SH_noise = trace_SH.data[noise_start_index : noise_end_index + 1]

    return P_data, SV_data, SH_data, P_noise, SV_noise, SH_noise


def calculate_moment_magnitude(
    wave_path: Path, 
    source_df: pd.DataFrame, 
    pick_df: pd.DataFrame,
    calibration_path: Path, 
    source_id: int, 
    figure_path: Path, 
    lqt_mode: bool = True,
    figure_statement: bool = False
    ) -> Tuple[Dict[str, str], Dict[str,List]]:
    
    """
    This function processes moment magnitude calculation for an earthquake from given
    hypocenter dataframe and picking dataframe. This function handle the waveform instrument 
    response removal, seismogram rotation, spectral fitting, moment magnitude calculation, and
    figure creation. It return two dictionary objects, magnitude and fitting result.      

    Args:
        wave_path (Path): Path to the directory containing waveform files.
        source_df (pd.DataFrame): DataFrame containing hypocenter information (latitude, longitude, depth).
        pick_df (pd.DataFrame): DataFrame containing pick information (arrival times).
        calibration_path (Path): Path to the calibration files for instrument response removal.
        source_id (int): Unique identifier for the earthquake.
        figure_path (Path): Path to save the generated figures.
        lqt_mode (bool): If True, perform LQT rotation; otherwise, use ZRT for very local earthquakes
                            default to True.
        figure_statement (bool): Boolean statement to generate and save figures (default is False).

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
                Mw = (2/3) * (log10(M_0) - 6.07), where M_0 is in Nm.
    """ 

    # Validate all config parameter before doing calculation
    required_config = [
        "LAYER_BOUNDARIES", "VELOCITY_VP", "VELOCITY_VS", "DENSITY", "SNR_THRESHOLD",
        "R_PATTERN_P", "R_PATTERN_S", "FREE_SURFACE_FACTOR", "K_P", "K_S",
        "PADDING_BEFORE_ARRIVAL", "NOISE_DURATION", "NOISE_PADDING", "F_MIN", "F_MAX"
    ]
    missing_config = [attr for attr in required_config if not hasattr(CONFIG.magnitude, attr) or not hasattr(CONFIG.spectral, attr)]
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
    if figure_statement:
        all_streams, all_p_times, all_s_times = [], [], []
        all_freqs = {
            "P": [],  "SV":[], "SH":[], "N_P":[], "N_SV":[], "N_SH":[] 
        }
        all_specs = {
            "P": [],  "SV":[], "SH":[], "N_P":[], "N_SV":[], "N_SH":[] 
        }
        all_fits = {
            "P":[], "SV":[], "SH":[]
        }
        station_names = []

    # Get hypocenter details
    source_info = source_df.iloc[0]
    source_origin_time = UTCDateTime(source_info.source_origin_time)
    source_lat, source_lon , source_depth_m =  source_info.source_lat, source_info.source_lon, source_info.source_depth_m
    source_type = source_info.earthquake_type

    # Find the correct velocity and DENSITY value for the specific layer depth
    velocity_P, velocity_S, density_value = None, None, None
    for (top, bottom), vp, vs, rho in zip(CONFIG.magnitude.LAYER_BOUNDARIES, CONFIG.magnitude.VELOCITY_VP, CONFIG.magnitude.VELOCITY_VS, CONFIG.magnitude.DENSITY):
        if (top*1000)   <= source_depth_m <= (bottom*1000):
            velocity_P, velocity_S, density_value = vp*1000, vs*1000, rho
            break
        else:
            logger.warning(f"Earthquake_{source_id}: Hypocenter depth not within the defined layers.")
            return {}, fitting_result
    
    # Start spectrum fitting and magnitude estimation
    moments, corner_frequencies, source_radius = [],[],[]
    for station in pick_df.get("station_code").unique():
        # Get the station coordinate
        station_info = pick_df[pick_df.station_code == station].iloc[0]
        station_lat, station_lon, station_elev_m = station_info.station_lat, station_info.station_lon, station_info.station_elev_m
        p_arr_time = UTCDateTime(station_info.p_arr_time)
        s_arr_time = UTCDateTime(station_info.s_arr_time)
        
        # Calculate the source distance and the azimuth (hypo to station azimuth)
        epicentral_distance, azimuth, _ = gps2dist_azimuth(source_lat, source_lon, station_lat, station_lon)
        epicentral_distance_degrees = locations2degrees(source_lat, source_lon, station_lat, station_lon)
        source_distance_m = np.sqrt(epicentral_distance**2 + ((source_depth_m + station_elev_m)**2))
            
        # Read the waveform 
        stream = read_waveforms(wave_path, source_id, station)
        stream_copy = stream.copy()
        if len(stream_copy) < 3:
            logger.warning(f"Earthquake_{source_id}: Not all components available for station {station} to calculate earthquake {source_id} moment magnitude")
            continue
        
        # Perform the instrument removal
        try:
            stream_displacement = instrument_remove(stream_copy, calibration_path, figure_path, figure_statement=False)
        except (ValueError, IOError) as e:
            logger.warning(f"Earthquake_{source_id}: An error occurred when correcting instrument for station {station}: {e}", exc_info=True)
            continue
        
        # Perform station rotation form ZNE to LQT in earthquake type dependent
        source_coordinate = [source_lat, source_lon , -1*source_depth_m]  # depth must be in negative notation
        station_coordinate = [station_lat, station_lon, station_elev_m]
        try:
            if source_type == 'very_local_earthquake' and not lqt_mode:
                stream_zrt = stream_displacement.copy()
                stream_zrt.rotate(method="NE->RT", back_azimuth=azimuth)
                p, sv, sh = stream_zrt.traces # Z, R, T components
            elif source_type =='teleseismic_earthquake':
                model = TauPyModel(model=CONFIG.magnitude.TAUP_MODEL)
                arrivals = model.get_travel_times(
                    source_depth_in_km=(source_depth_m/1e3),
                    distance_in_degree=epicentral_distance_degrees,
                    phase_list=["P", "S"]
                )
                # Deg to m conversion for teleseismic earthquake
                deg_to_m = (2*np.pi*6371e3)/360
                source_distance_m = arrivals[0].distance * deg_to_m
                incidence_angle_p = arrivals[0].incident_angle
                incidence_angle_s = arrivals[1].incident_angle
                stream_lqt_p = stream_displacement.copy()
                stream_lqt_s = stream_displacement.copy()
                stream_lqt_p.rotate(method="ZNE->LQT", back_azimuth=azimuth, inclination=incidence_angle_p)
                stream_lqt_s.rotate(method="ZNE->LQT", back_azimuth=azimuth, inclination=incidence_angle_s)
                p, _, _ = stream_lqt_p.traces # L, Q, T components
                _, sv, sh = stream_lqt_s.traces
            else:
                _, _, incidence_angle_p = calculate_inc_angle(source_coordinate, station_coordinate,
                                                                CONFIG.magnitude.LAYER_BOUNDARIES,
                                                                CONFIG.magnitude.VELOCITY_VP)
                _, _, incidence_angle_s = calculate_inc_angle(source_coordinate, station_coordinate,
                                                                CONFIG.magnitude.LAYER_BOUNDARIES,
                                                                CONFIG.magnitude.VELOCITY_VS)
                stream_lqt_p = stream_displacement.copy()
                stream_lqt_s = stream_displacement.copy()
                stream_lqt_p.rotate(method="ZNE->LQT", back_azimuth=azimuth, inclination=incidence_angle_p)
                stream_lqt_s.rotate(method="ZNE->LQT", back_azimuth=azimuth, inclination=incidence_angle_s)
                p, _, _ = stream_lqt_p.traces # L, Q, T components
                _, sv, sh = stream_lqt_s.traces
            rotated_stream = Stream(traces=[p, sv, sh])
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Earthquake_{source_id}: An error occurred when rotating component for station {station}.", exc_info=True)
            continue
 
        # Window the trace
        p_window_data, sv_window_data, sh_window_data, p_noise_data, sv_noise_data, sh_noise_data = window_trace(rotated_stream, p_arr_time, s_arr_time, lqt_mode=lqt_mode)
        
        # Check the data quality (SNR must be above or equal to 1)
        if any(trace_snr(data, noise) <= CONFIG.magnitude.SNR_THRESHOLD for data, noise in zip ([p_window_data, sv_window_data, sh_window_data], [p_noise_data, sv_noise_data, sh_noise_data])):
            logger.warning(f"Earthquake_{source_id}: SNR below threshold for station {station} to calculate moment magnitude")
            continue
            
        # check sampling rate
        fs = 1 / rotated_stream[0].stats.delta
        try:
            # Calculate source spectra
            freq_P , spec_P  = calculate_seismic_spectra(p_window_data, fs)
            freq_SV, spec_SV = calculate_seismic_spectra(sv_window_data, fs)
            freq_SH, spec_SH = calculate_seismic_spectra(sh_window_data, fs)
            
            # Calculate the noise spectra
            freq_N_P,  spec_N_P  = calculate_seismic_spectra(p_noise_data, fs)
            freq_N_SV, spec_N_SV = calculate_seismic_spectra(sv_noise_data, fs)
            freq_N_SH, spec_N_SH = calculate_seismic_spectra(sh_noise_data, fs)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Earthquake_{source_id}: An error occurred during spectra calculation for station {station}, {e}.", exc_info=True)
            continue

        # Fitting the spectrum, find the optimal value of Omega_O, corner frequency and Q using systematic/stochastic algorithm available
        try:
            fit_P  = fit_spectrum_qmc(freq_P,  spec_P,  abs(float(p_arr_time - source_origin_time)), CONFIG.magnitude.F_MIN, CONFIG.magnitude.F_MAX, CONFIG.spectral.DEFAULT_N_SAMPLES)
            fit_SV = fit_spectrum_qmc(freq_SV, spec_SV, abs(float(s_arr_time - source_origin_time)), CONFIG.magnitude.F_MIN, CONFIG.magnitude.F_MAX, CONFIG.spectral.DEFAULT_N_SAMPLES)
            fit_SH = fit_spectrum_qmc(freq_SH, spec_SH, abs(float(s_arr_time - source_origin_time)), CONFIG.magnitude.F_MIN, CONFIG.magnitude.F_MAX, CONFIG.spectral.DEFAULT_N_SAMPLES)
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
            omega_S = ((Omega_0_SV**2 + Omega_0_SH**2)**0.5)*1e-9
         
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
        if figure_statement:
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

    if not moments:
        return {}, fitting_result
    
    # Calculate average and std of moment magnitude
    moment_average, moment_std  = np.mean(moments), np.std(moments)
    mw = ((2.0 / 3.0) * np.log10(moment_average)) - 6.07
    mw_std = (2.0 /3.0) * moment_std/(moment_average * np.log(10))
 
    results = {"source_id":[f"{source_id}"], 
                "fc_avg":[f"{np.mean(corner_frequencies):.3f}"],
                "fc_std":[f"{np.std(corner_frequencies):.3f}"],
                "src_rad_avg_m":[f"{np.mean(source_radius):.3f}"],
                "src_rad_std_m":[f"{np.std(source_radius):.3f}"],
                "stress_drop_bar":[f"{(7 * moment_average) / (16 * np.mean(source_radius)** 3) *1e-5:.3f}"],
                "mw_average":[f"{mw:.3f}"],
                "mw_std":[f"{mw_std:.3f}"]
                }
    
    # Create fitting spectral plot
    if figure_statement and all_streams:
        try:
            plot_spectral_fitting(source_id, all_streams, all_p_times, all_s_times, all_freqs, all_specs, all_fits, station_names, figure_path)
        except (ValueError, IOError) as e:
            logger.warning(f"Earthquake_{source_id}: Failed to create spectral fitting plot for event {source_id}, {e}.", exc_info=True)
    
    return results, fitting_result


def start_calculate(
    wave_path: Path,
    calibration_path: Path,
    figure_path: Path,
    catalog_data: pd.DataFrame,
    id_start:Optional[int] = None,
    id_end: Optional[int] = None,
    lqt_mode: Optional[bool] = None,
    figure_statement: Optional[bool] = None,
    ) -> Tuple [pd.DataFrame, pd.DataFrame, str]:
    """
    This function processes moment magnitude calculation by iterating over a user-specified range
    of earthquake IDs. For each event of earthquake, it extracts source and station data, and 
    computes moment magnitudes using waveform and response file, and aggregates results into
    two DataFrames: magnitude results and spectral fitting parameters.
    
    Args:
        wave_path (Path): Path to the directory containing waveforms file (.miniSEED format).
        calibration_path (Path) : Path to the directory containing calibration file (.RESP format).
        figure_path (Path) : Path to the directory where spectral fitting figures will be saved.
        catalog_data (pd.DataFrame): Catalog DataFrame in LQTMomentMag format.
        id_start (Optional[int]): Starting earthquake ID. If not provided, prompts user or uses min ID.
        id_end (Optional[int]): Ending earthquake ID. If not provided, prompts user or uses max ID.
        lqt_mode (Optional[bool]): Use LQT rotation if True, ZRT otherwise. If not provided, prompts user.
        figure_statement (Optional[bool]): Generate and save figures if True. If not provided, prompts user.

    Returns:
        Tuple [pd.Dataframe, pd.DataFrame]:
            - First DataFrame: Magnitude results with columns ['source_id', 'fc_avg', 'fc_std', ...].
            - Second DataFrame: Fitting results with columns ['source_id', 'station', 'f_corner_p', ...].
    
    Raises:
        ValueError: If catalog_data is empty or missing required columns.
    
    Example:
        >>> catalog = pd.read_excel("lqt_catalog.xlsx")
        >>> result_df, fitting_df, output_name = start_calculate(
        ...     Path("data/waveforms"), Path("data/calibration"),
        ...     Path("figures"), catalog)
    """
    
    # Set defaults ID range if not provided through API or CLI use.
    default_id_start = int(catalog_data["source_id"].min())
    default_id_end = int(catalog_data["source_id"].max())
    default_lqt_mode = True
    default_figure_statement = False
    # Use user arguments if provided, otherwise fall back to interactive input.
    if id_start is None or id_end is None or figure_statement is None or lqt_mode is None:
        from .utils import get_user_input
        try:
            id_start_input, id_end_input, lqt_mode_input, figure_statement_input = get_user_input()
            id_start = id_start if id_start is not None else id_start_input
            id_end = id_end if id_end is not None else id_end_input
            lqt_mode = lqt_mode if lqt_mode is not None else lqt_mode_input
            figure_statement = figure_statement if figure_statement is not None else figure_statement_input
        except ValueError as e:
            logger.error(f"Invalid interactive input: {e}")
            raise ValueError(f"Invalid interactive input: {e}")

    # Use defaults if still None after interactive input
    id_start = id_start if id_start is not None else default_id_start
    id_end = id_end if id_end is not None else default_id_end
    lqt_mode = lqt_mode if lqt_mode is not None else default_lqt_mode
    figure_statement = figure_statement if figure_statement is not None else default_figure_statement

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
    grouped_data = catalog_data.groupby("source_id")
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
            logging.info(f"Earthquake_{source_id}: Calculating moment magnitude for earthquakes ID {source_id}")
            
            # Extract data for the current event
            try:
                catalog_data_handler = grouped_data.get_group(source_id)
            except KeyError:
                logger.warning(f"Earthquake_{source_id}: No data for earthquake ID {source_id}")
                failed_events += 1
                pbar.set_postfix({"Failed": failed_events})
                pbar.update(1)
                continue

            source_data_handler = catalog_data_handler[["source_lat", "source_lon", 
                                                    "source_depth_m", "source_origin_time", 
                                                    "earthquake_type"]].drop_duplicates()
            pick_data_handler = catalog_data_handler[["station_code", "station_lat",
                                                      "station_lon", "station_elev_m",
                                                      "p_arr_time", "s_arr_time"]].drop_duplicates()
            
            # Check for  empty data frame
            if source_data_handler.empty or pick_data_handler.empty:
                logger.warning(f"Earthquake_{source_id}: No data for earthquake {source_id}")
                failed_events += 1
                pbar.set_postfix({"Failed": failed_events})
                pbar.update(1)
                continue

            # Calculate the moment magnitude
            try:
                mw_results, fitting_result = calculate_moment_magnitude(
                                            wave_path, source_data_handler,
                                            pick_data_handler, calibration_path,
                                            source_id, figure_path,
                                            lqt_mode, figure_statement 
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

    # Summary message
    sys.stdout.write(
        f"Finished. Proceed {total_earthquakes - failed_events} earthquakes successfully,"
        f"{failed_events} failed. Check runtime.log for details. \n"
    )
    return df_result, df_fitting