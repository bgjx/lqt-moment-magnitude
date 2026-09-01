"""
Core processing module for the lqt-moment-magnitude package.

This module implements the seismic moment magnitude calculation using the LQT component
system. It handles instrument response removal, waveform rotation (ZNE to LQT or ZRT),
spectral fitting with quasi-Monte Carlo optimization, and moment magnitude estimation
based on user-configured parameters from `config.ini`. The module processes waveforms,
calibrates data, and generates spectral fitting figures, aggregating results into
DataFrames.

Dependencies:
    - See `pyproject.toml` or `pip install lqtmoment` for required packages.

Note:
    This module is intended for internal use by the `lqt-moment-magnitude` API and CLI.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy.signal.detrend import simple as detrend_simple
from obspy.signal.util import _npts2nfft
from obspy.taup import TauPyModel
from scipy.signal import windows
from tqdm import tqdm

from .config import CONFIG
from .fitting_spectral import fit_spectrum_qmc
from .plotting import plot_spectral_fitting
from .refraction import calculate_inc_angle
from .utils import (
    EARTHQUAKE_TYPE,
    REQUIRED_CONFIG,
    instrument_remove,
    read_waveforms,
    trace_snr,
    wave_trim,
)

logger = logging.getLogger("lqtmoment")


def _validate_calculation_config() -> None:
    """Validate all required configuration parameters before calculation."""
    missing_config = []
    config_keys = ["wave", "magnitude", "spectral", "performance"]

    for config_key in config_keys:
        config = getattr(CONFIG, config_key)
        for attr in REQUIRED_CONFIG[config_key]:
            if not hasattr(config, attr):
                missing_config.extend(
                    [f"{attr} (missing in {config_key} config)"]
                    if not hasattr(config, attr)
                    else []
                )

    if missing_config:
        logger.error(f"Missing config attributes: {missing_config}")
        raise ValueError(f"Missing config attributes: {missing_config}")


def _initialize_result_collectors(generate_figure: bool) -> tuple[dict, dict]:
    """Initialize result and fitting result collectors."""
    fitting_result = {
        "source_id": [],
        "station": [],
        "f_corner_p": [],
        "f_corner_sv": [],
        "f_corner_sh": [],
        "q_factor_p": [],
        "q_factor_sv": [],
        "q_factor_sh": [],
        "omega_0_p_nms": [],
        "omega_0_sv_nms": [],
        "omega_0_sh_nms": [],
        "rms_e_p_nms": [],
        "rms_e_sv_nms": [],
        "rms_e_sh_nms": [],
        "moment_p_Nm": [],
        "moment_s_Nm": [],
    }

    plot_data = None
    if generate_figure:
        plot_data = {
            "all_streams": [],
            "all_p_arr_times": [],
            "all_s_arr_times": [],
            "all_time_after_p": [],
            "all_time_after_s": [],
            "all_freqs": {
                "P": [],
                "SV": [],
                "SH": [],
                "N_P": [],
                "N_SV": [],
                "N_SH": [],
            },
            "all_specs": {
                "P": [],
                "SV": [],
                "SH": [],
                "N_P": [],
                "N_SV": [],
                "N_SH": [],
            },
            "all_fits": {"P": [], "SV": [], "SH": []},
            "station_names": [],
        }

    return fitting_result, plot_data


def _extract_hypocenter_details(source_df: pd.DataFrame) -> tuple:
    """Extract and validate hypocenter details from source dataframe."""
    source_info = source_df.iloc[0]
    source_origin_time = UTCDateTime(source_info.source_origin_time)
    source_lat, source_lon, source_depth_m = (
        source_info.source_lat,
        source_info.source_lon,
        source_info.source_depth_m,
    )
    source_type = source_info.earthquake_type

    if source_type not in EARTHQUAKE_TYPE:
        logger.error(
            f"Your earthquake type ({source_type}) is not in acceptable types "
            f"(e.g., {EARTHQUAKE_TYPE})."
        )
        raise TypeError(
            f"Your earthquake type ({source_type}) is not in acceptable types "
            f"(e.g., {EARTHQUAKE_TYPE})."
        )

    return source_origin_time, source_lat, source_lon, source_depth_m, source_type


def _get_velocity_and_density(
    source_depth_m: float,
) -> tuple[float, float, float] | None:
    """Find velocity and density values for the source depth layer."""
    for layer_idx, (top, bottom) in enumerate(CONFIG.magnitude.LAYER_BOUNDARIES):
        top_m, bottom_m = top * 1000, bottom * 1000
        if top_m <= source_depth_m < bottom_m:
            velocity_P = CONFIG.magnitude.VELOCITY_VP[layer_idx] * 1000
            velocity_S = CONFIG.magnitude.VELOCITY_VS[layer_idx] * 1000
            density_value = CONFIG.magnitude.DENSITY[layer_idx]
            return velocity_P, velocity_S, density_value

    logger.warning("Hypocenter depth not within the defined layers.")
    return None


def _prepare_waveform(
    wave_path: Path,
    calibration_path: Path,
    pick_df: pd.DataFrame,
    station: str,
    source_id: int,
    p_arr_time: UTCDateTime,
    coda_time: UTCDateTime | None,
    figure_path: Path | None,
) -> Stream | None:
    """Prepare waveform: read, trim, resample, remove instrument response."""
    # Read waveform
    stream = read_waveforms(wave_path, source_id, station)
    if len(stream) < 3:
        logger.warning(
            f"Not all components available for station {station} "
            f"to calculate moment magnitude"
        )
        return None

    # Trim waveform
    try:
        if CONFIG.wave.TRIM_MODE == "dynamic":
            trimmed_stream = wave_trim(
                stream,
                p_arr_time,
                coda_time,
                CONFIG.wave.SEC_BF_P_ARR_TRIM,
                CONFIG.wave.SEC_AF_P_ARR_TRIM,
            )
        else:
            trimmed_stream = wave_trim(
                stream,
                p_arr_time,
                CONFIG.wave.SEC_BF_P_ARR_TRIM,
                CONFIG.wave.SEC_AF_P_ARR_TRIM,
            )
    except ValueError as e:
        logger.warning(f"Failed to trim wave data from {station}: {e}.", exc_info=True)
        return None

    # Resample if specified
    if CONFIG.wave.RESAMPLE_DATA is not None:
        logger.info(f"Resampling to {CONFIG.wave.RESAMPLE_DATA} sps.")
        trimmed_stream.resample(CONFIG.wave.RESAMPLE_DATA)

    # Remove instrument response
    station_info = pick_df[pick_df.station_code == station].iloc[0]
    network_code = station_info.network_code

    try:
        stream_displacement = instrument_remove(
            trimmed_stream,
            calibration_path,
            figure_path,
            network_code,
            pre_filter=CONFIG.wave.PRE_FILTER,
            water_level=CONFIG.wave.WATER_LEVEL,
            generate_figure=False,
        )
    except Exception as e:
        logger.warning(
            f"Error correcting instrument for station {station}: {e}.",
            exc_info=True,
        )
        return None

    # Apply post-instrument removal filtering
    if CONFIG.wave.APPLY_POST_INSTRUMENT_REMOVAL_FILTER:
        logger.info(
            f"Post-instrument filtering: F_MIN={CONFIG.wave.POST_FILTER_F_MIN} Hz, "
            f"F_MAX={CONFIG.wave.POST_FILTER_F_MAX} Hz"
        )
        stream_displacement.filter(
            "bandpass",
            freqmin=CONFIG.wave.POST_FILTER_F_MIN,
            freqmax=CONFIG.wave.POST_FILTER_F_MAX,
            corners=4,
            zerophase=True,
        )

    return stream_displacement


def _calculate_and_fit_spectra(
    rotated_stream: Stream,
    p_window_data: np.ndarray,
    sv_window_data: np.ndarray,
    sh_window_data: np.ndarray,
    p_noise_data: np.ndarray,
    sv_noise_data: np.ndarray,
    sh_noise_data: np.ndarray,
    p_arr_time: UTCDateTime,
    s_arr_time: UTCDateTime,
    source_origin_time: UTCDateTime,
) -> tuple:
    """Calculate spectra and fit them."""
    fs = 1 / rotated_stream[0].stats.delta

    try:
        # Calculate source spectra
        freq_P, spec_P = calculate_seismic_spectra(
            p_window_data,
            fs,
            freq_min=CONFIG.spectral.F_MIN,
            freq_max=CONFIG.spectral.F_MAX,
            smooth_window_size=CONFIG.spectral.SMOOTH_WINDOW_SIZE,
        )
        freq_SV, spec_SV = calculate_seismic_spectra(
            sv_window_data,
            fs,
            freq_min=CONFIG.spectral.F_MIN,
            freq_max=CONFIG.spectral.F_MAX,
            smooth_window_size=CONFIG.spectral.SMOOTH_WINDOW_SIZE,
        )
        freq_SH, spec_SH = calculate_seismic_spectra(
            sh_window_data,
            fs,
            freq_min=CONFIG.spectral.F_MIN,
            freq_max=CONFIG.spectral.F_MAX,
            smooth_window_size=CONFIG.spectral.SMOOTH_WINDOW_SIZE,
        )

        freq_N_P, spec_N_P = calculate_seismic_spectra(
            p_noise_data,
            fs,
            freq_min=CONFIG.spectral.F_MIN,
            freq_max=CONFIG.spectral.F_MAX,
            smooth_window_size=CONFIG.spectral.SMOOTH_WINDOW_SIZE,
        )
        freq_N_SV, spec_N_SV = calculate_seismic_spectra(
            sv_noise_data,
            fs,
            freq_min=CONFIG.spectral.F_MIN,
            freq_max=CONFIG.spectral.F_MAX,
            smooth_window_size=CONFIG.spectral.SMOOTH_WINDOW_SIZE,
        )
        freq_N_SH, spec_N_SH = calculate_seismic_spectra(
            sh_noise_data,
            fs,
            freq_min=CONFIG.spectral.F_MIN,
            freq_max=CONFIG.spectral.F_MAX,
            smooth_window_size=CONFIG.spectral.SMOOTH_WINDOW_SIZE,
        )
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Error during spectra calculation: {e}.", exc_info=True)
        return None

    # Fit spectra
    try:
        fit_P = fit_spectrum_qmc(
            freq_P,
            spec_P,
            abs(float(p_arr_time - source_origin_time)),
            CONFIG.spectral.F_MIN,
            CONFIG.spectral.F_MAX,
            CONFIG.spectral.DEFAULT_N_SAMPLES,
        )
        fit_SV = fit_spectrum_qmc(
            freq_SV,
            spec_SV,
            abs(float(s_arr_time - source_origin_time)),
            CONFIG.spectral.F_MIN,
            CONFIG.spectral.F_MAX,
            CONFIG.spectral.DEFAULT_N_SAMPLES,
        )
        fit_SH = fit_spectrum_qmc(
            freq_SH,
            spec_SH,
            abs(float(s_arr_time - source_origin_time)),
            CONFIG.spectral.F_MIN,
            CONFIG.spectral.F_MAX,
            CONFIG.spectral.DEFAULT_N_SAMPLES,
        )
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Error during spectral fitting: {e}.", exc_info=True)
        return None

    if any(f is None for f in [fit_P, fit_SV, fit_SH]):
        logger.warning("None values returned from spectral fitting.")
        return None

    return (
        freq_P,
        spec_P,
        freq_SV,
        spec_SV,
        freq_SH,
        spec_SH,
        freq_N_P,
        spec_N_P,
        freq_N_SV,
        spec_N_SV,
        freq_N_SH,
        spec_N_SH,
        fit_P,
        fit_SV,
        fit_SH,
    )


def _collect_plot_data(
    plot_data: dict,
    rotated_stream: Stream,
    p_arr_time: UTCDateTime,
    s_arr_time: UTCDateTime,
    time_after_p: np.ndarray,
    time_after_s: np.ndarray,
    freq_P: np.ndarray,
    spec_P: np.ndarray,
    freq_SV: np.ndarray,
    spec_SV: np.ndarray,
    freq_SH: np.ndarray,
    spec_SH: np.ndarray,
    freq_N_P: np.ndarray,
    spec_N_P: np.ndarray,
    freq_N_SV: np.ndarray,
    spec_N_SV: np.ndarray,
    freq_N_SH: np.ndarray,
    spec_N_SH: np.ndarray,
    fit_P: tuple,
    fit_SV: tuple,
    fit_SH: tuple,
    station: str,
) -> None:
    """Collect and append plot data."""
    plot_data["all_streams"].append(rotated_stream)
    plot_data["all_p_arr_times"].append(p_arr_time)
    plot_data["all_s_arr_times"].append(s_arr_time)
    plot_data["all_time_after_p"].append(time_after_p)
    plot_data["all_time_after_s"].append(time_after_s)
    for key, freq, spec in [
        ("P", freq_P, spec_P),
        ("SV", freq_SV, spec_SV),
        ("SH", freq_SH, spec_SH),
    ]:
        plot_data["all_freqs"][key].append(freq)
        plot_data["all_specs"][key].append(spec)
    for key, freq in [("N_P", freq_N_P), ("N_SV", freq_N_SV), ("N_SH", freq_N_SH)]:
        plot_data["all_freqs"][key].append(freq)
        plot_data["all_specs"][key].append(
            [spec_N_P, spec_N_SV, spec_N_SH][["N_P", "N_SV", "N_SH"].index(key)]
        )
    plot_data["all_fits"]["P"].append(fit_P)
    plot_data["all_fits"]["SV"].append(fit_SV)
    plot_data["all_fits"]["SH"].append(fit_SH)
    plot_data["station_names"].append(station)


def _process_station_data(
    wave_path: Path,
    calibration_path: Path,
    pick_df: pd.DataFrame,
    station: str,
    source_id: int,
    source_origin_time: UTCDateTime,
    source_coordinate: list[float],
    velocity_P: float,
    velocity_S: float,
    density_value: float,
    lqt_mode: bool,
    generate_figure: bool,
    figure_path: Path | None,
    plot_data: dict | None,
) -> tuple[float | None, float | None, float | None, dict]:
    """Process waveform data for a single station."""
    field_names = [
        "f_corner_p",
        "f_corner_sv",
        "f_corner_sh",
        "q_factor_p",
        "q_factor_sv",
        "q_factor_sh",
        "omega_0_p_nms",
        "omega_0_sv_nms",
        "omega_0_sh_nms",
        "rms_e_p_nms",
        "rms_e_sv_nms",
        "rms_e_sh_nms",
        "moment_p_Nm",
        "moment_s_Nm",
    ]
    empty_result = dict.fromkeys(field_names)

    def _empty_with_ids():
        return {**empty_result, "source_id": source_id, "station": station}

    # Get station info and times
    station_info = pick_df[pick_df.station_code == station].iloc[0]
    station_lat, station_lon, station_elev_m = (
        station_info.station_lat,
        station_info.station_lon,
        station_info.station_elev_m,
    )
    p_arr_time, s_arr_time, s_p_lag_time_sec = (
        UTCDateTime(station_info.p_arr_time),
        UTCDateTime(station_info.s_arr_time),
        station_info.s_p_lag_time_sec,
    )
    coda_time = (
        UTCDateTime(station_info.coda_time)
        if not pd.isna(station_info.coda_time)
        else None
    )
    epicentral_distance, azimuth, _ = gps2dist_azimuth(
        source_coordinate[0], source_coordinate[1], station_lat, station_lon
    )

    # Prepare waveform
    stream_displacement = _prepare_waveform(
        wave_path,
        calibration_path,
        pick_df,
        station,
        source_id,
        p_arr_time,
        coda_time,
        figure_path,
    )
    if stream_displacement is None:
        return None, None, None, _empty_with_ids()

    # Rotate and window
    logger.info(f"Station {station} rotated, LQT mode {'ON' if lqt_mode else 'OFF'}.")
    try:
        rotated_stream = _rotate_stream(
            stream_displacement,
            source_coordinate[3],
            source_coordinate[:3],
            [station_lat, station_lon, station_elev_m],
            azimuth,
            s_p_lag_time_sec,
            p_arr_time,
            s_arr_time,
            lqt_mode,
        )
        (
            p_window_data,
            sv_window_data,
            sh_window_data,
            p_noise_data,
            sv_noise_data,
            sh_noise_data,
            time_after_p,
            time_after_s,
        ) = window_trace(rotated_stream, p_arr_time, s_arr_time, lqt_mode=lqt_mode)
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Failed to rotate/window for {station}: {e}.", exc_info=True)
        return None, None, None, _empty_with_ids()

    # Check SNR
    if any(
        trace_snr(d, n) <= CONFIG.wave.SNR_THRESHOLD
        for d, n in zip(
            [p_window_data, sv_window_data, sh_window_data],
            [p_noise_data, sv_noise_data, sh_noise_data],
        )
    ):
        return None, None, None, _empty_with_ids()

    # Calculate and fit spectra
    spectra_result = _calculate_and_fit_spectra(
        rotated_stream,
        p_window_data,
        sv_window_data,
        sh_window_data,
        p_noise_data,
        sv_noise_data,
        sh_noise_data,
        p_arr_time,
        s_arr_time,
        source_origin_time,
    )
    if spectra_result is None:
        return None, None, None, _empty_with_ids()

    (
        freq_P,
        spec_P,
        freq_SV,
        spec_SV,
        freq_SH,
        spec_SH,
        freq_N_P,
        spec_N_P,
        freq_N_SV,
        spec_N_SV,
        freq_N_SH,
        spec_N_SH,
        fit_P,
        fit_SV,
        fit_SH,
    ) = spectra_result

    # Extract and calculate moments
    Omega_0_P, Q_factor_p, f_c_P, err_P, _, _ = fit_P
    Omega_0_SV, Q_factor_SV, f_c_SV, err_SV, _, _ = fit_SV
    Omega_0_SH, Q_factor_SH, f_c_SH, err_SH, _, _ = fit_SH

    try:
        omega_P = Omega_0_P * 1e-9
        omega_S = np.sqrt(Omega_0_SV**2 + Omega_0_SH**2) * 1e-9
        if source_coordinate[3] in ("very_local_earthquake", "local_earthquake"):
            source_distance_m = np.sqrt(
                epicentral_distance**2 + ((source_coordinate[2] + station_elev_m) ** 2)
            )
        else:
            source_distance_m = epicentral_distance
        M_0_P = (
            4.0 * np.pi * density_value * (velocity_P**3) * source_distance_m * omega_P
        ) / (CONFIG.magnitude.R_PATTERN_P * CONFIG.magnitude.FREE_SURFACE_FACTOR)
        M_0_S = (
            4.0 * np.pi * density_value * (velocity_S**3) * source_distance_m * omega_S
        ) / (CONFIG.magnitude.R_PATTERN_S * CONFIG.magnitude.FREE_SURFACE_FACTOR)
        moment = (M_0_P + M_0_S) / 2
        source_rad = (
            (CONFIG.magnitude.K_P * velocity_P) / f_c_P
            + (2 * CONFIG.magnitude.K_S * velocity_S) / (f_c_SV + f_c_SH)
        ) / 2
        corner_freq = (f_c_P + (f_c_SV + f_c_SH) / 2) / 2
    except (ValueError, ZeroDivisionError) as e:
        logger.warning(f"Failed to calc moment for {station}: {e}.", exc_info=True)
        return None, None, None, _empty_with_ids()

    fitting_result = _empty_with_ids()
    fitting_result.update(
        {
            "f_corner_p": f_c_P,
            "f_corner_sv": f_c_SV,
            "f_corner_sh": f_c_SH,
            "q_factor_p": Q_factor_p,
            "q_factor_sv": Q_factor_SV,
            "q_factor_sh": Q_factor_SH,
            "omega_0_p_nms": Omega_0_P,
            "omega_0_sv_nms": Omega_0_SV,
            "omega_0_sh_nms": Omega_0_SH,
            "rms_e_p_nms": err_P,
            "rms_e_sv_nms": err_SV,
            "rms_e_sh_nms": err_SH,
            "moment_p_Nm": M_0_P,
            "moment_s_Nm": M_0_S,
        }
    )

    if generate_figure and plot_data:
        _collect_plot_data(
            plot_data,
            rotated_stream,
            p_arr_time,
            s_arr_time,
            time_after_p,
            time_after_s,
            freq_P,
            spec_P,
            freq_SV,
            spec_SV,
            freq_SH,
            spec_SH,
            freq_N_P,
            spec_N_P,
            freq_N_SV,
            spec_N_SV,
            freq_N_SH,
            spec_N_SH,
            fit_P,
            fit_SV,
            fit_SH,
            station,
        )

    return moment, source_rad, corner_freq, fitting_result


def calculate_seismic_spectra(
    trace_data: np.ndarray,
    sampling_rate: float,
    freq_min: float | None = None,
    freq_max: float | None = None,
    smooth_window_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the single-sided displacement amplitude spectrum of a seismogram
    using FFT. Always applies a Hann Window to reduce spectral leakage.

    Args:
        trace_data (np.ndarray): Array of displacement signal (in meters).
        sampling_rate (float): Sampling rate of the signal in Hz.
        freq_min (float | None): Minimum frequency to include in output (Hz).
            Default to None.
        freq_max (float | None): Maximum frequency to include in output (Hz).
            Default to None.
        smooth_window_size (int | None): Size of the moving average window for
            smoothing. If None, no smoothing is applied. Must be odd positive.
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - frequencies: Array of sample frequencies in Hz.
            - amplitudes: Array of displacement amplitudes in nm (nanometers).
    Raises:
        ValueError: If trace_data is empty or invalid sampling rate
    """
    if not trace_data.size or sampling_rate <= 0:
        raise ValueError(
            "Trace data cannot be empty and sampling rate must be positive"
        )

    # Apply Hann window
    window = windows.hann(len(trace_data))
    trace_data_processed = trace_data * window

    # Hann window correction
    window_correction = np.sqrt(8.0 / 3.0)

    # Zero pad to next power of 2 for FFT efficiency and resolution
    n_samples = len(trace_data_processed)
    nfft = _npts2nfft(n_samples)
    padded_data = np.pad(trace_data_processed, (0, nfft - n_samples), mode="constant")

    # Compute the FFT and single-sided spectrum
    fft_data = np.fft.rfft(padded_data)
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / sampling_rate)

    # Scale amplitudes: 2.0 for negative frequencies, 1/nfft for FFT normalization,
    # window_correction for Hann window
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
        smoothing_window = np.ones(smooth_window_size) / smooth_window_size
        amplitudes = np.convolve(amplitudes, smoothing_window, mode="same")

    return frequencies, amplitudes


def window_trace(
    streams: Stream,
    p_arr_time: UTCDateTime,
    s_arr_time: UTCDateTime,
    lqt_mode: bool = True,
) -> tuple[np.ndarray, ...]:
    """
    Windows seismic trace data around P, SV, and SH phase and extracts noise data.

    Args:

        streams (Stream): A stream object containing the seismic data.
        p_arr_time (UTCDateTime): Arrival time in UTCDateTime of the P phase.
        s_arr_time (UTCDateTime): Arrival time in UTCDateTime of the S phase.
        lqt_mode (bool): Use LQT components if True, ZRT if false. Default to True.

    Returns:
        tuple [
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            float,
            float
            ]:
            - P_data: The data windowed around the P phase in the L/Z component
                (displacement in meters).
            - SV_data: The data windowed around the S phase in the Q/R component
                (displacement in meters).
            - SH_data: The data windowed around the S phase in the T component
                (displacement in meters).
            - P_noise: The noise data before the P phase in the L/Z component
                (displacement in meters).
            - SV_noise: The noise data before the P phase in the Q/R component
                (displacement in meters).
            - SH_noise: The noise data before the P phase in the T component
                (displacement in meters).
            - time_after_pick_p: Time after P arrival in seconds.
            - time_after_pick_s: Time after S arrival in seconds.

    Raises:
        ValueError: If component traces are missing.

    Notes:
        Windows are dynamically calculated based on S-P time with configurable padding.
    """
    components = ["L", "Q", "T"] if lqt_mode else ["Z", "R", "T"]
    try:
        trace_P, trace_SV, trace_SH = [
            streams.select(component=comp)[0] for comp in components
        ]
    except IndexError as e:
        raise ValueError(f"Missing {components} components in stream") from e

    # Verify trace starttime consistency
    ref_starttime = trace_P.stats.starttime
    for tr in [trace_SV, trace_SH]:
        if tr.stats.starttime != ref_starttime:
            logger.warning(f"Traces {tr.id} have inconsistent starttime")

    # Dynamic window parameters
    s_p_time = s_arr_time - p_arr_time
    p_factor = 0.75 if s_p_time > 1.0 else 1.0
    s_factor = 2.25 if s_p_time > 1.0 else 3.0
    time_after_pick_p = min(
        max(p_factor * s_p_time, CONFIG.wave.MIN_P_WINDOW), CONFIG.wave.MAX_P_WINDOW
    )
    time_after_pick_s = min(
        max(s_factor * s_p_time, CONFIG.wave.MIN_S_WINDOW), CONFIG.wave.MAX_S_WINDOW
    )

    # Prevent P/S window overlap
    p_phase_end_time = p_arr_time + time_after_pick_p
    s_phase_start_time = s_arr_time - CONFIG.wave.PADDING_BEFORE_ARRIVAL
    if p_phase_end_time > s_phase_start_time:
        time_after_pick_p = s_phase_start_time - p_arr_time
        logger.info(
            f"Windows adjusted to prevent P/S overlap: P ends at "
            f"{p_arr_time + time_after_pick_p}, S starts at {s_phase_start_time}"
        )

    # Find the data index for phase windowing
    p_phase_start_index = int(
        (p_arr_time - trace_P.stats.starttime - CONFIG.wave.PADDING_BEFORE_ARRIVAL)
        / trace_P.stats.delta
    )
    p_phase_end_index = int(
        (p_arr_time - trace_P.stats.starttime + time_after_pick_p) / trace_P.stats.delta
    )
    s_phase_start_index = int(
        (s_phase_start_time - trace_SV.stats.starttime) / trace_SV.stats.delta
    )
    s_phase_end_index = int(
        (s_arr_time - trace_SV.stats.starttime + time_after_pick_s)
        / trace_SV.stats.delta
    )
    noise_start_index = int(
        (p_arr_time - trace_P.stats.starttime - CONFIG.wave.NOISE_DURATION)
        / trace_P.stats.delta
    )
    noise_end_index = int(
        (p_arr_time - trace_P.stats.starttime - CONFIG.wave.NOISE_PADDING)
        / trace_P.stats.delta
    )

    # Slicing method
    def _slice(data, start_idx, end_idx):
        """Helper function to do data slicing."""
        start_idx = max(0, min(start_idx, len(data) - 1))
        end_idx = max(start_idx, min(end_idx + 1, len(data)))
        return data[start_idx:end_idx]

    # Window the data by the index
    P_data = _slice(trace_P.data, p_phase_start_index, p_phase_end_index)
    SV_data = _slice(trace_SV.data, s_phase_start_index, s_phase_end_index)
    SH_data = _slice(trace_SH.data, s_phase_start_index, s_phase_end_index)
    P_noise = _slice(trace_P.data, noise_start_index, noise_end_index)
    SV_noise = _slice(trace_SV.data, noise_start_index, noise_end_index)
    SH_noise = _slice(trace_SH.data, noise_start_index, noise_end_index)

    # Preprocess data, apply detrending, demean, and taper.
    def _preprocess_data(data: np.ndarray) -> np.ndarray:
        """Helper function to preprocess data, apply de-trending, demean, and taper."""
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

    return (
        P_data,
        SV_data,
        SH_data,
        P_noise,
        SV_noise,
        SH_noise,
        time_after_pick_p,
        time_after_pick_s,
    )


def _rotate_stream(
    stream: Stream,
    source_type: str,
    source_coordinate: list[float],
    station_coordinate: list[float],
    azimuth: float,
    s_p_lag_time_sec: float,
    p_arr_time: UTCDateTime,
    s_arr_time: UTCDateTime,
    lqt_mode: bool,
) -> Stream:
    """
    Rotate the stream from ZNE to LQT or ZRT based on earthquake type and lqt_mode flag.

    Args:
        stream (Stream): Input stream in ZNE components.
        source_type (str): Type of the earthquake (e.g, 'very_local_earthquake').
        source_coordinate (list[float]): Source coordinate [lat, lon, depth].
        station_coordinate (list[float]): Station coordinate [lat, lon, elev].
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
    if source_type == "very_local_earthquake" and lqt_mode is False:
        stream_zrt = stream.copy()
        stream_zrt.rotate(method="NE->RT", back_azimuth=azimuth)
        sh_trace, sv_trace, p_trace = stream_zrt.traces  # T, R, Z components
    elif (
        source_type == "very_local_earthquake" and lqt_mode is True
    ) or source_type == "local_earthquake":
        trace_Z = stream.select(component="Z")[0]
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
            s_arr_time,
        )
        stream_lqt_p = stream.copy()
        stream_lqt_s = stream.copy()
        stream_lqt_p.rotate(
            method="ZNE->LQT", back_azimuth=azimuth, inclination=incidence_angle_p
        )
        stream_lqt_s.rotate(
            method="ZNE->LQT", back_azimuth=azimuth, inclination=incidence_angle_s
        )
        _, _, p_trace = stream_lqt_p.traces  # T, Q, L components
        sh_trace, sv_trace, _ = stream_lqt_s.traces  # T, Q, L components
    else:
        model = TauPyModel(model=CONFIG.magnitude.TAUP_MODEL)
        arrivals = model.get_travel_times(
            source_depth_in_km=(source_coordinate[2] * -1e-3),
            distance_in_degree=locations2degrees(
                source_coordinate[0],
                source_coordinate[1],
                station_coordinate[0],
                station_coordinate[1],
            ),
            phase_list=["P", "S"],
        )
        incidence_angle_p = arrivals[0].incident_angle
        incidence_angle_s = arrivals[1].incident_angle
        stream_lqt_p = stream.copy()
        stream_lqt_s = stream.copy()
        stream_lqt_p.rotate(
            method="ZNE->LQT", back_azimuth=azimuth, inclination=incidence_angle_p
        )
        stream_lqt_s.rotate(
            method="ZNE->LQT", back_azimuth=azimuth, inclination=incidence_angle_s
        )
        _, _, p_trace = stream_lqt_p.traces  # T, Q, L components
        sh_trace, sv_trace, _ = stream_lqt_s.traces  # T, Q, L components

    return Stream(traces=[p_trace, sv_trace, sh_trace])


def calculate_moment_magnitude(
    wave_path: Path,
    calibration_path: Path,
    source_df: pd.DataFrame,
    pick_df: pd.DataFrame,
    source_id: int,
    lqt_mode: bool = True,
    generate_figure: bool = False,
    figure_path: Path | None = None,
) -> tuple[dict[str, str], dict[str, list]]:
    """
    Processes moment magnitude calculation for an earthquake from given hypocenter
    dataframe and picking dataframe. This function handles waveform instrument
    response removal, seismogram rotation, spectral fitting, moment magnitude
    calculation, and figure creation. It returns two dictionary objects.

    Args:
        wave_path (Path): Path to the directory containing waveform files.
        calibration_path (Path): Path to the calibration files for instrument response.
        source_df (pd.DataFrame): DataFrame containing hypocenter information.
        pick_df (pd.DataFrame): DataFrame containing pick information.
        source_id (int): Unique identifier for the earthquake.
        lqt_mode (bool): If True, perform LQT rotation; otherwise, use ZRT.
        generate_figure (bool): Boolean to generate and save figures (default False).
        figure_path (Optional[Path]): Path to save figures. Defaults to None.

    Returns:
        Tuple[Dict[str, str], Dict[str, List]]:
            - results: A dictionary containing moment magnitude and related metrics.
            - fitting_result: A dictionary of detailed fitting results per station.

    Raises:
        ValueError: if source_df or pick_df are empty or wrong format.
        OSError: If waveform or calibration files cannot be read.
    """
    # Validate configuration
    _validate_calculation_config()

    # Initialize result collectors
    fitting_result, plot_data = _initialize_result_collectors(generate_figure)

    # Extract and validate hypocenter details
    source_origin_time, source_lat, source_lon, source_depth_m, source_type = (
        _extract_hypocenter_details(source_df)
    )

    # Get velocity and density for source depth
    velocity_info = _get_velocity_and_density(source_depth_m)
    if velocity_info is None:
        return {}, fitting_result

    velocity_P, velocity_S, density_value = velocity_info

    # Process each station
    moments, corner_frequencies, source_radius = [], [], []
    source_coordinate = [source_lat, source_lon, -1 * source_depth_m, source_type]

    for station in pick_df.get("station_code").unique():
        moment, source_rad, corner_freq, station_fitting = _process_station_data(
            wave_path,
            calibration_path,
            pick_df,
            station,
            source_id,
            source_origin_time,
            source_coordinate,
            [source_lat, source_lon, source_depth_m],
            velocity_P,
            velocity_S,
            density_value,
            lqt_mode,
            generate_figure,
            figure_path,
            plot_data,
        )

        if moment is not None:
            moments.append(moment)
            source_radius.append(source_rad)
            corner_frequencies.append(corner_freq)

        # Add station results to fitting result
        for key in station_fitting:
            if key in fitting_result:
                fitting_result[key].append(station_fitting[key])

    # Return early if no valid results
    if not moments or not corner_frequencies or not source_radius:
        return {}, fitting_result

    # Calculate average moment magnitude
    moment_average = np.mean(moments)
    moment_std = np.std(moments)
    mw = ((2.0 / 3.0) * np.log10(moment_average)) - CONFIG.magnitude.MW_CONSTANT
    mw_std = (2.0 / 3.0) * moment_std / (moment_average * np.log(10))

    results = {
        "source_id": [source_id],
        "fc_avg": [np.mean(corner_frequencies)],
        "fc_std": [np.std(corner_frequencies)],
        "src_rad_avg_m": [np.mean(source_radius)],
        "src_rad_std_m": [np.std(source_radius)],
        "stress_drop_bar": [
            (7 * moment_average) / (16 * np.mean(source_radius) ** 3) * 1e-5
        ],
        "mw_average": [mw],
        "mw_std": [mw_std],
    }

    # Create spectral fitting plot if data exists
    if generate_figure and plot_data and plot_data["all_streams"]:
        try:
            plot_spectral_fitting(
                source_id,
                plot_data["all_streams"],
                plot_data["all_p_arr_times"],
                plot_data["all_s_arr_times"],
                plot_data["all_time_after_p"],
                plot_data["all_time_after_s"],
                plot_data["all_freqs"],
                plot_data["all_specs"],
                plot_data["all_fits"],
                plot_data["station_names"],
                lqt_mode,
                figure_path,
            )
        except (ValueError, OSError) as e:
            logger.warning(
                f"Failed to create spectral fitting plot for event {source_id}: {e}.",
                exc_info=True,
            )

    return results, fitting_result


def _setup_id_range(
    catalog_data: pd.DataFrame, id_start: int | None, id_end: int | None
) -> tuple[int, int]:
    """Setup and validate ID range for earthquake processing."""
    default_id_start = int(catalog_data["source_id"].min())
    default_id_end = int(catalog_data["source_id"].max())

    id_start = id_start if id_start is not None else default_id_start
    id_end = id_end if id_end is not None else default_id_end

    if not (
        isinstance(id_start, int) and isinstance(id_end, int) and id_start <= id_end
    ):
        logger.error(f"Invalid ID range: id_start={id_start}, id_end={id_end}")
        raise ValueError(f"Invalid ID range: id_start={id_start}, id_end={id_end}")

    if not (
        id_start in catalog_data["source_id"].values
        and id_end in catalog_data["source_id"].values
    ):
        logger.error(f"ID range {id_start} - {id_end} not found in catalog")
        raise ValueError(f"ID range {id_start} - {id_end} not found in catalog")

    return id_start, id_end


def _initialize_result_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Initialize empty result dataframes with proper column names."""
    df_result = pd.DataFrame(
        columns=[
            "source_id",
            "fc_avg",
            "fc_std",
            "Src_rad_avg_m",
            "Src_rad_std_m",
            "Stress_drop_bar",
            "mw_average",
            "mw_std",
        ]
    )
    df_fitting = pd.DataFrame(
        columns=[
            "source_id",
            "station",
            "f_corner_p",
            "f_corner_sv",
            "f_corner_sh",
            "q_factor_p",
            "q_factor_sv",
            "q_factor_sh",
            "omega_0_P_nms",
            "omega_0_sv_nms",
            "omega_0_sh_nms",
            "rms_e_p_nms",
            "rms_e_sv_nms",
            "rms_e_sh_nms",
            "moment_p_Nm",
            "moment_s_Nm",
        ]
    )
    return df_result, df_fitting


def _process_earthquake_range(
    wave_path: Path,
    calibration_path: Path,
    grouped_data,
    id_start: int,
    id_end: int,
    lqt_mode: bool,
    generate_figure: bool,
    figure_path: Path | None,
) -> tuple[list, list, int]:
    """Process earthquakes within the ID range."""
    result_list = []
    fitting_list = []
    failed_events = 0
    total_earthquakes = id_end - id_start + 1

    with tqdm(
        total=total_earthquakes,
        file=sys.stderr,
        position=0,
        leave=True,
        desc="Processing earthquakes",
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        ),
        ncols=80,
        smoothing=0.1,
    ) as pbar:
        for source_id in range(id_start, id_end + 1):
            logger.info("\n\n")
            logger.info(f"**========** Earthquake ID: {source_id} **========**")

            try:
                catalog_data = grouped_data.get_group(source_id)
            except KeyError as e:
                logger.warning(
                    f"No data for earthquake ID {source_id}, check catalog: {e}.",
                    exc_info=True,
                )
                failed_events += 1
                pbar.set_postfix({"Failed": failed_events})
                pbar.update(1)
                continue

            source_data = catalog_data[
                [
                    "source_lat",
                    "source_lon",
                    "source_depth_m",
                    "source_origin_time",
                    "earthquake_type",
                ]
            ].drop_duplicates()

            pick_data = catalog_data[
                [
                    "network_code",
                    "station_code",
                    "station_lat",
                    "station_lon",
                    "station_elev_m",
                    "p_arr_time",
                    "s_arr_time",
                    "s_p_lag_time_sec",
                    "coda_time",
                ]
            ].drop_duplicates()

            if source_data.empty or pick_data.empty:
                logger.warning("Hypocenter and picking data not completely available.")
                failed_events += 1
                pbar.set_postfix({"Failed": failed_events})
                pbar.update(1)
                continue

            try:
                mw_results, fitting_result = calculate_moment_magnitude(
                    wave_path=wave_path,
                    calibration_path=calibration_path,
                    source_df=source_data,
                    pick_df=pick_data,
                    source_id=source_id,
                    lqt_mode=lqt_mode,
                    generate_figure=generate_figure,
                    figure_path=figure_path,
                )
                result_list.append(pd.DataFrame.from_dict(mw_results))
                fitting_list.append(pd.DataFrame.from_dict(fitting_result))
            except (ValueError, OSError) as e:
                logger.error(f"Calculation failed: {e}", exc_info=True)
                failed_events += 1
                pbar.set_postfix({"Failed": failed_events})
                pbar.update(1)
                continue

            pbar.set_postfix({"Failed": failed_events})
            pbar.update(1)

    return result_list, fitting_list, failed_events


def _finalize_results(
    catalog_data_copy: pd.DataFrame,
    df_result: pd.DataFrame,
    result_list: list,
    fitting_list: list,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Finalize and merge results."""
    df_result = pd.concat(result_list, ignore_index=True) if result_list else df_result
    df_fitting = (
        pd.concat(fitting_list, ignore_index=True) if fitting_list else pd.DataFrame()
    )

    merged_catalog = pd.merge(
        catalog_data_copy,
        df_result[["source_id", "mw_average"]],
        on="source_id",
        how="left",
    )

    merged_catalog = merged_catalog.rename(columns={"mw_average": "magnitude"})

    columns_name = merged_catalog.columns.to_list()
    columns_name.remove("magnitude")
    columns_name.insert(5, "magnitude")
    merged_catalog = merged_catalog[columns_name]

    merged_catalog = merged_catalog.reset_index(drop=True)

    return merged_catalog, df_result, df_fitting


def start_calculate(
    wave_path: Path,
    calibration_path: Path,
    catalog_data: pd.DataFrame,
    id_start: int | None = None,
    id_end: int | None = None,
    lqt_mode: bool = True,
    generate_figure: bool = False,
    figure_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function processes moment magnitude calculation by iterating over a
    user-specified range of earthquake IDs. For each event of earthquake, it extracts
    source and station data, and computes moment magnitudes using waveform and response
    file, and aggregates results intotwo DataFrames: magnitude results and spectral
    fitting parameters.

    Args:
        wave_path (Path): Path to the directory containing waveforms file
            (.miniSEED format).
        calibration_path (Path) : Path to the directory containing calibration file
            (.RESP format).
        catalog_data (pd.DataFrame): Catalog DataFrame in LQTMomentMag format.
        id_start (Optional[int]): Starting earthquake ID. If not provided, id_start set
        to min ID in catalog.
            id_end (Optional[int]): Ending earthquake ID. If not provided, id_end set
            to max ID in catalog.
        lqt_mode (bool): Use LQT rotation if True, ZRT otherwise. Defaults to True.
        generate_figure (bool): Generate and save figures if True. Defaults to False.
        figure_path (Optional[Path]) : Path to the directory where spectral fitting
            figures will be saved. Defaults to None, then generate folder at current
            directory.

    Returns:
        Tuple [pd.Dataframe, pd.DataFrame]:
            - First DataFrame: Magnitude results with columns [
                'source_id',
                'fc_avg',
                'fc_std',
                ...].
            - Second DataFrame: Fitting results with columns [
                'source_id',
                'station',
                'f_corner_p',
                ...].

    Raises:
        ValueError: If catalog_data is empty or missing required columns.

    Example:
        >>> catalog = pd.read_excel("lqt_catalog.xlsx")
        >>> result_df, fitting_df = start_calculate(
        ...     Path("data/waveforms"), Path("data/calibration"),
        ...     catalog)
    """
    # Setup and validate ID range
    id_start, id_end = _setup_id_range(catalog_data, id_start, id_end)

    # Initialize result dataframes
    df_result, df_fitting = _initialize_result_dataframes()

    # Pre-group catalog for efficiency
    catalog_data_copy = catalog_data.copy()
    grouped_data = catalog_data_copy.groupby("source_id")

    # Process earthquake range
    result_list, fitting_list, failed_events = _process_earthquake_range(
        wave_path,
        calibration_path,
        grouped_data,
        id_start,
        id_end,
        lqt_mode,
        generate_figure,
        figure_path,
    )

    # Finalize results and merge
    merged_catalog, df_result, df_fitting = _finalize_results(
        catalog_data_copy, df_result, result_list, fitting_list
    )

    # Output summary
    total_earthquakes = id_end - id_start + 1
    sys.stdout.write(
        f"Finished. Processed {total_earthquakes - failed_events} earthquakes "
        f"successfully, {failed_events} failed. Check lqt_runtime.log for details.\n"
    )

    return merged_catalog, df_result, df_fitting
