"""
Refraction module for lqt-moment-magnitude package.

This module calculates incidence angles, travel times, and ray paths for seismic waves
(P-waves, S-waves) using a layered 1-D velocity model and Snell's Law-based shooting
method, suitable for shallow borehole 3-C sensor.

Dependencies:
    - See `pyproject.toml` or `pip install lqtmoment` for required packages.

References:
- Aki, K., & Richards, P. G. (2002). Quantitative Seismology, 2nd Edition.
    University Science Books.

"""

import logging
from pathlib import Path

import numpy as np
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from scipy.optimize import brentq
from scipy.signal import welch

from .config import CONFIG
from .plotting import plot_rays

logger = logging.getLogger("lqtmoment")

# Global parameters
ANGLE_BOUNDS = (0.01, 89.99)


def compute_dominant_period(
    trace: np.ndarray,
    arrival_time: UTCDateTime,
    window_length: float = 5,
    f_min: float = 0.1,
    f_max: float = 15,
) -> float:
    """
    This function calculate the dominant period from a segment of a trace.

    Args:
        trace (np.ndarray): Array of trace data.
        arrival_time (UTCDataTime): Arrival time of a phase in UTCDateTime format.
        window_length (float): The length of trace segment in seconds, default to
                                10 seconds.
        f_min (float): Minimum frequency in Hz for calculating dominant period,
                        default to 0.1 Hz.
        f_max (float): Maximum frequency in Hz for calculating dominant period,
                        default to 15 Hz.

    Returns:
        float: Dominant period of a trace segment in second.

    Raises:
        ValueError: If trace is empty or arrival time is invalid.

    """
    if trace is None or len(trace.data) == 0:
        raise ValueError("Trace data cannot be None or Empty")
    if arrival_time < trace.stats.starttime or arrival_time > trace.stats.endtime:
        raise ValueError(
            f"Arrival time {arrival_time} is outside trace time range"
            f"({trace.stats.starttime} to {trace.stats.endtime})"
        )

    sampling_rate = trace.stats.sampling_rate
    idx_start = int((arrival_time - trace.stats.starttime) * sampling_rate)
    idx_end = int(idx_start + (window_length * sampling_rate))

    idx_start = max(0, min(len(trace.data) - 1, idx_start))
    idx_end = max(0, min(len(trace.data), idx_end))
    data = trace.data[idx_start:idx_end]
    if len(data) == 0:
        return 2.0

    freqs, psd = welch(
        data, fs=sampling_rate, nperseg=min(256, len(data)), scaling="density"
    )
    mask = (freqs >= f_min) & (freqs <= f_max)
    freqs = freqs[mask]
    psd = psd[mask]
    if len(freqs) == 0:
        return 2.0

    f_max = freqs[np.argmax(psd)]
    return 1 / f_max


def build_raw_model(
    layer_boundaries: list[list[float]], velocities: list[float]
) -> list[list[float]]:
    """
    Build a model of layers from the given layer boundaries and velocities.

    Args:
        layer_boundaries (list[list[float]]): List of lists where each sublist contains
            top and bottom depths for a layer.vvelocities (list[float]): List of layer
            velocities.

    Returns:
        list[list[float]]: List of [top_depth_m, thickness_m, velocity_m_s], where
            depths and thickness are in meters, and velocity is in m/s.

    Raises:
        ValueError: If lengths of layer boundaries and velocities don't match.

    Notes:
        Assumes layer_boundaries and velocities are ordered top-down (shallow to deep).
    """
    if len(layer_boundaries) != len(velocities):
        raise ValueError("Length of layer_boundaries must match velocities")

    model = []
    for (top_km, bottom_km), velocity_km_s in zip(layer_boundaries, velocities):
        top_m = top_km * -1000
        thickness_m = (top_km - bottom_km) * 1000
        velocity_m_s = velocity_km_s * 1000
        model.append([top_m, thickness_m, velocity_m_s])
    return model


def upward_model(
    hypo_depth_m: float, sta_elev_m: float, raw_model: list[list[float]]
) -> list[list[float]]:
    """
    Build a modified model for direct upward-refracted waves from the raw model by
    evaluating the hypo depth and the station elevation.

    Args:
        hypo_depth_m (float) : Hypocenter depth in meters (negative).
        sta_elev_m (float): Station elevation in meters (positive).
        raw_model (list[list[float]]): List of [top_m, thickness_m, velocity_m_s]

    Returns:
        list[list[float]] : A subset of the raw model, adjusted for station elevation
            and hypocenter depth, containing layers between the station elevation and
            hypocenter depth.
    """
    if hypo_depth_m >= sta_elev_m:
        raise ValueError(
            f"Hypocenter depth {hypo_depth_m} must be below station "
            f"elevation {sta_elev_m}"
        )
    # correct upper model boundary and last layer thickness
    sta_idx, hypo_idx = -1, -1
    for layer in raw_model:
        if layer[0] >= max(sta_elev_m, hypo_depth_m):
            sta_idx += 1
            hypo_idx += 1
        elif layer[0] >= hypo_depth_m:
            hypo_idx += 1
        else:
            pass

    modified_model = raw_model[sta_idx : hypo_idx + 1]
    modified_model[0][0] = sta_elev_m  # adjust top to station elevation
    if len(modified_model) > 1:
        modified_model[0][1] = (
            modified_model[1][0] - sta_elev_m
        )  # adjust first layer thickness (corrected by station elevation)
        modified_model[-1][1] = (
            hypo_depth_m - modified_model[-1][0]
        )  # adjust last layer thickness (corrected by hypo depth)
    else:
        modified_model[0][1] = hypo_depth_m - sta_elev_m
    return modified_model


def downward_model(
    hypo_depth_m: float, raw_model: list[list[float]]
) -> list[list[float]]:
    """
    Build a modified model for downward critically refracted waves from the raw model
    by evaluating the hypo depth and the station elevation.

    Args:
        hypo_depth_m (float) : Hypocenter depth in meters (negative).
        raw_model (list[list[float]]): List containing sublist where each sublist
            represents top depth, thickness, and velocity of each layer.

    Returns:
        list[list[float]] :  A subset of the raw model, containing layers from the
            hypocenter depth downward.
    """
    hypo_idx = -1
    for layer in raw_model:
        if layer[0] >= hypo_depth_m:
            hypo_idx += 1
    modified_model = raw_model[hypo_idx:]
    modified_model[0][0] = hypo_depth_m
    if len(modified_model) > 1:
        modified_model[0][1] = (
            float(modified_model[1][0]) - hypo_depth_m
        )  # adjust first layer thickness relative to the hypo depth
    return modified_model


def up_refract(
    epi_dist_m: float, up_model: list[list[float]], take_off: float | None = None
) -> tuple[dict[str, list], float]:
    """
    Calculate refracted angles, distances, and travel times for upward refracted waves.
    If take_off is provided, use it; otherwise, compute it using root-finding.

    Args:
        epi_dist_m (float): Epicentral distance in meters.
        up_model (list[list[float]]): List of [top_m, thickness_m, velocity_m_s],
            ordered top-down.
        take_off (float | None): User-specified take-off angle input in degrees;
            if None, computed via brentq.

    Returns:
        tuple[dict[str, list], float]:
            - result (dict[str, list]): A dictionary mapping take-off angles to
                {'refract_angles': [], 'distances': [], 'travel_times': []}.
            - take_off (float): The computed take-off angle (degrees) of the
                refracted-wave reaches the station.

    """
    # Convert upmodel to thickness and velocitites array
    thicknesses = np.array([layer[1] for layer in up_model[::-1]])
    velocities = np.array([layer[2] for layer in up_model[::-1]])

    def _distance_error(take_off_angle: float) -> float:
        """Compute the difference between cumulative distance and epi_dist_m."""
        angles = np.zeros(len(thicknesses))
        angles[0] = take_off_angle
        for i in range(1, len(thicknesses)):
            angles[i] = np.degrees(
                np.arcsin(
                    np.sin(np.radians(angles[i - 1]))
                    * velocities[i]
                    / velocities[i - 1]
                )
            )

        # Vectorized distance calculation
        distances = np.tan(np.radians(angles)) * np.abs(thicknesses)
        return np.sum(distances) - epi_dist_m

    # Find the take-off angle where distance_error = 0, between 0 and 90 degrees
    if take_off is None:
        try:
            take_off = brentq(_distance_error, *ANGLE_BOUNDS)
        except ValueError as e:
            raise ValueError(
                f"Failed to find take-off angle: {e}. Check velocity model and "
                f"epicentral distance"
            ) from e
    elif not 0 <= take_off < 90:
        raise ValueError("The take_off angle must be between 0 and 90 degrees.")

    # Compute full ray path (vectorized computing)
    angles = np.zeros(len(thicknesses))
    angles[0] = take_off
    for i in range(1, len(angles)):
        angles[i] = np.degrees(
            np.arcsin(
                np.sin(np.radians(angles[i - 1])) * velocities[i] / velocities[i - 1]
            )
        )

    # Vectorized distance and travel time calculation
    distances = np.tan(np.radians(angles)) * np.abs(thicknesses)
    travel_times = np.abs(thicknesses) / (np.cos(np.radians(angles)) * velocities)
    cumulative_distances = np.cumsum(distances)

    result = {
        "refract_angles": angles.tolist(),
        "distances": cumulative_distances.tolist(),
        "travel_times": travel_times.tolist(),
    }

    return {f"take_off_{take_off}": result}, take_off


def down_refract(
    epi_dist_m: float, up_model: list[list[float]], down_model: list[list[float]]
) -> tuple[dict[str, dict[str, list]], dict[str, dict[str, list]]]:
    """
    Calculate the refracted angle (relative to the normal line), the cumulative
    distance traveled, and the total travel time for all layers based on the downward
    critically refracted wave.

    Args:
        epi_dist_m (float): Epicenter distance in meters.
        up_model (list[list[float]]): List of sublist containing modified raw model
            results from the 'upward_model' function.
        down_model (list[list[float]]): List of sublist containing modified raw model
            results from the 'downward_model' function.

    Returns:
        tuple[dict[str, dict[str, list]], dict[str, dict[str, list]]]:
            - Downward segment results (dict[str, dict[str, list]]): Dict mapping
                take-off angles to {
                'refract_angles': [],
                'distances': [],
                'travel_times': []
                }.
            - Upward segment results (dict[str, dict[str, list]]): Dict for second half
                of critically refracted rays.
    Notes:
        Assumes velocity generally increases with depth for critical refraction to
        occur. Low-velocity zones are not supported.
    """
    half_dist = epi_dist_m / 2
    thicknesses = np.array([layer[1] for layer in down_model])
    velocities = np.array([layer[2] for layer in down_model])

    critical_angles = []
    if len(down_model) > 1:
        critical_angles = np.degrees(
            np.arcsin(velocities[:-1] / velocities[1:])
        ).tolist()

    take_off_angles = []
    for i, crit_angle in enumerate(critical_angles):
        angle = crit_angle
        for j in range(i, -1, -1):
            angle = np.degrees(
                np.arcsin(
                    np.sin(np.radians(angle)) * down_model[j][2] / down_model[j + 1][2]
                )
            )
        take_off_angles.append(angle)
    take_off_angles.sort()

    down_seg_result = {}
    up_seg_result = {}
    for angle in take_off_angles:
        angles = [angle]
        distances = []
        travel_times = []
        cumulative_dist = 0.0

        for i in range(len(thicknesses)):
            thickness = thicknesses[i]
            velocity = velocities[i]
            current_angle = angles[-1]

            dist = np.tan(np.radians(current_angle)) * abs(thickness)
            tt = abs(thickness) / (np.cos(np.radians(current_angle)) * velocity)
            cumulative_dist += dist

            distances.append(dist)
            travel_times.append(tt)

            if cumulative_dist > half_dist:
                break

            if i + 1 < len(thicknesses):
                sin_next = (
                    np.sin(np.radians(current_angle))
                    * velocities[i + 1]
                    / velocities[i]
                )
                if sin_next < 1:
                    angles.append(np.degrees(np.arcsin(sin_next)))
                elif sin_next == 1:
                    angles.append(90.0)
                    break
                else:
                    break

        cumulative_distances = np.cumsum(distances).tolist()
        down_data = {
            "refract_angles": angles,
            "distances": cumulative_distances,
            "travel_times": travel_times,
        }

        down_seg_result[f"take_off_{angle}"] = down_data

        if angles[-1] == 90.0:
            up_data, _ = up_refract(epi_dist_m, up_model, angle)
            up_seg_result.update(up_data)
            dist_up = up_data[f"take_off_{angle}"]["distances"][-1]
            dist_critical = epi_dist_m - (2 * cumulative_distances[-1]) - dist_up
            if dist_critical >= 0:
                tt_critical = dist_critical / velocities[len(angles) - 1]
                down_data["refract_angles"].append(90.0)
                down_data["distances"].append(dist_critical + cumulative_distances[-1])
                down_data["travel_times"].append(tt_critical)
    return down_seg_result, up_seg_result


def _extract_refracted_ray_data(ray_dict: dict, key: str) -> dict:
    """Extract ray data from dictionary by key."""
    return ray_dict.get(f"take_off_{key}", {})


def _extract_critical_refracted_data(
    ref_dict: dict,
) -> tuple[float, float, float] | tuple[None, None, None]:
    """Extract fastest critical refracted data, returning (take_off, tt, inc_angle)."""
    if not ref_dict:
        return None, None, None

    fastest_tt = min((v["total_tt"][0] for v in ref_dict.values()), default=None)
    if fastest_tt is None:
        return None, None, None

    fastest_key = next(
        (k for k, v in ref_dict.items() if v["total_tt"][0] == fastest_tt), None
    )
    if fastest_key is None:
        return None, None, None

    return (
        float(fastest_key.split("_")[-1]),
        fastest_tt,
        ref_dict[fastest_key]["incidence_angle"][0],
    )


def _compute_snr_value(
    trace: np.ndarray,
    arrival_time: UTCDateTime,
    signal_window: float = 2.0,
    noise_window: float = 2.0,
) -> float:
    """Compute signal-to-noise ratio for given trace."""
    if trace is None or len(trace.data) == 0:
        raise ValueError("Trace data is empty or None")
    if arrival_time < trace.stats.starttime or arrival_time > trace.stats.endtime:
        raise ValueError(
            f"Arrival time {arrival_time} outside trace range "
            f"({trace.stats.starttime} to {trace.stats.endtime})"
        )

    sampling_rate = trace.stats.sampling_rate
    noise_start, noise_end = arrival_time - noise_window, arrival_time
    idx1 = int((noise_start - trace.stats.starttime) * sampling_rate)
    idx2 = int((noise_end - trace.stats.starttime) * sampling_rate)
    idx1 = max(0, min(len(trace.data) - 1, idx1))
    idx2 = max(0, min(len(trace.data), idx2))
    noise = np.mean(np.abs(trace.data[idx1:idx2]))

    signal_start, signal_end = (
        arrival_time - signal_window / 2,
        arrival_time + signal_window / 2,
    )
    idx3 = int((signal_start - trace.stats.starttime) * sampling_rate)
    idx4 = int((signal_end - trace.stats.starttime) * sampling_rate)
    idx3 = max(0, min(len(trace.data) - 1, idx3))
    idx4 = max(0, min(len(trace.data), idx4))
    signal = np.mean(np.abs(trace.data[idx3:idx4]))
    return signal / noise


def _compute_phase_energy_value(
    trace: np.ndarray,
    arrival_time: UTCDateTime,
    window_before: float,
    window_after: float,
    f_min: float,
    f_max: float,
) -> float:
    """Compute the total energy from a time window of a trace."""
    if trace is None or len(trace.data) == 0:
        raise ValueError("Trace data is empty or None")
    if arrival_time < trace.stats.starttime or arrival_time > trace.stats.endtime:
        raise ValueError(
            f"Arrival time {arrival_time} outside trace range "
            f"({trace.stats.starttime} to {trace.stats.endtime})"
        )

    trace_filt = trace.copy()
    trace_filt.filter("bandpass", freqmin=f_min, freqmax=f_max, zerophase=True)
    sampling_rate = trace.stats.sampling_rate
    idx1 = int((arrival_time - trace.stats.starttime - window_before) * sampling_rate)
    idx2 = int((arrival_time - trace.stats.starttime + window_after) * sampling_rate)
    idx1 = max(0, min(len(trace.data) - 1, idx1))
    idx2 = max(0, min(len(trace.data), idx2))
    window_trace = trace_filt.data[idx1:idx2]
    return np.sum(window_trace**2)


def _handle_local_earthquake_phase_selection(
    trace_z: np.ndarray,
    p_arr_time: UTCDateTime,
    hypo_depth_m: float,
    critical_refract_tt_p: float,
    upward_refract_tt_p: float,
    dominant_period: float,
    take_off_upward_refract_p: float,
    upward_incidence_angle_p: float,
    take_off_upward_refract_s: float,
    upward_refract_tt_s: float,
    upward_incidence_angle_s: float,
    take_off_critical_p: float,
    critical_incidence_angle_p: float,
    take_off_critical_s: float,
    critical_refract_tt_s: float,
    critical_incidence_angle_s: float,
) -> tuple[float, float, float, float, float, float]:
    """Handle local earthquake phase selection with energy comparison."""
    gap = abs(critical_refract_tt_p - upward_refract_tt_p)
    threshold = 1.5 * dominant_period if hypo_depth_m > -10000 else 2 * dominant_period

    if gap < threshold:
        return (
            take_off_upward_refract_p,
            upward_refract_tt_p,
            upward_incidence_angle_p,
            take_off_upward_refract_s,
            upward_refract_tt_s,
            upward_incidence_angle_s,
        )

    arrival_time_pg, arrival_time_pn = (
        p_arr_time,
        p_arr_time + (critical_refract_tt_p - upward_refract_tt_p),
    )
    window_length = max(min(gap * 0.75, 5), 3)

    try:
        snr_pg = _compute_snr_value(
            trace_z, arrival_time_pg, window_length, 0.75 * window_length
        )
        snr_pn = _compute_snr_value(
            trace_z, arrival_time_pn, window_length, 0.75 * window_length
        )
    except (ValueError, RuntimeError):
        return (
            take_off_upward_refract_p,
            upward_refract_tt_p,
            upward_incidence_angle_p,
            take_off_upward_refract_s,
            upward_refract_tt_s,
            upward_incidence_angle_s,
        )

    if snr_pg < CONFIG.wave.SNR_THRESHOLD or snr_pn < CONFIG.wave.SNR_THRESHOLD:
        return (
            take_off_upward_refract_p,
            upward_refract_tt_p,
            upward_incidence_angle_p,
            take_off_upward_refract_s,
            upward_refract_tt_s,
            upward_incidence_angle_s,
        )

    window_before, window_after = window_length / 3, window_length / 3

    try:
        pg_energy = _compute_phase_energy_value(
            trace_z,
            arrival_time_pg,
            window_before,
            window_after,
            CONFIG.spectral.F_MIN,
            CONFIG.spectral.F_MAX,
        )
        pn_energy = _compute_phase_energy_value(
            trace_z,
            arrival_time_pn,
            window_before,
            window_after,
            CONFIG.spectral.F_MIN,
            CONFIG.spectral.F_MAX,
        )
    except ValueError:
        return (
            take_off_upward_refract_p,
            upward_refract_tt_p,
            upward_incidence_angle_p,
            take_off_upward_refract_s,
            upward_refract_tt_s,
            upward_incidence_angle_s,
        )

    if pn_energy > pg_energy:
        return (
            take_off_critical_p,
            critical_refract_tt_p,
            critical_incidence_angle_p,
            take_off_critical_s,
            critical_refract_tt_s,
            critical_incidence_angle_s,
        )

    return (
        take_off_upward_refract_p,
        upward_refract_tt_p,
        upward_incidence_angle_p,
        take_off_upward_refract_s,
        upward_refract_tt_s,
        upward_incidence_angle_s,
    )


def _compute_wave_refraction_data(
    epicentral_distance: float,
    hypo_depth_m: float,
    sta_elev_m: float,
    model: list[list],
    velocities: list,
    wave_type: str = "P",
) -> tuple[
    dict,
    dict | None,
    dict | None,
    float,
    float,
    float,
    dict,
    list[list],
    list[list],
    list[list],
]:
    """Compute refraction data (P or S waves) and extract results."""
    raw_model = build_raw_model(model, velocities)
    up_model = upward_model(hypo_depth_m, sta_elev_m, raw_model.copy())
    down_model = downward_model(hypo_depth_m, raw_model.copy())

    try:
        up_ref, final_take_off = up_refract(epicentral_distance, up_model)
    except (RuntimeError, ValueError) as e:
        wave_name = "Pg" if wave_type == "P" else "Sg"
        raise ValueError(
            f"Failed to compute upward-refracted ray ({wave_name}): {e!s}"
        ) from e

    try:
        down_ref, down_up_ref = down_refract(epicentral_distance, up_model, down_model)
    except (RuntimeError, ValueError):
        down_ref, down_up_ref = None, None

    last_ray = up_ref[f"take_off_{final_take_off}"]
    take_off_upward = 180 - last_ray["refract_angles"][0]
    upward_tt = np.sum(last_ray["travel_times"])
    upward_inc_angle = last_ray["refract_angles"][-1]

    critical_ref = {}
    if down_ref:
        for k in down_ref:
            if down_ref[k]["refract_angles"][-1] == 90:
                critical_ref[k] = {
                    "total_tt": [
                        sum(down_ref[k]["travel_times"])
                        + sum(down_up_ref[k]["travel_times"])
                    ],
                    "incidence_angle": [down_up_ref[k]["refract_angles"][-1]],
                }

    return (
        last_ray,
        down_ref,
        down_up_ref,
        take_off_upward,
        upward_tt,
        upward_inc_angle,
        critical_ref,
        raw_model,
        up_model,
        down_model,
    )


def calculate_inc_angle(
    hypo: list[float],
    station: list[float],
    model: list[list],
    velocities_p: list,
    velocities_s: list | None = None,
    source_type: str | None = None,
    trace_z: np.ndarray | None = None,
    s_p_lag_time: float | None = None,
    p_arr_time: UTCDateTime | None = None,
    generate_figure: bool = False,
    figure_path: Path | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Calculate take-off angle, travel time, and incidence angle at the station."""
    hypo_lat, hypo_lon, hypo_depth_m = hypo
    sta_lat, sta_lon, sta_elev_m = station
    epicentral_distance, _, _ = gps2dist_azimuth(hypo_lat, hypo_lon, sta_lat, sta_lon)

    # Compute P-wave refraction data
    (
        last_ray_p,
        down_ref_p,
        down_up_ref_p,
        take_off_upward_refract_p,
        upward_refract_tt_p,
        upward_incidence_angle_p,
        critical_ref_p,
        raw_model_p,
        up_model_p,
        down_model_p,
    ) = _compute_wave_refraction_data(
        epicentral_distance, hypo_depth_m, sta_elev_m, model, velocities_p, "P"
    )

    (take_off_critical_p, critical_refract_tt_p, critical_incidence_angle_p) = (
        _extract_critical_refracted_data(critical_ref_p)
    )
    if take_off_critical_p is None:
        (take_off_critical_p, critical_refract_tt_p, critical_incidence_angle_p) = (
            take_off_upward_refract_p,
            upward_refract_tt_p,
            upward_incidence_angle_p,
        )

    # Compute S-wave refraction data
    if velocities_s is None:
        velocities_s = [v / np.sqrt(3) for v in velocities_p]

    (
        _last_ray_s,
        _down_ref_s,
        _down_up_ref_s,
        take_off_upward_refract_s,
        upward_refract_tt_s,
        upward_incidence_angle_s,
        critical_ref_s,
        _raw_model_s,
        _up_model_s,
        _down_model_s,
    ) = _compute_wave_refraction_data(
        epicentral_distance, hypo_depth_m, sta_elev_m, model, velocities_s, "S"
    )

    (take_off_critical_s, critical_refract_tt_s, critical_incidence_angle_s) = (
        _extract_critical_refracted_data(critical_ref_s)
    )
    if take_off_critical_s is None:
        (take_off_critical_s, critical_refract_tt_s, critical_incidence_angle_s) = (
            take_off_upward_refract_s,
            upward_refract_tt_s,
            upward_incidence_angle_s,
        )

    # Compute S-P lag time
    t_p = min(upward_refract_tt_p, critical_refract_tt_p)
    t_s = (
        upward_refract_tt_s
        if upward_refract_tt_p <= critical_refract_tt_p
        else critical_refract_tt_s
    )
    if s_p_lag_time is None:
        s_p_lag_time = t_s - t_p if np.inf not in (t_s, t_p) else None
        if s_p_lag_time is None:
            raise ValueError("Failed to compute S-P lag time dynamically.")

    # Compute dominant period
    dominant_period = 2.0
    if trace_z and p_arr_time and p_arr_time >= 0:
        window_length = 0.75 * s_p_lag_time if s_p_lag_time else 5
        dominant_period_psd = compute_dominant_period(
            trace_z,
            p_arr_time,
            window_length,
            CONFIG.spectral.F_MIN,
            CONFIG.spectral.F_MAX,
        )
        dominant_period_sp = 0.2 * s_p_lag_time if s_p_lag_time else 2.0
        dominant_period = max(dominant_period_psd, dominant_period_sp)
        dominant_period = max(
            1 / CONFIG.spectral.F_MAX, min(CONFIG.spectral.F_MIN, dominant_period)
        )

    # Determine phase selection
    if source_type == "very_local_earthquake":
        result = (
            take_off_upward_refract_p,
            upward_refract_tt_p,
            upward_incidence_angle_p,
            take_off_upward_refract_s,
            upward_refract_tt_s,
            upward_incidence_angle_s,
        )
    else:
        if not (trace_z and p_arr_time and p_arr_time >= 0):
            raise ValueError(
                "Vertical trace and valid P arrival time required for local earthquake."
            )
        result = _handle_local_earthquake_phase_selection(
            trace_z,
            p_arr_time,
            hypo_depth_m,
            critical_refract_tt_p,
            upward_refract_tt_p,
            dominant_period,
            take_off_upward_refract_p,
            upward_incidence_angle_p,
            take_off_upward_refract_s,
            upward_refract_tt_s,
            upward_incidence_angle_s,
            take_off_critical_p,
            critical_incidence_angle_p,
            take_off_critical_s,
            critical_refract_tt_s,
            critical_incidence_angle_s,
        )

    if generate_figure:
        plot_rays(
            hypo_depth_m,
            sta_elev_m,
            epicentral_distance,
            velocities_p,
            raw_model_p,
            up_model_p,
            down_model_p,
            last_ray_p,
            critical_ref_p,
            down_ref_p,
            down_up_ref_p,
            figure_path,
        )

    return result
