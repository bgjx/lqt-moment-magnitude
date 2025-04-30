"""
Refraction module for lqt-moment-magnitude package.

Version: 0.2.1

This module calculates incidence angles, travel times, and ray paths for seismic waves (P-waves, S-waves)
using a layered 1-D velocity model and Snell’s Law-based shooting method, suitable for shallow borehole 
3-C sensor.

Dependencies:
    - See `pyproject.toml` or `pip install lqtmoment` for required packages.

References:
- Aki, K., & Richards, P. G. (2002). Quantitative Seismology, 2nd Edition. University Science Books.

"""

import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from scipy.optimize import brentq
from scipy.signal import welch

from .plotting import plot_rays
from .config import CONFIG


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
    
    freqs, psd = welch(data, fs=sampling_rate, nperseg=min(256, len(data)), scaling='density')
    mask = (freqs >= f_min) & (freqs<= f_max)
    freqs = freqs[mask]
    psd = psd[mask]
    if len(freqs) == 0:
        return 2.0
    
    f_max = freqs[np.argmax(psd)]
    return 1/f_max


def build_raw_model(layer_boundaries: List[List[float]], velocities: List) -> List[List[float]]:
    """
    Build a model of layers from the given layer boundaries and velocities.

    Args:
        layer_boundaries (List[List[float]]): List of lists where each sublist contains top and bottom depths for a layer.
        velocities (List): List of layer velocities.

    Returns:
        List[List[float]]: List of [top_depth_m, thickness_m, velocity_m_s], where depths and thickness are in meters,
                            and velocity is in m/s.
    
    Raises:
        ValueError: If lengths of layer boundaries and velocities don't match.

    Notes:
        Assumes layer_boundaries and velocities are ordered top-down (shallow to deep).
    """
    if len(layer_boundaries) != len(velocities):
        raise ValueError("Length of layer_boundaries must match velocities")
    
    model = []
    for (top_km, bottom_km), velocity_km_s in zip(layer_boundaries, velocities):
        top_m = top_km*-1000
        thickness_m = (top_km - bottom_km)* 1000
        velocity_m_s = velocity_km_s * 1000
        model.append([top_m, thickness_m, velocity_m_s])
    return model


def upward_model(hypo_depth_m: float, sta_elev_m: float, raw_model: List[List[float]]) -> List[List[float]]:
    """
    Build a modified model for direct upward-refracted waves from the raw model by evaluating 
    the hypo depth and the station elevation.

    Args:
        hypo_depth_m (float) : Hypocenter depth in meters (negative).
        sta_elev_m (float): Station elevation in meters (positive).
        raw_model (List[List[float]]): List of [top_m, thickness_m, velocity_m_s]

    Returns:
        List[List[float]] : A subset of the raw model, adjusted for station elevation and hypocenter depth,
                             containing layers between the station elevation and hypocenter depth.
    """
    if hypo_depth_m >= sta_elev_m:
        raise ValueError(f"Hypocenter depth {hypo_depth_m} must be below station elevation {sta_elev_m}")
    # correct upper model boundary and last layer thickness
    sta_idx, hypo_idx = -1, -1
    for layer in raw_model:
        if layer[0] >= max(sta_elev_m, hypo_depth_m):
            sta_idx+=1
            hypo_idx+=1
        elif layer[0] >= hypo_depth_m:
            hypo_idx+=1
        else:
            pass

    modified_model = raw_model[sta_idx:hypo_idx+1]
    modified_model[0][0] = sta_elev_m  # adjust top to station elevation
    if len(modified_model) > 1:
        modified_model[0][1] = modified_model[1][0] - sta_elev_m # adjust first layer thickness (corrected by station elevation)
        modified_model[-1][1] = hypo_depth_m - modified_model[-1][0] # adjust last layer thickness (corrected by hypo depth)
    else:
        modified_model[0][1] =  hypo_depth_m - sta_elev_m
    return modified_model
 

def downward_model(hypo_depth_m: float, raw_model: List[List[float]]) -> List[List[float]]:
    """
    Build a modified model for downward critically refracted waves from the raw model by evaluating 
    the hypo depth and the station elevation.

    Args:
        hypo_depth_m (float) : Hypocenter depth in meters (negative).
        raw_model (List[List[float]]): List containing sublist where each sublist represents top depth,
                                thickness, and velocity of each layer.

    Returns:
        List[List[float]] :  A subset of the raw model, containing layers from the hypocenter depth downward.
    """
    hypo_idx = -1
    for layer in raw_model:
        if layer[0] >= hypo_depth_m:
            hypo_idx+=1
    modified_model = raw_model[hypo_idx:]
    modified_model[0][0] = hypo_depth_m
    if len(modified_model) > 1:
        modified_model[0][1] = float(modified_model[1][0]) - hypo_depth_m # adjust first layer thickness relative to the hypo depth
    return modified_model
   
   
def up_refract(
    epi_dist_m: float, 
    up_model: List[List[float]],
    take_off: Optional[float] = None
    ) -> Tuple[Dict[str, List], float]:
    """
    Calculate refracted angles, distances, and travel times for upward refracted waves.
    If take_off is provided, use it; otherwise, compute it using root-finding.

    Args:
        epi_dist_m (float): Epicentral distance in meters.
        up_model (List[List[float]]): List of [top_m, thickness_m, velocity_m_s], ordered top-down.
        take_off (Optional[float]): User-specified take-off angle input in degrees; if None, computed via brentq.

    Returns:
        Tuple[Dict[str, List], float]:
            - result (Dict[str, List]): A dictionary mapping take-off angles to {'refract_angles': [], 'distances': [], 'travel_times': []}.
            - take_off (float): The computed take-off angle (degrees) of the refracted-wave reaches the station.
        
    """
    # Convert upmodel to thickness and velocitites array
    thicknesses = np.array([layer[1] for layer in up_model[::-1]])
    velocities = np.array([layer[2] for layer in up_model[::-1]])

    def _distance_error(take_off_angle: float) -> float:
        """ Compute the difference between cumulative distance and epi_dist_m."""
        angles = np.zeros(len(thicknesses))
        angles[0] = take_off_angle
        for i in range(1, len(thicknesses)):
            angles[i] = np.degrees(np.arcsin(np.sin(np.radians(angles[i - 1])) * velocities[i]/velocities[i-1]))

        # Vectorized distance calculation
        distances = np.tan(np.radians(angles))* np.abs(thicknesses)
        return np.sum(distances) - epi_dist_m

    # Find the take-off angle where distance_error = 0, between 0 and 90 degrees
    if take_off is None:
        try:
            take_off = brentq(_distance_error, *ANGLE_BOUNDS)
        except ValueError as e:
            raise ValueError("Failed to find take-off angle: {e}. Check velocity model and epicentral distance")
    else:
        if not 0 <= take_off < 90:
            raise ValueError("The take_off angle must be between 0 and 90 degrees.")
    
    # Compute full ray path (vectorized computing)
    angles = np.zeros(len(thicknesses))
    angles[0] = take_off
    for i in range(1, len(angles)):
        angles[i] = np.degrees(np.arcsin(np.sin(np.radians(angles[i - 1])) * velocities[i]/velocities[i-1]))
    
    # Vectorized distance and travel time calculation
    distances = np.tan(np.radians(angles)) * np.abs(thicknesses)
    travel_times = np.abs(thicknesses)/(np.cos(np.radians(angles))*velocities)
    cumulative_distances = np.cumsum(distances)

    result = {
        "refract_angles": angles.tolist(),
        "distances": cumulative_distances.tolist(),
        "travel_times": travel_times.tolist(),
    }

    return {f"take_off_{take_off}": result}, take_off
      
         
def down_refract(
    epi_dist_m: float,
    up_model: List[List[float]],
    down_model: List[List[float]]
    ) -> Tuple[Dict[str, Dict[str, List]], Dict[str, Dict[str, List]]] :
    """
    Calculate the refracted angle (relative to the normal line), the cumulative distance traveled, 
    and the total travel time for all layers based on the downward critically refracted wave.

    Args:
        epi_dist_m (float): Epicenter distance in meters.
        up_model (List[List[float]]): List of sublist containing modified raw model results from the 'upward_model' function.
        down_model (List[List[float]]): List of sublist containing modified raw model results from the 'downward_model' function.

    Returns:
        Tuple[Dict[str, List], Dict[str, List]]:
            - Downward segment results (Dict[str, List]): Dict mapping take-off angles to {'refract_angles': [], 'distances': [], 'travel_times': []}.
            - Upward segment results (Dict[str, List]): Dict for second half of critically refracted rays.
    Notes:
        Assumes velocity generally increases with depth for critical refraction to occur. Low-velocity zones are not supported.
    """
    half_dist = epi_dist_m/2
    thicknesses = np.array([layer[1] for layer in down_model])
    velocities = np.array([layer[2] for layer in down_model])

    critical_angles = []
    if len(down_model) > 1:
        critical_angles = np.degrees(np.arcsin(velocities[:-1]/velocities[1:])).tolist()
    
    take_off_angles=[]
    for i, crit_angle in enumerate(critical_angles):
        angle = crit_angle
        for j in range(i, -1, -1) :
            angle =  np.degrees(np.arcsin(np.sin(np.radians(angle))*down_model[j][2]/down_model[j+1][2]))
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

            dist = np.tan(np.radians(current_angle))*abs(thickness)
            tt = abs(thickness) / (np.cos(np.radians(current_angle))*velocity)
            cumulative_dist += dist

            distances.append(dist)
            travel_times.append(tt)

            if cumulative_dist > half_dist:
                break
            
            if i + 1 < len(thicknesses):
                sin_next = np.sin(np.radians(current_angle)) * velocities[i+1] / velocities[i]
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
            "travel_times": travel_times
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
    return  down_seg_result, up_seg_result


def calculate_inc_angle(
    hypo: List[float],
    station: List[float],
    model: List[List],
    velocities_p: List,
    velocities_s: Optional[List] = None,
    source_type: Optional[str] = None,
    trace_z: Optional[np.ndarray] = None,
    s_p_lag_time: Optional[float] = None,
    p_arr_time: Optional[UTCDateTime] = None,
    s_arr_time: Optional[UTCDateTime] = None,
    generate_figure: bool = False,
    figure_path: Optional[Path] = None
    ) -> Tuple [float, float, float, float, float, float]:
    """
    Calculate the take-off angle, total travel-time and the incidence angle at the station for 
    refracted angle using Snell's shooting method.

    Args:
        hypo (List[float]): A list containing the latitude, longitude, and depth of the hypocenter (depth in negative notation).
        station (List[float]): A list containing the latitude, longitude, and elevation of the station.
        model (List[List[float]]): List of list where each sublist contains top and bottom depths for a layer.
        velocities_p (List[float]): List of P-wave velocities.
        velocities_s (Optional[List]): List of S-wave velocities, optional only used fos S incidence angle calculation
                                        default to None.
        source_type (Optional[str]): Earthquake type to determine the calculation method, optional, default to None.
        trace_z (Optional[np.ndarray]): Vertical trace data, optional for energy comparison, default to None.
        s_p_lag_time (Optional[float]): Pre-calculated s-p lag time (seconds). If provided, 
                                        skips s-wave travel time calculation.
        p_arr_time (Optional[UTCDateTime]): Arrival time in UTCDateTime of the P phase, optional, default to None.
        s_arr_time (Optional[UTCDateTime]): Arrival time in UTCDateTime of the S phase, optional, default to None.
        generate_figure (bool): Whether to generate and save figures, default to False
        figure_path (Optional[Path]): A directory to save plot figures, optional, default to None
        
    Returns:
        Tuple[float, float, ...]: take-off angle P-wave, total travel time P-wave, incidence angle P-wave, 
                                    take-off angle S-wave, total travel time S-wave, incidence angle S-wave.
    
    Notes:
        The function compares direct upward-refracted and critically refracted ray paths,
        selecting the fastest path to determine the final take-off angle, travel time, and incidence angle.
    """

    # initialize hypocenter, station, model, and calculate the epicentral distance
    hypo_lat,hypo_lon, hypo_depth_m = hypo
    sta_lat, sta_lon, sta_elev_m = station
    epicentral_distance, azimuth, back_azimuth = gps2dist_azimuth(hypo_lat, hypo_lon, sta_lat, sta_lon)
    
    # build raw model and modified models for P-wave
    raw_model_p = build_raw_model(model, velocities_p)
    up_model_p = upward_model (hypo_depth_m, sta_elev_m, raw_model_p.copy())
    down_model_p = downward_model(hypo_depth_m, raw_model_p.copy())
    
    #  start calculating all refracted waves for all layers they may propagate through
    try:
        up_ref_p, final_take_off_p = up_refract(epicentral_distance, up_model_p)
    except (RuntimeError, ValueError) as e:
        raise ValueError(f"Failed to compute upward-refracted ray (Pg): {str(e)}")

    try:
        down_ref_p, down_up_ref_p = down_refract(epicentral_distance, up_model_p, down_model_p)
    except (RuntimeError, ValueError) as e:
        down_ref_p, down_up_ref_p = None, None
    
    # Result from direct upward-refracted P-wave (Pg)
    if up_ref_p is not None:
        last_ray_p = up_ref_p[f"take_off_{final_take_off_p}"]
        take_off_upward_refract_p = 180 - last_ray_p['refract_angles'][0]
        upward_refract_tt_p = np.sum(last_ray_p['travel_times'])
        upward_incidence_angle_p = last_ray_p['refract_angles'][-1]
    else:
        raise ValueError("Upward-refracted ray (Pg) computation failed.")

    # Result from critically refracted P-wave (Pn)
    critical_ref_p = {} # list of downward critically refracted ray (take_off_angle, total_tt, incidence_angle)
    if down_ref_p is not None:
        for take_off_key in down_ref_p:
            if down_ref_p[take_off_key]["refract_angles"][-1] == 90:
                tt_down = sum(down_ref_p[take_off_key]['travel_times'])
                tt_up_seg = sum(down_up_ref_p[take_off_key]['travel_times'])
                total_tt = tt_down + tt_up_seg
                inc_angle = down_up_ref_p[take_off_key]["refract_angles"][-1]
                critical_ref_p[take_off_key] = {"total_tt": [total_tt], "incidence_angle": [inc_angle]}
    if critical_ref_p:
        fastest_tt = min(data["total_tt"][0] for data in critical_ref_p.values())
        fastest_key = next(k for k, v in critical_ref_p.items() if v['total_tt'][0] == fastest_tt)
        take_off_critical_refract_p = float(fastest_key.split("_")[-1])
        critical_refract_tt_p = fastest_tt
        critical_incidence_angle_p = critical_ref_p[fastest_key]["incidence_angle"][0]
    else:
        take_off_critical_refract_p = take_off_upward_refract_p
        critical_refract_tt_p = upward_refract_tt_p
        critical_incidence_angle_p = upward_incidence_angle_p
        
    # Compute S-wave travel time and incidence angles (needed for S-wave incidence angle)
    if velocities_s is None:
        velocities_s = [v / np.sqrt(3) for v in velocities_p]
    
    raw_model_s = build_raw_model(model, velocities_s)
    up_model_s = upward_model(hypo_depth_m, sta_elev_m, raw_model_s.copy())
    down_model_s = downward_model(hypo_depth_m, raw_model_s.copy())

    # Compute Sg (direct upward-refracted S-wave)
    try:
        up_ref_s, final_take_off_s = up_refract(epicentral_distance, up_model_s)
    except (RuntimeError, ValueError) as e:
        raise ValueError(f"Failed to compute upward-refracted S-wave (Sg): {str(e)}")
    
    if up_ref_s is not None:
        last_ray_s = up_ref_s[f"take_off_{final_take_off_s}"]
        take_off_upward_refract_s = 180 - last_ray_s['refract_angles'][0]
        upward_refract_tt_s = np.sum(last_ray_s['travel_times'])
        upward_incidence_angle_s = last_ray_s['refract_angles'][-1]
    else:
        raise ValueError("Upward-refracted S-wave (sg) computation failed.")
    
    # Compute Sn (critically refracted S-wave)
    try:
        down_ref_s, down_up_ref_s = down_refract(epicentral_distance, up_model_s, down_model_s)
    except(RuntimeError, ValueError) as e:
        down_ref_s, down_up_ref_s = None, None
    
    critical_ref_s = {}
    if down_ref_s is not None:
        for take_off_key in down_ref_s:
            if down_ref_s[take_off_key]["refract_angles"][-1] == 90:
                tt_down = sum(down_ref_s[take_off_key]['travel_times'])
                tt_up_seg = sum(down_up_ref_s[take_off_key]['travel_times'])
                total_tt = tt_down + tt_up_seg
                inc_angle = down_up_ref_s[take_off_key]["refract_angles"][-1]
                critical_ref_s[take_off_key] = {"total_tt": [total_tt], "incidence_angle": [inc_angle]}
    
    if critical_ref_s:
        fastest_tt_s = min(data["total_tt"][0] for data in critical_ref_s.values())
        fastest_key_s = next(k for k, v in critical_ref_s.items() if v["total_tt"][0] == fastest_tt_s)
        take_off_critical_refract_s = float(fastest_key.split("_")[-1])
        critical_refract_tt_s = fastest_tt_s
        critical_incidence_angle_s = critical_ref_s[fastest_key_s]["incidence_angle"][0]
    else:
        take_off_critical_refract_s = take_off_upward_refract_s
        critical_refract_tt_s = upward_refract_tt_s
        critical_incidence_angle_s = upward_incidence_angle_s
    
    # Determine the P and S phases for S-P lag time calculation
    t_p = min(upward_refract_tt_p, critical_refract_tt_p)
    p_phase = "Pg" if upward_refract_tt_p <= critical_refract_tt_p else "Pn"
    if p_phase == "Pg":
        t_s = upward_refract_tt_s
        s_phase = "Sg"
    else:
        t_s = critical_refract_tt_s
        s_phase = "Sn"
    
    # Compute S-P lag time dynamically (only if not provided)
    if s_p_lag_time is None:
        s_p_lag_time = t_s - t_p if t_s != np.inf and t_p != np.inf else None
        if s_p_lag_time is None:
            raise ValueError("Failed to compute S-P lag time dynamically.")

    # Compute dominant period
    dominant_period = 2.0 # default fallback
    if trace_z is not None and p_arr_time is not None and p_arr_time >=0 :
        window_length = 0.75*s_p_lag_time if s_p_lag_time is not None else 5
        dominant_period_psd = compute_dominant_period(trace_z, p_arr_time, window_length, CONFIG.spectral.F_MIN, CONFIG.spectral.F_MAX)
        dominant_period_sp = 0.2 * s_p_lag_time if s_p_lag_time is not None else 2.0
        dominant_period = max(dominant_period_psd, dominant_period_sp)
        dominant_period = max(1/CONFIG.spectral.F_MAX, min(CONFIG.spectral.F_MIN, dominant_period))

    # For very local earthquake we directly use Pg
    if source_type == "very_local_earthquake":
        take_off_p = take_off_upward_refract_p
        total_tt_p = upward_refract_tt_p
        inc_angle_p = upward_incidence_angle_p
        take_off_s = take_off_upward_refract_s
        total_tt_s = upward_refract_tt_s
        inc_angle_s = upward_incidence_angle_s
        logger.info(f"Very local earthquake, at distance {epicentral_distance/1000:.1f} km, using Pg, "
              f"P-Incidence Angle: {inc_angle_p:.2f}°, and Sg, S-Incidence Angle: {inc_angle_s:.2f}°")
    
    # For local earthquake, we perform the energy comparison with Pg as fallback if requirements unsatisfied
    else:
        if trace_z is None:
            raise ValueError("Vertical trace is required for local earthquake energy comparison.")
        if p_arr_time is None or p_arr_time < 0:
            raise ValueError("Valid P arrival time is required for local earthquake energy comparison.")
        if not critical_ref_p or not up_ref_p:
            raise ValueError("Both Pg and Pn ray paths are required for local earthquake comparison.")
        
        gap = abs(critical_refract_tt_p - upward_refract_tt_p)
        threshold = 1.5 * dominant_period if hypo_depth_m > -10000 else 2 * dominant_period
        if gap < threshold:
            # Default to Pg when the gap is too small
            take_off_p = take_off_upward_refract_p
            total_tt_p = upward_refract_tt_p
            inc_angle_p = upward_incidence_angle_p
            take_off_s = take_off_upward_refract_s
            total_tt_s = upward_refract_tt_s
            inc_angle_s = upward_incidence_angle_s
            logger.info(f"Local earthquake, with Gap ({gap:.2f}) < {threshold}s at distance {epicentral_distance}Km, defaulting to Pg."
                        f"P-Incidence Angle: {inc_angle_p:.2f}°, and Sg, S-Incidence Angle: {inc_angle_s:.2f}°")
        else:
            def _compute_snr(
                trace: np.ndarray, 
                arrival_time: UTCDateTime,
                signal_window: float = 2.0,
                noise_window: float = 2.0
                ) -> float:
                """
                Computes signal-to-noise ratio for given trace.

                Args:
                    trace (np.ndarray): Array of trace data.
                    arrival_time (UTCDateTime): Arrival time of the phase.
                    signal_window (float): The width of signal window used for 
                                            calculating the SNR.
                    noise_window (float): The width of nose window used for
                                            calculating the SNR.
                
                Returns:
                    float: Signal to Noise ratio of given trace.
                
                Raises:
                    ValueError: If the trace or arrival time is invalid.
                """
                if trace is None or len(trace.data) == 0:
                    raise ValueError("Trace data is empty or None")
                if arrival_time < trace.stats.starttime or arrival_time > trace.stats.endtime:
                    raise ValueError(f"Arrival time {arrival_time} is outside trace time range"
                                     f"({trace.stats.starttime} to {trace.stats.endtime})")
                
                sampling_rate = trace.stats.sampling_rate
                noise_start = arrival_time - noise_window
                noise_end = arrival_time
                idx1 = int((noise_start - trace.stats.starttime) * sampling_rate)
                idx2 = int((noise_end - trace.stats.starttime) * sampling_rate)
                idx1 = max(0, min(len(trace.data) - 1, idx1))
                idx2 = max(0, min(len(trace.data), idx2))
                noise = np.mean(np.abs(trace.data[idx1:idx2])) 

                signal_start = arrival_time - signal_window / 2
                signal_end = arrival_time + signal_window / 2
                idx3 = int((signal_start - trace.stats.starttime) * sampling_rate)
                idx4 = int((signal_end - trace.stats.starttime) * sampling_rate)
                idx3 = max(0, min(len(trace.data) - 1, idx3))
                idx4 = max(0, min(len(trace.data), idx4))
                signal = np.mean(np.abs(trace.data[idx3:idx4]))
                return signal/noise

            # Compute arrival times for Pg and Pn(based on ray path simulation)
            arrival_time_pg = p_arr_time
            arrival_time_pn = p_arr_time + (critical_refract_tt_p - upward_refract_tt_p) 
            
            # Compute SNR for Pg and Pn (SNR need to be good enough for energy comparison)
            try:
                snr_pg = _compute_snr(trace_z, arrival_time_pg, window_length, 0.75*window_length)
                snr_pn = _compute_snr(trace_z, arrival_time_pn, window_length, 0.75*window_length)
            except (ValueError, RuntimeError) as e:
                logger.info(f"Local earthquake, SNR computation failed at distance {epicentral_distance/1000:.1f} km ({str(e)}),"
                            f"defaulting to Pg, P-Incidence Angle: {upward_incidence_angle_p:.2f}°, S-Incidence Angle: {upward_incidence_angle_s:.2f}°")
                take_off_p = take_off_upward_refract_p
                total_tt_p = upward_refract_tt_p
                inc_angle_p = upward_incidence_angle_p
                take_off_s = take_off_upward_refract_s
                total_tt_s = upward_refract_tt_s
                inc_angle_s = upward_incidence_angle_s
                if generate_figure:
                    plot_rays(hypo_depth_m, sta_elev_m, epicentral_distance, velocities_p, raw_model_p,
                              up_model_p, down_model_p, last_ray_p, critical_ref_p, down_ref_p, down_up_ref_p,
                              figure_path)
                return take_off_p, total_tt_p, inc_angle_p, take_off_s, total_tt_s, inc_angle_s

            if snr_pg < CONFIG.wave.SNR_THRESHOLD or snr_pn < CONFIG.wave.SNR_THRESHOLD:
                # Default to Pg when SNR is too low
                take_off_p = take_off_upward_refract_p
                total_tt_p = upward_refract_tt_p
                inc_angle_p = upward_incidence_angle_p
                take_off_s = take_off_upward_refract_s
                total_tt_s = upward_refract_tt_s
                inc_angle_s = upward_incidence_angle_s
                logger.info(f"Local earthquake, SNR too low (Pn: {snr_pn:.2f}, Pg: {snr_pg:.2f}) at distance {epicentral_distance/1000:.1f} km, "
                      f"defaulting to Pg, P-Incidence Angle: {inc_angle_p:.2f}°, S-Incidence Angle: {inc_angle_s:.2f}°")
            else:
                def _compute_phase_energy(
                    trace: np.ndarray,
                    arrival_time: UTCDateTime,
                    window_before: float,
                    window_after: float,
                    f_min: float,
                    f_max: float
                    ) -> float:

                    """
                    Compute the total energy from give window of a trace.

                    Args:
                        trace (np.ndarray): Array of trace data.
                        arrival_time (UTCDateTime): Arrival time of the phase.
                        window_before (float): A padding in second before the phase.
                        window_after (float): A padding in second after the phase.
                        f_min: Low corner frequecny for filtering the trace.
                        f_max: High corner frequency for filtering the trace.
                    
                    Return:
                        float: Energy of the trace.
                    
                    Raises:
                        ValueError: If the trace or arrival time is invalid.
                    """
                    if trace is None or len(trace.data) == 0:
                        raise ValueError("Trace data is empty or None")
                    if arrival_time < trace.stats.starttime or arrival_time > trace.stats.endtime:
                        raise ValueError(f"Arrival time {arrival_time} is outside trace time range "
                                         f"({trace.stats.starttime} to {trace.stats.endtime})")
                    trace_filt = trace.copy()
                    trace_filt.filter("bandpass", freqmin=f_min, freqmax=f_max, zerophase=True)
                    sampling_rate = trace.stats.sampling_rate
                    idx1 = int((arrival_time - trace.stats.starttime - window_before) * sampling_rate)
                    idx2 = int((arrival_time - trace.stats.starttime + window_after) * sampling_rate)
                    idx1 = max(0, min(len(trace.data) - 1, idx1))
                    idx2 = max(0, min(len(trace.data), idx2))
                    window_trace = trace_filt.data[idx1:idx2]
                    return np.sum(window_trace**2)
                
                window_length = max(min(gap * 0.75, 5), 3)
                window_before = window_length / 3
                window_after = window_length / 3

                try:
                    pg_energy = _compute_phase_energy(
                        trace_z, arrival_time_pg, window_before,
                        window_after, CONFIG.spectral.F_MIN,
                        CONFIG.spectral.F_MAX
                    )
                    pn_energy = _compute_phase_energy(
                        trace_z, arrival_time_pn, window_before,
                        window_after, CONFIG.spectral.F_MIN,
                        CONFIG.spectral.F_MAX
                    )
                except ValueError as e:
                    logger.info(f"Local earthquake, Energy computation failed at distance {epicentral_distance/1000:.1f} km ({str(e)}), "
                          f"defaulting to Pg, P-Incidence Angle: {upward_incidence_angle_p:.2f}°, S-Incidence Angle: {upward_incidence_angle_s:.2f}°")
                    take_off_p = take_off_upward_refract_p
                    total_tt_p = upward_refract_tt_p
                    inc_angle_p = upward_incidence_angle_p
                    take_off_s = take_off_upward_refract_s
                    total_tt_s = upward_refract_tt_s
                    inc_angle_s = upward_incidence_angle_s
                    if generate_figure:
                        plot_rays(hypo_depth_m, sta_elev_m, epicentral_distance, velocities_p, raw_model_p, up_model_p, down_model_p,
                                  last_ray_p, critical_ref_p, down_ref_p, down_up_ref_p, figure_path)
                    return take_off_p, total_tt_p, inc_angle_p, take_off_s, total_tt_s, inc_angle_s
                
                if pn_energy > pg_energy:
                    take_off_p = take_off_critical_refract_p
                    total_tt_p = critical_refract_tt_p
                    inc_angle_p = critical_incidence_angle_p
                    take_off_s = take_off_critical_refract_s
                    total_tt_s = critical_refract_tt_s
                    inc_angle_s = critical_incidence_angle_s
                    logger.info(f"Local earthquake, at distance {epicentral_distance/1000:.1f} km ({str(e)}), After energy comparison,"
                          f" Pn energy greater than Pg, use Pn, P-Incidence Angle: {inc_angle_p:.2f}°, and Sn, S-Incidence Angle: {inc_angle_s:.2f}°")
                else:
                    take_off_p = take_off_upward_refract_p
                    total_tt_p = upward_refract_tt_p
                    inc_angle_p = upward_incidence_angle_p
                    take_off_s = take_off_upward_refract_s
                    total_tt_s = upward_refract_tt_s
                    inc_angle_s = upward_incidence_angle_s  
                    logger.info(f"Local earthquake, at distance {epicentral_distance/1000:.1f} km ({str(e)}), After energy comparison,"
                          f" Pg energy greater than Pn, use Pg, P-Incidence Angle: {inc_angle_p:.2f}°, and Sg, S-Incidence Angle: {inc_angle_s:.2f}°")
    if generate_figure:
        plot_rays(hypo_depth_m, sta_elev_m, epicentral_distance,
                  velocities_p, raw_model_p, up_model_p, down_model_p,
                  last_ray_p, critical_ref_p, down_ref_p, down_up_ref_p,
                  figure_path)
    return take_off_p, total_tt_p, inc_angle_p, take_off_s, total_tt_s, inc_angle_s