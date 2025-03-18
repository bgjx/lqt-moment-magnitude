#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 19:32:03 2023.
Python code for seismic wave refraction calculations in layered medium.

Developed by Arham Zakki Edelo.
Version: 0.2.0
License: MIT

Contact:
- edelo.arham@gmail.com
- https://github.com/bgjx

Pre-requisite modules:
- [numpy, scipy, matplotlib, obspy, configparser]

This module calculates incidence angles, travel times, and ray paths for seismic waves (P-waves, S-waves)
using a layered velocity model and Snell’s Law-based shooting method, suitable for shallow borehole data
in local earthquake monitoring.

References:
- Aki, K., & Richards, P. G. (2002). Quantitative Seismology, 2nd Edition. University Science Books.

"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
from obspy.geodetics import gps2dist_azimuth
from scipy.optimize import brentq

from .plotting import plot_rays
from .config import CONFIG


# Global parameters
ANGLE_BOUNDS = (0.01, 89.99)
def build_raw_model(layer_boundaries: List[List[float]], velocities: List) -> List[List[float]]:
    """
    Build a model of layers from the given layer boundaries and velocities.

    Args:
        layer_boundaries (List[List[float]]): List of lists where each sublist contains top and bottom depths for a layer.
        velocities (List): List of layer velocities.

    Returns:
        List[List[float]]: List of [top_depth_m, thickness_m, velocity_m_s]
    
    Raises:
        ValueError: If lengths of layer boundaries and velocities don't match.
    """

    if len(layer_boundaries) != len(velocities):
        raise ValueError("Length of layer_boundaries must match velocites")
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
        List[List[float]] : List of modified model corrected by station elevation and hypocenter depth.
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
        List[List[float]] : List of modified model from hypocenter depth downward.
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
   
   
def up_refract(epi_dist_m: float, 
                up_model: List[List[float]],
                take_off: Optional[float] = None
                ) -> Tuple[Dict[str, List], float]:
    """
    Calculate refracted angles, distances, and travel times for upward refracted waves.
    If take_off is provided, use it; otherwise, compute it using root-finding.

    Args:
        epi_dist_m (float): Epicentral distance in meters.
        up_model (List[List[float]]): List of [top_m, thickness_m, velocity_m_s], ordered top-down.
        take_off (Optional[float]): User-spesified take-off angle input in degrees; if None, computed via brentq.

    Returns:
        Tuple[Dict[str, List], float]:
            - result (Dict[str, List]): A dictionary mapping take-off angles to {'refract_angles': [], 'distances': [], 'travel_times': []}.
            - take_off (float): The computed take-off angle (degrees) of the refracted-wave reaches the station.
        
    """

    # Convert upmodel to thickness and velocitites array
    thicknesses = np.array([layer[1] for layer in up_model[::-1]])
    velocities = np.array([layer[2] for layer in up_model[::-1]])

    def distance_error(take_off_angle: float) -> float:
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
        take_off = brentq(distance_error, *ANGLE_BOUNDS)
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
      
         
def down_refract(epi_dist_m: float,
                    up_model: List[List[float]],
                    down_model: List[List[float]]
                    ) -> Tuple[Dict[str, List], Dict[str, List]] :
    """
    Calculate the refracted angle (relative to the normal line), the cumulative distance traveled, 
    and the total travel time for all layers based on the downward critically refracted wave.

    Args:
        epi_dist_m (float): Epicenter distance in m.
        up_model (List[List[float]]): List of sublists containing modified raw model results from the 'upward_model' function.
        down_model (List[List[float]]): List of sublists containing modified raw model results from the 'downward_model' function.

    Returns:
        Tuple[Dict[str, List], Dict[str, List]]:
            - Downward segment results (Dict[str, List]): Dict mapping take-off angles to {'refract_angles': [], 'distances': [], 'travel_times': []}.
            - Upward segment results (Dict[str, List]): Dict for second half of critically refracted rays.
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


def calculate_inc_angle(hypo: List[float],
                        station: List[float],
                        model: List[List],
                        velocity: List, 
                        figure_statement: bool = False,
                        figure_path: Path = None
                        ) -> Tuple [float, float, float]:
    """
    Calculate the take-off angle, total travel-time and the incidence angle at the station for 
    refracted angle using Snell's shooting method.

    Args:
        hypo (List[float]): A list containing the latitude, longitude, and depth of the hypocenter (depth in negative notation).
        sta (List[float]): A list containing the latitude, longitude, and elevation of the station.
        model (List[List[float]]): List of list where each sublist contains top and bottom depths for a layer.
        velocity (List[float]): List of layer velocities.
        figure_statement (bool): Whether to generate and save figures (default is False).
        figure_path (Path): A directory to save plot figures.
        
    Returns:
        Tuple[float, float, float]: take-off angle, total travel time and incidence angle.
    """
    # initialize hypocenter, station, model, and calculate the epicentral distance
    hypo_lat,hypo_lon, hypo_depth_m = hypo
    sta_lat, sta_lon, sta_elev_m = station
    epicentral_distance, azimuth, back_azimuth = gps2dist_azimuth(hypo_lat, hypo_lon, sta_lat, sta_lon)
    
    # build raw model and modified models
    raw_model = build_raw_model(model, velocity)
    up_model = upward_model (hypo_depth_m, sta_elev_m, raw_model.copy())
    down_model = downward_model(hypo_depth_m, raw_model.copy())
    
    #  start calculating all refracted waves for all layers they may propagate through
    up_ref, final_take_off = up_refract(epicentral_distance, up_model)
    down_ref, down_up_ref = down_refract(epicentral_distance, up_model, down_model)
    
    # result from direct upward refracted wave only
    last_ray = up_ref[f"take_off_{final_take_off}"]
    take_off_upward_refract = 180 - last_ray['refract_angles'][0]
    upward_refract_tt = np.sum(last_ray['travel_times'])
    upward_incidence_angle = last_ray['refract_angles'][-1]

    critical_ref = {} # list of downward critically refracted ray (take_off_angle, total_tt, incidence_angle)
    for take_off_key in down_ref:
        if down_ref[take_off_key]["refract_angles"][-1] == 90:
            tt_down = sum(down_ref[take_off_key]['travel_times'])
            tt_up_seg = sum(down_up_ref[take_off_key]['travel_times'])
            total_tt = tt_down + tt_up_seg
            inc_angle = down_up_ref[take_off_key]["refract_angles"][-1]
            critical_ref[take_off_key] = {"total_tt": [total_tt], "incidence_angle": [inc_angle]}
    if critical_ref:
        fastest_tt = min(data["total_tt"][0] for data in critical_ref.values())
        fastest_key = next(k for k, v in critical_ref.items() if v['total_tt'][0] == fastest_tt)
        if fastest_tt < upward_refract_tt:
            take_off = float(fastest_key.split("_")[-1])
            total_tt = fastest_tt
            inc_angle = critical_ref[fastest_key]["incidence_angle"][0]
        else:
            take_off = take_off_upward_refract
            total_tt = upward_refract_tt
            inc_angle = upward_incidence_angle
    else:
        take_off = take_off_upward_refract
        total_tt = upward_refract_tt
        inc_angle = upward_incidence_angle
    
    if figure_statement:
        figure_path = figure_path or "."
        plot_rays(hypo_depth_m, sta_elev_m, velocity, raw_model, up_model, down_model, last_ray, critical_ref, down_ref, down_up_ref, epicentral_distance, figure_path)

    return take_off, total_tt, inc_angle
