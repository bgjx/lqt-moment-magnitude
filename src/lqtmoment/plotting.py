"""
Plotting module for lqt-moment-magnitude package.

Version: 0.2.1

This module provides internal modular function for plotting purpose in lqt-moment-magnitude package.
Users can generate spectral fitting and 1-D refraction figures from the earthquake during calculation

Dependencies:
    - See `pyproject.toml` or `pip install lqtmoment` for required packages.
"""

import logging
from obspy import Stream
from typing import List, Dict, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from .config import CONFIG


logger = logging.getLogger("lqtmoment")

def plot_spectral_fitting(
        event_id: int,
        streams: List[Stream],
        p_arr_times: List[float],
        s_arr_times: List[float],
        freqs: Dict[str, List],
        specs: Dict[str, List],
        fits: Dict[str, list],
        stations: List[str],
        figure_path: Path):
    """
    Plot phase windows and spectral fitting profiles for all stations in an event.

    Args:
        event_id (int): Unique identifier for the earthquake event.
        streams (Stream): A stream object containing the seismic data.
        p_arr_times (List[float]): List of all P arrival time for an event for each station.
        s_arr_times (List[float]): List of all S arrival time for an event for each station.
        freqs (Dict[str,List]): A dictionary of frequency arrays for P, SV, SH, and noise per station.
        specs (Dict[str, List]): A dictionary of spectral arrays for P, SV, SH, and noise per station.
        stations (List[str]): List of station names.
        figure_path(Path): Directory to save the plot.

    """
    # Initiate plotting dimension
    num_stations = len(streams)
    fig, axs = plt.subplots(num_stations*3, 2, figsize=(30, num_stations*15), squeeze=False)
    plt.subplots_adjust(hspace=0.5)
    axs[0,0].set_title("Phase Window", fontsize=20)
    axs[0,1].set_title("Spectra Fitting Profile", fontsize=20)

    for station_idx, (stream, p_time, s_time, station) in  enumerate(zip(streams, p_arr_times, s_arr_times, stations)):
        # Dinamic window parameter
        counter = station_idx*3
        s_p_time = float(s_time - p_time)    
        time_after_pick_p = 0.80 * s_p_time
        time_after_pick_s = 1.75 * s_p_time

        for comp, label in zip(["L", "Q", "T"], ["P", "SV", "SH"]):
            trace = stream.select(component=comp)[0]
            start_time = trace.stats.starttime
            trace.trim(start_time+(p_time - start_time) - 2.0, start_time+(s_time - start_time)+6.0)

            ax = axs[counter, 0]
            ax.plot(trace.times(), trace.data, "k")
            ax.axvline(p_time - trace.stats.starttime, color='r', linestyle='-', label='P arrival')
            ax.axvline(s_time - trace.stats.starttime, color='b', linestyle='-', label='S arrival')
            ax.axvline((p_time if comp == "L" else s_time) - CONFIG.wave.PADDING_BEFORE_ARRIVAL - trace.stats.starttime, color='g', linestyle='--')
            ax.axvline((p_time if comp == "L" else s_time) + (time_after_pick_p if comp == "L" else time_after_pick_s) - trace.stats.starttime, color='g', linestyle='--', label=f"{label} phase window")
            ax.axvline(p_time - CONFIG.wave.NOISE_DURATION - trace.stats.starttime, color='gray', linestyle='--')
            ax.axvline(p_time - CONFIG.wave.NOISE_PADDING - trace.stats.starttime, color='gray', linestyle='--', label='Noise Window')
            ax.set_title(f"{station}_BH{comp}", loc='right', va='center')
            ax.legend()
            ax.set_xlabel("Relative Time (s)")
            ax.set_ylabel("Amp (m)")

            ax = axs[counter, 1]
            ax.loglog(freqs[label][station_idx], specs[label][station_idx], "k", label=f"{label} spectral")
            ax.loglog(freqs[f"N_{label}"][station_idx], specs[f"N_{label}"][station_idx], "gray", label="Noise Spectra")
            ax.loglog(fits[label][station_idx][4], fits[label][station_idx][5], "b-", label=f"Fitted {label} Spectra")
            ax.set_title(f"{station}_BH{comp}", loc="right")
            ax.legend()
            ax.set_xlabel("Frequencies (Hz)")
            ax.set_ylabel("Amp (nms)")

            counter += 1
    
    fig.suptitle(f"Event {event_id} Spectral Fitting Profile", fontsize=24, fontweight='bold')
    plt.savefig(figure_path/f"event_{event_id}.png")
    plt.close(fig)


def plot_rays (hypo_depth_m: float, 
                sta_elev_m: float,
                epi_dist_m: float,
                velocity: List, 
                base_model: List[List[float]],
                up_model: List[List[float]],
                down_model: List[List[float]],
                reached_up_ref: Optional[Dict[str, List]] = None,
                critical_ref: Optional[Dict[str, List]] = None,
                down_ref: Optional[Dict[str, List]] = None,
                down_up_ref: Optional[Dict[str, List]] = None,
                figure_path: Optional[Path] = None
                ) -> None:
    """
    Plot the raw/base model, hypocenter, station, and the relative distance between the hypocenter and station
    and also plot all waves that reach the station.

    Args:
        hypo_depth_m (float) : Hypocenter depth in meters (negative).
        sta_elev_m (float): Elevation of station in meters.
        epi_dist_m (float): Epicenter distance in meters.
        velocity: List of velocities. 
        base_model (List[List[float]]): List of [top_m, thickness_m, velocity_m_s]
        up_model (List[List]): List of [top_m, thickness_m, velocity_m_s] from the 'upward_model' function.
        down_model (List[List]): List of [top_m, thickness_m, velocity_m_s] from the 'downward_model' function.
        reached_up_ref (Optional[Dict[str, List]]): A dictionary of 
                                                    {'refract_angles': [], 'distances': [], 'travel_times': []} 
                                                    from all direct upward refracted waves that reach the station.
        critical_ref (Optional[Dict[str, List]]): A dictionary of
                                                    {'refract_angles': [], 'distances': [], 'travel_times': []} 
                                                    from all critically refracted waves.
        down_ref (Optional[Dict[str, List]]): A dictionary of 
                                                {'refract_angles': [], 'distances': [], 'travel_times': []}
                                                from all downward segments of critically refracted waves.
        down_up_ref (Optional[Dict[str, List]]): A dictionary of
                                                {'refract_angles': [], 'distances': [], 'travel_times': []}
                                                from all upward segments of downward critically refracted waves.
        
        figure_path(Optional[Path]): Directory to save the plot. Defaults to None.
    """
    
    fig, axs = plt.subplots(figsize=(10,8))
    
    # Define colormaps and normalization
    cmap = cm.Oranges
    norm = mcolors.Normalize(vmin=min(velocity), vmax=max(velocity))
    
    max_width = epi_dist_m + 2000
    for layer in base_model:
        color = cmap(norm(layer[2]))
        rect = patches.Rectangle((-1000, layer[0]), max_width, layer[1], linewidth=1, edgecolor= 'black', facecolor = color)
        axs.add_patch(rect)
    
    # Plot only the last ray of direct upward wave that reaches the station
    if reached_up_ref:
        x, y = 0, hypo_depth_m
        for dist, layer in zip(reached_up_ref['distances'], reversed(up_model)):
            x_next = dist
            y_next = layer[0]
            axs.plot([x, x_next], [y, y_next], 'k')
            x, y = x_next, y_next

    if critical_ref:
        for take_off in critical_ref:
            x, y = 0, hypo_depth_m
            for i , (dist, angle) in enumerate(zip(down_ref[take_off]['distances'], down_ref[take_off]['refract_angles'])):
                x_next = dist
                y_next = down_model[i][0] if i == 0 else down_model[i - 1][0] + down_model[i - 1][1]
                axs.plot([x, x_next], [y,y_next], 'b')
                x, y = x_next, y_next
                if angle == 90:
                    for j, dist_up in enumerate(down_up_ref[take_off]['distances']):
                        x_next = x + dist_up
                        y_next = up_model[-j - 1][0]
                        axs.plot([x,x_next], [y, y_next], 'b')
                        x, y = x_next, y_next
                    break

    axs.plot(epi_dist_m, sta_elev_m, marker = 'v', color = 'black', markersize = 15, label='Station')
    axs.plot(0, hypo_depth_m, marker = '*', color = 'red', markersize = 12)
    axs.set_xlim(-2000, max_width)
    axs.set_ylim((hypo_depth_m-5000), (sta_elev_m+1000))
    axs.set_ylabel('Depth (m)')
    axs.set_xlabel('Distance (m)')
    axs.set_title("Seismic Ray Paths (Snell's Shooting Method)")
    axs.legend()
    if figure_path is None:
        figure_path = '.'
    plt.savefig(f"{figure_path}/ray_path_event.png")
    plt.close(fig)