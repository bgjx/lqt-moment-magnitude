#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Dec 15 19:32:03 2022.
Python code for spectral fitting of seismic data using Abercrombie/Boatwright model.

Developed by Arham Zakki Edelo.
Version: 0.2.0
License: MIT

Contact:
- edelo.arham@gmail.com
- https://github.com/bgjx

Pre-requisite modules:
- [numpy, scipy, scikit-optimize]

This module fits seismic displacement spectra to estimate Omega_0, corner frequency, and quality factor
using Quasi-Monte Carlo (QMC) sampling, Bayesian optimization, and grid search, based on Abercrombie
(1995) and Boatwright (1980) models for volcanic geothermal systems.

References:
- Abercrombie, R. E. (1995). Earthquake locations using single-station deep borehole recordings:
  Implications for microseismicity on the San Andreas fault in southern California. JGR, 100, 24003–24013.
- Boatwright, J. (1980). A spectral theory for circular seismic sources. BSSA, 70(1).

"""

from typing import Dict, Tuple, List, Optional
import logging
import numpy as np
from skopt import gp_minimize
from skopt.space import Real , Integer
from skopt.utils import use_named_args
from scipy.stats import qmc
from scipy import optimize
from .config import CONFIG

logger = logging.getLogger("mw_calculator")


def window_band(frequencies: np.ndarray, spectrums: np.ndarray, f_min: float, f_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts a subset of the frequency spectrum within a specified frequency band.

    Args:
        frequencies (np.ndarray): Array of frequency values.
        spectrums (np.ndarray): Array of spectral values corresponding to the frequencies.
        f_min (float): Minimum frequency of the band to extract (inclusive).
        f_max (float): Maximum frequency of the band to extract (inclusive).

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - freq: Array of frequencies within the specified band.
            - spec: Array of spectral values corresponding to the extracted frequencies.
    
    Raises:
        ValueError: If arrays are mismatched, emtpy, or contain Nan/inf, or if f_min >= fmax.
    """
    
    if not (frequencies.size and spectrums.size):
        raise ValueError("Frequencies and spectrums must be non-empty")
    if frequencies.shape != spectrums.shape or not np.isfinite(frequencies).all() or not np.isfinite(spectrums).all():
        raise ValueError("Mismatched or invalid frequency and spectrum arrays")
    if f_min >= f_max:
        raise ValueError("f_min must be lest than f_max")

    indices = np.where((frequencies >= f_min) & (frequencies <= f_max))
    return frequencies[indices], spectrums[indices]


def calculate_source_spectrum(
    frequencies: np.ndarray,
    omega_0: float,
    q_factor: float,
    corner_frequency: float,
    traveltime: float
    ) -> np.ndarray:
    
    """
    Calculate theoretical source spectrum using Abercrombie (1995) and Boatwright (1980) model.
    
    Args:
        frequencies (np.ndarray): Array of frequency values.
        omega_0 (float): Value of the omega zero (spectral flat level).
        q_factor (float): Quality factor for amplitude attenuation.
        corner_frequency (float): The corner frequency.
        traveltime(float): Value of the phase travel time.
        
    Returns:
        np.ndarray: Theoretical spectrum in nm·s.
    
    Notes:
        Model: A(f) = Ω0 * exp(-π f t / Q) / (1 + (f/f_c)^(2n))^(1/y), where n and y are configurable
        (default n=2, y=1 per Boatwright, 1980).
    """
    
    n, y  = CONFIG.spectral.N_FACTOR, CONFIG.spectral.Y_FACTOR
    num = omega_0 * np.exp(-np.pi * frequencies * traveltime / q_factor)
    denom = (1 + (frequencies / corner_frequency) ** (y*n))**(1/y)
    spectrums = num/denom
    return spectrums


def fit_spectrum_grid_search (
    frequencies: np.ndarray,
    spectrums: np.ndarray,
    traveltime: float,
    f_min: float,
    f_max: float
    ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    
    """
    Fit seismic spectrum systematically using grid search (deprecated for performance, provided for user option).
    
    Args:
        frequencies (np.ndarray): Frequency values in Hz.
        spectrums (np.ndarray): Spectral amplitudes in nm·s.
        traveltime (float): Travel time in seconds for attenuation (must be positive).
        f_min (float): Minimum frequency in Hz for fitting.
        f_max (float): Maximum frequency in Hz for fitting.
        
    Returns:
        Tuple[Optional[float], ...]: (Omega_0, Q, corner_frequency, RMS_error, fitted_freq, fitted_spec) or (None, ...) if fitting fails.
    
    Notes:
        This method is computationally expensive and should be used only for small datasets or validation.
        
    """
    # windowing frequencies and spectrum within f band    
    freq, spec = window_band(frequencies, spectrums, f_min, f_max)
    if not (freq.size and spec.size):
        return None, None, None, None, np.array([]), np.array([])
    
    
    peak_omega = spec.max()
    omega_0_range = np.linspace(peak_omega/100, peak_omega, 50)
    q_factor_range = np.linspace(CONFIG.spectral.Q_RANGE_MIN, CONFIG.spectral.Q_RANGE_MAX, 50)
    f_c_range = np.linspace(f_min, f_max, 50)
    
    # rms and error handler
    best_rms_e = np.inf
    omega_0_fit, q_factor_fit, f_c_fit = None, None, None
    
    # define callable function
    def objective(omega, q_val, f_cor):
        return calculate_source_spectrum(freq, omega, q_val, f_cor, traveltime)
        
    # start guessing
    for omega in omega_0_range:
        for q_val in q_factor_range:
            for f_c in f_c_range:
                theoretical = objective(omega, q_val, f_c)
                rms_e = np.sqrt(np.mean((theoretical - spec)**2))
                if rms_e < best_rms_e:
                    best_rms_e = rms_e
                    omega_0_fit, q_factor_fit, f_c_fit = omega, q_val, f_c

    if best_rms_e == np.inf or any(v is None for v in [omega_0_fit, q_factor_fit, f_c_fit]):
        return None, None, None, None, np.array([]), np.array([])

                    
    # calculate the fitted power spectral density from tuned parameter
    x_tuned = np.linspace(max(CONFIG.spectral.FC_RANGE_BUFFER, f_min), f_max*1.75, 100)
    y_tuned = calculate_source_spectrum(x_tuned, omega_0_fit, q_factor_fit, f_c_fit, traveltime)
    return omega_0_fit, q_factor_fit, f_c_fit, best_rms_e, x_tuned, y_tuned


def fit_spectrum_qmc (
    frequencies: np.ndarray,
    spectrums: np.ndarray,
    traveltime: float,
    f_min: float,
    f_max: float,
    n_samples: int = CONFIG.spectral.DEFAULT_N_SAMPLES
    ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    
    """
    Fit seismic spectrum stochastically using Quasi-Monte Carlo (QMC) sampling.
    
    Args:
        frequencies (np.ndarray): Array of frequency values in Hz.
        spectrums (np.ndarray): Spectral amplitude in nms.
        traveltime (float): Travel time in seconds for attenuation (must be positive).
        f_min (float): Minimum frequency in Hz for fitting.
        f_max (float): Maximum frequency in Hz for fitting.
        n_samples(int): Number of samples for QMC sampling (default: 2000).
        
    Returns:
        Tuple[Optional[float], ...]: (Omega_0, Q, corner_frequency, RMS_error, fitted_freq, fitted_spec) or (None, ...) if fitting fails.

    Raises:
        ValueError: if inputs are invalid or fitting fails.
    """
    
    # windowing frequencies and spectrum within f band    
    freq, spec = window_band(frequencies, spectrums, f_min, f_max)
    if not (freq.size and spec.size):
        return None, None, None, None, np.array([]), np.array([])

    # setting initial guess
    peak_omega = spec.max()
    omega_0_range = (peak_omega/100, peak_omega)
    q_factor_range = (CONFIG.spectral.Q_RANGE_MIN, CONFIG.spectral.Q_RANGE_MAX)
    f_c_range = (f_min, f_max)
    
    try:
        # QMC sampling with latin Hypercube
        sampler = qmc.LatinHypercube(d=3)
        samples = sampler.random(n=n_samples)
        scaled_samples = qmc.scale(samples, [omega_0_range[0], q_factor_range[0], f_c_range[0]], 
                               [omega_0_range[1], q_factor_range[1], f_c_range[1]])
        omega_0 = scaled_samples[:, 0]
        q_factor = scaled_samples[:, 1]
        f_c = scaled_samples[:, 2]
        
        best_rms_e = np.inf
        omega_0_fit, q_factor_fit, f_c_fit = None, None, None
        last_rms = np.inf
        
        # define callable function
        def objective(omega, q_val, f_cor):
            return calculate_source_spectrum(freq, omega, q_val, f_cor, traveltime)
            
        # random search with convergence check
        for i in range(len(omega_0)):
            theoretical = objective(omega_0[i], q_factor[i], f_c[i])
            rms_e = np.sqrt(np.mean((theoretical - spec)**2))
            if rms_e < best_rms_e:
                best_rms_e = rms_e
                omega_0_fit, q_factor_fit, f_c_fit = omega_0[i], q_factor[i], f_c[i]
                
            # check convergence
            if i > 100 and abs(last_rms - best_rms_e)/last_rms < 0.01:
                break
            last_rms = best_rms_e
        if best_rms_e == np.inf or any(v is None for v in [omega_0_fit, q_factor_fit, f_c_fit]):
            raise ValueError("QMC fitting failed to converge")
        
        # generate fitted curve
        x_tuned = np.linspace(max(CONFIG.spectral.FC_RANGE_BUFFER, f_min), f_max*1.75, 100)
        y_tuned = calculate_source_spectrum(x_tuned, omega_0_fit, q_factor_fit, f_c_fit, traveltime)
        return omega_0_fit, q_factor_fit, f_c_fit, best_rms_e, x_tuned, y_tuned
    
    except Exception:
        return None, None, None, None, np.array([]), np.array([])


def fit_spectrum_bayesian(
    frequencies: np.ndarray,
    spectrums: np.ndarray,
    traveltime: float,
    f_min: float,
    f_max: float,
    n_samples: int= CONFIG.spectral.DEFAULT_N_SAMPLES
    ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """
    Fit seismic spectrum using Bayesian optimization.
    
    Args:
        frequencies (np.ndarray): Frequency values in Hz.
        spectrums (np.ndarray): Spectral amplitudes in nm·s.
        traveltime (float): Travel time in seconds for attenuation (must be positive).
        f_min (float): Minimum frequency in Hz for fitting.
        f_max (float): Maximum frequency in Hz for fitting.
        n_samples (int): Number of calls for Bayesian optimization (default: 1000).
        
    Returns:
        Tuple[Optional[float], ...]: (Omega_0, Q, corner_frequency, RMS_error, fitted_freq, fitted_spec) or (None, ...) if fitting fails.
        
    Raises:
        ValueError: if inputs are invalid or fitting fails.
    """
    
    # windowing frequencies and spectrum within f band    
    freq, spec = window_band(frequencies, spectrums, f_min, f_max)

    
    if not (freq.size and spec.size):
        return None, None, None, None, np.array([]), np.array([])
   
   # setting initial guess from peak spectrum
    peak_omega = spec.max()
    space = [
        Real((peak_omega/100), peak_omega, name='omega_0', prior='uniform'),
        Real(50, 250, name='q_factor', prior='uniform'),
        Real(f_min, f_max, name='f_c', prior='uniform')
    ]
   # define the objective function with fixed parameters(freq, spec, traveltime)
    @use_named_args(space)
    def objective_func(omega_0:float, q_factor:float, f_c:float) -> float:
        try:
            theoretical = calculate_source_spectrum(freq, omega_0, q_factor, f_c, traveltime)
            if not np.isfinite(theoretical).all():
                raise ValueError("Non-finite theoretical spectrum")
            rms_error = np.sqrt(np.mean((theoretical - spec)**2))
            return rms_error
        except Exception:
            return np.inf
            
    try:
        # Bayesian optimization with scikit-optimize
        res = gp_minimize(
            objective_func,
            dimensions = space,
            n_calls=100,
            n_initial_points=20,
            acq_func = 'LCB',
            acq_optimizer = "lbfgs",
            n_points = 500,
            random_state=42,
            noise=1e-3,
            n_jobs = -1
        )
        
        omega_0_fit, q_factor_fit, f_c_fit = res.x
        best_rms_e = res.fun
        
        if best_rms_e == np.inf or not np.isfinite([omega_0_fit, q_factor_fit, f_c_fit]).all():
            raise ValueError("Non-finite or infinite fitting parameters")
        
        # generate fitted curve
        x_tuned = np.linspace(max(CONFIG.spectral.FC_RANGE_BUFFER, f_min), f_max*1.75, 100)
        y_tuned = calculate_source_spectrum(x_tuned, omega_0_fit, q_factor_fit, f_c_fit, traveltime)
        return omega_0_fit, q_factor_fit, f_c_fit, best_rms_e, x_tuned, y_tuned
        
    except Exception :
        return None, None, None, None, np.ndarray([]), np.ndarray([])
