""" Unit test for fitting_spectral.py """

import pytest
import numpy as np

from LQTMomentMag.fitting_spectral import fit_spectrum_qmc
from LQTMomentMag.config import CONFIG


@pytest.fixture
def sample_spectrum():
    """ Fixture providing consistent test data."""
    freq = np.linspace(2, 100, 1000)
    num_amp = 50 * np.exp(-np.pi * freq * 1.75/ 100)
    denom_amp = (1 + (freq/30) ** 2)
    spectrum = num_amp/denom_amp
    return freq, spectrum


def test_fit_spectrum(sample_spectrum):
    freq, spectrum = sample_spectrum
    omega_0,  q_factor,  f_c,  err,  x_fit,  y_fit = fit_spectrum_qmc(freq, spectrum, 1.75, CONFIG.magnitude.F_MIN, CONFIG.magnitude.F_MAX, CONFIG.spectral.DEFAULT_N_SAMPLES )
    assert isinstance(omega_0, float)
    assert isinstance(q_factor, float)
    assert isinstance(f_c, float)
    assert 7 < f_c < 60


