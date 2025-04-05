""" Unit test for checking data integrity of parameters built by config.py"""

from lqtmoment.config import CONFIG


def test_magnitude_params():
    """ Check few default parameters in package default config.ini"""
    expected_taup_model  = 'iasp91'
    expected_snr = 1.75
    expected_water_level = 60
    assert CONFIG.magnitude.TAUP_MODEL == expected_taup_model
    assert CONFIG.magnitude.SNR_THRESHOLD == expected_snr
    assert CONFIG.magnitude.WATER_LEVEL == expected_water_level

def test_spectral_params():
    """ Check few default parameters in package default config.ini"""
    expected_n_samples = 3000
    assert CONFIG.spectral.DEFAULT_N_SAMPLES == expected_n_samples

def test_performance_params():
    """ Check few default parameters in package default config.ini"""
    expected_logging_level = 'INFO'
    assert CONFIG.performance.LOGGING_LEVEL == expected_logging_level