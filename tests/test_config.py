""" Unit test for checking data integrity of parameters built by config.py"""

from lqtmoment.config import CONFIG


def test_magnitude_params():
    """ Check few default parameters in package default config.ini"""
    expected_taup_model  = 'iasp91'
    expected_snr = 1.75
    expected_pre_filter = [0.1, 0.2, 55, 60]
    expected_water_level = 60
    expected_velocity_vp = [2.68, 2.99, 3.95, 4.50]
    expected_velocity_vs = [1.60, 1.79, 2.37, 2.69]
    assert CONFIG.magnitude.TAUP_MODEL == expected_taup_model
    assert CONFIG.magnitude.SNR_THRESHOLD == expected_snr
    assert CONFIG.magnitude.PRE_FILTER == expected_pre_filter
    assert CONFIG.magnitude.WATER_LEVEL == expected_water_level
    assert CONFIG.magnitude.VELOCITY_VP == expected_velocity_vp
    assert CONFIG.magnitude.VELOCITY_VS == expected_velocity_vs


def test_spectral_params():
    """ Check few default parameters in package default config.ini"""
    expected_n_samples = 3000
    assert CONFIG.spectral.DEFAULT_N_SAMPLES == expected_n_samples

def test_performance_params():
    """ Check few default parameters in package default config.ini"""
    expected_logging_level = 'INFO'
    assert CONFIG.performance.LOGGING_LEVEL == expected_logging_level