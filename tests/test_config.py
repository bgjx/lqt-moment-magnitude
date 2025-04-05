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