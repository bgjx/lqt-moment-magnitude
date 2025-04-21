""" Unit test for checking data integrity of parameters built by config.py"""

from lqtmoment.config import CONFIG


def test_wave_params():
    """ Check few default wave parameters in package default config.ini """
    expected_snr = 1.75
    expected_water_level = 60
    expected_pre_filter = [0.01, 0.02, 55, 60]
    expected_post_filter_statement = 'yes'
    expected_post_filter_f_min = 0.1
    expected_post_filter_f_max = 50
    expected_trim_method = 'dynamic'
    expected_sec_bf_p = 10
    expected_sec_af_p = 50
    assert CONFIG.wave.SNR_THRESHOLD == expected_snr
    assert CONFIG.wave.WATER_LEVEL == expected_water_level
    assert CONFIG.wave.PRE_FILTER == expected_pre_filter
    assert CONFIG.wave.APPLY_POST_INSTRUMENT_REMOVAL_FILTER == expected_post_filter_statement
    assert CONFIG.wave.POST_FILTER_F_MIN == expected_post_filter_f_min
    assert CONFIG.wave.POST_FILTER_F_MAX == expected_post_filter_f_max
    assert CONFIG.wave.TRIM_METHOD == expected_trim_method
    assert CONFIG.wave.SEC_BF_P_ARR_TRIM == expected_sec_bf_p
    assert CONFIG.wave.SEC_AF_P_ARR_TRIM == expected_sec_af_p

def test_magnitude_params():
    """ Check few default magnitude parameters in package default config.ini """
    expected_padding_bf_arrival = 0.1
    expected_noise_duration = 0.5
    expected_noise_padding = 0.2
    expected_r_pattern_p = 0.52
    expected_r_pattern_s = 0.63
    expected_free_surface = 2.0
    expected_k_p = 0.32
    expected_k_s = 0.21
    expected_mw_constant = 6.07
    expected_taup_model  = 'iasp91'
    expected_velocity_vp = [2.68, 2.99, 3.95, 4.50]
    expected_velocity_vs = [1.60, 1.79, 2.37, 2.69]
    assert CONFIG.magnitude.PADDING_BEFORE_ARRIVAL == expected_padding_bf_arrival
    assert CONFIG.magnitude.NOISE_DURATION == expected_noise_duration
    assert CONFIG.magnitude.NOISE_PADDING == expected_noise_padding
    assert CONFIG.magnitude.R_PATTERN_P == expected_r_pattern_p
    assert CONFIG.magnitude.R_PATTERN_S == expected_r_pattern_s
    assert CONFIG.magnitude.FREE_SURFACE_FACTOR == expected_free_surface
    assert CONFIG.magnitude.K_P == expected_k_p
    assert CONFIG.magnitude.K_S == expected_k_s
    assert CONFIG.magnitude.MW_CONSTANT == expected_mw_constant
    assert CONFIG.magnitude.TAUP_MODEL == expected_taup_model
    assert CONFIG.magnitude.VELOCITY_VP == expected_velocity_vp
    assert CONFIG.magnitude.VELOCITY_VS == expected_velocity_vs

def test_spectral_params():
    """ Check few default spectral parameters in package default config.ini """
    expected_f_min = 0.1
    expected_f_max = 45
    expected_omega_min = 0.01
    expected_omega_max = 2000
    expected_q_min = 50
    expected_q_max = 250
    expected_fc_buffer = 1
    expected_n_samples = 3000
    assert CONFIG.spectral.F_MIN == expected_f_min
    assert CONFIG.spectral.F_MAX == expected_f_max
    assert CONFIG.spectral.OMEGA_0_RANGE_MIN == expected_omega_min
    assert CONFIG.spectral.OMEGA_0_RANGE_MAX == expected_omega_max
    assert CONFIG.spectral.Q_RANGE_MIN == expected_q_min
    assert CONFIG.spectral.Q_RANGE_MAX == expected_q_max
    assert CONFIG.spectral.FC_RANGE_BUFFER == expected_fc_buffer
    assert CONFIG.spectral.DEFAULT_N_SAMPLES == expected_n_samples

def test_performance_params():
    """ Check few default performance parameters in package default config.ini"""
    expected_logging_level = 'INFO'
    assert CONFIG.performance.LOGGING_LEVEL == expected_logging_level