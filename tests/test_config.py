"""Unit test for checking data integrity of parameters built by config.py"""

from lqtmoment.config import CONFIG


def test_wave_params():
    """Check few default wave parameters in package default config.ini"""
    expected_snr = 2
    expected_water_level = 60
    expected_pre_filter = [0.001, 0.005, 55, 60]
    expected_post_filter_statement = True
    expected_post_filter_f_min = 0.01
    expected_post_filter_f_max = 30
    expected_trim_mode = "dynamic"
    expected_sec_bf_p = 10
    expected_sec_af_p = 50
    expected_padding_bf_arrival = 0.2
    expected_min_p_window = 1.0
    expected_max_p_window = 10.0
    expected_min_s_window = 2.0
    expected_max_s_window = 20.0
    expected_noise_duration = 1.0
    expected_noise_padding = 0.2
    assert expected_snr == CONFIG.wave.SNR_THRESHOLD
    assert expected_water_level == CONFIG.wave.WATER_LEVEL
    assert expected_pre_filter == CONFIG.wave.PRE_FILTER
    assert (
        expected_post_filter_statement
        == CONFIG.wave.APPLY_POST_INSTRUMENT_REMOVAL_FILTER
    )
    assert expected_post_filter_f_min == CONFIG.wave.POST_FILTER_F_MIN
    assert expected_post_filter_f_max == CONFIG.wave.POST_FILTER_F_MAX
    assert expected_trim_mode == CONFIG.wave.TRIM_MODE
    assert expected_sec_bf_p == CONFIG.wave.SEC_BF_P_ARR_TRIM
    assert expected_sec_af_p == CONFIG.wave.SEC_AF_P_ARR_TRIM
    assert expected_padding_bf_arrival == CONFIG.wave.PADDING_BEFORE_ARRIVAL
    assert expected_min_p_window == CONFIG.wave.MIN_P_WINDOW
    assert expected_max_p_window == CONFIG.wave.MAX_P_WINDOW
    assert expected_min_s_window == CONFIG.wave.MIN_S_WINDOW
    assert expected_max_s_window == CONFIG.wave.MAX_S_WINDOW
    assert expected_noise_duration == CONFIG.wave.NOISE_DURATION
    assert expected_noise_padding == CONFIG.wave.NOISE_PADDING


def test_magnitude_params():
    """Check few default magnitude parameters in package default config.ini"""
    expected_r_pattern_p = 0.52
    expected_r_pattern_s = 0.63
    expected_free_surface = 2.0
    expected_k_p = 0.32
    expected_k_s = 0.21
    expected_mw_constant = 6.07
    expected_taup_model = "iasp91"
    expected_velocity_vp = [3.82, 4.50, 4.60, 6.20, 8.00]
    expected_velocity_vs = [2.30, 2.53, 2.53, 3.44, 4.44]
    assert expected_r_pattern_p == CONFIG.magnitude.R_PATTERN_P
    assert expected_r_pattern_s == CONFIG.magnitude.R_PATTERN_S
    assert expected_free_surface == CONFIG.magnitude.FREE_SURFACE_FACTOR
    assert expected_k_p == CONFIG.magnitude.K_P
    assert expected_k_s == CONFIG.magnitude.K_S
    assert expected_mw_constant == CONFIG.magnitude.MW_CONSTANT
    assert expected_taup_model == CONFIG.magnitude.TAUP_MODEL
    assert expected_velocity_vp == CONFIG.magnitude.VELOCITY_VP
    assert expected_velocity_vs == CONFIG.magnitude.VELOCITY_VS


def test_spectral_params():
    """Check few default spectral parameters in package default config.ini"""
    expected_smooth_window = 3
    expected_f_min = 0.01
    expected_f_max = 30
    expected_omega_min = 0.01
    expected_omega_max = 2000
    expected_q_min = 50
    expected_q_max = 300
    expected_n_samples = 3000
    assert expected_smooth_window == CONFIG.spectral.SMOOTH_WINDOW_SIZE
    assert expected_f_min == CONFIG.spectral.F_MIN
    assert expected_f_max == CONFIG.spectral.F_MAX
    assert expected_omega_min == CONFIG.spectral.OMEGA_0_RANGE_MIN
    assert expected_omega_max == CONFIG.spectral.OMEGA_0_RANGE_MAX
    assert expected_q_min == CONFIG.spectral.Q_RANGE_MIN
    assert expected_q_max == CONFIG.spectral.Q_RANGE_MAX
    assert expected_n_samples == CONFIG.spectral.DEFAULT_N_SAMPLES


def test_performance_params():
    """Check few default performance parameters in package default config.ini"""
    expected_logging_level = "INFO"
    assert expected_logging_level == CONFIG.performance.LOGGING_LEVEL
