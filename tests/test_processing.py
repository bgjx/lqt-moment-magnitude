""" Unit tests for processing.py """

import pytest
from lqtmoment import magnitude_estimator

@pytest.fixture
def test_data():
    """ Fixture for sample test data """
    dir_collector = {
            'wave_dir': r"tests/sample_tests_data/data/wave",
            'calibration_dir': r"tests/sample_tests_data/data/calibration",
            'catalog_file': r"tests/sample_tests_data/results/lqt_catalog/lqt_catalog_test.csv",
            'config_file': r"tests/sample_tests_data/config/config_test.ini"
            }
    
    return dir_collector

def test_magnitude_estimator(test_data):
    """ Unit test for magnitude estimator function """
    dirs = test_data
    lqt_catalog_result, moment_result, fitting_result = magnitude_estimator(
                                                        wave_dir=dirs['wave_dir'],
                                                        cal_dir=dirs['calibration_dir'],
                                                        catalog_file=dirs['catalog_file'],
                                                        config_file=dirs['config_file'],
                                                        id_start=1001,
                                                        id_end=1005,
                                                        lqt_mode=True
                                                        )
    
    # check expected columns for all output
    expected_lqt_catalog_columns = [
        "source_id", "source_lat", "source_lon", "source_depth_m", 
        "network_code", "magnitude", "station_code", "station_lat",
        "station_lon", "station_elev_m", "source_origin_time",
        "p_arr_time", "p_travel_time_sec", "p_polarity", "p_onset",
        "s_arr_time", "s_travel_time_sec", "s_p_lag_time_sec","coda_time",
        "source_err_rms_s", "n_phases", "gap_degree", 
        "x_horizontal_err_m", "y_horizontal_err_m", "z_depth_err_m",
        "earthquake_type", "remarks"
    ]

    expected_moment_result_columns = [
        "source_id", "fc_avg", "fc_std",
        "src_rad_avg_m", "src_rad_std_m",
        "stress_drop_bar", "mw_average", "mw_std"
    ]

    expected_fitting_result_columns = [
        "source_id", "station", "f_corner_p", "f_corner_sv", "f_corner_sh",
        "q_factor_p", "q_factor_sv", "q_factor_sh",
        "omega_0_p_nms", "omega_0_sv_nms", "omega_0_sh_nms",
        "rms_e_p_nms", "rms_e_sv_nms", "rms_e_sh_nms",
        "moment_p_Nm", "moment_s_Nm"
    ]

    expected_moment_magnitude_result = [0.871, 0.975, 1.128, 1.147, 0.784]

    assert list(lqt_catalog_result.columns) == expected_lqt_catalog_columns
    assert list(moment_result.columns) == expected_moment_result_columns
    assert list(fitting_result.columns) == expected_fitting_result_columns
    assert lqt_catalog_result['magnitude'].drop_duplicates().tolist() == pytest.approx(expected_moment_magnitude_result, abs=0.2)


    