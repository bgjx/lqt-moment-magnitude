""" Unit tests for processing.py """

import pytest
from lqtmoment import magnitude_estimator

@pytest.fixture
def test_data():
    """ Fixture for sample test data """
    dir_collector = {
            'wave_dir': r"..\tests\sample_tests_data\data\wave",
            'calibration_dir': r"..\tests\sample_tests_data\data\calibration",
            'catalog_file': r"..\tests\sample_tests_data\results\lqt_catalog\lqt_catalog_test.csv",
            'config_file': r"..\tests\sample_tests_data\config\config_test.ini"
            }
    
    return dir_collector

def test_magnitude_estimator():
    """ Unit test for magnitude estimator function """
    dirs = test_data
    lqt_catalog_result, moment_result, fitting_result = magnitude_estimator(
                                                        dirs['wave_dir'],
                                                        dirs['calibration_dir'],
                                                        dirs['catalog_file'],
                                                        dirs['config_file'],
                                                        id_start=1001,
                                                        id_end=1005,
                                                        lqt_mode=True
                                                        )
    


    