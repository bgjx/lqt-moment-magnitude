import pytest
import pandas as pd
from lqtmoment.lqt_analysis import load_catalog, LqtAnalysis, Statistic

@pytest.fixture
def test_data():
    """ Fixture for sample DataFrame"""
    data_dict = {
        'source_id': [  1000, 1000, 1000, 1001, 1001, 1001, 
                        1002, 1002, 1002, 1003, 1003, 1003,
                        1004, 1004, 1004, 1005, 1005, 1005,
                        1006, 1006, 1006, 1007, 1007, 1007,
                        1008, 1008, 1008, 1009, 1009, 1009
                          ],
        'lat': [38.088, 38.088, 38.088, 38.086, 38.086, 38.086,
                38.084, 38.084, 38.084, 38.085, 38.085, 38.085,
                38.088, 38.088, 38.088, 38.084, 38.084, 38.084,
                38.086, 38.086, 38.086, 38.088, 38.088, 38.088,
                38.085, 38.085, 38.085, 38.084, 38.084, 38.084,
        ],
        'lon': [126.596, 126.596, 126.596, 126.591, 126.591, 126.591,
                126.597, 126.597, 126.597, 126.602, 126.602, 126.602,
                126.594, 126.594, 126.594, 126.596, 126.596, 126.596,
                126.597, 126.597, 126.597, 126.594, 126.594, 126.594,
                126.591, 126.591, 126.591, 126.596, 126.596, 126.596,
                ],
        'depth': [1252, 1252, 1252, 1035, 1035, 1035,
                    705, 705, 705, 770, 770, 770,
                    1005, 1005, 1005, 670, 670, 670,
                    1135, 1135, 1135, 876, 876, 676,
                    1035, 1035, 1135, 976, 976, 976,
                  ],
        'magnitude': [  0.220,0.220,0.220, 1.220,1.220,1.220,
                        0.720,0.720,0.720, 1.520,1.520,1.520,
                        0.520,0.520,0.520, 0.950,0.950,0.950,
                        1.320,1.320,1.320, 2.950,2.950,2.950,
                        0.320,0.320,0.320, 0.250,0.250,0.250,
                    ]
    }

    return pd.DataFrame(data_dict)

def test_class(test_data):
    """ Test LqtAnalysis instantiation with valid data. """
    data = LqtAnalysis(test_data)
    assert isinstance(data, LqtAnalysis)

def test_statistic_mean(test_data):
    """ Test compute_statistic for mean. """   
    data = LqtAnalysis(test_data)
    expected_mean = 0.9990000000000002
    assert data.compute_statistic('magnitude', Statistic.MEAN) == pytest.approx(expected_mean, rel=1e-5)

def test_statistic_median(test_data):
    """ Test compute_statistic for median. """   
    data = LqtAnalysis(test_data)
    expected_median = 0.835
    assert data.compute_statistic('magnitude', Statistic.MEDIAN) == pytest.approx(expected_median, rel=1e-5)
