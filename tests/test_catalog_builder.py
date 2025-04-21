""" Unit test for catalog builder.py"""

from pathlib import Path
import pytest
import pandas as pd

from lqtmoment import build_catalog

@pytest.fixture
def test_data():
    """ Fixture for sample test data """
    hypo_dir = Path(r"tests/sample_tests_data/data/catalog/hypo_catalog.xlsx")
    pick_dir = Path(r"tests/sample_tests_data/data/catalog/picking_catalog.xlsx")
    sta_dir = Path(r"tests/sample_tests_data/data/station/station.xlsx")
    return hypo_dir, pick_dir, sta_dir

def test_catalog_builder(test_data):
    hypo_path, pick_path, sta_path = test_data
    built_dataframe = build_catalog(hypo_path, pick_path, sta_path)
    assert isinstance(built_dataframe, pd.DataFrame)
    assert not built_dataframe.empty

    # check the dataframe structure
    expected_columns = [
    "source_id", "source_lat", "source_lon", "source_depth_m", 
    "network_code", "station_code", "station_lat", "station_lon", "station_elev_m",
    "source_origin_time", "p_arr_time", "p_travel_time_sec",
    "p_polarity", "p_onset",
    "s_arr_time", "s_travel_time_sec", "s_p_lag_time_sec","coda_time",
    "source_err_rms_s", "n_phases", "gap_degree", 
    "x_horizontal_err_m", "y_horizontal_err_m", "z_depth_err_m",
    "earthquake_type", "remarks"
    ]
    assert list(built_dataframe.columns) == expected_columns, "Missing or extra columns"
    assert len(built_dataframe) > 1, "Expected more than one row"

