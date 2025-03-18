""" Unit test for catalog builder.py"""

from pathlib import Path
import pytest
import pandas as pd

from LQTMomentMag.catalog_builder import build_catalog


@pytest.fixture
def test_data():
    hypo_dir = Path(r"tests/sample_tests_data/catalog/hypo_catalog.xlsx")
    pick_dir = Path(r"tests/sample_tests_data/catalog/picking_catalog.xlsx")
    sta_dir = Path(r"tests/sample_tests_data/station/station.xlsx")
    return hypo_dir, pick_dir, sta_dir

def test_catalog_builder(test_data):
    hypo_path, pick_path, sta_path = test_data
    built_dataframe = build_catalog(hypo_path, pick_path, sta_path, "TEST01")
    assert isinstance(built_dataframe, pd.DataFrame)
    assert not built_dataframe.empty
    assert built_dataframe.network.iloc[0] == "TEST01"

    # check the dataframe structure
    expected_columns = [
        "network", "event_id", "source_lat", "source_lon", "source_depth_m", "source_origin_time",
        "station_code", "station_lat", "station_lon", "station_elev_m", "p_arr_time",
        "p_onset", "p_polarity", "s_arr_time", "remarks"
    ]
    assert list(built_dataframe.columns) == expected_columns, "Missing or extra columns"
    assert len(built_dataframe) > 1, "Expected more than one row"

