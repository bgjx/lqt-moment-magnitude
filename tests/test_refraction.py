""" Unit test for refraction.py """

import pytest
import numpy as np

from LQTMomentMag.refraction import (
    build_raw_model,
    upward_model,
    downward_model,
    up_refract,
    calculate_inc_angle
)

from LQTMomentMag.config import CONFIG 

@pytest.fixture
def test_data():
    "Fixture providing consistent test data."
    boundaries = [
                    [-3.00, -1.90], [-1.90, -0.59], [-0.59, 0.22], [0.22, 2.50], [2.50, 7.00]
    ]
    velocities = [2.68, 2.99, 3.95, 4.50, 4.99]
    hypo = [37.916973, 126.651613, 200]
    station = [ 37.916973, 126.700882, 2200]
    epi_dist_m = 4332.291
    return hypo, station, epi_dist_m, boundaries, velocities


@pytest.fixture
def sample_model(test_data):
    _, _, _, boundaries, velocities = test_data
    return build_raw_model(boundaries, velocities)


def test_build_raw_model(sample_model):
    """ Test build_raw_model creates correct layer structure."""
    expected = [[3000.0, -1100.0, 2680], [1900.0, -1310.0, 2990], [590.0, -809.9999999999999, 3950], [-220.0, -2280.0, 4500], [-2500.0, -4500.0, 4990]]
    assert(len(sample_model)) == len(expected)
    for layer, exp_layer in zip(sample_model, expected):
        assert layer == pytest.approx(exp_layer, rel=1e-5)


def test_upward_model(sample_model):
    """ Test the upward model adjusts the layers correctly. """
    hypo_depth_m = 200
    sta_elev_m = 2200
    up_model = upward_model(hypo_depth_m, sta_elev_m, sample_model.copy())
    expected_upward = [[2200, -300.0, 2680], [1900.0, -1310.0, 2990], [590.0, -390.0, 3950]]
    assert len(up_model) ==   len(expected_upward)
    for layer, exp_upward in zip(up_model, expected_upward):
        assert layer == pytest.approx(exp_upward, rel=1e-5)


def test_downward_model(sample_model):
    """ Test the downward model adjusts layers correctly. """
    hypo_depth_m = 200
    down_model = downward_model(hypo_depth_m, sample_model)
    expected_downward = [[200, -420.0, 3950], [-220.0, -2280.0, 4500], [-2500.0, -4500.0, 4990]]
    assert len(down_model) == len(expected_downward)
    for layer, exp_downward in zip(down_model, expected_downward):
        assert layer == pytest.approx(exp_downward, rel=1e-5)


def test_up_refract(sample_model):
    hypo_depth_m = 200
    sta_elev_m = 2200
    epi_dist_m = 4332.291
    up_model = upward_model(hypo_depth_m, sta_elev_m, sample_model.copy())
    result_dict, final_take_off = up_refract(epi_dist_m, up_model)
    final_key = f"take_off_{final_take_off}"
    assert isinstance(result_dict, dict)
    assert isinstance(final_take_off, float)
    assert result_dict[final_key]["distances"][-1] == pytest.approx(epi_dist_m, rel=1e-5)
    assert all(0 <= angle <= 90 for angle in result_dict[final_key]["refract_angles"])


def test_calculate_inc_ange(test_data, tmp_path):
    hypo, station, epi_dist_m, boundaries, velocities = test_data
    figure_path = tmp_path / "figures"
    figure_path.mkdir()
    take_off, tt, inc_angle = calculate_inc_angle(hypo, station, boundaries, velocities, figure_statement=True, figure_path=str(figure_path))
    expected_value = [98.64864864864865, 1.4680449010337573, 42.12621600327373]
    assert take_off == pytest.approx(expected_value[0], rel=1e-1)
    assert tt == pytest.approx(expected_value[1], rel=1e-1)
    assert inc_angle == pytest.approx(expected_value[2], rel=1e-1)
    assert isinstance(take_off, float)
    assert isinstance(tt, float)
    assert isinstance(inc_angle, float)
    assert 0 <= take_off <= 180
    assert tt > 0
    assert 0 <= inc_angle <=90
    plot_file = figure_path/ "ray_path_event.png"
    assert plot_file.exists(), f"plot file {plot_file} was not created"

