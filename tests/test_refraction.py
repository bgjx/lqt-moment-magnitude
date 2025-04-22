""" Unit test for refraction.py """

import pytest
import numpy as np

from lqtmoment.refraction import (
    build_raw_model,
    upward_model,
    downward_model,
    up_refract,
    calculate_inc_angle
)

@pytest.fixture
def test_data():
    "Fixture providing consistent test data."
    boundaries = [
                    [-3.00, -1.90], [-1.90, -0.59], [-0.59, 0.22], [0.22, 2.50], [2.50, 7.00]
    ]
    velocity_p = [2.68, 2.99, 3.95, 4.50, 4.99]
    velocity_s = [1.60, 1.79, 2.37, 2.69, 2.99]
    hypo = [37.916973, 126.651613, 200]
    station = [ 37.916973, 126.700882, 2200]
    epi_dist_m = 4332.291
    return hypo, station, epi_dist_m, boundaries, velocity_p, velocity_s


@pytest.fixture
def sample_model(test_data):
    _, _, _, boundaries, velocity_p, velocity_s = test_data
    return build_raw_model(boundaries, velocity_p)


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
    hypo, station, epi_dist_m, boundaries, velocity_p, velocity_s = test_data
    figure_path = tmp_path / "figures"
    figure_path.mkdir()
    take_off_p, total_tt_p, inc_angle_p, take_off_s, total_tt_s, inc_angle_s = calculate_inc_angle(
                                                                                hypo, station,
                                                                                boundaries,
                                                                                velocity_p,
                                                                                velocity_s,
                                                                                source_type='very_local_earthquake',
                                                                                generate_figure=True,
                                                                                figure_path=str(figure_path))
    expected_value = [98.64864864864865, 1.4680449010337573, 42.12621600327373, 98.55855855855856, 2.4581347184387163, 41.88115419698122]
    assert take_off_p == pytest.approx(expected_value[0], rel=1e-1)
    assert total_tt_p == pytest.approx(expected_value[1], rel=1e-1)
    assert inc_angle_p == pytest.approx(expected_value[2], rel=1e-1)
    assert isinstance(take_off_p, float)
    assert isinstance(total_tt_p, float)
    assert isinstance(inc_angle_p, float)
    assert 0 <= take_off_p <= 180
    assert total_tt_p > 0
    assert 0 <= inc_angle_p <=90
    assert take_off_s == pytest.approx(expected_value[3], rel=1e-1)
    assert total_tt_s == pytest.approx(expected_value[4], rel=1e-1)
    assert inc_angle_s == pytest.approx(expected_value[5], rel=1e-1)
    assert isinstance(take_off_s, float)
    assert isinstance(total_tt_s, float)
    assert isinstance(inc_angle_s, float)
    assert 0 <= take_off_s <= 180
    assert total_tt_s > 0
    assert 0 <= inc_angle_s <=90
    plot_file = figure_path/ "ray_path_event.png"
    assert plot_file.exists(), f"plot file {plot_file} was not created"

