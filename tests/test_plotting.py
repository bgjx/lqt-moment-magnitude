"""Unit tests for plotting.py"""

from typing import ClassVar

from lqtmoment.plotting import plot_rays


class TestPlotting:
    hypo_depth_m = 200
    sta_elev_m = 2200
    velocity: ClassVar[list] = [2.68, 2.99, 3.95, 4.50, 4.99]
    epicentral_dist = 4332.290
    raw_model: ClassVar[list] = [
        [3000.0, -1100.0, 2.680],
        [1900.0, -1310.0, 2.990],
        [590.0, -810.0, 3.950],
        [-220.0, -2280.0, 4.500],
        [-2500.0, -4500.0, 4.990],
        [-7000.0, -2000.0, 5.600],
        [-9000.0, -6000.0, 5.800],
        [-15000.0, -18000.0, 6.400],
        [-33000.0, -99966000.0, 8.000],
    ]
    up_model: ClassVar[list] = [
        [2200.0, -300.0, 2.680],
        [1900.0, -1310.0, 2.990],
        [590.0, -390.0, 3.950],
    ]
    down_model: ClassVar[list] = [
        [200.0, -420.0, 3.950],
        [-220.0, -2280.0, 4.500],
        [-2500.0, -4500.0, 4.990],
        [-7000.0, -2000.0, 5.600],
        [-9000.0, -6000.0, 5.800],
        [-15000.0, -18000.0, 6.400],
        [-33000.0, -99966000.0, 8.000],
    ]
    last_ray: ClassVar[dict] = {
        "refract_angles": [81.351, 48.448, 42.126],
        "distances": [2564.029, 4042.012, 4313.332],
        "travel_times": [0.657, 0.661, 0.151],
    }
    critical_ref: ClassVar[dict] = {
        "take_off_61.375": {
            "total_tt": [1.529],
            "incidence_angle": [36.552],
        }
    }
    down_ref: ClassVar[dict] = {
        "take_off_29.587": {
            "refract_angles": [29.587, 34.229],
            "distances": [238.471, 1789.637],
            "travel_times": [0.122, 0.613],
        },
        "take_off_38.111": {
            "refract_angles": [38.111],
            "distances": [329.453],
            "travel_times": [0.135],
        },
        "take_off_42.925": {
            "refract_angles": [42.925],
            "distances": [390.623],
            "travel_times": [0.145],
        },
        "take_off_44.858": {
            "refract_angles": [44.858],
            "distances": [417.929],
            "travel_times": [0.150],
        },
        "take_off_52.334": {
            "refract_angles": [52.334],
            "distances": [544.078],
            "travel_times": [0.174],
        },
        "take_off_61.375": {
            "refract_angles": [61.375, 90.0],
            "distances": [769.550, 1461.051],
            "travel_times": [0.222, 0.154],
        },
    }
    down_up_ref: ClassVar[dict] = {
        "take_off_61.375": {
            "refract_angles": [
                61.375,
                41.640,
                36.552,
            ],
            "distances": [714.582, 1879.279, 2101.691],
            "travel_times": [
                0.206,
                0.586,
                0.139,
            ],
        }
    }

    def test_plot_rays(self, tmp_path):
        output_path = tmp_path / "ray_path_event.png"
        plot_rays(
            self.hypo_depth_m,
            self.sta_elev_m,
            self.epicentral_dist,
            self.velocity,
            self.raw_model,
            self.up_model,
            self.down_model,
            self.last_ray,
            self.critical_ref,
            self.down_ref,
            self.down_up_ref,
            tmp_path,
        )
        assert output_path.exists()
