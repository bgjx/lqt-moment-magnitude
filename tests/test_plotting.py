""" Unit tests for plotting.py """

import pytest
import matplotlib.pyplot as plt 
from LQTMomentMag.plotting import plot_rays

@pytest.fixture
def ray_data():
    """ Fixture for sample ray data. """
    hypo_depth_m = 200
    sta_elev_m = 2200
    velocities = [2.68, 2.99, 3.95, 4.50, 4.99]
    return hypo_depth_m, sta_elev_m, velocities

def test_plot_rays(ray_data, tmp_path):
    hypo_depth_m, sta_elev_m, velocity = ray_data
    raw_model = [[3000.0, -1100.0, 2.68], [1900.0, -1310.0, 2.99], [590.0, -809.9999999999999, 3.95], [-220.0, -2280.0, 4.5], [-2500.0, -4500.0, 4.99], [-7000.0, -2000.0, 5.6], [-9000.0, -6000.0, 5.8], [-15000.0, -18000.0, 6.4], [-33000.0, -99966000.0, 8.0]]
    up_model = [[2200, -300.0, 2.68], [1900.0, -1310.0, 2.99], [590.0, -390.0, 3.95]]
    down_model = [[200, -420.0, 3.95], [-220.0, -2280.0, 4.5], [-2500.0, -4500.0, 4.99], [-7000.0, -2000.0, 5.6], [-9000.0, -6000.0, 5.8], [-15000.0, -18000.0, 6.4], [-33000.0, -99966000.0, 8.0]]
    last_ray = {'refract_angles': [81.35135135135135, 48.44805320754633, 42.12621600327373], 'distances': [2564.028523533721, 4042.0118697394314, 4313.332110425197], 'travel_times': [0.6565871601653936, 0.6605274773413181, 0.15093026352704556]}
    critical_ref = {'take_off_61.37544904657465': {'total_tt': [1.5292617553510635], 'incidence_angle': [36.552246534262366]}}
    down_ref = {'take_off_29.587357652947006': {'refract_angles': [29.587357652947006, 34.228866327812575], 'distances': [238.47064248419497, 1789.636839714063], 'travel_times': [0.12227304213257874, 0.612806398905627]}, 'take_off_38.111040455259975': {'refract_angles': [38.111040455259975], 'distances': [329.452774210178], 'travel_times': [0.1351384556926864]}, 'take_off_42.924534158176805': {'refract_angles': [42.924534158176805], 'distances': [390.6233623666296], 'travel_times': [0.14520849233946176]}, 'take_off_44.85840430510595': {'refract_angles': [44.85840430510595], 'distances': [417.9292120362051], 'travel_times': [0.15000183223219024]}, 'take_off_52.33370794610822': {'refract_angles': [52.33370794610822], 'distances': [544.0777388918488], 'travel_times': [0.1740072371139449]}, 'take_off_61.37544904657465': {'refract_angles': [61.37544904657465, 90], 'distances': [769.5497240838614, 1461.050852023914], 'travel_times': [0.22194992843309577, 0.15366691732001173]}}
    down_up_ref = {'take_off_61.37544904657465': {'refract_angles': [61.37544904657465, 41.63971879245696, 36.552246534262366], 'distances': [714.5818866492999, 1879.2792745925904, 2101.69123218957], 'travel_times': [0.20609636211644605, 0.5862505168560538, 0.13934810219236024]}}
    epicentral_dist = 4332.29
    output_path = tmp_path/"ray_path_event.png"
    plot_rays(hypo_depth_m, sta_elev_m, velocity, raw_model, up_model, down_model, last_ray, critical_ref, down_ref, down_up_ref, epicentral_dist, tmp_path)
    assert output_path.exists()
    plt.close()