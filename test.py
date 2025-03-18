import numpy as np


layer_boundaries = "-3.00,-1.90; -1.90,-0.59; -0.59, 0.22; 0.22, 2.50; 2.50, 7.00; 7.00,9.00;  9.00,15.00 ; 15.00,33.00; 33.00,9999"

layer_boundary = [[float(v) for v in layer.split(",")] for layer in layer_boundaries.split(";")]
print(layer_boundary)

angles = np.zeros(len(layer_boundary))
print(angles)

angles_vector = np.array([layer[1] for layer in layer_boundary[::-1]])

cum_sum = np.cumsum(angles_vector)
print(cum_sum)
print(angles_vector)

veloc = [2.68, 2.99, 3.95, 4.50, 4.99, 5.60, 5.80, 6.40, 8.00]
velocities = np.array(veloc)
crittical_angles = np.degrees(np.arcsin(velocities[:-1]/velocities[1:])).tolist()
print(crittical_angles)

down_seg_result = {}
down_seg_result[f"take_off"] = veloc
veloc.append(crittical_angles)

print(down_seg_result)