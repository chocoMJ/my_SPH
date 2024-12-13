from basic_math import *
from numba import jit, prange
from neighbour_search import *
import numpy as np

@jit(nopython = True, parallel = True)
def compute_force(positions, velocities, pre, den, h, mass, visc, gravity, forces, cells, x_cells, y_cells) :
    for i in prange(len(positions)) :
        neighbours = get_neighbours(cells, x_cells, y_cells, positions[i], h)
        force_pressure = np.zeros(2, dtype=np.float64)
        force_viscosity = np.zeros(2, dtype=np.float64)

        for j in neighbours :
            if i == j : 
                continue

            r_vector = positions[j] - positions[i]
            r = np.sqrt(r_vector[0] ** 2 + r_vector[1] ** 2)

            if 0 < r < h :
                force_pressure += -mass / den[j] * (pre[i] + pre[j]) / 2 * Spiky_Kernel_gradient(r, h, r_vector)

                force_viscosity += visc * mass * (velocities[j] - velocities[i]) / den[j] * Viscosity_Kernel_laplacian(r, h)
    
        forces[i] = force_pressure + force_viscosity + gravity