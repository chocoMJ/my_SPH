from numba import jit, prange
import numpy as np
from neighbour_search import *
from basic_math import poly6_kernel

@jit(nopython = True, parallel = True)
def compute_density_pressure(positions, den, pre, h, mass, k, rest_density, cells, x_cells, y_cells) :
    for i in prange(len(positions)) : 
        neighbours = get_neighbours(cells, x_cells, y_cells, positions[i], h)
        den[i] = 0
        for j in neighbours :
            if i == j : pass
            r = positions[j] - positions[i]
            r_2 = r[0] ** 2 + r[1] ** 2

            if r_2 <= h**2 :
                den[i] += mass * poly6_kernel(r_2, h)
                

        pre[i] = k * (den[i] - rest_density)
