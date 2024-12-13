from boundary import apply_bound
from numba import jit, prange
from neighbour_search import *

def update(positions, velocities, forces, den, time_step, xlim, ylim, cells, x_cells, y_cells, smoothing_length) :
    for i in prange(len(positions)) :
        old_pos = positions[i]
        velocities[i] += time_step * forces[i] / den[i]
        positions[i] += time_step * velocities[i]
        apply_bound(positions,velocities, xlim, ylim, i)
        
        move_node(cells, x_cells, y_cells, old_pos, positions[i], i, smoothing_length)