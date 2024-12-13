import numpy as np
from numba import njit

@njit
def create_spatial_hash(width, height, cellsize):
    x_cells = int(np.ceil(width / cellsize))
    y_cells = int(np.ceil(height / cellsize))
    
    MAX_PARTICLES_PER_CELL = 10000
    
    cells = np.full((x_cells, y_cells, MAX_PARTICLES_PER_CELL), -1, dtype=np.int32)
    
    return cells, x_cells, y_cells

@njit
def coords_to_index(coords, cellsize):
    return int(coords[0] // cellsize), int(coords[1] // cellsize)

@njit
def add_to_cell(cells, x_cells, y_cells, coords, id, cellsize):
    x_idx, y_idx = coords_to_index(coords, cellsize)
    if 0 <= x_idx < x_cells and 0 <= y_idx < y_cells:
        for k in range(cells.shape[2]):
            if cells[x_idx, y_idx, k] == -1:
                cells[x_idx, y_idx, k] = id
                break

@njit
def move_node(cells, x_cells, y_cells, old_coords, new_coords, id, cellsize):
    old_x_idx, old_y_idx = coords_to_index(old_coords, cellsize)
    new_x_idx, new_y_idx = coords_to_index(new_coords, cellsize)
    
    if 0 <= old_x_idx < x_cells and 0 <= old_y_idx < y_cells:
        for k in range(cells.shape[2]):
            if cells[old_x_idx, old_y_idx, k] == id:
                cells[old_x_idx, old_y_idx, k] = -1
                break
    
    add_to_cell(cells, x_cells, y_cells, new_coords, id, cellsize)

@njit
def get_neighbours(cells, x_cells, y_cells, coords, cellsize):
    x_idx, y_idx = coords_to_index(coords, cellsize)
    neighbours = np.empty(cells.shape[2] * 9, dtype=np.int32)
    neighbour_count = 0
    
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            nx, ny = x_idx + dx, y_idx + dy
            if 0 <= nx < x_cells and 0 <= ny < y_cells:
                for k in range(cells.shape[2]):
                    particle_id = cells[nx, ny, k]
                    if particle_id != -1:
                        neighbours[neighbour_count] = particle_id
                        neighbour_count += 1
    
    return neighbours[:neighbour_count]