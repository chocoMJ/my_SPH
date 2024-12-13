import numpy as np
from numba import jit
@jit(nopython=True)
def poly6_kernel(r2, smoothing_length):
    coefficient = 315 / (64 * np.pi * smoothing_length**9)
    
    return coefficient * (smoothing_length**2 - r2)**3

@jit(nopython=True)
def poly6_kernel_gradient(r, h, r_vector):
    coefficient = - 945 / (32 * np.pi * h**9)
    grad_scalar = coefficient * (h**2 - r**2)**2
    
    return grad_scalar * r_vector

@jit(nopython=True)
def poly6_kernel_laplacian(r, h) :
    coefficient = 945 / (8 * np.pi * h**9)
    return coefficient * (h**2 - r**2) * (r**2 - (3/4) * (h**2 - r**2))


@jit(nopython = True)
def Spiky_Kernel_gradient(r,h, r_vector) :
    coefficient = -45 / (np.pi * h**6)

    scalar = coefficient * (h - r)**2 / r
    gradient = scalar * r_vector

    return gradient

@jit(nopython = True)
def Viscosity_Kernel_laplacian(r, h):
    if r > h or r < 0:  
        return 0.0
    coefficient = 45 / (np.pi * h**6)
    laplacian = coefficient * (h - r)

    return laplacian