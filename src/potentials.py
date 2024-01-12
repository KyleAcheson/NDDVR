import numpy as np

def load_potential(pot_file, grid_file, ndim, order):
    pot = np.genfromtxt(pot_file, skip_header=1)
    grids, grid_sizes = [], []
    for i in range(ndim):
        grid = np.genfromtxt(grid_file, skip_header=1, usecols=(i))
        grids.append(grid)
        grid_sizes.append(len(grid))
    grid_sizes = tuple(grid_sizes)
    pot = pot.reshape(grid_sizes, order=order)
    return pot, grids

def harmonic(x, k=1.0):
    return 0.5 * k * x**2

def harmonic_potential_2d(x, y, kx=1.0, ky=1.0, mx=1.0, my=1.0):
    return 0.5 * (kx * x ** 2 / mx + ky * y ** 2 / my)

def harmonic_potential_3d(x, y, z, kx=1.0, ky=1.0, kz=1.0, mx=1.0, my=1.0, mz=1.0):
    return 0.5 * (kx * x**2 / mx + ky * y**2 / my + kz * z**2 / mz)