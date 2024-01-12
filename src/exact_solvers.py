import numpy as np
import warnings

def colbert_miller(grid, mass, hbar=1):
    ngrid = len(grid)
    dg = grid[1] - grid[0]
    indicies = np.arange(ngrid)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        T_n = ((hbar ** 2) * (-1.0)**(indicies[None] - indicies[:, None])) / (mass * dg**2 * (indicies[None] - indicies[:, None])**2)
    np.fill_diagonal(T_n, ((hbar ** 2) * np.pi ** 2) / (6 * mass * dg ** 2))
    return T_n