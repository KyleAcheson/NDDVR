import numpy as np
import fast_dvr.dvr as dvr
import fast_dvr.potentials as pot
from fast_dvr.exact_solvers import *
import fast_dvr.grids as grd

BOHR = 0.529177
AU2WAVNUM = 219474.63


if __name__ == "__main__":

    ndims = 2
    ngrids = [41, 41]
    nbases = [25, 25]
    q_mins = [-5, -5]
    q_maxs = [5, 5]
    masses = [1, 1]
    neig = 6

    basis_func = ('sine', get_pib_basis)

    grids = []
    for i in range(ndims):
        grids.append(np.linspace(q_mins[i], q_maxs[i], ngrids[i]))

    # sine-DVR - first get quadrature points by diagonalising the coordinate operator,
    # then use this to evaluate the potential - this is used to construct a diagonal
    # potential operator. The KEO is evaluated in the FBR using the grids specified
    # by `grids` - this is then transformed to the DVR by application of the
    # transformation matrices that result from diagonalising the coordinate operator.

    quad_grid = grd.get_quadrature_points(grids, q_mins, q_maxs, nbases, basis_func)
    v = pot.harmonic_potential_2d(quad_grid[:, 0], quad_grid[:, 1], kx=1.0, ky=1.0)
    calc = dvr.Calculator(sine_dvr)
    energies, wfs = calc.solve_nd(grids, masses, v, neig, nbases, ndim=2)
    print(energies)

    # compare to CM-DVR - here one does not have to diagonalise the coordinate operator first
    # and the number of basis functions isn't passed as an argument as it
    # is assumed to be the same as the number of grid points along each axis.
    calc = dvr.Calculator(colbert_miller)
    v = pot.harmonic_potential_2d(grids[0][:, None], grids[1][None, :], kx=1.0, ky=1.0)
    energies_cm, wfs_cm = calc.solve_nd(grids, masses, v, neig, ngrids, ndim=2)
    print(energies_cm)