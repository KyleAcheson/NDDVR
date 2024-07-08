import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import fast_dvr.dvr as dvr
import fast_dvr.potentials as pot
from fast_dvr.synthesised_solvers import *
from fast_dvr.exact_solvers import *
import fast_dvr.wf_utils as wfu
import fast_dvr.grids as grids

BOHR = 0.529177
AU2WAVNUM = 219474.63


if __name__ == "__main__":

    ndims = 2
    ngrids = [41, 41]
    nbases = [25, 25]
    q_mins = [-5, -5]
    q_maxs = [5, 5]

    basis_func = ('sine', get_pib_basis)

    grids = []
    for i in range(ndims):
        grids.append(np.linspace(q_mins[i], q_maxs[i], ngrids[i]))

    quad_grid = grids.get_quadrature_points(grids, q_mins, q_maxs, nbases, basis_func)
    v = pot.harmonic_potential_2d(quad_grid[:, 0], quad_grid[:, 1], kx=1.0, ky=1.0)
    breakpoint()