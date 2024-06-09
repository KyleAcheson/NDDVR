import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import fast_dvr.dvr as dvr
import fast_dvr.potentials as potf
from fast_dvr.synthesised_solvers import *
from fast_dvr.exact_solvers import *
import fast_dvr.wf_utils as wfu
from natsort import natsorted
import timeit


def run_cm_dvr(grids, masses, v, neig, ndims, ops):
    calculator = dvr.Calculator(colbert_miller, use_operators=ops)
    exact_energies, exact_wfs = calculator.solve_nd(grids, masses, v, neig, ndim=ndims)


def run_algorithm(algorithm, grids, masses, v, neig, ndims, ops):
    calculator = dvr.Calculator(algorithm, use_operators=ops)
    ps_energies, ps_wfs = calculator.solve_nd(grids, masses, v, neig, ndim=ndims)
    ps_energies, ps_wfs = wfu.evaluate_energies(ps_wfs, grids, v, masses, neig, ndim=ndims, normalise=True)

def get_grid(xmin, xmax, grid_size, ndim):
    grids = []
    for i in range(ndim):
        g = np.linspace(xmin, xmax, grid_size)
        grids.append(g)
    grids = tuple(grids)
    nd_grids = np.meshgrid(*grids)
    return nd_grids, grids

def ho_5d(x1, x2, x3, x4, x5):
    ng = len(x1)
    v = np.zeros((ng, ng, ng, ng, ng))
    for i in range(ng):
        for j in range(ng):
            for k in range(ng):
                for m in range(ng):
                    for n in range(ng):
                        v[i, j, k, m, n] = 0.5 * (x1[i]**2 + x2[j]**2 + x3[k]**2 + x4[m]**2 + x5[n]**2)
    return v



def ho_6d(x1, x2, x3, x4, x5, x6):
    ng = len(x1)
    v = np.zeros((ng, ng, ng, ng, ng, ng))
    for i in range(ng):
        for j in range(ng):
            for k in range(ng):
                for m in range(ng):
                    for n in range(ng):
                        for l in range(ng):
                            v[i, j, k, m, n, l] = 0.5 * (x1[i]**2 + x2[j]**2 + x3[k]**2 + x4[m]**2 + x5[n]**2 + x6[l]**2)
    return v


def ho_7d(x1, x2, x3, x4, x5, x6, x7):
    ng = len(x1)
    v = np.zeros((ng, ng, ng, ng, ng, ng, ng))
    for i in range(ng):
        for j in range(ng):
            for k in range(ng):
                for m in range(ng):
                    for n in range(ng):
                        for l in range(ng):
                            for a in range(ng):
                                v[i, j, k, m, n, l, a] = 0.5 * (x1[i]**2 + x2[j]**2 + x3[k]**2 + x4[m]**2 + x5[n]**2 + x6[l]**2 + x7[a]**2)
    return v


def get_potential(ks, xmin, xmax, grid_size, ndim):
    if ndim < 5:
        grids, grids_1d = get_grid(xmin, xmax, grid_size, ndim)
        v = potf.harmonic_potential_nd(grids, ks)
    elif ndim == 5:
        grids_1d = []
        for i in range(ndim):
            grids_1d.append(np.linspace(xmin, xmax, grid_size))
        v = ho_5d(*tuple(grids_1d))
    elif ndim == 6:
        grids_1d = []
        for i in range(ndim):
            grids_1d.append(np.linspace(xmin, xmax, grid_size))
        v = ho_6d(*tuple(grids_1d))
    elif ndim == 7:
        grids_1d = []
        for i in range(ndim):
            grids_1d.append(np.linspace(xmin, xmax, grid_size))
        v = ho_7d(*tuple(grids_1d))
    return v, grids_1d

def time_ps_algorithms(wdir, xmin, xmax, grid_size, algorithms, ndims, neig, operators=False, nruns=3):
    num_dims = len(ndims)
    for i in range(num_dims):
        ndim = ndims[i]
        masses = np.ones(ndim)
        ks = np.ones(ndim)
        out_dir = f'{wdir}/D{ndim}'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        v, grids = get_potential(ks, xmin, xmax, grid_size, ndim)
        for algo_name, algorithm in algorithms.items():
            algorithm_times = timeit.repeat(lambda: run_algorithm(algorithm, grids, masses, v, neig, ndim, operators), repeat=nruns, number=1)
            np.savetxt(f'{out_dir}/{algo_name}_timings.tab', algorithm_times)


def time_cmdvr(wdir, xmin, xmax, grid_size, ndims, neig, operators=False, nruns=3):

    num_dims = len(ndims)
    for i in range(num_dims):
        ndim = ndims[i]
        masses = np.ones(ndim)
        ks = np.ones(ndim)
        out_dir = f'{wdir}/D{ndim}'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        v, grids = get_potential(ks, xmin, xmax, grid_size, ndim)
        exact_times = timeit.repeat(lambda: run_cm_dvr(grids, masses, v, neig, ndim, operators), repeat=nruns, number=1)
        np.savetxt(f'{out_dir}/cm_dvr_timings.tab', exact_times)




if __name__ == "__main__":

    wdir = '/storage/chem/msszxt/ND_Tests/output/ND_timings/matvec/timings/ngrid_21'

    CM_DVR = True
    op = True
    ndims = [4]

    nruns = 20
    neig = 3
    xmin, xmax = -5, 5
    grid_size = 21

    algorithms = {'A116': algorithm_116}

    time_ps_algorithms(wdir, xmin, xmax, grid_size, algorithms, ndims, neig, nruns=nruns, operators=op)
    if CM_DVR:
        time_cmdvr(wdir, xmin, xmax, grid_size, ndims, neig, nruns=nruns, operators=op)
