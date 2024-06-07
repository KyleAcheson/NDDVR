import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import src.dvr as dvr
import src.potentials as potf
from src.synthesised_solvers import *
from src.exact_solvers import *
import src.wf_utils as wfu
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

def get_potential(ks, xmin, xmax, grid_size, ndim):
    grids, grids_1d = get_grid(xmin, xmax, grid_size, ndim)
    v = potf.harmonic_potential_nd(grids, ks)
    return v, grids_1d

def time_ps_algorithms(wdir, xmin, xmax, grid_size, algorithms, ndims, neig, operators=False, nruns=3):
    ps_algorithm_times = {k: [] for k in algorithms.keys()}
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
            best_algorithm_time = min(algorithm_times)
            ps_algorithm_times[algo_name].append(best_algorithm_time)

    for algo_name in algorithms.keys():
        algo_times = ps_algorithm_times[algo_name]
        algo_times = [str(i) for i in algo_times]
        with open(f'{wdir}/{algo_name}_timings.tab', 'w+') as f:
            f.write(', '.join(algo_times))
            
            
def time_cmdvr(wdir, xmin, xmax, grid_size, ndims, neig, operators=False, nruns=3):

    cm_dvr_times = []
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
        best_exact_time = min(exact_times)
        cm_dvr_times.append(best_exact_time)

    cm_dvr_times = [str(i) for i in cm_dvr_times]
    with open(f'{wdir}/cm_dvr_timings.tab', 'w+') as f:
        f.write(', '.join(cm_dvr_times))



if __name__ == "__main__":

    wdir = '/storage/chem/msszxt/ND_Tests/output/ND/N10_rms_tfunc/timings'
    wdir = '/home/kyle/PycharmProjects/NDDVR/examples/ND_tests/timings'

    CM_DVR = True
    op = False
    ndims = [2, 3, 4]

    nruns = 2
    neig = 3
    xmin, xmax = -5, 5
    grid_size = 31

    algorithms = {'A116': algorithm_116}

    time_ps_algorithms(wdir, xmin, xmax, grid_size, algorithms, ndims, neig, nruns=nruns, operators=op)
    if CM_DVR:
        time_cmdvr(wdir, xmin, xmax, grid_size, ndims, neig, nruns=nruns, operators=op)
