import os
import sys

import matplotlib.pyplot as plt
import numpy as np
sys.path.extend([os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '..'),
                 os.path.dirname(os.path.dirname(os.path.realpath(__file__)))])

import src.dvr as dvr
import src.potentials as potf
from src.synthesised_solvers import *
from src.exact_solvers import *
import src.wf_utils as wfu
from natsort import natsorted
import timeit

def get_potential_files(pdir):
    all_files = os.listdir(pdir)
    file_paths = natsorted([os.path.join(pdir, file) for file in all_files if file.endswith('.tab') and 'grid_ngrid' not in file])
    return file_paths[0]


def get_grid_files(pdir):
    all_files = os.listdir(pdir)
    file_paths = [os.path.join(pdir, file) for file in all_files if file.endswith('.tab') and 'grid_ngrid' in file]
    return file_paths[0]


def run_cm_dvr(grids, masses, v, neig, ndims):
    calculator = dvr.Calculator(colbert_miller)
    exact_energies, exact_wfs = calculator.solve_nd(grids, masses, v, neig, ndim=ndims)


def run_algorithm(algorithm, grids, masses, v, neig, ndims):
    calculator = dvr.Calculator(algorithm, tridiag=True)
    ps_energies, ps_wfs = calculator.solve_nd(grids, masses, v, neig, ndim=ndims)
    ps_energies, ps_wfs = wfu.evaluate_energies(ps_wfs, grids, v, masses, neig, ndim=ndims, normalise=True)


def time_algorithms(pdir, wdir, grid_sizes, algorithms, masses, ndims, neig, nruns=5):

    cm_dvr_times = []
    ps_algorithm_times = {k: [] for k in algorithms.keys()}
    for grid_size in grid_sizes:
        pot_dir = f'{pdir}/{grid_size}'
        potential_file = get_potential_files(pot_dir)
        grid_file = get_grid_files(pot_dir)
        out_dir = f'{wdir}/{grid_size}'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        v, grids = potf.load_potential(potential_file, grid_file, ndims, order='F')
        with open(f'{out_dir}/input_files.txt', 'w') as f:
            f.write(f'{potential_file}\n')
            f.write(f'{grid_file}')
            exact_times = timeit.repeat(lambda: run_cm_dvr(grids, masses, v, neig, ndims), repeat=nruns, number=1)
            best_exact_time = min(exact_times)
            cm_dvr_times.append(best_exact_time)
            for algo_name, algorithm in algorithms.items():
                algorithm_times = timeit.repeat(lambda: run_algorithm(algorithm, grids, masses, v, neig, ndims), repeat=nruns, number=1)
                best_algorithm_time = min(algorithm_times)
                ps_algorithm_times[algo_name].append(best_algorithm_time)


    grids = np.arange(len(grid_sizes))
    fig, ax = plt.subplots()
    ax.plot(grids, cm_dvr_times, '-o', label='CM-DVR')
    cm_dvr_times = [str(i) for i in cm_dvr_times]
    with open(f'{wdir}/cm_dvr_timings.tab', 'w+') as f:
        f.write(', '.join(cm_dvr_times))

    for algo_name in algorithms.keys():
        algo_times = ps_algorithm_times[algo_name]
        ax.plot(grids, algo_times, '-o', label=f'{algo_name}')
        algo_times = [str(i) for i in algo_times]
        with open(f'{wdir}/{algo_name}_timings.tab', 'w+') as f:
            f.write(', '.join(algo_times))


    ax.set_xlabel('$N_{\mathrm{g}}$')
    ax.set_ylabel('$t (s)$')
    ax.set_xticks(grids, grid_sizes)
    ax.legend(loc='upper left', ncols=2, frameon=False)
    fig.savefig(f'{wdir}/algorithm_timings.png')
    fig.clear()
    plt.close(fig)


if __name__ == "__main__":

    pdir = '/storage/chem/msszxt/ND_Tests/potentials/3D/harmonic'
    wdir = '/storage/chem/msszxt/ND_Tests/output/3D/N10_rms_tfunc/simple/timings'
    #pdir = '/home/kyle/PycharmProjects/Potential_Generator/potentials/3D'
    #wdir = '/home/kyle/PycharmProjects/NDDVR/examples/3D_tests/outputs'


    grid_sizes = ['21x21x21', '31x31x31', '41x41x41', '51x51x51', '61x61x61',
                  '71x71x71', '81x81x81', '91x91x91', '101x101x101']

    masses = [1, 1, 1]
    ndims = 3
    neig = 3
    nruns = 2

    algorithms = rms_tfunc_N10_algorithms
    #algorithms = {'131': algorithm_131}

    time_algorithms(pdir, wdir, grid_sizes, algorithms, masses, ndims, neig, nruns)
