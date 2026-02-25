import os
import sys

import matplotlib.pyplot as plt
import numpy as np
sys.path.extend([os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '..'),
                 os.path.dirname(os.path.dirname(os.path.realpath(__file__)))])

import fast_dvr.dvr as dvr
import fast_dvr.potentials as potf
from fast_dvr.synthesised_solvers import *
from fast_dvr.exact_solvers import *
import fast_dvr.wf_utils as wfu
from natsort import natsorted


AU2WAVNUM = 219474.63
def get_potential_files(pdir):
    all_files = os.listdir(pdir)
    file_paths = natsorted([os.path.join(pdir, file) for file in all_files if file.endswith('.tab') and 'grid_ngrid' not in file])
    return file_paths


def get_grid_files(pdir):
    all_files = os.listdir(pdir)
    file_paths = [os.path.join(pdir, file) for file in all_files if file.endswith('.tab') and 'grid_ngrid' in file]
    return file_paths[0]


def write_energies(energies, out_dir, algorithm, format='%10.12f'):
    fname = f'{out_dir}/{algorithm}_energies.dat'
    with open(fname, 'a') as f:
        np.savetxt(f, energies, fmt=format)


def write_wfs(wfs, neig, out_dir, algorithm, format='%10.12f'):
    fname = f'{out_dir}/{algorithm}_wfs.dat'
    labels = [f'eig{eig}' for eig in range(neig)]
    labels = '\t'.join(labels)

    for eig in range(neig):
        s = np.average(wfs[:, eig])
        wfs[:, eig] = wfs[:, eig] * np.sign(s)

    with open(fname, 'a') as f:
        np.savetxt(f, wfs, fmt=format, header=labels)


def test_algorithms(wdir, pdir, x, algorithms, neig):

    ngrid = len(x)
    mass = 1
    potential_files = get_potential_files(pdir)
    for i, file in enumerate(potential_files):
        out_dir = f'{wdir}/P{i}/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        v = np.genfromtxt(file, skip_header=1)
        calculator = dvr.Calculator(colbert_miller)
        exact_energies, exact_wfs = calculator.solve_1d(x, v, mass, neig)
        exact_energies -= exact_energies[0]
        write_energies(exact_energies, out_dir, 'cm_dvr')
        write_wfs(exact_wfs, neig, out_dir, 'cm_dvr')
        for algo_name, algorithm in algorithms.items():
            calculator.algorithm = algorithm
            ps_energies, ps_wfs = calculator.solve_1d(x, v, mass, neig)
            ps_energies, ps_wfs = wfu.evaluate_energies(ps_wfs, x, v, mass, neig, normalise=True)
            ps_energies -= ps_energies[0]
            write_energies(ps_energies, out_dir, algo_name)
            write_wfs(ps_wfs, neig, out_dir, algo_name)


if __name__ == "__main__":

    # these paths will need editing
    pdir = '/home/kyle/PycharmProjects/NDDVR/data/potentials/tunnelling_test/double_well'
    wdir = '/home/kyle/PycharmProjects/NDDVR/examples/1D_tests/outputs'

    neig = 4

    x = np.linspace(-5, 5, 501)
    algorithms = var_N10_algorithms

    test_algorithms(wdir, pdir, x, algorithms, neig)