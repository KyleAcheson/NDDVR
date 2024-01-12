import os
import sys
import numpy as np
sys.path.extend([os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '../../src'),
                 os.path.dirname(os.path.dirname(os.path.realpath(__file__)))])
import src.dvr as dvr
import src.potentials as potf
from src.synthesised_solvers import *
from src.exact_solvers import *
import src.wf_utils as wfu


def get_potential_files(pdir):
    all_files = os.listdir(pdir)
    file_paths = [os.path.join(pdir, file) for file in all_files if file.endswith('.tab') and 'grid_ngrid' not in file]
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


def test_algorithms(wdir, pdir, ptypes, grid_size, algorithms, masses, ndims, neig):

    for ptype in ptypes:
        pot_dir = f'{pdir}/{ptype}/{grid_size}'
        potential_files = get_potential_files(pot_dir)
        grid_file = get_grid_files(pot_dir)
        for i, file in enumerate(potential_files):
            out_dir = f'{wdir}/{ptype}/{grid_size}/P{i}/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            v, grids = potf.load_potential(file, grid_file, ndims, order='F')
            with open(f'{out_dir}/input_files.txt', 'w') as f:
                f.write(f'{file}\n')
                f.write(f'{grid_file}')
            calculator = dvr.Calculator(colbert_miller)
            exact_energies, exact_wfs = calculator.solve_nd(grids, masses, v, neig, ndim=ndims)
            write_energies(exact_energies, out_dir, 'cm_dvr')
            write_wfs(exact_wfs, neig, out_dir, 'cm_dvr')
            for algo_name, algorithm in algorithms.items():
                calculator.algorithm = algorithm
                ps_energies, ps_wfs = calculator.solve_nd(grids, masses, v, neig, ndim=ndims)
                ps_energies, ps_wfs = wfu.evaluate_energies(ps_wfs, grids, v, masses, neig, ndim=ndims, normalise=True)
                write_energies(ps_energies, out_dir, algo_name)
                write_wfs(ps_wfs, neig, out_dir, algo_name)


if __name__ == "__main__":

    pdir = '/storage/chem/msszxt/ND_Tests/potentials/harmonic'
    wdir = '/storage/chem/msszxt/ND_Tests/output/N10_rms_tfunc/simple'

    masses = [1, 1]
    ndims = 2
    neig = 3

    algorithms = {'A116': algorithm_116, 'A129': algorithm_129, 'A152': algorithm_152, 'A175': algorithm_175, 'A131': algorithm_131}
    ptypes = ['harmonic', 'anharmonic', 'morse', 'double_well', 'asym_double_well']
    grid_size = '31x31'

    test_algorithms(wdir, pdir, ptypes, grid_size, algorithms, masses, ndims, neig)
