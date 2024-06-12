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


def test_algorithms(wdir, pdir, grids, algorithms, masses, ndims, neig):

    v = np.genfromtxt(f'{pdir}/energies_cut_ev.txt')
    #v = e - np.min(e)
    #v *= 219474.63
    v /= 27.2114
    print(np.min(v), np.max(v))
    v = v.reshape(41, 31, 31)
    out_dir = f'{wdir}'
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

def plot_results(wdir, pdir, ptypes, algorithms, neig, conv_thresh):

    nalgorithms = len(algorithms.keys())
    for ptype in ptypes:
        parent_dir = f'{pdir}/{ptype}'
        out_dir = f'{wdir}/{ptype}'
        subdirectories = natsorted([d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))])
        grid_labels = []
        ngrids = len(subdirectories)
        grids = np.arange(ngrids)
        ps_energies_all = []
        dvr_energies_all = []
        for subdir in subdirectories:
            grid_dir = f'{out_dir}/{subdir}'
            pot_dir = f'{parent_dir}/{subdir}'
            grid_labels.append(subdir)
            potential_files = get_potential_files(pot_dir)
            npots = len(potential_files)
            ps_energies = np.zeros((nalgorithms, npots, neig))
            dvr_energies = np.zeros((npots, neig))
            for i, potential in enumerate(potential_files):
                problem_dir = f'{grid_dir}/P{i}'
                exact_file = f'{problem_dir}/cm_dvr_energies.dat'
                dvr_energies[i, :] = np.genfromtxt(exact_file)[:neig]
                for j, algorithm in enumerate(algorithms.keys()):
                    ps_file = f'{problem_dir}/{algorithm}_energies.dat'
                    ps_energies[j, i, :] = np.genfromtxt(ps_file)[:neig]
            ps_energies_all.append(ps_energies)
            dvr_energies_all.append(dvr_energies)
        dvr_energies_all = np.asarray(dvr_energies_all)
        ps_energies_all = np.asarray(ps_energies_all)

        best_algorithms = []
        converged_algorithms = {k: [] for k in algorithms.keys()}
        for eig in range(neig):
            for j, algorithm in enumerate(algorithms.keys()):
                fig1, ax1 = plt.subplots()
                fig2, ax2 = plt.subplots()
                conv_pot = []
                for i in range(npots):
                    ref = dvr_energies_all[-1, i, eig]
                    dvr = dvr_energies_all[:, i, eig]
                    ps = ps_energies_all[:, j, i, eig]
                    dE_ps = (ps - ref) / ref
                    if np.abs(dE_ps[-1]) <= conv_thresh:
                        conv_pot.append(True)
                    else:
                        conv_pot.append(False)
                    dE_dvr = (dvr - ref) / ref
                    ax1.plot(grids, dE_ps, '-o')
                    ax2.plot(grids, dE_dvr, '-o')
                ax1.set_xlabel('$N_{\mathrm{g}}$')
                ax1.set_ylabel('$\Delta E_{g}$')
                ax2.set_xlabel('$N_{\mathrm{g}}$')
                ax2.set_ylabel('$\Delta E_{g}$')
                ax1.set_xticks(grids, grid_labels)
                ax2.set_xticks(grids, grid_labels)
                fig1.savefig(f'{out_dir}/{algorithm}_grid_conv_{ptype}_neig{eig}.png')
                fig2.savefig(f'{out_dir}/exact_grid_conv_{ptype}_neig{eig}.png')
                fig1.clear()
                plt.close(fig1)
                fig2.clear()
                plt.close(fig2)

                if all(conv_pot):
                    converged_algorithms[algorithm].append(True)
                else:
                    converged_algorithms[algorithm].append(False)
        for algorithm in algorithms.keys():
            if all(converged_algorithms[algorithm]):
                best_algorithms.append(algorithm)

        with open(f'{out_dir}/converged_algorithms_{ptype}.txt', 'w+') as f:
            f.write(', '.join(best_algorithms))






if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-g', dest="grid_size", required=False, type=str)
    parser.add_argument('-p', dest='plot', required=False, type=bool)

    args = parser.parse_args()

    pdir = '/home/kyle/PycharmProjects/NDDVR/examples/H2O'
    wdir = '/home/kyle/PycharmProjects/NDDVR/examples/H2O/outputs'

    masses = [1, 1, 1]
    ndims = 3
    neig = 3
    conv_thresh = 0.01

    qgrid1 = np.linspace(-1, 1, 41)
    qgrid2 = np.linspace(-1, 0.5, 31)
    qgrid3 = np.linspace(-0.75, 0.75, 31)
    grids = [qgrid1, qgrid2, qgrid3]

    algorithms = rms_tfunc_N10_algorithms

    grid_size = args.grid_size

    test_algorithms(wdir, pdir, grids, algorithms, masses, ndims, neig)
