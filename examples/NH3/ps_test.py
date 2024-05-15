import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.extend([os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '../src'),
                 os.path.dirname(os.path.dirname(os.path.realpath(__file__)))])
import src.dvr as dvr
import src.potentials as pot
from src.synthesised_solvers import *
from src.exact_solvers import *
import src.wf_utils as wfu

BOHR = 0.529177
AU2WAVNUM = 219474.63

algorithms = {'A116': algorithm_116}


def plot_transition_convergence(wdir, solver_name, ngrids, ntransitions, exact=True):
    tot_grids = len(ngrids)
    exp_transitions = np.genfromtxt('./results/exp_transitions.txt')[:ntransitions]
    all_errors = np.zeros((tot_grids, ntransitions))
    for i, ngrid in enumerate(ngrids):
        if exact:
            out_dir = f'{wdir}/ngrid_{ngrid}'
        else:
            out_dir = f'{wdir}/2pow{ngrid}'
        transitions = np.genfromtxt(f'{out_dir}/{solver_name}_transitions.txt')
        frac_error = np.abs(transitions - exp_transitions)
        all_errors[i, :] = frac_error
    fig = plt.figure()
    plt.plot(ngrids, all_errors)
    plt.xlim([ngrids[0], ngrids[-1]])
    if exact:
        plt.xlabel('$N_{\mathrm{grid}}$')
    else:
        plt.xlabel('$N_{\mathrm{grid}} (2^n)$')
    plt.ylabel('$\Delta v$')
    fig.savefig(f'{wdir}/{solver_name}_convergence.png')


def full_cmdvr(q_grids, v, neig):
    masses = [1, 1, 1, 1, 1, 1]
    calculator = dvr.Calculator(colbert_miller)
    exact_energies, exact_wfs = calculator.solve_nd(q_grids, masses, v, neig, ndim=6)
    exact_energies *= AU2WAVNUM
    return exact_energies


def full_dvr(q_grids, v, neig, solver):
    masses = [1, 1, 1, 1, 1, 1]
    ngrid = len(q_grids[0])
    calculator = dvr.Calculator(solver)
    exact_energies, exact_wfs = calculator.solve_nd(q_grids, masses, v, neig, ndim=6)
    v = v.reshape(ngrid, ngrid, ngrid, ngrid, ngrid, ngrid)
    exact_energies, exact_wfs = wfu.evaluate_energies(exact_wfs, q_grids, v, masses, neig, ndim=6, normalise=True)
    exact_energies *= AU2WAVNUM
    return exact_energies


def run_full_dvr(wdir, v, q_grids, solver_name, neig):
    """ run DVR on the whole 3D H2O potential in normal coordinates. """
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    if solver_name == 'cm_dvr':
        exact_energies = full_cmdvr(q_grids, v, neig)
    else:
        solver = algorithms[solver_name]
        exact_energies = full_dvr(q_grids, v, neig, solver)
    np.savetxt(f'{wdir}/{solver_name}_energies.txt', exact_energies)
    tranistion_energies = exact_energies - exact_energies[0]
    tranistion_energies = tranistion_energies[1:]
    np.savetxt(f'{wdir}/{solver_name}_transitions.txt', tranistion_energies)
    return tranistion_energies


def dvr_exact_pot(solver_name):
    ngrids = [15]
    neig = 20
    qmins = [-95, -55, -50, -22.5, -22.5, -20]
    qmaxs = [60, 55, 55, 30, 22.5, 25]
    variable_modes = [0, 1, 2, 3, 4, 5]

    tot_grids = len(ngrids)
    transitions_all = np.zeros((tot_grids, neig-1))
    for i, ngrid in enumerate(ngrids):
        q0 = np.linspace(qmins[0], qmaxs[0], ngrid)
        q1 = np.linspace(qmins[1], qmaxs[1], ngrid)
        q2 = np.linspace(qmins[2], qmaxs[2], ngrid)
        q3 = np.linspace(qmins[3], qmaxs[3], ngrid)
        q4 = np.linspace(qmins[4], qmaxs[4], ngrid)
        q5 = np.linspace(qmins[5], qmaxs[5], ngrid)
        q_grids = [q0, q1, q2, q3, q4, q5]
        input_dir = f'./inputs/ngrid_{ngrid}'
        t1 = time.time()
        v = np.genfromtxt(f'{input_dir}/exact_potential.txt')
        t2 = time.time()
        print('pot read time: ', t2-t1)
        out_dir = f'./results/exact_pot/ngrid_{ngrid}'
        te = run_full_dvr(out_dir, v, q_grids, solver_name, neig)
        transitions_all[i, :] = te
    np.savetxt(f'./results/exact_pot/{solver_name}_exact_transitions.txt', transitions_all)
    print('done all calculations')
    breakpoint()


if __name__ == "__main__":
    import time
    solver = 'A116'
    #solver = 'cm_dvr'
    dvr_exact_pot(solver)