import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import fast_dvr.dvr as dvr
import fast_dvr.potentials as pot
from fast_dvr.synthesised_solvers import *
from fast_dvr.exact_solvers import *
import fast_dvr.wf_utils as wfu

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
    calculator = dvr.Calculator(colbert_miller, use_operators=True)
    exact_energies, exact_wfs = calculator.solve_nd(q_grids, masses, v, neig, ndim=6)
    exact_energies *= AU2WAVNUM
    return exact_energies


def full_dvr(q_grids, v, neig, solver):
    masses = [1, 1, 1, 1, 1, 1]
    ngrid = len(q_grids[0])
    calculator = dvr.Calculator(solver, use_operators=True)
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


def run_dvr_1d(solver_name, in_dir, out_dir):
    ngrid = 101
    neig = 2
    #qmins = [-95, -55, -50, -22.5, -25, -25]
    #qmaxs = [60, 55, 55, 30, 25, 25]
    qmins = [-60, -40, -40, -20, -20, -20]
    qmaxs = [40, 40, 40, 20, 20, 20]

    variable_modes = [0, 1, 2, 3, 4, 5]
    tot_modes = len(variable_modes)
    transitions = np.zeros(tot_modes)
    output_dir = f'{out_dir}/ngrid_{ngrid}'
    input_dir = f'{in_dir}/ngrid_{ngrid}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(tot_modes):
        q = np.linspace(qmins[i], qmaxs[i], ngrid)
        v = np.genfromtxt(f'{input_dir}/nm{i}_potential.txt')
        calculator = dvr.Calculator(solver)
        energies, wfs = calculator.solve_1d(q, v, 1, neig)
        energies *= AU2WAVNUM
        transition_energy = energies[1] - energies[0]
        transitions[i] = transition_energy
    np.savetxt(f'{output_dir}/nm_energies.txt', transitions)


def dvr_exact_pot(solver_name, in_dir, out_dir):
    ngrids = [11]
    neig = 3
    #qmins = [-95, -55, -50, -22.5, -22.5, -20]
    #qmaxs = [60, 55, 55, 30, 22.5, 25]
    qmins = [-60, -40, -40, -20, -20, -20]
    qmaxs = [40, 40, 40, 20, 20, 20]
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
        input_dir = f'{in_dir}/ngrid_{ngrid}'
        t1 = time.time()
        v = np.genfromtxt(f'{input_dir}/exact_potential.txt')
        t2 = time.time()
        print('pot read time: ', t2-t1)
        output_dir = f'{out_dir}/ngrid_{ngrid}'
        t1 = time.time()
        te = run_full_dvr(output_dir, v, q_grids, solver_name, neig)
        t2 = time.time()
        print('dvr time: ', t2-t1)
        transitions_all[i, :] = te
    np.savetxt(f'{output_dir}/{solver_name}_exact_transitions.txt', transitions_all)
    print('done all calculations')


if __name__ == "__main__":
    import time
    #solver = 'A116'
    solver = colbert_miller
    #in_dir = '/home/kyle/DVR_Applications/NH3/inputs'
    in_dir = '/home/kyle/DVR_Applications/NH3/normal_modes/smaller_range/pots'
    #out_dir = '/home/kyle/DVR_Applications/NH3/results'
    out_dir = '/home/kyle/DVR_Applications/NH3/normal_modes/smaller_range/results'
    run_dvr_1d(solver, in_dir, out_dir)
    #dvr_exact_pot(solver, in_dir, out_dir)