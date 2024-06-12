import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.extend([os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '../../fast_dvr'),
                 os.path.dirname(os.path.dirname(os.path.realpath(__file__)))])
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
    masses = [1, 1, 1]
    calculator = dvr.Calculator(colbert_miller)
    exact_energies, exact_wfs = calculator.solve_nd(q_grids, masses, v, neig, ndim=3)
    exact_energies *= AU2WAVNUM
    return exact_energies


def full_dvr(q_grids, v, neig, solver):
    masses = [1, 1, 1]
    ngrid = len(q_grids[0])
    calculator = dvr.Calculator(solver)
    exact_energies, exact_wfs = calculator.solve_nd(q_grids, masses, v, neig, ndim=3)
    v = v.reshape(ngrid, ngrid, ngrid)
    exact_energies, exact_wfs = wfu.evaluate_energies(exact_wfs, q_grids, v, masses, neig, ndim=3, normalise=True)
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
    ngrids = [21, 31, 41, 51, 61, 71]
    neig = 20
    qmins = [-50, -30, -25]
    qmaxs = [50, 20, 25]
    #qmins = [-80, -50, -30]
    #qmaxs = [70, 25, 30]

    tot_grids = len(ngrids)
    transitions_all = np.zeros((tot_grids, neig-1))
    transitions_error = np.zeros((tot_grids, neig-1))
    exp_transitions = np.genfromtxt('./results/exp_transitions.txt')[:neig-1]
    for i, ngrid in enumerate(ngrids):
        q0 = np.linspace(qmins[0], qmaxs[0], ngrid)
        q1 = np.linspace(qmins[1], qmaxs[1], ngrid)
        q2 = np.linspace(qmins[2], qmaxs[2], ngrid)
        q_grids = [q0, q1, q2]
        input_dir = f'./inputs/ngrid_{ngrid}'
        v = np.genfromtxt(f'{input_dir}/exact_potential.txt')
        out_dir = f'./results/exact_pot/ngrid_{ngrid}'
        te = run_full_dvr(out_dir, v, q_grids, solver_name, neig)
        transitions_all[i, :] = te
        transitions_error[i, :] = np.abs(te - exp_transitions)
    np.savetxt(f'./results/exact_pot/{solver_name}_exact_transitions.txt', transitions_all)
    np.savetxt(f'./results/exact_pot/{solver_name}_transitions_error.txt', transitions_error, fmt='%.2f')
    plot_transition_convergence('./results/exact_pot', solver_name, ngrids, neig-1)
    print('done all calculations')
    breakpoint()


def dvr_gpr_pot(solver_name, ngrid):
    neig = 14
    exponents = [8, 9, 10, 11]
    qmins = [-50, -30, -25]
    qmaxs = [50, 20, 25]

    tot_grids = len(exponents)
    transitions_all = np.zeros((tot_grids, neig-1))
    transitions_error = np.zeros((tot_grids, neig-1))
    exp_transitions = np.genfromtxt('./results/exp_transitions.txt')[:neig-1]
    for i, exponent in enumerate(exponents):
        q0 = np.linspace(qmins[0], qmaxs[0], ngrid)
        q1 = np.linspace(qmins[1], qmaxs[1], ngrid)
        q2 = np.linspace(qmins[2], qmaxs[2], ngrid)
        q_grids = [q0, q1, q2]
        input_dir = f'./inputs/gpr/exact_ngrid_{ngrid}/2pow{exponent}'
        v = np.genfromtxt(f'{input_dir}/predicted_potential.txt')
        out_dir = f'./results/gpr_pot/ngrid_{ngrid}/2pow{exponent}'
        te = run_full_dvr(out_dir, v, q_grids, solver_name, neig)
        transitions_all[i, :] = te
        transitions_error[i, :] = np.abs(te - exp_transitions)
    np.savetxt(f'./results/gpr_pot/ngrid_{ngrid}/{solver_name}_pred_transitions.txt', transitions_all)
    np.savetxt(f'./results/gpr_pot/ngrid_{ngrid}/{solver_name}_transitions_error.txt', transitions_error, '%.2f')
    plot_transition_convergence(f'./results/gpr_pot/ngrid_{ngrid}', solver_name, exponents, neig-1, exact=False)
    print('done all calculations')
    breakpoint()


if __name__ == "__main__":
    #solver = 'A116'
    solver = 'cm_dvr'
    #dvr_exact_pot(solver)
    dvr_gpr_pot(solver, ngrid=21)