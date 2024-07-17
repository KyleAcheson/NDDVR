import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import fast_dvr.dvr as dvr
import fast_dvr.potentials as pot
from fast_dvr.synthesised_solvers import *
from fast_dvr.exact_solvers import *
import fast_dvr.wf_utils as wfu
import fast_dvr.transforms as tf
import fast_dvr.grids as grids
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF
from pyscf import gto, hessian, dft
from pyscf.hessian.thermo import harmonic_analysis

ELEC_MASS = 9.1093837E-31
AU2EV = 27.2114
AU2WAVNUM = 219474.63

BOHR = 0.52917721092  # Angstroms
BOHR_SI = BOHR * 1e-10
ATOMIC_MASS = 9.109E-31
HARTREE2J = 4.359744650e-18
HARTREE2EV = 27.21138602
LIGHT_SPEED_SI = 299792458

AU2Hz = ((HARTREE2J / (ATOMIC_MASS * BOHR_SI ** 2)) ** 0.5 / (2 * np.pi))


def write_to_xyz(coords, atom_labels, fname):
    natoms, _, npoints = coords.shape
    with open(fname, 'a+') as f:
        for i in range(npoints):
            coord = coords[:, :, i] * BOHR
            f.write(f'{natoms}\n')
            f.write(f'grid point {i + 1}\n')
            array_string = '\n'.join(
                [f'{atom_labels[ind]}\t' + '\t'.join(map(str, row)) for ind, row in enumerate(coord)]) + '\n'
            f.write(array_string)


def run_full_dvr(pot_dir, wdir, qmins, qmaxs, ngrids, nbases, neig, solver_name, use_ops=True):
    solvers = {'cm_dvr': colbert_miller, 'sine_dvr': sine_dvr, 'A116': algorithm_116}
    solver = solvers.get(solver_name)
    if solver_name == 'sine_dvr':
        ngrid_prod = np.prod(nbases)
        pot_size = tuple(nbases)
    else:
        ngrid_prod = np.prod(ngrids)
        pot_size = tuple(ngrids)
        nbases = None

    masses = np.array([1, 1, 1])
    ndims = 3
    v = np.genfromtxt(f'{pot_dir}/ngrid_{ngrid_prod}/exact_potential.txt')
    if not os.path.exists(wdir):
        os.makedirs(wdir)

    q_grids = []
    for i in range(ndims):
        q_grids.append(np.linspace(qmins[i], qmaxs[i], ngrids[i]))
    calc = dvr.Calculator(solver, use_operators=use_ops)
    exact_energies, wfs = calc.solve_nd(q_grids, masses, v, neig, nbases, ndim=ndims)
    if solver_name != 'cm_dvr' and solver_name != 'sine_dvr':
        v = v.reshape(*pot_size)
        exact_energies, exact_wfs = wfu.evaluate_energies(wfs, q_grids, v, masses, neig, ndim=ndims, normalise=True)

    np.savetxt(f'{wdir}/{solver_name}_energies.txt', exact_energies)
    tranistion_energies = exact_energies - exact_energies[0]
    tranistion_energies = tranistion_energies[1:] * AU2WAVNUM
    np.savetxt(f'{wdir}/{solver_name}_transitions.txt', tranistion_energies)
    return tranistion_energies


if __name__ == "__main__":


    pot_dir = f'/home/kyle/DVR_Applications/NH3/sine_dvr/whole_pot/'
    out_dir = f'/home/kyle/DVR_Applications/NH3/sine_dvr/results/'

    neig = 9
    solver_name = 'cm_dvr'
    use_ops = True
    variable_modes = np.array([0, 1, 2])
    ngrids = np.array([41, 31, 31])
    nbases = np.array([41, 31, 31])
    q_mins = np.array([-80, -50, -40])
    q_maxs = np.array([80, 40, 40])
    
    natoms = 3
    labels = ['S', 'O', 'O']
    masses = np.array([32.065, 15.999, 15.999])

    eq_coords = np.array([[-0.009001, -0.015486, 0.00],
                          [1.454979, -0.003042, 0.00],
                          [-0.719129, 1.264878, 0.00]])

    eq_coords *= (1 / BOHR)
    ngrid_prod = np.prod(ngrids)

    run_full_dvr(pot_dir, out_dir, q_mins, q_maxs, ngrids, nbases, neig, solver_name, use_ops)