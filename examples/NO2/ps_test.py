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
from sklearn.gaussian_process.kernels import RBF

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

def fit_gpr(v, qmins, qmaxs, ngrids_train, ngrids_interp, **kwargs):

    ndof = len(qmins)
    q_train = grids.direct_product_grid(qmins, qmaxs, ngrids_train, ndof=ndof)
    q_pred = grids.direct_product_grid(qmins, qmaxs, ngrids_interp, ndof=ndof)
    kernel = RBF(length_scale=kwargs['length_scale'], length_scale_bounds=kwargs['length_scale_bounds'])
    v_pred, v_std, gp = pot.fit_potential(q_train, q_pred, v, ndof, kernel)
    print(f'opt. length scale: {gp.kernel_.length_scale}')
    print(f'max std. dev.: {np.max(v_std)}')
    return v_pred


def run_full_dvr(wdir, v, qmins, qmaxs, ngrids, nbases, neig, solver_name, use_ops=True):
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


    pot_dir = f'/home/kyle/DVR_Applications/NO2/ND_dvr'
    out_dir = f'/home/kyle/DVR_Applications/NO2/ND_dvr/results/gpr'

    solver_name = 'cm_dvr'
    use_ops = True
    do_gpr = False

    neig = 9
    ngrids = np.array([41, 31, 31])
    nbases = np.array([41, 31, 31])
    nbases_pred = np.array([51, 31, 31])
    q_mins = np.array([-70, -45, -30])
    q_maxs = np.array([70, 35, 30])

    length_scale = 1
    #length_scale_bounds = (2.0, 25.0)
    length_scale_bounds = "fixed"

    natoms = 3
    variable_modes = np.array([0, 1, 2])
    labels = ['N', 'O', 'O']
    masses = np.array([14.007, 15.999, 15.999])

    # def2-tzvp
    eq_coords = np.array([[0.001011, 0.002440, 0.00],
                          [1.191918, -0.000971, 0.00],
                          [-0.830153, 0.855256, 0.00]])

    eq_coords *= (1 / BOHR)
    ngrid_prod = np.prod(nbases)
    v = np.genfromtxt(f'{pot_dir}/ngrid_{ngrid_prod}/energies_raw.txt')

    if do_gpr:
        v = fit_gpr(v, q_mins, q_maxs, nbases, nbases_pred, length_scale=length_scale, length_scale_bounds=length_scale_bounds)
        nbases = nbases_pred
        ngrids = nbases_pred

    run_full_dvr(out_dir, v, q_mins, q_maxs, ngrids, nbases, neig, solver_name, use_ops)