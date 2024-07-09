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
            f.write(f'grid point {i+1}\n')
            array_string = '\n'.join([f'{atom_labels[ind]}\t' + '\t'.join(map(str, row)) for ind, row in enumerate(coord)]) + '\n'
            f.write(array_string)


def slice_1d(variable_modes, qmins, qmaxs, ngrid_prod, nbasis):

    basis_func = ('sine', get_pib_basis)
    masses = np.array([25527.03399, 1833.3516, 1833.3516, 1833.3516])
    eq_coords = np.array([[0.0, 0.0, 0.0],
                          [1.0128, 0.0, 0.0],
                          [-0.296621, 0.968390, 0.0],
                          [-0.296621, -0.401080, -0.881427]])

    dh3_coords = np.array([[0.0, 0.0, 0.0],
                          [0.9988, 0.0, 0.0],
                          [-0.499400, 0.864986, 0.0],
                          [-0.499400, -0.864986, 0.0]])


    eq_coords *= (1 / BOHR)
    ndof = len(variable_modes)
    variable_modes = np.array(variable_modes)
    inds = variable_modes + 6

    dh3_coords *= (1 / BOHR)
    hessian = pot.ammpot4_hessian(dh3_minima=True)
    freqs_wavenums, norm_modes, tmat = tf.get_normal_modes(hessian, masses, 6)
    tmat[:, [0, 6]] = tmat[:, [6, 0]]

    grid = [np.linspace(qmins[0], qmaxs[0], ngrid_prod[0])]
    quad_grid = grids.get_quadrature_points(grid, qmins, qmaxs, nbasis, basis_func)
    q_prod = np.zeros((nbasis[0], 12))
    q_prod[:, inds[0]] = quad_grid

    cart_coords_prod = tf.norm2cart_grid(q_prod, dh3_coords, masses, tmat)
    v_prod = pot.ammpot4_cart(cart_coords_prod)
    q_prod = q_prod[:, inds]
    return cart_coords_prod, q_prod, v_prod


def run_1d_test(wdir):

    ngrids = [41, 41, 41, 41, 41, 41]
    nbases = [25, 25, 25, 25, 25, 25]
    qmins = np.array([-80, -40, -40, -30, -20, -20])
    qmaxs = np.array([80, 40, 40, 20, 20, 20])

    variable_modes = [0, 1, 2, 3, 4, 5]
    n = len(variable_modes)
    outdir = f'{wdir}/pots/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fig = plt.subplots()
    for i in range(n):
        qmin = [qmins[i]]
        qmax = [qmaxs[i]]
        ngrid = [ngrids[i]]
        variable_mode = [variable_modes[i]]
        nbasis = [nbases[i]]
        cart, q, v = slice_1d(variable_mode, qmin, qmax, ngrid, nbasis)
        zero_ind = np.argmin(np.abs(q - 0))
        write_to_xyz(cart, ['N', 'H', 'H', 'H'], fname=f'{outdir}/nmode{variable_modes[i]}.xyz')
        np.savetxt(f'{outdir}/nm{i}_potential.txt', v)
        np.savetxt(f'{outdir}/nm{i}_grid.txt', q)
        plt.plot(q, v*AU2WAVNUM, label=i)
    plt.xlabel('$q$')
    plt.ylabel('$v$ (cm^-1)')
    plt.legend()
    plt.show()


def generate_whole_potential(wdir, coords, variable_modes, qmins, qmaxs, ngrids, nbases):

    basis_func = ('sine', get_pib_basis)
    ngrid_prod = np.prod(nbases)
    outdir = f'{wdir}/ngrid_{ngrid_prod}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    masses = np.array([25527.03399, 1833.3516, 1833.3516, 1833.3516])
    ndof = len(variable_modes)
    variable_modes = np.array(variable_modes)
    inds = variable_modes + 6

    q0 = np.linspace(qmins[0], qmaxs[0], ngrids[0])
    q1 = np.linspace(qmins[1], qmaxs[1], ngrids[1])
    q2 = np.linspace(qmins[2], qmaxs[2], ngrids[2])
    q3 = np.linspace(qmins[3], qmaxs[3], ngrids[3])
    q4 = np.linspace(qmins[4], qmaxs[4], ngrids[4])
    q5 = np.linspace(qmins[5], qmaxs[5], ngrids[5])
    q_grids = [q0, q1, q2, q3, q4, q5]

    hessian = pot.ammpot4_hessian(dh3_minima=True)
    freqs_wavenums, norm_modes, tmat = tf.get_normal_modes(hessian, masses, 6)
    tmat[:, [0, 6]] = tmat[:, [6, 0]]

    q_prod = grids.get_quadrature_points(q_grids, qmins, qmaxs, nbases, basis_func)
    cart_coords_prod = tf.norm2cart_grid(q_prod[0:1, :], coords, masses, tmat) # for JIT compilation
    cart_coords_prod = tf.norm2cart_grid(q_prod, coords, masses, tmat)
    v_prod = pot.ammpot4_cart(cart_coords_prod)

    q_prod = q_prod[:, inds]
    min_pot_ind = np.argmin(v_prod)
    print(f'min vpot: {v_prod[min_pot_ind]}')
    print(f'q coords: {q_prod[min_pot_ind, :]}')
    np.savetxt(f'{outdir}/exact_potential.txt', v_prod)
    np.savetxt(f'{outdir}/exact_grid.txt', q_prod)
    
    
def run_full_dvr(pot_dir, wdir, qmins, qmaxs, ngrids, nbases, neig):
    ngrid_prod = np.prod(nbases)
    v = np.genfromtxt(f'{pot_dir}/ngrid_{ngrid_prod}/exact_potential.txt')
    masses = np.array([1, 1, 1, 1, 1, 1])
    ndims = 6
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    q_grids = []
    for i in range(ndims):
        q_grids.append(np.linspace(qmins[i], qmaxs[i], ngrids[i]))
    calc = dvr.Calculator(sine_dvr)
    exact_energies, wfs = calc.solve_nd(q_grids, masses, v, neig, nbases, ndim=ndims)
    np.savetxt(f'{wdir}/sine_dvr_energies.txt', exact_energies)
    tranistion_energies = exact_energies - exact_energies[0]
    tranistion_energies = tranistion_energies[1:]
    np.savetxt(f'{wdir}/sine_dvr_transitions.txt', tranistion_energies)
    return tranistion_energies

if __name__ == "__main__":

    POT_GEN = True

    pot_dir = f'/home/kyle/DVR_Applications/NH3/sine_dvr/whole_pot/'
    out_dir = f'/home/kyle/DVR_Applications/NH3/sine_dvr/results/'

    neig = 9
    qmins = np.array([-80, -40, -40, -30, -20, -20])
    qmaxs = np.array([80, 40, 40, 20, 20, 20])
    ngrids = np.array([51, 51, 51, 51, 51, 51])
    nbases = np.array([31, 25, 25, 25, 21, 21])
    variable_modes = np.array([0, 1, 2, 3, 4, 5])

    ngrid_prod = np.prod(ngrids)
    
    dh3_coords = np.array([[0.0, 0.0, 0.0],
                          [0.9988, 0.0, 0.0],
                          [-0.499400, 0.864986, 0.0],
                          [-0.499400, -0.864986, 0.0]])

    dh3_coords *= (1/BOHR)

    if POT_GEN:
        generate_whole_potential(pot_dir, dh3_coords, variable_modes, qmins, qmaxs, ngrids, nbases)
    else:
        run_full_dvr(pot_dir, out_dir, qmins, qmaxs, ngrids, nbases, neig)


