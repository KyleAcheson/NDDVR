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

BOHR = 0.52917741092  # Angstroms
ATOMIC_MASS = 1E-3 / 6.022140857E23
ELEC_MASS = 9.1093837E-31
AU2EV = 27.2114
AU2WAVNUM = 219474.63

def write_to_xyz(coords, atom_labels, fname):
    natoms, _, npoints = coords.shape
    with open(fname, 'a+') as f:
        for i in range(npoints):
            coord = coords[:, :, i] * BOHR
            f.write(f'{natoms}\n')
            f.write(f'grid point {i+1}\n')
            array_string = '\n'.join([f'{atom_labels[ind]}\t' + '\t'.join(map(str, row)) for ind, row in enumerate(coord)]) + '\n'
            f.write(array_string)


def slice_1d(variable_modes, qmins, qmaxs, ngrid_prod):

    #au_masses = np.array([14.007, 1.008, 1.008, 1.008])
    #masses = (au_masses * ATOMIC_MASS) / ELEC_MASS
    masses = np.array([25527.03399, 1833.3516, 1833.3516, 1833.3516])
    #eq_coords = np.array([[0.0, 0.0, 0.0],
    #                   [0.0, -0.9377, -0.3816],
    #                   [0.8121, 0.4689, -0.3816],
    #                   [-0.8121, 0.4689, -0.3816]])
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

    hessian = pot.ammpot4_hessian()
    dh3_coords *= (1 / BOHR)
    freqs, norm_modes, tmat = tf.get_normal_modes(hessian, masses, 6)
    print(freqs)

    q_prod = grids.generate_grid(tmat, qmins, qmaxs, variable_modes, ngrid_prod, grid_type='product')
    cart_coords_prod = tf.norm2cart_grid(q_prod, eq_coords, masses, tmat)
    #write_to_xyz(cart_coords_prod, ['N', 'H', 'H', 'H'], fname=f'nmode{variable_modes[0]}.xyz')
    v_prod = pot.ammpot4_cart(cart_coords_prod)
    q_prod = q_prod[:, inds]
    return cart_coords_prod, q_prod, v_prod

def run_1d_test(wdir):

    # dh3 minima occurs at:
    # -42.62775198,  -6.55439153,   3.73244146,
    # -23.12649245, -14.35714   , -27.76418164

    ngrid = 101
    qmins = [-60, -40, -40, -20, -20, -20]
    qmaxs = [40, 40, 40, 20, 20, 20]
    #qmins = [-100, -40, -40, -25, -25, -25]
    #qmaxs = [100, 40, 40, 25, 25, 25]

    #qmins = [-200, -200, -200, -50, -50, -50]
    #qmaxs = [200, 200, 200, 50, 50, 50]
    variable_modes = [0, 1, 2, 3, 4, 5]
    n = len(variable_modes)
    outdir = f'{wdir}/pots/ngrid_{ngrid}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fig = plt.subplots()
    for i in range(n):
        qmin = [qmins[i]]
        qmax = [qmaxs[i]]
        variable_mode = [variable_modes[i]]
        cart, q, v = slice_1d(variable_mode, qmin, qmax, ngrid)
        zero_ind = np.argmin(np.abs(q - 0))
        write_to_xyz(cart, ['N', 'H', 'H', 'H'], fname=f'{outdir}/nmode{variable_modes[i]}.xyz')
        np.savetxt(f'{outdir}/nm{i}_potential.txt', v)
        np.savetxt(f'{outdir}/nm{i}_grid.txt', q)
        plt.plot(q, v*AU2WAVNUM, label=i)
    plt.xlabel('$q$')
    plt.ylabel('$v$ (cm^-1)')
    plt.legend()
    plt.show()
    
    
def generate_exact_potential(wdir, ngrid_prod):
    #qmins = [-95, -55, -50, -22.5, -22.5, -20]
    #qmaxs = [60, 55, 55, 30, 22.5, 25]
    qmins = [-60, -40, -40, -20, -20, -20]
    qmaxs = [40, 40, 40, 20, 20, 20]

    outdir = f'{wdir}/inputs/ngrid_{ngrid_prod}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    au_masses = np.array([14.007, 1.008, 1.008, 1.008])
    masses = np.array([25527.03399, 1833.3516, 1833.3516, 1833.3516])
    eq_coords = np.array([[0.0, 0.0, 0.0],
                          [1.0128, 0.0, 0.0],
                          [-0.296621, 0.968390, 0.0],
                          [-0.296621, -0.401080, -0.881427]])
    eq_coords *= (1 / BOHR)
    variable_modes = [0, 1, 2, 3, 4, 5]
    ndof = len(variable_modes)
    variable_modes = np.array(variable_modes)
    inds = variable_modes + 6

    q0 = np.linspace(qmins[0], qmaxs[0], ngrid_prod)
    q1 = np.linspace(qmins[1], qmaxs[1], ngrid_prod)
    q2 = np.linspace(qmins[2], qmaxs[2], ngrid_prod)
    q3 = np.linspace(qmins[3], qmaxs[3], ngrid_prod)
    q4 = np.linspace(qmins[4], qmaxs[4], ngrid_prod)
    q5 = np.linspace(qmins[5], qmaxs[5], ngrid_prod)
    q_grids = [q0, q1, q2, q3, q4, q5]

    hessian = pot.ammpot4_hessian()
    freqs, norm_modes, tmat = tf.get_normal_modes(hessian, masses, ndof)

    q_prod = grids.generate_grid(tmat, qmins, qmaxs, variable_modes, ngrid_prod, grid_type='product')
    cart_coords_prod = tf.norm2cart_grid(q_prod[0:1, :], eq_coords, masses, tmat) # for JIT compilation
    cart_coords_prod = tf.norm2cart_grid(q_prod, eq_coords, masses, tmat)
    v_prod = pot.ammpot4_cart(cart_coords_prod)

    q_prod = q_prod[:, inds]
    np.savetxt(f'{outdir}/exact_potential.txt', v_prod)
    np.savetxt(f'{outdir}/exact_grid.txt', q_prod)


def run_exact_pot(wdir):
    ngrids = [21]
    for ngrid in ngrids:
        generate_exact_potential(wdir, ngrid)


if __name__ == "__main__":
    import time
    out_dir = '/home/kyle/DVR_Applications/NH3/smaller_range'
    #run_1d_test(out_dir)
    run_exact_pot(out_dir)
    breakpoint()
