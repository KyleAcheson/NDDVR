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
import fast_dvr.transforms as tf
import fast_dvr.grids as grids
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.gaussian_process.kernels import RBF

BOHR = 0.52917741092  # Angstroms
AU2EV = 27.2114
AU2WAVNUM = 219474.63

def distance_hist(x_data, nbins, skip=1):
    dists = distance_matrix(x_data[::skip, :], x_data[::skip, :], p=2)
    plt.figure()
    plt.hist(dists.flatten(), bins=nbins)
    #plt.bar(bins, counts)
    plt.show()
    

def slice_1d(variable_modes, qmins, qmaxs, ngrid_prod):

    masses = np.array([29157.9765, 1833.3516, 1833.3516])
    eq_coords = np.array([[0, 0, 0], [0.95865, 0, 0], [-0.237556, 0.928750, 0]]) * (1 / BOHR)
    ndof = len(variable_modes)
    variable_modes = np.array(variable_modes)
    inds = variable_modes + 6

    hessian = pot.partridge_schwenke_hessian()
    freqs, norm_modes, tmat = tf.get_normal_modes(hessian, masses, 3)

    q_prod = grids.generate_grid(tmat, qmins, qmaxs, variable_modes, ngrid_prod, grid_type='product')
    cart_coords_prod = tf.norm2cart_grid(q_prod, eq_coords, masses, tmat)
    v_prod = pot.partridge_schwenke_cart(cart_coords_prod)
    q_prod = q_prod[:, inds]
    return q_prod, v_prod


def water_gpr(tmat, qmins, qmaxs, ngrid, ngrid_prod):
    masses = np.array([29157.9765, 1833.3516, 1833.3516])
    eq_coords = np.array([[0, 0, 0], [0.95865, 0, 0], [-0.237556, 0.928750, 0]]) * (1 / BOHR)
    variable_modes = [0, 1, 2]
    ndof = len(variable_modes)
    variable_modes = np.array(variable_modes)

    thresh = 0.005 # .5% error in gpr fit
    qcoords = grids.generate_grid(tmat, qmins, qmaxs, variable_modes, ngrid, grid_type='sobol', scramble=True, seed=42)
    cart_coords = tf.norm2cart_grid(qcoords, eq_coords, masses, tmat)
    v = pot.partridge_schwenke_cart(cart_coords)

    inds = variable_modes + 6

    q_prod = grids.generate_grid(tmat, qmins, qmaxs, variable_modes, ngrid_prod, grid_type='product')
    cart_coords_prod = tf.norm2cart_grid(q_prod, eq_coords, masses, tmat)
    v_prod = pot.partridge_schwenke_cart(cart_coords_prod)

    q_train = qcoords[:, inds]
    q_pred = q_prod[:, inds]
    kernel = RBF(length_scale=15, length_scale_bounds=(2.0, 45.0))
    v_pred, v_std, gp = pot.fit_potential(q_train, q_pred, v, ndof, kernel)

    frac_error = np.abs(v_pred - v_prod) / v_prod
    error_inds = np.argwhere(np.abs(frac_error) > thresh)
    nabove_thresh = len(error_inds)
    total_numel = q_pred.shape[0]

    print('GPR FIT:\n')
    print(f'opt. length scale: {gp.kernel_.length_scale}')
    print(f'max std. dev.: {np.max(v_std)}')
    print(f'max fractional error: {np.max(frac_error)}')
    print(f'{nabove_thresh} points of {total_numel} above error thresh of {thresh}.')

    return v_pred, v_prod

def generate_exact_potential(wdir, ngrid_prod):
    #qmins = [-60, -45, -25]
    #qmaxs = [60, 30, 25]
    qmins = [-80, -50, -30]
    qmaxs = [70, 25, 30]

    outdir = f'{wdir}/inputs/range2/ngrid_{ngrid_prod}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    masses = np.array([29157.9765, 1833.3516, 1833.3516])
    eq_coords = np.array([[0, 0, 0], [0.95865, 0, 0], [-0.237556, 0.928750, 0]]) * (1 / BOHR)
    variable_modes = [0, 1, 2]
    ndof = len(variable_modes)
    variable_modes = np.array(variable_modes)
    inds = variable_modes + 6

    q0 = np.linspace(qmins[0], qmaxs[0], ngrid_prod)
    q1 = np.linspace(qmins[1], qmaxs[1], ngrid_prod)
    q2 = np.linspace(qmins[2], qmaxs[2], ngrid_prod)
    q_grids = [q0, q1, q2]

    hessian = pot.partridge_schwenke_hessian()
    freqs, norm_modes, tmat = tf.get_normal_modes(hessian, masses, ndof)

    q_prod = grids.generate_grid(tmat, qmins, qmaxs, variable_modes, ngrid_prod, grid_type='product')
    cart_coords_prod = tf.norm2cart_grid(q_prod, eq_coords, masses, tmat)
    v_prod = pot.partridge_schwenke_cart(cart_coords_prod)
    q_prod = q_prod[:, inds]
    np.savetxt(f'{outdir}/exact_potential.txt', v_prod)
    np.savetxt(f'{outdir}/exact_grid.txt', q_prod)


def generate_gpr_potential(wdir, exponent, ngrid_prod):
    #qmins = [-50, -30, -25]
    #qmaxs = [50, 20, 25]
    qmins = [-80, -50, -30]
    qmaxs = [70, 25, 30]
    ngrid = 2**exponent

    outdir = f'{wdir}/inputs/range2/gpr/exact_ngrid_{ngrid_prod}/2pow{exponent}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    masses = np.array([29157.9765, 1833.3516, 1833.3516])
    eq_coords = np.array([[0, 0, 0], [0.95865, 0, 0], [-0.237556, 0.928750, 0]]) * (1 / BOHR)
    variable_modes = [0, 1, 2]
    ndof = len(variable_modes)
    variable_modes = np.array(variable_modes)
    inds = variable_modes + 6

    q0 = np.linspace(qmins[0], qmaxs[0], ngrid_prod)
    q1 = np.linspace(qmins[1], qmaxs[1], ngrid_prod)
    q2 = np.linspace(qmins[2], qmaxs[2], ngrid_prod)
    q_grids = [q0, q1, q2]

    hessian = pot.partridge_schwenke_hessian()
    freqs, norm_modes, tmat = tf.get_normal_modes(hessian, masses, ndof)

    v_pred, v_exact = water_gpr(tmat, qmins, qmaxs, ngrid, ngrid_prod)
    np.savetxt(f'{outdir}/predicted_potential.txt', v_pred)


def run_exact_pot():
    wdir = '.'
    ngrids = [21, 31, 41, 51, 61, 71, 81]
    for ngrid in ngrids:
        generate_exact_potential(wdir, ngrid)


def run_gpr_pot():
    wdir = '.'
    ngrid_exact = 41
    exponents = [8, 9, 10, 11, 12]
    exponents = [12]
    for exponent in exponents:
        generate_gpr_potential(wdir, exponent, ngrid_exact)


def run_1d_test():
    ngrid = 61
    qmins = [-80, -50, -30]
    qmaxs = [70, 25, 30]
    variable_modes = [0, 1, 2]
    mode_labels = ['bend', 'sym', 'asym']
    n = len(variable_modes)
    fig = plt.subplots()
    for i in range(n):
        qmin = [qmins[i]]
        qmax = [qmaxs[i]]
        variable_mode = [variable_modes[i]]
        q, v = slice_1d(variable_mode, qmin, qmax, ngrid)
        ind = np.argwhere(q == 0)
        print(ind)
        plt.plot(q, v, label=mode_labels[i])
    plt.xlabel('$q$')
    plt.ylabel('$v$ (a.u.)')
    plt.legend()
    plt.show()
    breakpoint()

if __name__ == "__main__":
    #run_1d_test()
    run_exact_pot()
    #run_gpr_pot()
