import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
import fast_dvr.h2opes as h2opes
import fast_dvr.nh3pes as nh3pes
from numba import njit

BOHR = 0.529177
AU2EV = 27.2114
AU2WAVNUM = 219474.63

def load_potential(pot_file, grid_file, ndim, order):
    pot = np.genfromtxt(pot_file, skip_header=1)
    grids, grid_sizes = [], []
    for i in range(ndim):
        grid = np.genfromtxt(grid_file, skip_header=1, usecols=(i))
        grids.append(grid)
        grid_sizes.append(len(grid))
    grid_sizes = tuple(grid_sizes)
    pot = pot.reshape(grid_sizes, order=order)
    return pot, grids


def harmonic(x, k=1.0):
    return 0.5 * k * x**2


def harmonic_potential_2d(x, y, kx=1.0, ky=1.0, mx=1.0, my=1.0):
    return 0.5 * (kx * x ** 2 / mx + ky * y ** 2 / my)


def harmonic_potential_3d(x, y, z, kx=1.0, ky=1.0, kz=1.0, mx=1.0, my=1.0, mz=1.0):
    return 0.5 * (kx * x**2 / mx + ky * y**2 / my + kz * z**2 / mz)


def ammpot4(rij):
    """
    Wrapper for AMMPOT4 (NH3 potential).

    NOTE: The AMMPOT4 Fortran library returns energy in cm^-1 - this
    python wrapper converts energies to a.u. and returns them.

    :param rij: the internal coordinates in order
                (R1, R2, R3, A12, A13, A23 (the 3 bond lengths and angles)),
                with bond lengths in angstroem and angles in degrees.
    :return: v: energy in Hartrees.
    """
    n, _ = rij.shape
    v = nh3pes.nh3pot(r=rij, np=n)
    return v / AU2WAVNUM


def partridge_schwenke_potential(rij):
    """ Wrapper for Partridge-Schwenke potential (H2O).
        Takes an array of size [n, 3], where n is the
        number of points and each column an internal coordinate.
        NOTE: rij[:, 0:2] = O-H distances in a.u.,
              rij[:, -1] = HOH angle in radians. """
    v = h2opes.vibpot(n=rij.shape[0], rij=rij)
    return v


def ammpot4_cart(coords):
    """
    A wrapper for the AMMPOT4 (NH3) potential
    in cartesian coordiantes (a.u.).
    """
    internals = _internal_coordinates_nh3(coords[:, :, 0:1]) # for JIT compilation
    internals = _internal_coordinates_nh3(coords)
    internals[:, :3] *= BOHR
    v = ammpot4(internals)
    return v


def partridge_schwenke_cart(coords):
    """ A wrapper for the Partridge-Schwenke potential
        in cartesian coordinates (a.u.). """
    internals = _internal_coordinates_h2o(coords)
    #internals[:, :2] *= (1 / BOHR)
    internals[:, 2] *= (np.pi / 180.0)
    v = partridge_schwenke_potential(internals)
    return v


def partridge_schwenke_hessian(fname=None):
    """ Get the numerical Hessian of the Partridge-Schwenke potential
        in cartesian coordinates (hartree / bohr**2).
        If fname is provided, the hessian is written to file. """
    eq_coords = np.array([[0, 0, 0], [0.95865, 0, 0], [-0.237556, 0.928750, 0]])
    hess = _hessian(eq_coords, partridge_schwenke_cart, 0.001*BOHR)
    if fname:
        np.savetxt(fname, hess)
    return hess

def ammpot4_hessian(fname=None, **kwargs):
    #rint = np.array([1.01, 1.01, 1.01, 108, 108, 108])
    try:
        coords = kwargs['coords']
    except KeyError:
        #coords = np.array([[0.0, 0.0, 0.0],
        #                   [0.0, -0.9377, -0.3816],
        #                   [0.8121, 0.4689, -0.3816],
        #                   [-0.8121, 0.4689, -0.3816]])
        coords = np.array([[0.0, 0.0, 0.0],
                          [1.0128, 0.0, 0.0],
                          [-0.296621, 0.968390, 0.0],
                          [-0.296621, -0.401080, -0.881427]])

    hess = _hessian(coords, ammpot4_cart, 0.001*BOHR)
    if fname:
        np.savetxt(fname, hess)
    return hess

def _hessian(eq_coords, potential_function, epsilon=0.001):
    natoms, _ = eq_coords.shape
    n = 3*natoms
    eq_coords *= (1 / BOHR) # CONVERTS TO AU - INPUT IN ANG
    epsilon *= (1 / BOHR)
    x = eq_coords.flatten()
    hess = np.zeros((n, n))
    for i in range(n): # a mess I know
        for j in range(n):
            x_plus_delta = x.copy()
            x_plus_delta[i] += epsilon
            x_plus_delta[j] += epsilon
            x_mp_delta = x.copy()
            x_mp_delta[i] -= epsilon
            x_mp_delta[j] += epsilon
            x_pm_delta = x.copy()
            x_pm_delta[i] += epsilon
            x_pm_delta[j] -= epsilon
            x_minus_delta = x.copy()
            x_minus_delta[i] -= epsilon
            x_minus_delta[j] -= epsilon
            x_plus_delta = x_plus_delta.reshape(natoms, 3)
            x_mp_delta = x_mp_delta.reshape(natoms, 3)
            x_pm_delta = x_pm_delta.reshape(natoms, 3)
            x_minus_delta = x_minus_delta.reshape(natoms, 3)
            x_disp = np.stack((x_plus_delta, x_mp_delta, x_pm_delta, x_minus_delta), axis=2)
            v = potential_function(x_disp)
            h = (v[0] - v[1] - v[2] + v[3]) / (4 * epsilon**2)
            hess[i, j] = h

    hess += hess.T
    hess /= 2
    return hess


@njit
def get_bond_length(coorindates, connectivity):
    rvec = coorindates[connectivity[0], :] - coorindates[connectivity[1], :]
    return np.linalg.norm(rvec)


@njit
def get_bond_angle(coordinates, connectivity):
    i, j, k = connectivity
    r_ij = coordinates[i, :] - coordinates[j, :]
    r_kj = coordinates[k, :] - coordinates[j, :]
    cosine_theta = np.dot(r_ij, r_kj)
    sin_theta = np.linalg.norm(np.cross(r_ij, r_kj))
    theta = np.arctan2(sin_theta, cosine_theta)
    theta = 180.0 * theta / np.pi
    return theta


@njit
def _internal_coordinates_h2o(coordinates):
    """ Converts cartesian coordinates of H2O to internals. """
    natoms, _, npoints = coordinates.shape
    internals = np.zeros((npoints, 3))
    for i in range(npoints):
        r1 = get_bond_length(coordinates[:, :, i], [1, 0])
        r2 = get_bond_length(coordinates[:, :, i], [2, 0])
        angle = get_bond_angle(coordinates[:, :, i], [2, 0, 1])
        internals[i, 0], internals[i, 1], internals[i, 2] = r1, r2, angle
    return internals


@njit
def _internal_coordinates_nh3(coordinates):
    """ Converts cartesian coordinates of NH3 to internals. """
    natoms, _, npoints = coordinates.shape
    internals = np.zeros((npoints, 6))
    for i in range(npoints):
        r1 = get_bond_length(coordinates[:, :, i], [1, 0])
        r2 = get_bond_length(coordinates[:, :, i], [2, 0])
        r3 = get_bond_length(coordinates[:, :, i], [3, 0])
        a1 = get_bond_angle(coordinates[:, :, i], [1, 0, 2])
        a2 = get_bond_angle(coordinates[:, :, i], [1, 0, 3])
        a3 = get_bond_angle(coordinates[:, :, i], [2, 0, 3])
        internals[i, 0], internals[i, 1], internals[i, 2] = r1, r2, r3
        internals[i, 3], internals[i, 4], internals[i, 5] = a1, a2, a3
    return internals


def fit_potential(x_train, x_pred, v_train, ndof, kernel):
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)
    if ndof == 1:
        x_train = x_train.reshape(-1, 1)
        x_pred = x_pred.reshape(-1, 1)
        v_train = v_train.reshape(-1, 1)
    gp.fit(x_train, v_train)
    v_mean, v_std = gp.predict(x_pred, return_std=True)
    return v_mean, v_std, gp


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    eq_coords = np.array([[0.0, 0.0, 0.0],
                          [0.0, -0.9377, -0.3816],
                          [0.8121, 0.4689, -0.3816],
                          [-0.8121, 0.4689, -0.3816]])
    disp_coords = np.array([[0.01, 0.0, 0.0],
                          [0.0, -0.9377, -0.3816],
                          [0.8121, 0.4689, -0.3816],
                          [-0.8121, 0.4689, -0.3816]])
    coords = np.stack([eq_coords, disp_coords, eq_coords, disp_coords, eq_coords], axis=2)
    coords *= (1 / BOHR)
    #v = ammpot4_cart(coords)
    t1 = time.time()
    internals = _internal_coordinates_nh3(coords)
    t2 = time.time()
    print(t2 - t1)
    t1 = time.time()
    internals = _internal_coordinates_nh3(coords)
    t2 = time.time()
    print(t2 - t1)
    breakpoint()
