import numpy as np
from numba import njit

BOHR = 0.52917721092  # Angstroms
BOHR_SI = BOHR * 1e-10
ATOMIC_MASS = 9.109E-31
HARTREE2J = 4.359744650e-18
HARTREE2EV = 27.21138602
LIGHT_SPEED_SI = 299792458

AU2Hz = ((HARTREE2J / (ATOMIC_MASS * BOHR_SI ** 2)) ** 0.5 / (2 * np.pi))


@njit
def norm2cart_grid(qcoordinates, eq_coordinates, masses, transformation_matrix):
    ngrid_total, ndof = qcoordinates.shape
    natoms, _ = eq_coordinates.shape
    cartesian_coords = np.zeros((natoms, 3, ngrid_total))
    for i in range(ngrid_total):
        cartesian_coords[:, :, i] = norm2cart(qcoordinates[i, :], eq_coordinates, masses, transformation_matrix)
    return cartesian_coords

@njit
def cart2norm_grid(cartesian_potential, eq_coordinates, masses, transformation_matrix):
    natoms, _, ngrid_total = cartesian_potential.shape
    ndof_total = natoms * 3
    qcoordinates = np.zeros((ndof_total, ngrid_total))
    for i in range(ngrid_total):
        qcoordinates[:, i] = cart2norm(cartesian_potential[:, :, i], eq_coordinates, masses, transformation_matrix)
    return qcoordinates


def get_normal_modes(hessian, masses, nvibs):
    """ Diagonalise the Hessian (gets mass weighted)
        and calculate frequencies (cm^-1) and normal modes.
        Parameters
        ----------
        hessian : numpy.ndarrary
            hessian in a.u.
        masses : numpy.ndarray
            masses of atoms in a.u.
        nvibs : number of vib modes to return

        Returns
        -------
        freqs_wavenums : numpy.ndarray
            nvibs frequencies in cm^-1
        normal_modes : numpy.ndarray
            nvibs normal modes of vibration
        modes: numpy.ndarray
            raw eigenvectors / modes of Hessian
        """
    natoms = len(masses)
    fconstants_au, modes = _diag_hessian(hessian, masses)
    freqs = np.sqrt(np.abs(fconstants_au))  # in a.u.
    freqs_wavenums = freqs * AU2Hz / LIGHT_SPEED_SI * 1e-2
    normal_modes = np.einsum('z,zri->izr', masses ** -.5, modes.reshape(natoms, 3, -1))
    return freqs_wavenums[-nvibs:], normal_modes, modes


def _construct_mass_matrix(atom_masses):
    masses = np.array(atom_masses)
    mass_vec = np.repeat(masses, 3)
    mass_mat = np.sqrt(np.outer(mass_vec, mass_vec))
    return mass_mat


def _diag_hessian(hessian, masses):
    sq_mass_matrix = _construct_mass_matrix(masses)
    weighted_hessian = hessian * (1 / sq_mass_matrix)
    fconstants, modes = np.linalg.eigh(weighted_hessian)
    return fconstants, modes


@njit
def norm2cart(qcoordinates, eq_coordinates, masses, transformation_matrix):
    natoms = len(masses)
    mw_dcoords = transformation_matrix @ qcoordinates
    umw_dcoords = mw_dcoords.reshape((natoms, 3)) * (1 / np.sqrt(masses[:, np.newaxis]))
    cart_coords = umw_dcoords + eq_coordinates
    return cart_coords


@njit
def cart2norm(coordinates, eq_coordinates, masses, transformation_matrix):
    coords = _mass_weighted_displacements(coordinates, eq_coordinates, masses)
    return transformation_matrix.T @ coords.flatten()


@njit
def _mass_weighted_displacements(coordinates, eq_coordinates, masses):
    mw_dcoords = (coordinates - eq_coordinates) * np.sqrt(masses[:, np.newaxis])
    return mw_dcoords
