from __future__ import annotations
import numpy as np
import warnings
from scipy import sparse
from typing import Callable
import numpy.typing as npt


def colbert_miller(grid, mass, ngrid, hbar=1):
    dg = grid[1] - grid[0]
    indicies = np.arange(ngrid)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        T_n = ((hbar ** 2) * (-1.0)**(indicies[None] - indicies[:, None])) / (mass * dg**2 * (indicies[None] - indicies[:, None])**2)
    np.fill_diagonal(T_n, ((hbar ** 2) * np.pi ** 2) / (6 * mass * dg ** 2))
    return sparse.csr_matrix(T_n)


def get_pib_basis(x, nbasis, ngrid):
    x_min, x_max = np.min(x), np.max(x)
    L = x_max - x_min
    basis_funcs = np.zeros((nbasis, ngrid))
    for i in range(nbasis):
        y_i = np.sqrt(2.0 / L) * np.sin(((float(i+1) * np.pi) / L) * (x - x_min))
        basis_funcs[i, :] = y_i
    return basis_funcs

def sine_dvr(x, mass, nbasis, hbar=1):

    x_min, x_max = np.min(x), np.max(x)
    dx = x[1] - x[0]
    L = x_max - x_min
    ngrid = len(x)

    basis_funcs = get_pib_basis(x, nbasis, ngrid)
    indices = np.arange(1, nbasis+1)
    derivs = (indices * np.pi / L)**2
    Tm = np.einsum('ik,jk,j->ij', basis_funcs, basis_funcs, derivs) * dx
    Tm *= (hbar ** 2) / (2 * mass)
    _, tmat = HEG_procedure(x, x_min, x_max, nbasis, basis_func=('sine', basis_funcs))
    Tm = tmat.T @ Tm @ tmat
    return Tm


def HEG_procedure(grid: npt.NDArray, x_min: float, x_max: float, nbasis: int, basis_func: tuple[str, Callable]):

    """
    Diagonalises the matrix of the 1D position operator evaluated in the basis of basis functions
    defined by `basis_func.get('basis_name')`. If `basis_name == 'sine'`, one diagonalises a function
    of the position operator, specifically $f(x) = \cos(\frac{\pi (x - x_0)}{L})$.
    The resulting quadrature points for the given dimension are returned as a 1D array.

    :param grid:
    :param x_min:
    :param x_max:
    :param nbasis:
    :param basis_func:
    :return:
    """

    ngrid = len(grid)
    dx = grid[1] - grid[0]
    L = x_max - x_min

    func_type, get_basis = basis_func # get basis functions or callable for constructing them
    if hasattr(get_basis, '__call__'):
        basis = get_basis(grid, nbasis, ngrid)
        basis = basis.T
    else:
        try:
            basis = get_basis.T
        except AttributeError:
            raise TypeError('basis_func[1] must be a npt.NDarray containing the basis if not a Callable.')

    if func_type == 'sine':  # in sine-DVR one diagonalises this function of the position operator
        x_op = np.cos(np.pi * (grid - x_min) / L)
    else:
        x_op = grid

    Xmat = basis.T @ (x_op.reshape(-1, 1) * basis) * dx  # evaluate position operator matrix in the basis
    eigvals, tmat = np.linalg.eigh(Xmat)

    if func_type == 'sine':  # we have diagonalised a function of position, therefore have to invert it
        quad_points = x_min + (L / np.pi) * np.arccos(eigvals)
    else:
        quad_points = eigvals  # otherwise quadrature points are just the eigenvalues

    if np.argmin(quad_points) != 0:
        quad_points = np.flipud(quad_points)

    return quad_points, tmat


basis_funcs = {'sine': get_pib_basis}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import fast_dvr.potentials as potf
    ngrid = 81
    x = np.linspace(-20, 20, ngrid)
    nbasis = 50
    mass = 1

    x_min, x_max = np.min(x), np.max(x)
    dx = x[1] - x[0]
    L = x_max - x_min
    basis_funcs = get_pib_basis(x, nbasis, ngrid)
    basis_funcs = basis_funcs.T

    v = potf.harmonic(x, k=1)

    T = sine_dvr(x, mass, nbasis)
    V = np.zeros((nbasis, nbasis))
    for i in range(nbasis):
        for j in range(nbasis):
            V[i, j] = np.trapz(basis_funcs[:, i] * v * basis_funcs[:, j], x)

    H = T + V
    E, coeffs = np.linalg.eigh(H)
    print(E[:3])


    fx = np.cos(np.pi * (x - x_min) / L)
    Xfm = basis_funcs.T @ (fx.reshape(-1, 1) * basis_funcs) * dx
    eigvals, eigvecs = np.linalg.eigh(Xfm)
    Xd = eigvecs.T @ Xfm @ eigvecs
    x_diag = np.diag(Xd)
    xq = x_min + (L / np.pi) * np.arccos(x_diag)

    vd = potf.harmonic(np.flipud(xq), k=1)
    Vd = np.eye(nbasis) * vd
    Td = eigvecs.T @ T @ eigvecs
    H = Td + Vd

    E, coeffs = np.linalg.eigh(H)
    print(E[:3])

    Vd = eigvecs.T @ V @ eigvecs
    H = Td + Vd
    E, coeffs = np.linalg.eigh(H)
    print(E[:3])


    breakpoint()
