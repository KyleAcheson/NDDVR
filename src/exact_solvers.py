import numpy as np
import warnings
from scipy import sparse


def colbert_miller(grid, mass, hbar=1):
    ngrid = len(grid)
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

def sine_dvr(x, nbasis, mass=1, hbar=1):

    x_min, x_max = np.min(x), np.max(x)
    dx = x[1] - x[0]
    L = x_max - x_min
    ngrid = len(x)

    basis_funcs = get_pib_basis(x, nbasis, ngrid)
    indices = np.arange(1, nbasis+1)
    derivs = (indices * np.pi / L)**2
    T = np.einsum('ik,jk,j->ij', basis_funcs, basis_funcs, derivs) * dx
    T *= (hbar ** 2) / (2 * mass)
    return T


basis_funcs = {'sine': get_pib_basis}

if __name__ == "__main__":
    import fast_dvr.potentials as potf
    ngrid = 41
    x = np.linspace(-5, 5, ngrid)
    nbasis = 25

    x_min, x_max = np.min(x), np.max(x)
    dx = x[1] - x[0]
    L = x_max - x_min
    basis_funcs = get_pib_basis(x, nbasis, ngrid)
    basis_funcs = basis_funcs.T

    T = sine_dvr(x, nbasis)

    fx = np.cos(np.pi * (x - x_min) / L)
    Xfm = basis_funcs.T @ (fx.reshape(-1, 1) * basis_funcs) * dx
    eigvals, eigvecs = np.linalg.eigh(Xfm)
    Xd = eigvecs.T @ Xfm @ eigvecs
    x_diag = np.diag(Xd)
    xq = x_min + (L / np.pi) * np.arccos(x_diag)

    v = potf.harmonic(xq, k=1)
    Vd = np.eye(nbasis) * v
    H = T + Vd

    E, coeffs = np.linalg.eigh(H)
    print(E[:3])

    breakpoint()
