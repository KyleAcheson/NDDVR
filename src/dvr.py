import numpy as np
import wf_utils as wfu
import plotting as pltg
import potential_functions as potf

def cm_dvr(x, v, mass, neig):
    ngrid = len(x)
    hbar = 1.0
    dx = x[1] - x[0]
    V = np.zeros((ngrid, ngrid))
    for i in range(ngrid):
        V[i, i] = v[i]

    T = np.zeros((ngrid, ngrid))
    for i in range(ngrid):
        for j in range(ngrid):
            if i == j:
                T[i, j] = ((hbar ** 2) * np.pi ** 2) / (6 * mass * dx ** 2)
            else:
                T[i, j] = ((hbar ** 2) * (-1.0) ** (i - j)) / (mass * dx ** 2 * (i - j) ** 2)

    H = T + V
    E, c = np.linalg.eigh(H)
    for i in range(ngrid):
        csum = np.trapz(np.conj(c[:,i]) * c[:,i], x)
        c[:, i] = c[:, i] / np.sqrt(csum)
        E[i] = np.real(E[i])

    return c[neig, :], E[:neig], H

def cm_dvr_2d(x, y, v, hbar=1.0, mx=1.0, my=1.0):

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    Nx = len(x)
    Ny = len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    V = np.zeros((Nx * Ny, Nx * Ny))
    for i in range(Nx):
        for j in range(Ny):
            n = i * Ny + j
            V[n, n] = v[i, j]

    T_x = np.zeros((Nx, Nx))
    T_y = np.zeros((Ny, Ny))

    for i in range(Nx):
        for i_prime in range(Nx):
            if i == i_prime:
                T_x[i, i_prime] = ((hbar ** 2) * np.pi ** 2) / (6 * mx * dx ** 2)
            else:
                T_x[i, i_prime] = ((hbar ** 2) * (-1.0) ** (i - i_prime)) / (mx * dx ** 2 * (i - i_prime) ** 2)

    for j in range(Ny):
        for j_prime in range(Ny):
            if j == j_prime:
                T_y[j, j_prime] = ((hbar ** 2) * np.pi ** 2) / (6 * my * dy ** 2)
            else:
                T_y[j, j_prime] = ((hbar ** 2) * (-1.0) ** (j - j_prime)) / (my * dy ** 2 * (j - j_prime) ** 2)

    Tx = np.kron(T_x, np.eye(Ny))
    Ty = np.kron(np.eye(Nx), T_y)
    T = Tx + Ty

    H = T + V

    energies, wfs = np.linalg.eigh(H)

    return energies[:neig], wfs[:, :neig], H


def cm_dvr_3d(x, y, z, v, hbar=1.0, mx=1.0, my=1.0, mz=1.0):
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    zmin, zmax = np.min(z), np.max(z)

    Nx = len(x)
    Ny = len(y)
    Nz = len(z)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    V = np.zeros((Nx * Ny * Nz, Nx * Ny * Nz))

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                n = (i * Ny + j) * Nz + k
                V[n, n] = v[i, j, k]

    T_x = np.zeros((Nx, Nx))
    T_y = np.zeros((Ny, Ny))
    T_z = np.zeros((Nz, Nz))

    for i in range(Nx):
        for i_prime in range(Nx):
            if i == i_prime:
                T_x[i, i_prime] = ((hbar ** 2) * np.pi ** 2) / (6 * mx * dx ** 2)
            else:
                T_x[i, i_prime] = ((hbar ** 2) * (-1.0) ** (i - i_prime)) / (mx * dx ** 2 * (i - i_prime) ** 2)

    for j in range(Ny):
        for j_prime in range(Ny):
            if j == j_prime:
                T_y[j, j_prime] = ((hbar ** 2) * np.pi ** 2) / (6 * my * dy ** 2)
            else:
                T_y[j, j_prime] = ((hbar ** 2) * (-1.0) ** (j - j_prime)) / (my * dy ** 2 * (j - j_prime) ** 2)

    for k in range(Nz):
        for k_prime in range(Nz):
            if k == k_prime:
                T_z[k, k_prime] = ((hbar ** 2) * np.pi ** 2) / (6 * mz * dz ** 2)
            else:
                T_z[k, k_prime] = ((hbar ** 2) * (-1.0) ** (k - k_prime)) / (mz * dz ** 2 * (k - k_prime) ** 2)

    T = np.kron(np.kron(T_x, np.eye(Ny)), np.eye(Nz)) + np.kron(np.kron(np.eye(Nx), T_y), np.eye(Nz)) + np.kron(
        np.kron(np.eye(Nx), np.eye(Ny)), T_z)

    H = T + V
    energies, wfs = np.linalg.eigh(H)

    return energies[:neig], wfs[:, :neig], H


if __name__ == "__main__":

    # TEST 2D ISOTROPIC HARMONIC OSCILLATOR

    neig = 3
    Nx = 35  # Number of grid points in x
    Ny = 35  # Number of grid points in y
    xmin, xmax = -5.0, 5.0
    ymin, ymax = -5.0, 5.0
    x = np.linspace(xmin, xmax, Nx)
    y = np.linspace(ymin, ymax, Ny)

    v = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Nx):
            v[i, j] = potf.harmonic_potential_2d(x[i], y[j])

    energies, wfs, H = cm_dvr_2d(x, y, v)
    print(energies[:neig]) # energies from diagonalisation of H (2D)

    wfs = wfu.normalise_wf2(wfs, x, y, neig)
    energies = wfu.evaluate_energies_2d(wfs, x, y, v, neig)
    print(energies) # compare w energies evaluated as expectation value of \hat{V} and \hat{T}
                    # \hat{T} is evaluated in momemntum space - using 2D FFT

    pltg.plot_wavefunctions_2d(x, y, wfs, energies, neig)

    # TEST 3D ISOTROPIC HARMONIC OSCILLATOR

    neig = 3
    Nx = 11  # Number of grid points in x
    Ny = 11  # Number of grid points in y
    Nz = 11  # Number of grid points in y
    xmin, xmax = -5.0, 5.0
    ymin, ymax = -5.0, 5.0
    zmin, zmax = -5.0, 5.0
    x = np.linspace(xmin, xmax, Nx)
    y = np.linspace(ymin, ymax, Ny)
    z = np.linspace(zmin, zmax, Nz)

    v = np.zeros((Nx, Ny, Nz))
    for i in range(Nx):
        for j in range(Nx):
            for k in range(Nz):
                v[i, j, k] = potf.harmonic_potential_3d(x[i], y[j], z[k])

    energies, wfs, H = cm_dvr_3d(x, y, z, v)
    print(energies[:neig]) # energies for 3D oscillator
    wfs = wfs.reshape(Nx, Ny, Nz, neig)

    pltg.plot_wavefunctions_3d(x, y, z, wfs, energies, neig)
