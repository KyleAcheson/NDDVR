import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import wf_utils as wfu
import potential_functions as potf

def algorithm_36(x, v, mass, neig):

    ngrid = len(x)
    dx = x[1] - x[0]
    L = x[-1] - x[0]

    H = np.zeros((ngrid, ngrid))
    for i in range(ngrid): # init tridiagonal unit matrix
        y = np.pi / (3 * mass)
        #H[i, i] = 1.5*mass * ( mass * (((np.pi * np.tanh(y)) / 4) - L) + (v[i] - np.pi) )
        H[i, i] = - 3*mass * (1/8) * (mass*((4*L) - (np.pi*np.tanh(y))) - 4*v[i] + 4*np.pi)
        if i > 0:
            j = i - 1
            H[i, j] = ((-3 * mass) * (np.exp(0.5*(x[i] - x[j])**2) * np.tanh(np.pi))) / (4 * dx * np.abs(x[i] - x[j]))
        if i < ngrid - 1:
            j = i + 1
            H[i, j] = ((-3 * mass) * (np.exp(0.5*(x[i] - x[j])**2) * np.tanh(np.pi))) / (4 * dx * np.abs(x[i] - x[j]))

    eigval, eigvec = np.linalg.eigh(H)  # output - eigenvector of H

    wf = wfu.normalise_wf(eigvec, x, neig)
    energies = wfu.evaluate_energies(wf, x, v, neig)

    return wf, energies, H

def algorithm_36_2D(x, y, v, mass_x, mass_y, neig):

    Nx = len(x)
    Ny = len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Lx = x[-1] - x[0]
    Ly = y[-1] - y[0]

    H_x = np.zeros((Nx, Nx))
    H_y = np.zeros((Ny, Ny))

    a = np.pi / (3 * mass_x)
    for i in range(Nx):
        i_prime = i
        H_x[i, i_prime] = - 3*mass_x * (1/8) * (mass_x*((4*Lx) - (np.pi*np.tanh(a))) - 4*v[i, i_prime] + 4*np.pi)
        if i > 0:
            i_prime = i - 1
            H_x[i, i_prime] = ((-3 * mass_x) * (np.exp(0.5*(x[i] - x[i_prime])**2) * np.tanh(np.pi))) / (4 * dx * np.abs(x[i] - x[i_prime]))
        if i < Nx - 1:
            i_prime = i + 1
            H_x[i, i_prime] = ((-3 * mass_x) * (np.exp(0.5*(x[i] - x[i_prime])**2) * np.tanh(np.pi))) / (4 * dx * np.abs(x[i] - x[i_prime]))

    a = np.pi / (3 * mass_y)
    for j in range(Ny):
        j_prime = j
        H_y[j, j_prime] = - 3*mass_y * (1/8) * (mass_y*((4*Ly) - (np.pi*np.tanh(a))) - 4*v[j, j_prime] + 4*np.pi)
        if j > 0:
            j_prime = j - 1
            H_y[j, j_prime] = ((-3 * mass_y) * (np.exp(0.5*(y[j] - y[j_prime])**2) * np.tanh(np.pi))) / (4 * dy * np.abs(y[j] - y[j_prime]))
        if j < Ny - 1:
            j_prime = j + 1
            H_y[j, j_prime] = ((-3 * mass_y) * (np.exp(0.5*(y[j] - y[j_prime])**2) * np.tanh(np.pi))) / (4 * dy * np.abs(y[j] - y[j_prime]))

    Hx = np.kron(H_x, np.eye(Ny))
    Hy = np.kron(np.eye(Nx), H_y)
    H = Hx + Hy
    eigval, eigvec = np.linalg.eigh(H)

    wf = wfu.normalise_wf2(eigvec, x, y, neig)
    energies = wfu.evaluate_energies_2d(wf, x, y, v, neig)

    return energies, wf


def algorithm_36_sparse(x, v, mass, neig):
    ngrid = len(x)
    dx = x[1] - x[0]
    L = x[-1] - x[0]

    Hii = []
    Hij = []
    for i in range(ngrid):  # init tridiagonal unit matrix
        y = np.pi / (3 * mass)
        hii = - 3 * mass * (1 / 8) * (mass * ((4 * L) - (np.pi * np.tanh(y))) - 4 * v[i] + 4 * np.pi)
        Hii.append(hii)
        if i > 0:
            j = i - 1
            hij = ((-3 * mass) * (np.exp(0.5 * (x[i] - x[j]) ** 2) * np.tanh(np.pi))) / (
                        4 * dx * np.abs(x[i] - x[j]))
            Hij.append(hij)

    H = diags([Hii, Hij, Hij], [0, -1, +1])
    eigval, eigvec = np.linalg.eigh(H.toarray())
    #eigval, eigvec = eigsh(H, k=neig)

    wf = wfu.normalise_wf(eigvec, x, neig)
    energies = wfu.evaluate_energies(wf, x, v, neig)

    return wf, energies, H


def algorithm_100(x, v, mass, neig):

    ngrid = len(x)
    dx = x[1] - x[0]
    L = x[-1] - x[0]

    H = np.zeros((ngrid, ngrid))
    for i in range(ngrid):
        H[i, i] = (3 * L / np.pi) - ((9 * dx**2 * v[i]) / (np.pi**2 * mass)) + (3 * mass) + (2 * v[i]) + 3 + (3 * np.pi)
        #H[i, i] = (3 * mass) + (2 * v[i]) - ((3 / np.pi) * (((3*dx**2 * v[i]) / (np.pi * mass)) - np.pi - L)) + (3 * np.pi)
        if i > 0:
            j = i -1
            H[i, j] = (-3 * np.exp(0.5 * (x[i] - x[j]) ** 2)) / (np.pi * dx * abs(x[i] - x[j]))
        if i < ngrid - 1:
            j = i + 1
            H[i, j] = (-3 * np.exp(0.5*(x[i] - x[j])**2)) / (np.pi * dx * abs(x[i] - x[j]))

    eigval, eigvec = np.linalg.eigh(H)  # output - eigenvector of H

    wf = wfu.normalise_wf(eigvec, x, neig)
    energies = wfu.evaluate_energies(wf, x, v, neig)

    return wf, energies, H


def algorithm_100_sparse(x, v, mass, neig):

    ngrid = len(x)
    dx = x[1] - x[0]
    L = x[-1] - x[0]

    Hii = []
    Hij = []
    for i in range(ngrid):
        hii = (3 * L / np.pi) - ((9 * dx**2 * v[i]) / (np.pi**2 * mass)) + (3 * mass) + (2 * v[i]) + 3 + (3 * np.pi)
        Hii.append(hii)
        if i > 0:
            j = i -1
            hij = (-3 * np.exp(0.5 * (x[i] - x[j]) ** 2)) / (np.pi * dx * abs(x[i] - x[j]))
            Hij.append(hij)

    H = diags([Hii, Hij, Hij], [0, -1, +1])
    eigval, eigvec = np.linalg.eigh(H.toarray())
    #eigval, eigvec = eigsh(H, k=neig)

    wf = wfu.normalise_wf(eigvec, x, neig)
    energies = wfu.evaluate_energies(wf, x, v, neig)

    return wf, energies, H


def algorithm_100_2D(x, y, v, mass_x, mass_y, neig):

    Nx = len(x)
    Ny = len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Lx = x[-1] - x[0]
    Ly = y[-1] - y[0]

    H_x = np.zeros((Nx, Nx))
    H_y = np.zeros((Ny, Ny))

    for i in range(Nx):
        i_prime = i
        H_x[i, i_prime] = (3 * Lx / np.pi) - ((9 * dx ** 2 * v[i, i_prime]) / (np.pi ** 2 * mass_x)) + (3 * mass_x) + (2 * v[i, i_prime]) + 3 + (
                3 * np.pi)
        if i > 0:
            i_prime = i - 1
            H_x[i, i_prime] = (-3 * np.exp(0.5 * (x[i] - x[i_prime]) ** 2)) / (np.pi * dx * abs(x[i] - x[i_prime]))
        if i < Nx - 1:
            i_prime = i + 1
            H_x[i, i_prime] = (-3 * np.exp(0.5 * (x[i] - x[i_prime]) ** 2)) / (np.pi * dx * abs(x[i] - x[i_prime]))

    for j in range(Ny):
        j_prime = j
        H_y[j, j_prime] = (3 * Ly / np.pi) - ((9 * dy ** 2 * v[j, j_prime]) / (np.pi ** 2 * mass_y)) + (3 * mass_y) + (2 * v[j, j_prime]) + 3 + (
                3 * np.pi)
        if j > 0:
            j_prime = j - 1
            H_y[j, j_prime] = (-3 * np.exp(0.5 * (y[j] - y[j_prime]) ** 2)) / (np.pi * dy * abs(y[j] - y[j_prime]))
        if j < Ny - 1:
            j_prime = j + 1
            H_y[j, j_prime] = (-3 * np.exp(0.5 * (y[j] - y[j_prime]) ** 2)) / (np.pi * dy * abs(y[j] - y[j_prime]))


    Hx = np.kron(H_x, np.eye(Ny))
    Hy = np.kron(np.eye(Nx), H_y)
    H = Hx + Hy
    eigval, eigvec = np.linalg.eigh(H)

    wf = wfu.normalise_wf2(eigvec, x, y, neig)
    energies = wfu.evaluate_energies_2d(wf, x, y, v, neig)

    return energies, wf


def algorithm_29_2D(x, y, v, mass_x, mass_y, neig):

    Nx = len(x)
    Ny = len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Lx = x[-1] - x[0]
    Ly = y[-1] - y[0]

    H_x = np.zeros((Nx, Nx))
    H_y = np.zeros((Ny, Ny))

    a = np.i0(np.sin(1) + 4)
    for i in range(Nx):
        i_prime = i
        H_x[i, i_prime] = Lx**2 * ((2 * v[i, i_prime]) * (3 + 2**(a)) + (np.pi * (Lx + 3 + np.pi)))
        if i > 0:
            i_prime = i - 1
            H_x[i, i_prime] = Lx ** 2 * ((-1) ** (i + 1 - i_prime + 1) * (((3 * np.pi) * (x[i] - x[i_prime]) ** 4) + 3 + 2 ** (a)) + (
                (np.pi * Lx) * (x[i] - x[i_prime]) ** 4)) * (x[i] - x[i_prime]) ** (-2)
        if i < Nx - 1:
            i_prime = i + 1
            H_x[i, i_prime] = Lx ** 2 * ((-1) ** (i + 1 - i_prime + 1) * (((3 * np.pi) * (x[i] - x[i_prime]) ** 4) + 3 + 2 ** (a)) + (
                    (np.pi * Lx) * (x[i] - x[i_prime]) ** 4)) * (x[i] - x[i_prime]) ** (-2)

    a = np.i0(np.sin(1) + 4)
    for j in range(Ny):
        j_prime = j
        H_y[j, j_prime] = Ly**2 * ((2 * v[j, j_prime]) * (3 + 2**(a)) + (np.pi * (Ly + 3 + np.pi)))
        if j > 0:
            j_prime = j - 1
            H_y[j, j_prime] = Ly ** 2 * ((-1) ** (j + 1 - j_prime + 1) * (((3 * np.pi) * (y[j] - y[j_prime]) ** 4) + 3 + 2 ** (a)) + (
                    (np.pi * Ly) * (y[j] - y[j_prime]) ** 4)) * (y[j] - y[j_prime]) ** (-2)

        if j < Ny - 1:
            j_prime = j + 1
            H_y[j, j_prime] = Ly ** 2 * ((-1) ** (j + 1 - j_prime + 1) * (((3 * np.pi) * (y[j] - y[j_prime]) ** 4) + 3 + 2 ** (a)) + (
                    (np.pi * Ly) * (y[j] - y[j_prime]) ** 4)) * (y[j] - y[j_prime]) ** (-2)

    Hx = np.kron(H_x, np.eye(Ny))
    Hy = np.kron(np.eye(Nx), H_y)
    H = Hx + Hy
    eigval, eigvec = np.linalg.eigh(H)

    wf = wfu.normalise_wf2(eigvec, x, y, neig)
    energies = wfu.evaluate_energies_2d(wf, x, y, v, neig)

    return energies, wf


def algorithm_29(x, v, mass, neig):

    ngrid = len(x)
    dx = x[1] - x[0]
    L = x[-1] - x[0]

    H = np.zeros((ngrid, ngrid))
    y = np.i0(np.sin(1) + 4)
    for i in range(ngrid):
        H[i, i] = L**2 * ((2 * v[i]) * (3 + 2**(y)) + (np.pi * (L + 3 + np.pi)))
        if i > 0:
            j = i - 1
            H[i, j] = L**2 * ((-1)**(i+1 - j+1) * (((3 * np.pi) * (x[i] - x[j])**4) + 3 + 2**(y)) + ((np.pi * L) * (x[i] - x[j])**4)) * (x[i] - x[j])**(-2)
        if i < ngrid - 1:
            j = i + 1
            H[i, j] = L ** 2 * ((-1) ** (i+1 - j+1) * (((3 * np.pi) * (x[i] - x[j]) ** 4) + 3 + 2 ** (y)) + (
                    (np.pi * L) * (x[i] - x[j]) ** 4)) * (x[i] - x[j]) ** (-2)

    eigval, eigvec = np.linalg.eigh(H) # output - eigenvector of H

    wf = wfu.normalise_wf(eigvec, x, neig)
    energies = wfu.evaluate_energies(wf, x, v, neig)

    return wf, energies, H

def algorithm_29_sparse(x, v, mass, neig):

    ngrid = len(x)
    dx = x[1] - x[0]
    L = x[-1] - x[0]

    Hii = []
    Hij = []
    y = np.i0(np.sin(1) + 4)
    for i in range(ngrid):
        hi = L**2 * ((2 * v[i]) * (3 + 2**(y)) + (np.pi * (L + 3 + np.pi)))
        Hii.append(hi)
        if i > 0:
            j = i - 1
            hij = L**2 * ((-1)**(i+1 - j+1) * (((3 * np.pi) * (x[i] - x[j])**4) + 3 + 2**(y)) + ((np.pi * L) * (x[i] - x[j])**4)) * (x[i] - x[j])**(-2)
            Hij.append(hij)

    H = diags([Hii, Hij, Hij], [0, -1, +1])
    eigval, eigvec = np.linalg.eigh(H.toarray())
    #eigval, eigvec = eigsh(H, k=neig)

    wf = wfu.normalise_wf(eigvec, x, neig)
    energies = wfu.evaluate_energies(wf, x, v, neig)

    return wf, energies, H


if __name__ == "__main__":

    neig = 6
    Nx = 11
    Ny = 11
    x = np.linspace(-5, 5, Nx)
    y = np.linspace(-5, 5, Ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    mass_x, mass_y = 1, 1

    v = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            v[i, j] = potf.harmonic_potential_2d(x[i], y[j])

    energies, wf = algorithm_100_2D(x, y, v, mass_x, mass_y, neig)
    print(energies)
    energies, wf = algorithm_36_2D(x, y, v, mass_x, mass_y, neig)
    print(energies)
    energies, wf = algorithm_29_2D(x, y, v, mass_x, mass_y, neig)
    print(energies)

