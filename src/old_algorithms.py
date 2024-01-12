import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import wf_utils as wfu
import potentials as potf

def algorithm_80(x, v, mass, neig):
    ngrid = len(x)
    dx = x[1] - x[0]

    T = np.zeros((ngrid, ngrid))
    V = np.zeros(ngrid)
    for i in range(ngrid):
        for j in range(ngrid):
            if i == j:
                V[i] = 1/3 * (v[i] * ((np.sin(np.i0(1))) * np.pi)**3 + 3)
                T[i, j] = ((1 / dx)**3 * dx) / (3 * np.pi)
            elif j == i + 1 or j == i - 1:
                a = (-1)**(i-j)
                T[i, j] = (((2 / dx)**3 * a * np.pi * a + dx) * a * dx) / (3 * np.pi)
            #else:
            #    T[i, j] = (dx**2) / (3 * np.pi)

    H = T + np.diag(V)

    eigval, eigvec = np.linalg.eigh(H)  # output - eigenvector of H

    wf = wfu.normalise_wf(eigvec, x, neig)
    energies = wfu.evaluate_energies(wf, x, v, neig)

    return wf, energies, H


def algorithm_2(x, v, mass, neig):

    ngrid = len(x)
    dx = x[1] - x[0]
    L = x[-1] - x[0]

    T = np.zeros((ngrid, ngrid))
    V = np.zeros(ngrid)
    for i in range(ngrid):
        V[i] = 4 * v[i] - 4
        T[i, i] = (4 * mass) / (3 * dx**2)
        if i > 0:
            j = i - 1
            T[i, j] = (16 * (-1)**(i+1 - j+1) * mass * np.exp(0.5*(x[i] - x[j])**2)) / (3 * np.pi * dx**2)
        if i < ngrid - 1:
            j = i + 1
            T[i, j] = (16 * (-1)**(i+1 - j+1) * mass * np.exp(0.5*(x[i] - x[j])**2)) / (3 * np.pi * dx**2)

    H = T + np.diag(V)

    eigval, eigvec = np.linalg.eigh(H)  # output - eigenvector of H

    wf = wfu.normalise_wf(eigvec, x, neig)
    energies = wfu.evaluate_energies(wf, x, v, neig)

    return wf, energies, H


def algorithm_2_2D(x, y, v, mass_x, mass_y, neig):

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    Nx = len(x)
    Ny = len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    masses = [mass_x, mass_y]

    V = np.zeros((Nx * Ny, Nx * Ny))
    for i in range(Nx):
        for j in range(Ny):
            n = i * Ny + j
            V[n, n] = 4 * v[i, j] - 4

    T_x = np.zeros((Nx, Nx))
    T_y = np.zeros((Ny, Ny))

    for i in range(Nx):
        i_prime = i
        T_x[i, i_prime] = (4 * mass_x) / (3 * dx ** 2)
        if i > 0:
            i_prime = i - 1
            T_x[i, i_prime] = (16 * (-1)**(i - i_prime) * mass_x * np.exp(0.5*(x[i] - x[i_prime])**2)) / (3 * np.pi * dx**2)
        if i < Nx - 1:
            i_prime = i + 1
            T_x[i, i_prime] = (16 * (-1)**(i - i_prime) * mass_x * np.exp(0.5*(x[i] - x[i_prime])**2)) / (3 * np.pi * dx**2)

    for j in range(Ny):
        j_prime = j
        T_y[j, j_prime] = (4 * mass_y) / (3 * dy ** 2)
        if j > 0:
            j_prime = j - 1
            T_y[j, j_prime] = (16 * (-1)**(j - j_prime) * mass_y * np.exp(0.5*(y[j] - y[j_prime])**2)) / (3 * np.pi * dy**2)
        if j < Ny - 1:
            j_prime = j + 1
            T_y[j, j_prime] = (16 * (-1)**(j - j_prime) * mass_y * np.exp(0.5*(y[j] - y[j_prime])**2)) / (3 * np.pi * dy**2)

    Tx = np.kron(T_x, np.eye(Ny))
    Ty = np.kron(np.eye(Nx), T_y)
    T = Tx + Ty
    H = T + V

    energies, wfs = np.linalg.eigh(H)

    wf = wfu.normalise_wf2(wfs, x, y, neig)
    energies = wfu.evaluate_energies_2d(wf, x, y, v, masses, neig)

    return energies, wf, H


def algorithm_2_3D(x, y, z, v, mass_x, mass_y, mass_z, neig):

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    zmin, zmax = np.min(z), np.max(z)
    Nx = len(x)
    Ny = len(y)
    Nz = len(z)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    masses = [mass_x, mass_y, mass_z]

    V = np.zeros((Nx * Ny * Nz, Nx * Ny * Nz))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                n = (i * Ny + j) * Nz + k
                V[n, n] = 4 * v[i, j, k] - 4

    T_x = np.zeros((Nx, Nx))
    T_y = np.zeros((Ny, Ny))
    T_z = np.zeros((Nz, Nz))

    for i in range(Nx):
        i_prime = i
        T_x[i, i_prime] = (4 * mass_x) / (3 * dx ** 2)
        if i > 0:
            i_prime = i - 1
            T_x[i, i_prime] = (16 * (-1)**(i - i_prime) * mass_x * np.exp(0.5*(x[i] - x[i_prime])**2)) / (3 * np.pi * dx**2)
        if i < Nx - 1:
            i_prime = i + 1
            T_x[i, i_prime] = (16 * (-1)**(i - i_prime) * mass_x * np.exp(0.5*(x[i] - x[i_prime])**2)) / (3 * np.pi * dx**2)

    for j in range(Ny):
        j_prime = j
        T_y[j, j_prime] = (4 * mass_y) / (3 * dy ** 2)
        if j > 0:
            j_prime = j - 1
            T_y[j, j_prime] = (16 * (-1)**(j - j_prime) * mass_y * np.exp(0.5*(y[j] - y[j_prime])**2)) / (3 * np.pi * dy**2)
        if j < Ny - 1:
            j_prime = j + 1
            T_y[j, j_prime] = (16 * (-1)**(j - j_prime) * mass_y * np.exp(0.5*(y[j] - y[j_prime])**2)) / (3 * np.pi * dy**2)

    for k in range(Nz):
        k_prime = k
        T_z[k, k_prime] = (4 * mass_z) / (3 * dz ** 2)
        if k > 0:
            k_prime = k - 1
            T_z[k, k_prime] = (16 * (-1) ** (k - k_prime) * mass_z * np.exp(0.5 * (z[k] - z[k_prime]) ** 2)) / (3 * np.pi * dz**2)
        if k < Nz - 1:
            k_prime = k + 1
            T_z[k, k_prime] = (16 * (-1) ** (k - k_prime) * mass_z * np.exp(0.5 * (z[k] - z[k_prime]) ** 2)) / (3 * np.pi * dz**2)

    T = np.kron(np.kron(T_x, np.eye(Ny)), np.eye(Nz)) + np.kron(np.kron(np.eye(Nx), T_y), np.eye(Nz)) + np.kron(
        np.kron(np.eye(Nx), np.eye(Ny)), T_z)
    H = T + V

    energies, wfs = np.linalg.eigh(H)

    wf = wfu.normalise_wf3(wfs, x, y, z, neig)
    energies = wfu.evaluate_energies_3d(wf, x, y, z, v, masses, neig)

    return energies[:neig], wf, H



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
    masses = [mass_x, mass_y]

    H = np.zeros((Nx*Ny, Nx*Ny))

    a = np.pi / (3 * mass_x)
    ay = np.pi / (3 * mass_y)

    for i in range(Nx):
        for j in range(Ny):
            n = i * Ny + j
            for m in range(Nx):
                for l in range(Ny):
                    m_new = m * Ny + l
                    if i == m and j == l:
                        H[n, m_new] += -3*mass_x * (1/8) * (mass_x*((4*Lx) - (np.pi*np.tanh(a))) - 4*v[m, l] + 4*np.pi)
                        H[n, m_new] += -3*mass_y * (1/8) * (mass_y*((4*Ly) - (np.pi*np.tanh(ay))) - 4*v[m, l] + 4*np.pi)
                        #H[n, m_new] = v[m, l]
                    elif (l == j + 1 and i == m):
                        H[n, m_new] = ((-3 * mass_y) * (
                                np.exp(0.5 * (y[j] - y[l]) ** 2) * np.tanh(np.pi))) / (
                                                  4 * dy * np.abs(y[j] - y[l]))
                        H[m_new, n] = ((-3 * mass_y) * (
                                np.exp(0.5 * (y[l] - y[j]) ** 2) * np.tanh(np.pi))) / (
                                              4 * dy * np.abs(y[l] - y[j]))
                    elif m == i - 1 and j == l:
                        H[n, m_new] = ((-3 * mass_x) * (np.exp(0.5*(x[i] - x[m])**2) * np.tanh(np.pi))) / (4 * dx * np.abs(x[i] - x[m]))
                    elif m == i + 1 and j == l:
                        H[n, m_new] = ((-3 * mass_x) * (np.exp(0.5*(x[i] - x[m])**2) * np.tanh(np.pi))) / (4 * dx * np.abs(x[i] - x[m]))

    eigval, eigvec = np.linalg.eigh(H)

    wf = wfu.normalise_wf2(eigvec, x, y, neig)
    energies = wfu.evaluate_energies_2d(wf, x, y, v, masses, neig)

    return energies, wf, H


def algorithm_36_sparse(x, v, mass, neig):
    ngrid = len(x)
    dx = x[1] - x[0]
    L = x[-1] - x[0]
    masses = [mass_x, mass_y]

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

    return energies, wf, H


def algorithm_29_2D(x, y, v, mass_x, mass_y, neig):

    Nx = len(x)
    Ny = len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Lx = x[-1] - x[0]
    Ly = y[-1] - y[0]

    masses = [mass_x, mass_y]

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
    energies = wfu.evaluate_energies_2d(wf, x, y, v, masses, neig)

    return energies, wf, H


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

def algorithm_27(x, v, mass, neig):

    ngrid = len(x)
    dx = x[1] - x[0]
    L = x[-1] - x[0]

    H = np.zeros((ngrid, ngrid))
    for i in range(ngrid):
        H[i, i] = - ((v[i, i] * dx) * (L * (np.tanh(L + (4*dx)/3 - (dx/(3*v[i, i]*mass) )) - 2) - 2*np.pi) - np.pi**2) / np.pi
        if i > 0:
            j = i - 1
            H[i, j] = -((L * dx) * (np.tanh(((4 * mass) - np.exp((x[i]-x[j])**2)) / (3 * mass)) - 2) * np.exp(-(x[i]-x[j])**2)) / ( np.pi * (-1)**(i+1 - j+1) * abs((x[i]-x[j])**2))
        if i < ngrid - 1:
            j = i + 1
            H[i, j] = -((L * dx) * (np.tanh(((4 * mass) - np.exp((x[i]-x[j])**2)) / (3 * mass)) - 2) * np.exp(-(x[i]-x[j])**2)) / ( np.pi * (-1)**(i+1 - j+1) * abs((x[i]-x[j])**2))

    eigval, eigvec = np.linalg.eigh(H) # output - eigenvector of H

    wf = wfu.normalise_wf(eigvec, x, neig)
    energies = wfu.evaluate_energies(wf, x, v, neig)

    return wf, energies, H

def algorithm_27_2D(x, y, v, mass_x, mass_y, neig):

    Nx = len(x)
    Ny = len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Lx = x[-1] - x[0]
    Ly = y[-1] - y[0]

    H_x = np.zeros((Nx, Nx))
    H_y = np.zeros((Ny, Ny))

    for i in range(Nx):
        for i_prime in range(Nx):
            if i == i_prime:
                H_x[i, i_prime] = - ((v[i, i_prime] * dx) * (Lx * (np.tanh(Lx + (4*dx)/3 - (dx/(3*v[i, i_prime]*mass_x) )) - 2) - 2*np.pi) - np.pi**2) / np.pi
            else:
                H_x[i, i_prime] = -((Lx * dx) * (np.tanh(((4 * mass_x) - np.exp((x[i]-x[i_prime])**2)) / (3 * mass_x)) - 2) * np.exp(-(x[i]-x[i_prime])**2)) / ( np.pi * (-1)**(i+1 - i_prime+1) * abs((x[i]-x[i_prime])**2))

    for j in range(Ny):
        for j_prime in range(Ny):
            if j == j_prime:
                H_y[j, j_prime] = - ((v[j, j_prime] * dy) * (Ly * (np.tanh(Ly + (4*dy)/3 - (dy/(3*v[j, j_prime]*mass_y) )) - 2) - 2*np.pi) - np.pi**2) / np.pi
            else:
                H_y[j, j_prime] = -((Ly * dy) * (np.tanh(((4 * mass_y) - np.exp((y[j]-y[j_prime])**2)) / (3 * mass_y)) - 2) * np.exp(-(y[j]-y[j_prime])**2)) / ( np.pi * (-1)**(j+1 - j_prime+1) * abs((y[j]-y[j_prime])**2))

    Hx = np.kron(H_x, np.eye(Ny))
    Hy = np.kron(np.eye(Nx), H_y)
    H = Hx + Hy
    eigval, eigvec = np.linalg.eigh(H)

    wf = wfu.normalise_wf2(eigvec, x, y, neig)
    energies = wfu.evaluate_energies_2d(wf, x, y, v, neig)

    return energies, wf, H



if __name__ == "__main__":

    neig = 6
    Nx = 21
    Ny = 21
    x = np.linspace(-5, 5, Nx)
    y = np.linspace(-5, 5, Ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    mass_x, mass_y = 1, 1

    v = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            v[i, j] = potf.harmonic_potential_2d(x[i], y[j])

    energies, wf, H = algorithm_100_2D(x, y, v, mass_x, mass_y, neig)
    print(energies)
    energies, wf, H = algorithm_36_2D(x, y, v, mass_x, mass_y, neig)
    print(energies)
    energies, wf, H = algorithm_29_2D(x, y, v, mass_x, mass_y, neig)
    print(energies)
    energies, wf, H = algorithm_27_2D(x, y, v, mass_x, mass_y, neig)
    print(energies)
    breakpoint()

