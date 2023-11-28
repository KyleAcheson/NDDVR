import numpy as np
from scipy.fftpack import fft, fftshift, fft2, fftn

def normalise_wf(wf, x, neig):
    ngrid = len(x)
    for eig in range(neig):
        cprod = np.conj(wf[:, eig]) * wf[:, eig]
        csum = np.trapz(cprod, x=x)
        wf[:, eig] /= np.sqrt(csum)
    return wf

def position_to_momentum(wfx, x):
    dx = x[1] - x[0]
    ngrid = len(x)
    k = 2 * np.pi * np.fft.fftfreq(ngrid, d=dx)
    k = np.sort(k)
    wfp = fftshift(fft(wfx))
    return k, wfp

def evaluate_energies(wf, x, v, neig):
    energies = np.zeros(neig)
    for eig in range(neig):
        wfx = wf[:, eig]
        V = np.trapz(np.conj(wfx) * v * wfx, x=x)
        energies[eig] += np.real(V)
        k, wfp = position_to_momentum(wfx, x)
        T = 0.5 * np.trapz(k**2 * (np.conj(wfp) * wfp), x=k)
        T /= np.trapz(np.conj(wfp) * wfp, x=k)
        energies[eig] += np.real(T)
    return energies

def normalise_wf2(wf, x, y, neig):
    Nx, Ny = len(x), len(y)
    wfn = np.zeros((Nx, Ny, neig))
    for i in range(neig):
        wfun = wf[:, i].reshape(Nx, Ny)
        integral = np.trapz(np.trapz(np.conj(wfun) * wfun, x, axis=0), y)
        nconst = 1.0 / np.sqrt(integral)
        wfn[:, :, i] = nconst * wfun
    return wfn

def normalise_wf3(wf, x, y, z, neig):
    Nx, Ny, Nz = len(x), len(y), len(z)
    wfn = np.zeros((Nx, Ny, Nz, neig))
    for i in range(neig):
        wfun = wf[:, i].reshape(Nx, Ny, Nz)
        integral = np.trapz(np.trapz(np.trapz(np.conj(wfun) * wfun, x, axis=0), y, axis=1), z)
        nconst = 1.0 / np.sqrt(integral)
        wfn[:, :, :, i] = nconst * wfun
    return wfn

def position_to_momentum2(wfx, x, y):
    Nx, Ny = len(x), len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kx = np.sort(kx)
    ky = np.sort(ky)
    #wfp = np.zeros((Nx, Ny), dtype=complex)
    #for i in range(Nx):
    #    wfp[i, :] = fftshift(fft(wfx[i, :], n=Nx))
    wfp = fftshift(fft2(wfx))
    return kx, ky, wfp

def position_to_momentum3(wfx, x, y, z):
    Nx, Ny, Nz = len(x), len(y), len(z)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kz = 2 * np.pi * np.fft.fftfreq(Nz, d=dz)
    kx = np.sort(kx)
    ky = np.sort(ky)
    kz = np.sort(kz)
    wfp = fftshift(fftn(wfx))
    return kx, ky, kz, wfp


def evaluate_energies_2d(wf, x, y, v, masses, neig):
    energies = np.zeros(neig)
    mass_x, mass_y = masses[0], masses[1]
    Nx, Ny = len(x), len(y)
    for eig in range(neig):
        try:
            wfx = wf[:, :, eig]
        except IndexError:
            wfx = wf[:, eig]
            wfx = wfx.reshape(Nx, Ny)
        kx, ky, wfp = position_to_momentum2(wfx, x, y)
        V = np.trapz(np.trapz(np.conj(wfx) * (v * wfx), x, axis=0), y)
        energies[eig] += V.real

        kmat = np.zeros((Nx, Ny))
        for i in range(Nx):
            for j in range(Ny):
                kmat[i, j] = ((kx[i]**2 / mass_x) + (ky[j]**2 / mass_y))

        T = 0.5 * np.trapz(np.trapz(np.conj(wfp) * (kmat * wfp), kx, axis=0), ky)
        T /= np.trapz(np.trapz(np.conj(wfp) * wfp, kx, axis=0), ky)
        #print(f'neig: {eig}, V: {V.real}, T: {T.real}')
        energies[eig] += T.real
    return energies


def evaluate_energies_3d(wf, x, y, z, v, masses, neig):
    energies = np.zeros(neig)
    Nx, Ny, Nz = len(x), len(y), len(z)
    for eig in range(neig):
        try:
            wfx = wf[:, :, :, eig]
        except IndexError:
            wfx = wf[:, eig]
            wfx = wfx.reshape(Nx, Ny, Nz)
        mass_x, mass_y, mass_z = masses[0], masses[1], masses[2]
        kx, ky, kz, wfp = position_to_momentum3(wfx, x, y, z)
        V = np.trapz(np.trapz(np.trapz(np.conj(wfx) * (v * wfx), x, axis=0), y, axis=1), z)
        energies[eig] += V.real

        kmat = np.zeros((Nx, Ny, Nz))
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    kmat[i, j, k] = ((kx[i]**2 / mass_x) + (ky[j]**2 / mass_y) + (kz[k]**2 / mass_z))

        T = 0.5 * np.trapz(np.trapz(np.trapz(np.conj(wfp) * (kmat * wfp), kx, axis=0), ky, axis=1), kz)
        T /= np.trapz(np.trapz(np.trapz(np.conj(wfp) * wfp, kx, axis=0), ky, axis=1), kz)
        #print(f'neig: {eig}, V: {V.real}, T: {T.real}')
        energies[eig] += T.real
    return energies
