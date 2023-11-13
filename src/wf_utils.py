import numpy as np
from scipy.fftpack import fft, fftshift, fft2

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
        integral = np.trapz(np.trapz(np.conj(wfun) * wfun, x, axis=1), y)
        nconst = 1.0 / np.sqrt(integral)
        wfn[:, :, i] = nconst * wfun
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


def evaluate_energies_2d(wf, x, y, v, neig):
    energies = np.zeros(neig)
    Nx, Ny = len(x), len(y)
    for i in range(neig):
        wfx = wf[:, :, i]
        kx, ky, wfp = position_to_momentum2(wfx, x, y)
        a = np.zeros((Nx, Ny))
        #for j in range(Nx):
        #    a[j, :] = np.conj(wfx[j, :]) * v[j, :] * wfx[j, :]
        #V = np.trapz(np.trapz(a, x, axis=1), y)
        V = np.trapz(np.trapz(np.conj(wfx) * (v * wfx), x, axis=1), y)
        energies[i] += V.real
        b = np.zeros((Nx, Ny))
        for j in range(Nx):
            b[j, :] = (np.conj(wfp[j, :]) * (kx**2) * wfp[j, :]).real
        T = np.trapz(np.trapz(b, kx, axis=1), ky)
        #T = np.trapz(np.trapz(np.conj((wfp.T * kx**2).T) * wfp, kx, axis=1) * ky**2, ky)
        T /= np.trapz(np.trapz(np.conj(wfp) * wfp, kx, axis=1), ky)
        energies[i] += T.real
    return energies

