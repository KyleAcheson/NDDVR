import os
import sys
import numpy as np
sys.path.extend([os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '../src'),
                 os.path.dirname(os.path.dirname(os.path.realpath(__file__)))])
import src.dvr as dvr
import src.plotting as pltg
import src.potential_functions as potf
import src.synthesised_algorithms as sa
import matplotlib.pyplot as plt
import src.wf_utils as utils

def run_1D_comparison():

    mass = 1
    neig = 1
    xmin, xmax = -5.0, 5.0
    Nxs = [11, 21, 41]
    fig, ax = plt.subplots(nrows=6, figsize=(10, 10))
    for Nx in Nxs:

        x = np.linspace(xmin, xmax, Nx)
        v = np.zeros(Nx)
        for i in range(Nx):
            v[i] = potf.harmonic(x[i])

        wfs_dvr, energies_dvr, Hd = dvr.cm_dvr(x, v, mass, neig)
        wfs, energies, H2 = sa.algorithm_2(x, v, mass, neig)
        wfs, energies_80, H80 = sa.algorithm_80(x, v, mass, neig)
        wfs, energies, H36 = sa.algorithm_36(x, v, mass, neig)
        wfs, energies, H100 = sa.algorithm_100(x, v, mass, neig)
        wfs, energies, H29 = sa.algorithm_29(x, v, mass, neig)

        ax[0].plot(x, Hd[int((Nx-1)/2), :])
        ax[1].plot(x, H2[int((Nx-1)/2), :])
        ax[2].plot(x, H80[int((Nx-1)/2), :])
        ax[3].plot(x, H36[int((Nx-1)/2), :])
        ax[4].plot(x, H100[int((Nx-1)/2), :])
        ax[5].plot(x, H29[int((Nx-1)/2), :])

    fig.show()
    breakpoint()

def run_2D_comparison(x, y, neig, plot=False):

    Nx, Ny = len(x), len(y)
    mass_x, mass_y = 1, 1
    v = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            v[i, j] = potf.harmonic_potential_2d(x[i], y[j])

    masses = [mass_x, mass_y]
    energies_dvr, wfs_dvr, Hd = dvr.cm_dvr_2d(x, y, v, neig)
    energies_2, wfs_2, H2 = sa.algorithm_2_2D(x, y, v, mass_x, mass_y, neig)
    energies_29, wfs_29, H29 = sa.algorithm_29_2D(x, y, v, mass_x, mass_y, neig)
    energies_36, wfs_36, H36 = sa.algorithm_36_2D(x, y, v, mass_x, mass_y, neig)

    wfs_dvr = utils.normalise_wf2(wfs_dvr, x, y, neig)
    energies = utils.evaluate_energies_2d(wfs_dvr, x, y, v, masses, neig)

    print(f'dvr eigval: {energies_dvr}')
    print(f'dvr expec: {energies}')
    print(f'A2 expec: {energies_2}')
    print(f'A29 expec: {energies_29}')
    print(f'A36 expec: {energies_36}')

    if plot:
        pltg.plot_wavefunctions_2d(x, y, wfs_dvr, energies_dvr, num_to_plot=neig, fname=f'DVR_2DHO_Nxy{Nx}')
        pltg.plot_wavefunctions_2d(x, y, wfs_2, energies_2, num_to_plot=neig, fname=f'A2_2DHO_Nxy{Nx}')
        pltg.plot_wavefunctions_2d(x, y, wfs_29, energies_29, num_to_plot=neig, fname=f'A29_2DHO_Nxy{Nx}')
        pltg.plot_wavefunctions_2d(x, y, wfs_36, energies_36, num_to_plot=neig, fname=f'A36_2DHO_Nxy{Nx}')

    return energies_dvr, energies_2

def run_3D_comparison(x, y, z, v, mass_x, mass_y, mass_z, neig, plot=False):
    Nx, Ny, Nz = len(x), len(y), len(z)

    energies_dvr, wfs_dvr, Hd = dvr.cm_dvr_3d(x, y, z, v, neig)
    energies_2, wfs_2, H2 = sa.algorithm_2_3D(x, y, z, v, mass_x, mass_y, mass_z, neig)

    masses = [mass_x, mass_y, mass_z]
    wfs_dvr = utils.normalise_wf3(wfs_dvr, x, y, z, neig)
    energies = utils.evaluate_energies_3d(wfs_dvr, x, y, z, v, masses, neig)
    #wfs_dvr = wfs_dvr.reshape((Nx, Ny, Nz, neig))
    #wfs_2 = wfs_2.reshape((Nx, Ny, Nz, neig))
    print(f'dvr eigval: {energies_dvr}')
    print(f'dvr expec: {energies}')
    print(f'A2 expec: {energies_2}')
    if plot:
        pltg.plot_wavefunctions_3d(x, y, z, wfs_dvr, energies_dvr, num_to_plot=neig, fname=f'DVR_3DHO_Nxy{Nx}')
        pltg.plot_wavefunctions_3d(x, y, z, wfs_2, energies_2, num_to_plot=neig, fname=f'A2_3DHO_Nxy{Nx}')


def test_grid_convergence():
    grids = [11, 15, 21, 31, 51]
    ngrids = len(grids)
    neig = 3
    energies_dvr = np.zeros((ngrids, neig))
    energies_100 = np.zeros((ngrids, neig))
    energies_36 = np.zeros((ngrids, neig))
    energies_29 = np.zeros((ngrids, neig))
    for i, Nx in enumerate(grids):
        Ny = Nx
        x = np.linspace(-5, 5, Nx)
        y = np.linspace(-5, 5, Ny)
        edvr, e100, e36, e29 = run_2D_comparison(x, y, neig, plot=False)
        energies_dvr[i, :] = edvr
        energies_100[i, :] = e100
        energies_36[i, :] = e36
        energies_29[i, :] = e29

    for i in range(neig):
        fig, ax = plt.subplots()
        ax.plot(grids, energies_dvr[:, i], '-o', label='cm-dvr')
        ax.plot(grids, energies_100[:, i], '-o', label='A100')
        ax.plot(grids, energies_36[:, i], '-o', label='A36')
        ax.plot(grids, energies_29[:, i], '-o', label='A36')
        ax.set_xlabel('$N_{xy}$')
        ax.set_ylabel(f'$E_{i}$')
        fig.savefig(f'2D_convergence_eig{i}.png')



if __name__ == "__main__":
    neig = 3
    Nx = 21  # Number of grid points in x
    Ny = 21  # Number of grid points in y
    Nz = 21  # Number of grid points in y
    xmin, xmax = -5.0, 5.0
    ymin, ymax = -5.0, 5.0
    zmin, zmax = -5.0, 5.0
    x = np.linspace(xmin, xmax, Nx)
    y = np.linspace(ymin, ymax, Ny)
    z = np.linspace(zmin, zmax, Nz)

    run_2D_comparison(x, y, neig, True)

#    v = np.zeros((Nx, Ny, Nz))
#    for i in range(Nx):
#        for j in range(Nx):
#            for k in range(Nz):
#                v[i, j, k] = potf.harmonic_potential_3d(x[i], y[j], z[k])
#
#
#    run_3D_comparison(x, y, z, v, 1, 1, 1, neig, plot=True)
#
#