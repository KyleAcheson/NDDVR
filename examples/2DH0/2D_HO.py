import os
import sys
import numpy as np
sys.path.extend([os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'src'),
                 os.path.dirname(os.path.dirname(os.path.realpath(__file__)))])
import src.dvr as dvr
import src.plotting as pltg
import src.potential_functions as potf
import src.synthesised_algorithms as sa
import matplotlib.pyplot as plt



def run_2D_comparison(x, y, neig, plot=False):

    Nx, Ny = len(x), len(y)
    mass_x, mass_y = 1, 1
    v = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            v[i, j] = potf.harmonic_potential_2d(x[i], y[j])

    energies_dvr, wfs_dvr, _ = dvr.cm_dvr_2d(x, y, v, neig)
    energies_100, wfs_100 = sa.algorithm_100_2D(x, y, v, mass_x, mass_y, neig)
    energies_36, wfs_36 = sa.algorithm_36_2D(x, y, v, mass_x, mass_y, neig)

    if plot:
        wfs_dvr = wfs_dvr.reshape((Nx, Ny, neig))
        pltg.plot_wavefunctions_2d(x, y, wfs_dvr, energies_dvr, num_to_plot=neig, fname=f'DVR_2DHO_Nxy{Nx}')
        pltg.plot_wavefunctions_2d(x, y, wfs_100, energies_100, num_to_plot=neig, fname=f'A100_2DHO_Nxy{Nx}')
        pltg.plot_wavefunctions_2d(x, y, wfs_36, energies_100, num_to_plot=neig, fname=f'A36_2DHO_Nxy{Nx}')

    return energies_dvr, energies_100, energies_36


def test_grid_convergence():
    grids = [11, 15, 21, 31, 51]
    ngrids = len(grids)
    neig = 3
    energies_dvr = np.zeros((ngrids, neig))
    energies_100 = np.zeros((ngrids, neig))
    energies_36 = np.zeros((ngrids, neig))
    for i, Nx in enumerate(grids):
        Ny = Nx
        x = np.linspace(-5, 5, Nx)
        y = np.linspace(-5, 5, Ny)
        edvr, e100, e36 = run_2D_comparison(x, y, neig, plot=False)
        energies_dvr[i, :] = edvr
        energies_100[i, :] = e100
        energies_36[i, :] = e36

    for i in range(neig):
        fig, ax = plt.subplots()
        ax.plot(grids, energies_dvr[:, i], '-o', label='cm-dvr')
        ax.plot(grids, energies_100[:, i], '-o', label='A100')
        ax.plot(grids, energies_36[:, i], '-o', label='A36')
        ax.set_xlabel('$N_{xy}$')
        ax.set_ylabel(f'$E_{i}$')
        fig.savefig(f'2D_convergence_eig{i}.png')




if __name__ == "__main__":

    neig = 3
    Nx = 21
    Ny = 21
    x = np.linspace(-5, 5, Nx)
    y = np.linspace(-5, 5, Ny)
    edvr, e100, e36 = run_2D_comparison(x, y, neig, plot=True)
    print(edvr)
    print(e100)
    print(e36)
