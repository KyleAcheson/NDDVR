import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import fast_dvr.dvr as dvr
import fast_dvr.potentials as pot
from fast_dvr.synthesised_solvers import *
from fast_dvr.exact_solvers import *
import fast_dvr.wf_utils as wfu
import fast_dvr.transforms as tf
import fast_dvr.grids as grids
import matplotlib.pyplot as plt
from pyscf import gto, hessian, dft
from pyscf.hessian.thermo import harmonic_analysis
from sklearn.gaussian_process.kernels import RBF

ELEC_MASS = 9.10938356E-31
AU2EV = 27.2114
AU2WAVNUM = 219474.63
AVOGADRO = 6.022140857E23
BOHR = 0.52917721092  # Angstroms
BOHR_SI = BOHR * 1e-10
ATOMIC_MASS = 9.109E-31
HARTREE2J = 4.359744650e-18
HARTREE2EV = 27.21138602
LIGHT_SPEED_SI = 299792458
AMU2AU = (1E-3 / AVOGADRO) / ELEC_MASS

AU2Hz = ((HARTREE2J / (ATOMIC_MASS * BOHR_SI ** 2)) ** 0.5 / (2 * np.pi))


def write_to_xyz(coords, atom_labels, fname):
    natoms, _, npoints = coords.shape
    with open(fname, 'a+') as f:
        for i in range(npoints):
            coord = coords[:, :, i] * BOHR
            f.write(f'{natoms}\n')
            f.write(f'grid point {i + 1}\n')
            array_string = '\n'.join(
                [f'{atom_labels[ind]}\t' + '\t'.join(map(str, row)) for ind, row in enumerate(coord)]) + '\n'
            f.write(array_string)

def fit_potential(v_train, q_train, q_pred, **kwargs):
    ndof = 3
    kernel = RBF(length_scale=kwargs['length_scale'], length_scale_bounds=kwargs['length_scale_bounds'])
    v_pred, v_std, gp = pot.fit_potential(q_train, q_pred, v_train, ndof, kernel)
    return v_pred



def generate_ncoords(outdir, coords, masses, hessian, variable_modes, qmins, qmaxs, ngrids):
    ngrid_prod = np.prod(ngrids)
    ndof = len(variable_modes)
    variable_modes = np.array(variable_modes)
    inds = variable_modes + 6

    fconsts, tmat = tf._diag_hessian(hessian, masses)
    freqs = np.lib.scimath.sqrt(fconsts)  # in a.u.
    freqs_wavenums = freqs * AU2Hz / LIGHT_SPEED_SI * 1e-2
    print(freqs_wavenums[-3:])

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fig = plt.subplots()
    for i in range(ndof):
        qmin = [qmins[i]]
        qmax = [qmaxs[i]]
        variable_mode = [variable_modes[i]]
        grid = [ngrids[i]]
        q_prod = grids.generate_grid(tmat, qmin, qmax, variable_mode, grid, grid_type='product')
        cart = tf.norm2cart_grid(q_prod, coords, masses, tmat)
        v = pot.partridge_schwenke_cart(cart)
        q_prod = q_prod[:, inds[i]]
        write_to_xyz(cart, ['0', 'H', 'H'], fname=f'{outdir}/nmode{variable_modes[i]}.xyz')
        np.savetxt(f'{outdir}/nm{i}_grid.txt', q_prod)
        plt.plot(q_prod, v * AU2WAVNUM, label=i)
        print(np.min(v * AU2WAVNUM))
    plt.xlabel('$q$')
    plt.ylabel('$v$ (cm^-1)')
    plt.legend()
    plt.savefig(f'{outdir}/pot_1d_cuts.png')
    plt.show()

def generate_whole_potential(wdir, coords, masses, hessian, variable_modes, qmins, qmaxs, ngrids, nbases, get_quad=False, grid_type='product', **kwargs):
    if get_quad:
        ngrid_prod = np.prod(nbases)
    elif get_quad and grid_type == 'sobol':
        raise TypeError
    elif grid_type == 'sobol' and not get_quad:
        ngrid_prod = np.prod(ngrids)
        nsobol = kwargs.get('nsobol')
    elif grid_type == 'product' and not get_quad:
        ngrid_prod = np.prod(ngrids)
    outdir = f'{wdir}/ngrid_{ngrid_prod}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ndof = len(variable_modes)
    variable_modes = np.array(variable_modes)
    inds = variable_modes + 6


    fconsts, tmat = tf._diag_hessian(hessian, masses)
    freqs = np.lib.scimath.sqrt(fconsts)  # in a.u.
    freqs_wavenums = freqs * AU2Hz / LIGHT_SPEED_SI * 1e-2
    print(freqs_wavenums[-3:])

    if get_quad:
        q_grids = []
        for i in range(ndof):
            q_grids.append(np.linspace(qmins[i], qmaxs[i], ngrids[i]))
        q_prod = grids.get_quadrature_points(q_grids, qmins, qmaxs, nbases, ('sine', get_pib_basis))
        q_prod = np.concatenate([np.zeros((ngrid_prod, 6)), q_prod], axis=1)
        cart_coords_prod = tf.norm2cart_grid(q_prod[0:1, :], coords, masses, tmat)  # for JIT compilation
        cart_coords_prod = tf.norm2cart_grid(q_prod, coords, masses, tmat)
        v = pot.partridge_schwenke_cart(cart_coords_prod)
    else:
        if grid_type == 'product':
            q_prod = grids.generate_grid(tmat, qmins, qmaxs, variable_modes, ngrids, grid_type=grid_type)
            cart_coords_prod = tf.norm2cart_grid(q_prod[0:1, :], coords, masses, tmat)  # for JIT compilation
            cart_coords_prod = tf.norm2cart_grid(q_prod, coords, masses, tmat)
            v = pot.partridge_schwenke_cart(cart_coords_prod)
        elif grid_type == 'sobol':
            q_train = grids.generate_grid(tmat, qmins, qmaxs, variable_modes, nsobol, grid_type=grid_type)
            cart_coords_train = tf.norm2cart_grid(q_train[0:1, :], coords, masses, tmat)  # for JIT compilation
            cart_coords_train = tf.norm2cart_grid(q_train, coords, masses, tmat)
            v_train = pot.partridge_schwenke_cart(cart_coords_train)
            q_prod = grids.generate_grid(tmat, qmins, qmaxs, variable_modes, ngrids, grid_type='product')
            cart_coords_prod = tf.norm2cart_grid(q_prod, coords, masses, tmat)
            v = fit_potential(v_train, q_train, q_prod, **kwargs)




    q_prod = q_prod[:, inds]
    np.savetxt(f'{outdir}/exact_grid.txt', q_prod)
    np.savetxt(f'{outdir}/exact_potential.txt', v)



if __name__ == "__main__":
    out_dir = '/home/kyle/DVR_Applications/H2Oc/sobol/exp8'

    # If get_quad == True - diagonalises the position operator defined on
    # a direct product grid according to ngrids. This yields a sine DVR
    # basis defined by nbases, in which the potential is evaluated.
    # Otherwise, the potential is evaluated directly on the direct product
    # grid defined by the product of ngrids points.

    get_quad = False
    grid_type = 'sobol'

    ngrids = np.array([81, 61, 61])
    nsobol = 2**8 # if sampling using sobol grids
    nbases = np.array([81, 61, 61])
    q_mins = np.array([-80, -50, -30])
    q_maxs = np.array([70, 25, 30])
    variable_modes = np.array([0, 1, 2])

    natoms = 3
    labels = ['O', 'H', 'H']
    masses = np.array([15.999, 1.008, 1.008])
    masses *= AMU2AU

    # Partridge-Schwenke minima in Ang
    eq_coords = np.array([[0.00, 0.00, 0.00],
                          [0.95865, 0.00, 0.00],
                          [-0.237556, 0.928750, 0.00]])
    eq_coords *= (1 / BOHR)

    hessian = pot.partridge_schwenke_hessian()
    #generate_ncoords(out_dir, eq_coords, masses, hessian, variable_modes, q_mins, q_maxs, ngrids)
    generate_whole_potential(out_dir, eq_coords, masses, hessian, variable_modes, q_mins, q_maxs, ngrids, nbases,
                             get_quad=get_quad, grid_type=grid_type, nsobol=nsobol,
                             length_scale=15, length_scale_bounds=(2, 30))
    breakpoint()
