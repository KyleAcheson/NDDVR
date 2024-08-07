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
from scipy.spatial.distance import cdist

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
            f.write(f'grid point {i+1}\n')
            array_string = '\n'.join([f'{atom_labels[ind]}\t' + '\t'.join(map(str, row)) for ind, row in enumerate(coord)]) + '\n'
            f.write(array_string)


def fit_potential(v_train, q_train, q_pred, **kwargs):
    ndof = 3
    kernel = RBF(length_scale=kwargs['length_scale'], length_scale_bounds=kwargs['length_scale_bounds'])
    v_pred, v_std, gp = pot.fit_potential(q_train, q_pred, v_train, ndof, kernel)
    return v_pred


def generate_ncoords(outdir, coords, masses, hessian, variable_modes, qmins, qmaxs, ngrids, **kwargs):

    ngrid_prod = np.prod(ngrids)
    masses *= AMU2AU
    ndof = len(variable_modes)
    variable_modes = np.array(variable_modes)
    inds = variable_modes + 6

    coords *= (1 / BOHR)
    fconsts, tmat = tf._diag_hessian(hessian, masses)
    freqs = np.lib.scimath.sqrt(fconsts)  # in a.u.
    freqs_wavenums = freqs * AU2Hz / LIGHT_SPEED_SI * 1e-2
    print(freqs_wavenums[-3:])

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    v_eq = potential_calcultor_1d(labels, coords, basis=kwargs['basis'], xc=kwargs['xc'], units='Bohr')
    fig = plt.subplots()
    for i in range(ndof):
        qmin = [qmins[i]]
        qmax = [qmaxs[i]]
        variable_mode = [variable_modes[i]]
        grid = [ngrids[i]]
        q_prod = grids.generate_grid(tmat, qmin, qmax, variable_mode, grid, grid_type='product')
        cart = tf.norm2cart_grid(q_prod, coords, masses, tmat)
        v = potential_calcultor_1d(labels, cart, basis=kwargs['basis'], xc=kwargs['xc'], units='Bohr')
        v -= v_eq
        q_prod = q_prod[:, inds[i]]
        write_to_xyz(cart, ['S', 'O', 'O'], fname=f'{outdir}/nmode{variable_modes[i]}.xyz')
        np.savetxt(f'{outdir}/nm{i}_grid.txt', q_prod)
        plt.plot(q_prod, v * AU2WAVNUM, label=i)
        print(np.min(v*AU2WAVNUM))
    plt.xlabel('$q$')
    plt.ylabel('$v$ (cm^-1)')
    plt.legend()
    plt.savefig(f'{outdir}/pot_1d_cuts.png')
    plt.show()

def potential_calcultor_1d(labels, cart_coords, **kwargs):
    if len(cart_coords.shape) > 2:
        natoms, _, npoints = cart_coords.shape
    else:
        natoms, _ = cart_coords.shape
        npoints = 1
        cart_coords = cart_coords.reshape(natoms, 3, 1)
    v = np.zeros(npoints)
    for i in range(npoints):
        pyscf_coords = cart_coords[:, :, i].tolist()
        for j in range(natoms):
            pyscf_coords[j].insert(0, labels[j])
        mol = gto.M(atom=pyscf_coords, basis=kwargs['basis'], verbose=0, unit=kwargs['units'])
        mf = dft.RKS(mol)
        mf.xc = kwargs['xc']
        mf.kernel()
        v[i] = mf.e_tot
    if len(v) == 1:
        return v[0]
    else:
        return v
    
    

def generate_whole_potential(wdir, coords, masses, hessian, variable_modes, qmins, qmaxs, ngrids, nbases, get_quad=False, grid_type='product', **kwargs):

    if get_quad:
        ngrid_prod = np.prod(nbases)
    elif get_quad and grid_type == 'sobol':
        raise TypeError
    elif grid_type == 'sobol' and not get_quad:
        nsobol = kwargs.get('nsobol')
        ngrid_prod = nsobol
    elif grid_type == 'product' and not get_quad:
        ngrid_prod = np.prod(ngrids)
    outdir = f'{wdir}/ngrid_{ngrid_prod}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    masses *= AMU2AU
    coords *= (1/BOHR)
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
    else:
        if grid_type == 'product':
            q_prod = grids.generate_grid(tmat, qmins, qmaxs, variable_modes, ngrids, grid_type=grid_type)
            cart_coords_prod = tf.norm2cart_grid(q_prod[0:1, :], coords, masses, tmat) # for JIT compilation
            cart_coords_prod = tf.norm2cart_grid(q_prod, coords, masses, tmat)
        elif grid_type == 'sobol':
            q_prod = grids.generate_grid(tmat, qmins, qmaxs, variable_modes, nsobol, grid_type=grid_type)
            cart_coords_prod = tf.norm2cart_grid(q_prod[0:1, :], coords, masses, tmat)  # for JIT compilation
            cart_coords_prod = tf.norm2cart_grid(q_prod, coords, masses, tmat)
        

    q_prod = q_prod[:, inds]
    np.savetxt(f'{outdir}/exact_grid.txt', q_prod)
    write_to_xyz(cart_coords_prod, ['S', 'O', 'O'], fname=f'{outdir}/cartesian_grids_ngrid{ngrid_prod}.xyz')


def pyscf_freq(labels, masses, coords, **kwargs):
    natoms = len(labels)
    pyscf_coords = coords.tolist()
    for i in range(natoms):
        pyscf_coords[i].insert(0, labels[i])

    mol = gto.M(atom=pyscf_coords, basis=kwargs['basis'], verbose=0, unit=kwargs['units'])
    mf = dft.RKS(mol)
    mf.xc = kwargs['xc']
    mf.kernel()
    hessian = mf.Hessian().kernel()
    res = harmonic_analysis(mol, hessian)
    freqs = res['freq_wavenumber']
    norm_modes = res['norm_mode'].transpose(1, 2, 0).reshape(3*natoms, 3)
    modes = np.einsum('z,zri->izr', masses**.5, norm_modes.reshape(natoms,3,-1))
    modes = modes.transpose(1, 2, 0).reshape(3*natoms, 3)
    print(freqs)
    hessian = hessian.transpose(0, 2, 1, 3).reshape(natoms*3, natoms*3)
    return hessian

if __name__ == "__main__":

    out_dir = '/home/kyle/DVR_Applications/SO2/whole_pot/sobol/exp10'

    # If get_quad == True - diagonalises the position operator defined on
    # a direct product grid according to ngrids. This yields a sine DVR
    # basis defined by nbases, in which the potential is evaluated.
    # Otherwise, the potential is evaluated directly on the direct product
    # grid defined by the product of ngrids points.

    get_quad = False
    fit_sobol = True
    grid_type = 'sobol'
    
    variable_modes = np.array([0, 1, 2])
    ngrids = np.array([41, 31, 31])
    nsobol = 2**10
    nbases = np.array([41, 31, 31])
    q_mins = np.array([-80, -50, -40])
    q_maxs = np.array([80, 40, 40])

    natoms = 3
    labels = ['S', 'O', 'O']
    masses = np.array([32.065, 15.999, 15.999])

    # def2-tzvp
    eq_coords = np.array([[-0.003392, -0.005887, 0.00],
                          [1.435506, 0.001876, 0.00],
                          [-0.705264, 1.250360, 0.00]])

    hessian = pyscf_freq(labels, masses, eq_coords, xc='B3LYP', basis='def2-tzvp', units='Angstrom')
    #generate_ncoords(out_dir, eq_coords, masses, hessian, variable_modes, q_mins, q_maxs, ngrids, xc='B3LYP', basis='def2-tzvp')
    if fit_sobol:
        v_train = np.genfromtxt(f'{out_dir}/ngrid_{nsobol}/sobol_potential.txt')
        q_train = np.genfromtxt(f'{out_dir}/ngrid_{nsobol}/sobol_grid.txt')
        q_pred = grids.direct_product_grid(q_mins, q_maxs, ngrids, ndof=3)
        v = fit_potential(v_train, q_train, q_pred, length_scale=25, length_scale_bounds=(1e-2, 2e2))
        np.savetxt(f'{out_dir}/ngrid_{nsobol}/exact_grid.txt', q_pred)
        np.savetxt(f'{out_dir}/ngrid_{nsobol}/exact_potential.txt', v)

    else:
        generate_whole_potential(out_dir, eq_coords, masses, hessian, variable_modes, q_mins, q_maxs, ngrids, nbases,
                                 get_quad=get_quad, grid_type=grid_type, nsobol=nsobol)