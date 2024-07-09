from __future__ import annotations
from typing import Callable
import numpy as np
from scipy.stats import qmc
import numpy.typing as npt
from fast_dvr.exact_solvers import HEG_procedure

def generate_grid(transformation_matrix, qmins, qmaxs, mode_indicies, ngrids, grid_type='product', **kwargs):
    """
    Generate a grid in normal mode coordinates.
    Can be a direct product grid or Sobol grid.

    Parameters
    ----------
    transformation_matrix : numpy.ndarray
        transformation matrix from normal mode to cartesian coordinates, columns are eigenmodes.
    qmins: list
        minimum value of the normal mode coordinates to be varied - ordered according to mode_indicies
    qmaxs: list
        maximum value of the normal mode coordinates to be varied - ordered accoridng to mode_indicies
    mode_indicies: list
        which normal modes to vary. [0, 1, ... , n] are the first, second and nth normal mode according to
        energy ordering.
    ngrid: int
        if grid_type == 'product' - the number of points along each normal coordinate axis,
        else if grid_type == 'sobol' - the total number of points to sample in the n dimensional space.
        Warning - sobol grids balance properties are only guaranteed for powers of 2^n.
    grid_type: str
        selects the type of grid - either product or sobol.

    Takes any additional kwargs of scipy.qmc.Sobol - e.g. scramble, seed ect.
    """
    ndof = len(mode_indicies)
    ndof_total = transformation_matrix.shape[0]
    nvibs = ndof_total - 6
    ndof_rot_trans = ndof_total - nvibs
    if ndof != len(qmins) and len(qmaxs):
            raise Exception('Number of indices of variable vibrational modes must match the number of axis limits.')
    
    if grid_type == 'product':
        grid_points = direct_product_grid(qmins, qmaxs, ngrids, ndof)
    elif grid_type == 'sobol':
        grid_points = sobol_grid(qmins, qmaxs, ngrids, ndof, **kwargs)
    else:
        raise Exception('Grid type must be either product (direct) or Sobol.')

    ngrid_total = grid_points.shape[0]
    qvib_coords = np.zeros((ngrid_total, nvibs))
    qvib_coords[:, mode_indicies] = grid_points
    qcoordinates = np.concatenate([np.zeros((ngrid_total, ndof_rot_trans)), qvib_coords], axis=1)
    return qcoordinates
        

def direct_product_grid(qmins, qmaxs, ngrids, ndof):
    ngrid_total = np.prod(ngrids)
    grids = []
    for d in range(ndof):
        qgrid = np.linspace(qmins[d], qmaxs[d], ngrids[d])
        grids.append(qgrid)
    meshed_grids = np.meshgrid(*grids, indexing='ij')
    grid_points = np.column_stack([axis.flatten() for axis in meshed_grids])
    return grid_points


def sobol_grid(qmins, qmaxs, ngrid, ndof, **kwargs):
    exponent = int(np.ceil(np.log2(ngrid)))
    if ngrid != 2**exponent:
        raise Warning('Balance properties not guaranteed - ngrid is not a power of 2.')
    sampler = qmc.Sobol(d=ndof, **kwargs)
    sample = sampler.random_base2(m=exponent)
    #discrep = qmc.discrepancy(sample)
    scaled = qmc.scale(sample, qmins, qmaxs)
    scaled = scaled[:ngrid, :]
    #sample = np.squeeze(scaled)
    return scaled


def get_quadrature_points(grids: list[npt.NDArray], x_mins: npt.NDArray, x_maxs: npt.NDArray, nbases: npt.NDArray, basis_func: tuple[str, Callable]):
    """
    Gets the quadrature points along each dimension/ axis by diagonalising the position operator
    (or a function of) via the HEG procedure. If one dimension is requested, the array `quad_points`
    is simply 1D. In the case of $d$ dimensions, a 2D array of shape $[n^d, d]$ is returned,
    where $n$ is the number of basis functions along each axis. This defines the coordinates
    of the direct product grid of quadrature points in $d$ dimensions, and can be used to
    then evaluate the potential operator in the diagonal basis.

    NOTE: This should be constructed in basis of the coordinate system you intend to use
          in the subsequent DVR calculation e.g. normal coordinates.
          This can then be transformed back to Cartesian's, or whatever coordinate system required
          for evaluating the potential energy matrix elements.

    :param grids: list of 1D arrays that define the position operator of a direct product grid
    :param x_mins: minimum value along each axis as an array
    :param x_maxs: maximum value along each axis as an array
    :param nbases: number of basis functions per axis
    :param basis_func: dict key value pair of the type of basis and a callable function that constructs it
    :return: quad_points - the quadrature points flattened into a 2D array if > one dimension
    """
    ndims = len(grids)
    quad_grids = []
    for d in range(ndims):
        quad_points, _ = HEG_procedure(grids[d], x_mins[d], x_maxs[d], nbases[d], basis_func)
        quad_grids.append(quad_points)
    if ndims == 1:
        return quad_points
    else:
        meshed_grids = np.meshgrid(*quad_grids, indexing='ij') # construct direct product grid
        quad_points = np.column_stack([axis.flatten() for axis in meshed_grids])
        return quad_points




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import potentials as potf
    from fast_dvr.exact_solvers import get_pib_basis, basis_funcs

    ngrid = 41
    x = np.linspace(-5, 5, ngrid)
    v = potf.harmonic(x, k=1)
    nbasis = 25
    basis_func = ('sine', get_pib_basis)

    x_min, x_max = np.min(x), np.max(x)
    dx = x[1] - x[0]
    L = x_max - x_min

    grids = [x, x]
    xmins = [x_min, x_min]
    xmaxs = [x_max, x_max]
    nbases = [nbasis, nbasis]

    quad_grid = get_quadrature_points(grids, xmins, xmaxs, nbases, basis_func)

    breakpoint()