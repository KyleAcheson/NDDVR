import numpy as np
from scipy.stats import qmc

def generate_grid(transformation_matrix, qmins, qmaxs, mode_indicies, ngrid, grid_type='product', **kwargs):
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
        grid_points = direct_product_grid(qmins, qmaxs, ngrid, ndof)
    elif grid_type == 'sobol':
        grid_points = sobol_grid(qmins, qmaxs, ngrid, ndof, **kwargs)
    else:
        raise Exception('Grid type must be either product (direct) or Sobol.')

    ngrid_total = grid_points.shape[0]
    qvib_coords = np.zeros((ngrid_total, nvibs))
    qvib_coords[:, mode_indicies] = grid_points
    qcoordinates = np.concatenate([np.zeros((ngrid_total, ndof_rot_trans)), qvib_coords], axis=1)
    return qcoordinates
        

def direct_product_grid(qmins, qmaxs, ngrid, ndof):
    ngrid_total = ngrid**ndof
    grids = []
    for d in range(ndof):
        qgrid = np.linspace(qmins[d], qmaxs[d], ngrid)
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