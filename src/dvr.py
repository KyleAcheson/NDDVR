import numpy as np
#import line_profiler
import scipy.linalg as spyl
from scipy import sparse

class Calculator:

    def __init__(self, algorithm, tridiag=False):
        self.algorithm = algorithm
        self.tridiag = tridiag

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm):
        self._algorithm = algorithm

    def solve_1d(self, x, v, mass, neig, hbar=1):
        ngrid = len(x)
        V = np.zeros((ngrid, ngrid))
        for i in range(ngrid):
            V[i, i] = v[i]

        T = self.algorithm(x, mass, hbar)

        H = T + V
        energies, wfs = spyl.eigh(H, driver='evr', subset_by_index=[0, neig-1])

        return energies, wfs

    #@profile
    def solve_nd(self, grids, masses, v, neig, hbar=1, ndim=2):

        if len(grids) != ndim:
            raise TypeError('grids must be a list of arrays with length ndim')
        if len(masses) != ndim:
            raise TypeError('masses must be a list of real numbers with length ndim')

        grid_sizes = [len(grid) for grid in grids]
        total_size = np.prod(grid_sizes)
        vshape = v.shape

        if len(vshape) == ndim and vshape != tuple(grid_sizes):
            raise TypeError('if v is a nd array, it must have ndim dimensions each of length grids[n]')
        elif len(vshape) == 1 and len(v) != total_size:
            raise TypeError('if v is a 1d array, its length must be the product of all grid sizes')

        H = self.kinetic_matrix_nd(grids, masses, grid_sizes, dim=ndim)

        diag_inds = np.diag_indices(int(total_size))

        if len(v.shape) == ndim:
            H[diag_inds] += v.flatten()
        else:
            H[diag_inds] += v

        #if self.tridiag:
        #H = sparse.csr_matrix(H)
        energies, wfs = sparse.linalg.eigsh(H, k=neig, which='SA')
        #else:
        #    energies, wfs = spyl.eigh(H, driver='evr', subset_by_index=[0, neig-1])

        return energies, wfs

    #@profile
    def kinetic_matrix_nd(self, grids, masses, sizes, dim):

        if dim < 2:
            raise ValueError

        total_size = np.prod(sizes)
        #result_matrix = np.zeros((total_size, total_size))
        result_matrix = sparse.coo_matrix((total_size, total_size), dtype=np.float64)
        result_matrix = result_matrix.tocsr()

        for i in range(dim):
            matricies = [sparse.identity(sizes[d], dtype=np.float64).tocsr() for d in range(dim)]
            matricies[i] = self.algorithm(grids[i], masses[i])
            result = sparse.kron(matricies[0], matricies[1])
            for j in range(2, dim):
                result = sparse.kron(result, matricies[j])
            result_matrix += result

        return result_matrix


if __name__ == "__main__":
    import potentials as pots
    import synthesised_solvers as ss
    import exact_solvers as es
    import wf_utils as wfu
    import memory_profiler as mp

    # TEST 2D POTENTIAL FROM FILE
    pdir = '/home/kyle/PycharmProjects/Potential_Generator/potentials'
    #pot_file = f'{pdir}/harmonic/harmonic/101x101/isotropic_harmonic1_ngrid101.tab'
    #grid_file = f'{pdir}/harmonic/harmonic/101x101/isotropic_harmonic_grid_ngrid101.tab'
    pot_file = f'{pdir}/3D/harmonic/51x51x51/harmonic0_ngrid51.tab'
    grid_file = f'{pdir}/3D/harmonic/51x51x51/harmonic_grid_ngrid51.tab'

    neig = 3
    masses = [1, 1, 1]
    ndims = 3

    v, grids = pots.load_potential(pot_file, grid_file, ndim=ndims, order='F')


    calculator = Calculator(es.colbert_miller)
    start_mem = mp.memory_usage(max_usage=True)
    res = mp.memory_usage(proc=(calculator.solve_nd, [grids, masses, v, neig, 1, ndims]), max_usage=True, retval=True)

    print(start_mem)
    print(res[0])
    print(res[0] - start_mem)
    energies = res[1][0]
    wfs = res[1][1]
    print(energies)

    calculator = Calculator(ss.algorithm_116)
    start_mem = mp.memory_usage(max_usage=True)
    res = mp.memory_usage(proc=(calculator.solve_nd, [grids, masses, v, neig, 1, ndims]), max_usage=True, retval=True)
    energies = res[1][0]
    wfs = res[1][1]
    energies, wfs = wfu.evaluate_energies(wfs, grids, v, masses, neig, ndim=ndims, normalise=True)

    print(start_mem)
    print(res[0])
    print(res[0] - start_mem)
    print(energies)

    #calculator = Calculator(ss.algorithm_116)
    #energies, wfs = calculator.solve_nd(grids, masses, v, neig, ndim=ndims)
    #energies, wfs = wfu.evaluate_energies(wfs, grids, v, masses, neig, ndim=ndims, normalise=True)
    #print(energies)
