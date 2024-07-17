import numpy as np
#import line_profiler
import scipy.linalg as spyl
from scipy import sparse
import fast_dvr.operators as op
import warnings

class ArpackArnoldiIter(UserWarning):
    pass

class Calculator:

    def __init__(self, algorithm, use_operators=False):
        self.algorithm = algorithm
        self.use_operators = use_operators

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm):
        self._algorithm = algorithm

    def solve_1d(self, x, v, mass, neig, nbasis=None, hbar=1):
        if not nbasis:
            nbasis = len(x)

        V = np.zeros((nbasis, nbasis))
        for i in range(nbasis):
            V[i, i] = v[i]

        T = self.algorithm(x, mass, nbasis, hbar)

        H = T + V
        energies, wfs = spyl.eigh(H, driver='evr', subset_by_index=[0, neig-1])

        return energies, wfs
    
    def solve_nd(self, grids, masses, v, neig, nbases=None, hbar=1, ndim=2):
        
        if len(grids) != ndim:
            raise TypeError('grids must be a list of arrays with length ndim')
        if len(masses) != ndim:
            raise TypeError('masses must be a list of real numbers with length ndim')

        if type(nbases) == np.ndarray:  # if nbases not provided - assumed to be number of points along each axis
            pass
        else:
            nbases = [len(grid) for grid in grids]

        total_size = np.prod(nbases)
        
        if len(v.shape) > 1:
            v = v.flatten()

        if len(v) != total_size:
            raise TypeError('v must be a 1d array, its length must be the product of all grid sizes')

        if self.use_operators:
            energies, wfs = self._solve_nd_operator(grids, nbases, masses, v, neig, hbar, ndim)
        else:
            energies, wfs = self._solve_nd_full(grids, nbases, masses, v, neig, hbar, ndim)

        return energies, wfs


    #@profile
    def _solve_nd_operator(self, grids, nbases, masses, v, neig, hbar=1, ndim=2):

        kinetic_1d_mats = []
        for i in range(ndim):
            kinetic_1d_mats.append(self.algorithm(grids[i], masses[i], nbases[i]))

        total_size = np.prod(nbases)
        H = op.Hamiltonian(v, kinetic_1d_mats, nbases)

        max_iter = np.iinfo(np.int32).max
        niter = total_size * 10
        if niter > max_iter:
            niter = max_iter
            warnings.warn("Default maximum number of Arnoldi iterations (n*10 > int32 limit) - maxiter set to int32 limit.", ArpackArnoldiIter)

        energies, wfs = sparse.linalg.eigsh(H, k=neig, which='SA', maxiter=niter)
        return energies, wfs


    #@profile
    def _solve_nd_full(self, grids, nbases, masses, v, neig, hbar=1, ndim=2):

        total_size = np.prod(nbases)
        
        H = self.kinetic_matrix_nd(grids, masses, nbases, dim=ndim)
        
        diag_inds = np.diag_indices(int(total_size))
        H[diag_inds] += v

        max_iter = np.iinfo(np.int32)
        niter = total_size * 10
        if niter > max_iter:
            niter = max_iter
            warnings.warn("Default maximum number of Arnoldi iterations (n*10 > int32 limit) - maxiter set to int32 limit.", ArpackArnoldiIter)

        energies, wfs = sparse.linalg.eigsh(H, k=neig, which='SA', maxiter=niter)
        return energies, wfs

    #@profile
    def kinetic_matrix_nd(self, grids, masses, nbases, dim):

        if dim < 2:
            raise ValueError

        total_size = np.prod(nbases)
        result_matrix = sparse.coo_matrix((total_size, total_size), dtype=np.float32)
        result_matrix = result_matrix.tocsr()

        for i in range(dim):
            matricies = [sparse.identity(nbases[d], dtype=np.float32).tocsr() for d in range(dim)]
            matricies[i] = self.algorithm(grids[i], masses[i], nbases[i])
            result = sparse.kron(matricies[0], matricies[1])
            for j in range(2, dim):
                result = sparse.kron(result, matricies[j])
            result_matrix += result

        return result_matrix


if __name__ == "__main__":
    import fast_dvr.potentials as pots
    import fast_dvr.exact_solvers as es

    # TEST 2D POTENTIAL FROM FILE
    pdir = '/home/kyle/PycharmProjects/Potential_Generator/potentials'
    #pot_file = f'{pdir}/harmonic/harmonic/101x101/isotropic_harmonic1_ngrid101.tab'
    #grid_file = f'{pdir}/harmonic/harmonic/101x101/isotropic_harmonic_grid_ngrid101.tab'
    pot_file = f'{pdir}/3D/harmonic/21x21x21/harmonic0_ngrid21.tab'
    grid_file = f'{pdir}/3D/harmonic/21x21x21/harmonic_grid_ngrid21.tab'

    neig = 3
    masses = [1, 1, 1]
    ndims = 3

    #v, grids = pots.load_potential(pot_file, grid_file, ndim=ndims, order='F')
    x = np.linspace(-5, 5, 21)
    y = np.linspace(-5, 5, 21)
    z = np.linspace(-5, 5, 21)
    v = np.zeros(21**3)
    i = 0
    for xi in x:
        for yi in y:
            for zi in z:
                v[i] = pots.harmonic_potential_3d(xi, yi, zi)
                i += 1
    grids = [x, y, z]
    calculator = Calculator(es.colbert_miller)
    energies, wfs = calculator.solve_nd(grids, masses, v, neig, 1, ndims)
    calculator = Calculator(es.colbert_miller, use_operators=True)
    energies1, wfs1 = calculator.solve_nd(grids, masses, v, neig, 1, ndims)

    breakpoint()

    #calculator = Calculator(es.colbert_miller)
    #start_mem = mp.memory_usage(max_usage=True)
    #res = mp.memory_usage(proc=(calculator.solve_nd, [grids, masses, v, neig, 1, ndims]), max_usage=True, retval=True)

    #print(start_mem)
    #print(res[0])
    #print(res[0] - start_mem)
    #energies = res[1][0]
    #wfs = res[1][1]
    #print(energies)

    #calculator = Calculator(ss.algorithm_116)
    #start_mem = mp.memory_usage(max_usage=True)
    #res = mp.memory_usage(proc=(calculator.solve_nd, [grids, masses, v, neig, 1, ndims]), max_usage=True, retval=True)
    #energies = res[1][0]
    #wfs = res[1][1]
    #energies, wfs = wfu.evaluate_energies(wfs, grids, v, masses, neig, ndim=ndims, normalise=True)

    #print(start_mem)
    #print(res[0])
    #print(res[0] - start_mem)
    #print(energies)

    #calculator = Calculator(ss.algorithm_116)
    #energies, wfs = calculator.solve_nd(grids, masses, v, neig, ndim=ndims)
    #energies, wfs = wfu.evaluate_energies(wfs, grids, v, masses, neig, ndim=ndims, normalise=True)
    #print(energies)
