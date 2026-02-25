import numpy as np
import scipy.linalg as spyl
from scipy import sparse
import fast_dvr.operators as op
import warnings

class ArpackArnoldiIter(UserWarning):
    pass

class Calculator:
    '''
    DVR Calculator object, can use any of the algorithms in the exact_solvers and synthesised_solvers
    modules. The algorithm must be important and passed as an argument in construction.
    Mass-weighted normal mode coordinates and a direct product representation are assumed.

    WARNING: by default this will construct the full rank-2d Hamiltonian where d is the number of DOFs.
    This results in a Hamiltonian of size N x N, where N = n^d, with n the number of discrete basis functions
    or points for DOF d. This is not suitable for more than three DOFs due to the memory requirement.

    To compute wavefunctions and energies of higher dimensional systems use the Lanczos based eigensolver
    that iteratively computes matrix-vector products. This is requests by setting the use_operators
    keyword to use_operators=True. Here the d dimensional Hamiltonian is factorized into
    a sum of products of 1D operations. Matrix-vector products Hv are computed for each DOF k as H^kv,
    computing the product with n^{d-1} subvectors of v, each of length n. The potential provided as a vector
    is treated as a tensor, for each matrix H^k a premutation of tensor indices is performed, resulting in a
    potential matrix of shape n x n^{d-1}, which is matrix multiplied with H^k. The total complexity is thus
    O(d(n^{d+1})). For a discussion of this see: https://doi.org/10.1021/acs.jctc.4c01312.

    '''

    def __init__(self, algorithm, use_operators=False):
        '''
        :param algorithm:  solver to use from exact_solvers (e.g. CM-DVR) or synthesised_solvers
        :param use_operators: if True - use the iterative Lanczos eigensolver, for higher DOFs.

        If a PS algorithm from synthesised_solvers is used, one must renormalize the wfs and
        compute the energies as an expectation value of this using functions from the wf_utils module.
        WARNING: The output energies of the PS algorithm will not be correct if taken directly from
        solve_1d or solve_nd.
        '''
        self.algorithm = algorithm
        self.use_operators = use_operators

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm):
        self._algorithm = algorithm

    def solve_1d(self, x, v, mass, neig, nbasis=None, hbar=1):
        '''
        Computes the energies and eigenvectors of a discrete 1D potential.
        If the solver is a PS algorithm - the energies must be computed as an expectation value again.
        By default, this uses the evr driver.

        :param x: position coordinate axes defined on a grid
        :param v: potential evaluated on the grid
        :param mass: mass, should be 1 for normal coordinates
        :param neig: number of eigenstates to compute
        :param nbasis: number of basis functions - only required for exact solvers such as Sine DVR.
                        otherwise do not specify as this is computed from the length of x.
        :param hbar: 1 as normal coordinates
        :return: energies [neig], wfs[n, neig] - where n is len(x) or nbasis depending
        '''
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
        '''
        Computes the energies and eigenvectors of a generalised ND potential.
        If the solver is a PS algorithm - the energies must be computed as an expectation value again.

        WARNING: If larger than 2/3 DOFs ensure you set set_operators=True in construction of
                 the Calculator object and use the iterative approach.

        When using set_operators=True, the default number of Arnoldi iterations is n*10,
        with n the total number of grid/ basis points in the D dimensional direct product grid.
        If n*10 > Int32 max limit then the number of iterations is np.iinfo(np.int32).max.

        :param grids: list of each position coordinate axes (1D array) defined on a grid
        :param v: potential evaluated on the grid flattened in F-order (column-major for FORTRAN)
        :param mass: list of masses for each DOF, should be 1 for normal coordinates
        :param neig: number of eigenstates to compute
        :param nbasis: numpy nd array of number of basis functions for each DOF.
                       Only required for exact solvers such as Sine DVR.
                       Otherwise do not specify as this is computed from the length of grid in grids.
        :param hbar: 1 as normal coordinates
        :return: energies [neig], wfs[n, neig] - where n is the total number of basis/ grid points.
        '''

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

        max_iter = np.iinfo(np.int32).max
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

    # TEST 3D POTENTIAL

    neig = 3
    masses = [1, 1, 1]
    ndims = 3

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
    calculator = Calculator(es.colbert_miller, use_operators=True)
    energies, wfs = calculator.solve_nd(grids, masses, v, neig, 1, ndims)
    print(energies[:neig])
