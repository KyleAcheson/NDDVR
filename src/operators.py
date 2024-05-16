import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator


class Hamiltonian(LinearOperator):

    def __init__(self, v, kinetic_1d_mats, dims, dtype='float64'):
        n = np.prod(dims)
        self.shape = (n, n)
        self.dtype = np.dtype(dtype)
        self.kinetic = Kinetic(kinetic_1d_mats, dims, dtype)
        self.potential = Potential(v, dtype)
    
    def _matvec(self, x):
        return self.kinetic.matvec(x) + self.potential.matvec(x)


class Potential(LinearOperator):

    def __init__(self, diag, dtype='float64'):
        self.diag = diag
        self.shape = (len(self.diag), len(self.diag))
        self.dtype = np.dtype(dtype)

    def _matvec(self, x):
        return self.diag * x

    def _rmatvec(self, x):
        return self.diag * x
    

class Kinetic(LinearOperator):
    
    def __init__(self, kinetic_1d_mats, dims, dtype='float64'):
        n = np.prod(dims)
        self.kinetic_1d_mats = kinetic_1d_mats
        self.dims = dims
        self.shape = (n, n)
        self.ndims = len(dims)
        self.dtype = np.dtype(dtype)
        
    def _matvec(self, x):
        v_prod = np.zeros(self.shape[0])
        for i in range(self.ndims):
            v_prod += self._kron_sum_prod(self.kinetic_1d_mats[i], self.dims, x, i)
        return v_prod

    def _kron_sum_prod(self, mat, dims, x, i):
        xt = x.reshape(dims)
        unfolded = self._unfold(xt, i, dims)
        res = mat @ unfolded
        xt = self._refold(res, i, dims)
        return xt.ravel()

    @staticmethod
    def _unfold(tens, mode, dims):
        """
        Unfolds tensor into matrix.
        Parameters
        ----------
        tens : ndarray, tensor with shape == dims
        mode : int, which axis to move to the front
        dims : list, holds tensor shape
        Returns
        -------
        matrix : ndarray, shape (dims[mode], prod(dims[/mode]))
        """
        if mode == 0:
            return tens.reshape(dims[0], -1)
        else:
            return np.moveaxis(tens, mode, 0).reshape(dims[mode], -1)

    @staticmethod
    def _refold(vec, mode, dims):
        """
        Refolds vector into tensor.
        Parameters
        ----------
        vec : ndarray, tensor with len == prod(dims)
        mode : int, which axis was unfolded along.
        dims : list, holds tensor shape
        Returns
        -------
        tens : ndarray, tensor with shape == dims
        """
        if mode == 0:
            return vec.reshape(dims)
        else:
            # Reshape and then move dims[mode] back to its
            # appropriate spot (undoing the `unfold` operation).
            tens = vec.reshape(
                [dims[mode]] +
                [d for m, d in enumerate(dims) if m != mode]
            )
            return np.moveaxis(tens, 0, mode)


def full_hamiltonian(pot, grids, masses, sizes, dim):
    H = full_kinetic_matrix(grids, masses, sizes, dim)
    diag_inds = np.diag_indices(int(np.prod(sizes)))
    H[diag_inds] += pot
    return H


def full_kinetic_matrix(grids, masses, sizes, dim):

    total_size = np.prod(sizes)
    result_matrix = sparse.coo_matrix((total_size, total_size), dtype=np.float64)
    result_matrix = result_matrix.tocsr()

    for i in range(dim):
        matricies = [sparse.identity(sizes[d], dtype=np.float64).tocsr() for d in range(dim)]
        matricies[i] = es.colbert_miller(grids[i], masses[i])
        result = sparse.kron(matricies[0], matricies[1])
        for j in range(2, dim):
            result = sparse.kron(result, matricies[j])
        result_matrix += result

    return result_matrix


def construct_hamiltonians(n, nd, full=False):
    masses = np.array([1 for i in range(nd)])
    dims = np.array([n for i in range(nd)])
    ntotal = np.prod(dims)
    pot = np.random.rand(ntotal)
    vec = np.random.rand(ntotal)
    grids = []

    kinetic_1d_mats = []
    for i in range(nd):
        grid = np.linspace(-5, 5, n)
        kinetic_1d_mats.append(es.colbert_miller(grid, 1))
        grids.append(grid)

    H_iter = Hamiltonian(pot, kinetic_1d_mats, dims)
    if full:
        H_full = full_hamiltonian(pot, grids, masses, dims, nd)
        return  H_iter, H_full
    else:
        return H_iter


def time_eigsh_full(H, neig):
    e1, v1 = sparse.linalg.eigsh(H, k=neig, which='SA')

def time_matvec(H, x):
    H.matvec(x)

if __name__ == "__main__":
    import exact_solvers as es
    import timeit
    
    n = 21
    nds = [2, 3, 4, 5]
    nt = len(nds)
    neig = 1
    nr = 1

    tf = np.zeros(nt)
    ti = np.zeros(nt)
    for i in range(nt):
        H_iter = construct_hamiltonians(n, nds[i], full=False)
        #x = np.random.rand(n**nds[i])
        #t1 = min(timeit.repeat(lambda: time_matvec(H_iter, x), repeat=nr, number=1))
        #t1 = min(timeit.repeat(lambda: time_eigsh_full(H_full, neig), repeat=nr, number=1))
        t2 = min(timeit.repeat(lambda: time_eigsh_full(H_iter, neig), repeat=nr, number=1))
        #tf[i] = t1
        print(t2)
        ti[i] = t2

    print(ti)

    breakpoint()