import numpy as np

########################
# TRAINED ON RMS TFUNC #
########################

# N10 Algorithms #


def algorithm_116(grid, mass, hbar=1):

    ng = len(grid)
    dx = grid[1] - grid[0]
    L = grid[-1] - grid[0]
    cos1 = np.cos(1)

    diagonal = np.full(ng, -0.5 * (3 * cos1 - 3 + mass))
    diag_pm1 = np.full(ng - 1, (-1 / (2 * dx**2)) * (((3 * cos1 - 3) / L) + mass))
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n

def algorithm_129(grid, mass, hbar=1):

    ng = len(grid)
    dx = grid[1] - grid[0]
    L = grid[-1] - grid[0]
    sin1 = np.sin(1)

    diagonal = np.full(ng, 4.0)
    exp_term = np.diag(np.exp((grid[:, None] - grid[None, :])**2), k=-1)
    diag_pm1 = np.full(ng-1, (4 * (np.pi - 4) * L + sin1 * exp_term + 16) / (4 * L * dx**2))
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n


def algorithm_131(grid, mass, hbar=1):

    ng = len(grid)
    dx = grid[1] - grid[0]
    L = grid[-1] - grid[0]

    diagonal = np.full(ng, ((2 * np.pi) / dx ** 2) + 1)
    diag_pm1 = np.full(ng-1, (((0.5 * dx**2 + np.pi) / (mass * L)) - 4 + np.pi) / dx**2)
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n

def algorithm_152(grid, mass, hbar=1):

    ng = len(grid)
    dx = grid[1] - grid[0]
    L = grid[-1] - grid[0]
    sinh1 = np.sinh(1)

    diagonal = np.full(ng, (sinh1 / 3.0) - mass)
    diag_pm1 = np.full(ng-1, (1 / (2 * dx**2)) * ((1 / (3 * mass * L)) * sinh1 + (1 / L) - mass))
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n

def algorithm_175(grid, mass, hbar=1):

    ng = len(grid)
    dx = grid[1] - grid[0]
    L = grid[-1] - grid[0]
    sinh1 = np.sinh(1)

    diagonal = np.full(ng, np.log10((np.e * dx**2 + 4) / L) * (1 / dx**2))
    diag_pm1 = np.full(ng-1, np.log10((np.e * dx**2 + 4) / (3 * L)) * (1 / (2 * dx**2)))
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n


# N15 Algorithms #


def algorithm_36(grid, mass, hbar=1):

    ng = len(grid)
    indicies = np.arange(ng)
    dx = grid[1] - grid[0]
    L = grid[-1] - grid[0]
    coshtanh1 = np.cosh(np.tanh(1))
    exp_term1 = np.diag(np.exp(-0.5*(grid[:, None] - grid[None, :])**2), k=-1)
    exp_term2 = np.diag(np.exp((grid[:, None] - grid[None, :])**2), k=-1)
    npow_term = np.diag((-1.0)**(indicies[None] - indicies[:, None]), k=-1)

    diagonal = np.full(ng, (np.pi * ((1/3) + np.pi)) / 3)
    off_diag_term = (1/3) * (np.pi * npow_term * (np.pi + (exp_term1 * coshtanh1) / (3 * dx**2)) * exp_term2) + (np.pi/3) + 3
    diag_pm1 = np.full(ng - 1, off_diag_term)
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n


#class A1(DVR):
#
#
#    def solve_1d(self, x, v, mass, neig, hbar=1):
#        energies, wfs = super().solve_1d(x, v, mass, neig, self.calculate_kinetic_block, hbar)
#        return energies, wfs
#
#    def solve_nd(self, grids, masses, v, neig, hbar=1, ndim=2):
#        energies, wfs = super().solve_nd(grids, masses, v, neig, self.calculate_kinetic_block, hbar, ndim)
#        wfs = wfu.normalise_wf2(wfs, grids[0], grids[1], neig)
#        energies = wfu.evaluate_energies_2d(wfs, grids[0], grids[1], v, masses, neig)
#        return energies, wfs
#
#    def calculate_kinetic_block(self, grid, mass, hbar=1):
#
#        ng = len(grid)
#        dx = grid[1] - grid[0]
#        L = grid[-1] - grid[0]
#        T_n = np.zeros((ng, ng))
#
#        for i in range(ng):
#            ##T_n[i, i] = np.cos(-3 * dx**2)
#            ##T_n[i, i] = 2 * np.exp(0.5 * (grid[i] - grid[i])**2) / (dx**2 * np.pi)
#            ## N10
#            #T_n[i, i] = -1 * ((3 * np.cos(1) - 3 + mass) / 2)
#            #T_n[i, i] = 4
#            T_n[i, i] = ((2 * np.pi) / dx**2) + 1 # VGOOD!
#            #T_n[i, i] = (np.sinh(1) / 3) - mass
#            #T_n[i, i] = np.log10((np.e * dx**2 + 4) / L) * (1 / dx**2)
#            if i < ng - 1:
#                j = i + 1
#                ##T_n[i, j] = np.cos(((-3 * dx**4)/L) - 2) / dx**2
#                ##T_n[i, j] = (-1 - np.pi * np.exp(0.5 * (grid[i] - grid[j])**2)) / (dx**2 * 3 * np.pi)
#                ## N10
#                #T_n[i, j] = (-1 / (2*dx**2)) * (((3 * np.cos(1) - 3) / L) + mass)
#                #T_n[i, j] = ((4 * (np.pi - 4) * L) + np.sin(1) * np.exp((grid[i] - grid[j])**2) + 16) / (4 * L * dx**2)
#                T_n[i, j] = (((0.5 * dx**2 + np.pi) / (mass * L)) - 4 + np.pi) / dx**2
#                #T_n[i, j] = (1 / (2*dx**2)) * ((1 / (3 * mass * L)) * np.sinh(1) + (1 / L) - mass)
#                #T_n[i, j] = np.log10((np.e * dx**2 + 4) / (3 * L)) * (1 / (2*dx**2))
#
#            if i > 0:
#                j = i -1
#                ##T_n[i, j] = np.cos(((-3 * dx**4)/L) - 2) / dx**2
#                ##T_n[i, j] = (-1 - np.pi * np.exp(0.5 * (grid[i] - grid[j]) ** 2)) / (dx ** 2 * 3 * np.pi)
#                #T_n[i, j] = (-1 / (2*dx**2)) * (((3 * np.cos(1) - 3) / L) + mass)
#                #T_n[i, j] = ((4 * (np.pi - 4) * L) + np.sin(1) * np.exp((grid[i] - grid[j])**2) + 16) / (4 * L * dx**2)
#                T_n[i, j] = (((0.5 * dx**2 + np.pi) / (mass * L)) - 4 + np.pi) / dx**2
#                #T_n[i, j] = (1 / (2*dx**2)) * ((1 / (3 * mass * L)) * np.sinh(1) + (1 / L) - mass)
#                #T_n[i, j] = np.log10((np.e * dx**2 + 4) / (3 * L)) * (1 / (2*dx**2))
#
#
#        return T_n
