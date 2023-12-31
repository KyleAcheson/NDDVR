import numpy as np

def algorithm_131(grid, mass, hbar=1):

    ng = len(grid)
    dg = grid[1] - grid[0]
    L = grid[-1] - grid[0]
    T_n = np.zeros((ng, ng))

    for i in range(ng):
        T_n[i, i] = ((2 * np.pi) / dg ** 2) + 1
        if i < ng - 1:
            j = i + 1
            T_n[i, j] = (((0.5 * dg**2 + np.pi) / (mass * L)) - 4 + np.pi) / dg**2
        if i > 0:
            j = i - 1
            T_n[i, j] = (((0.5 * dg**2 + np.pi) / (mass * L)) - 4 + np.pi) / dg**2

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
#        dg = grid[1] - grid[0]
#        L = grid[-1] - grid[0]
#        T_n = np.zeros((ng, ng))
#
#        for i in range(ng):
#            ##T_n[i, i] = np.cos(-3 * dg**2)
#            ##T_n[i, i] = 2 * np.exp(0.5 * (grid[i] - grid[i])**2) / (dg**2 * np.pi)
#            ## N10
#            #T_n[i, i] = -1 * ((3 * np.cos(1) - 3 + mass) / 2)
#            #T_n[i, i] = 4
#            T_n[i, i] = ((2 * np.pi) / dg**2) + 1 # VGOOD!
#            #T_n[i, i] = (np.sinh(1) / 3) - mass
#            #T_n[i, i] = np.log10((np.e * dg**2 + 4) / L) * (1 / dg**2)
#            if i < ng - 1:
#                j = i + 1
#                ##T_n[i, j] = np.cos(((-3 * dg**4)/L) - 2) / dg**2
#                ##T_n[i, j] = (-1 - np.pi * np.exp(0.5 * (grid[i] - grid[j])**2)) / (dg**2 * 3 * np.pi)
#                ## N10
#                #T_n[i, j] = (-1 / (2*dg**2)) * (((3 * np.cos(1) - 3) / L) + mass)
#                #T_n[i, j] = ((4 * (np.pi - 4) * L) + np.sin(1) * np.exp((grid[i] - grid[j])**2) + 16) / (4 * L * dg**2)
#                T_n[i, j] = (((0.5 * dg**2 + np.pi) / (mass * L)) - 4 + np.pi) / dg**2
#                #T_n[i, j] = (1 / (2*dg**2)) * ((1 / (3 * mass * L)) * np.sinh(1) + (1 / L) - mass)
#                #T_n[i, j] = np.log10((np.e * dg**2 + 4) / (3 * L)) * (1 / (2*dg**2))
#
#            if i > 0:
#                j = i -1
#                ##T_n[i, j] = np.cos(((-3 * dg**4)/L) - 2) / dg**2
#                ##T_n[i, j] = (-1 - np.pi * np.exp(0.5 * (grid[i] - grid[j]) ** 2)) / (dg ** 2 * 3 * np.pi)
#                #T_n[i, j] = (-1 / (2*dg**2)) * (((3 * np.cos(1) - 3) / L) + mass)
#                #T_n[i, j] = ((4 * (np.pi - 4) * L) + np.sin(1) * np.exp((grid[i] - grid[j])**2) + 16) / (4 * L * dg**2)
#                T_n[i, j] = (((0.5 * dg**2 + np.pi) / (mass * L)) - 4 + np.pi) / dg**2
#                #T_n[i, j] = (1 / (2*dg**2)) * ((1 / (3 * mass * L)) * np.sinh(1) + (1 / L) - mass)
#                #T_n[i, j] = np.log10((np.e * dg**2 + 4) / (3 * L)) * (1 / (2*dg**2))
#
#
#        return T_n
