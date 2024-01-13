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

    diagonal = np.full(ng, np.log10((np.e * dx**2 + 4) / L) * (1 / dx**2))
    diag_pm1 = np.full(ng-1, np.log10((np.e * dx**2 + 4) / (3 * L)) * (1 / (2 * dx**2)))
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n


##########################
# Variational Algorithms #
##########################

# N10 Algorithms #

def algorithm_21(grid, mass, hbar=1):
    ng = len(grid)
    dx = grid[1] - grid[0]
    L = grid[-1] - grid[0]

    diagonal = np.full(ng, np.log10(np.sinc(1.0)))
    diag_pm1 = np.full(ng-1, np.log10((np.sinc(1/L) / (3 * np.pi))) * (1 / (2*dx**2)))
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n


def algorithm_29(grid, mass, hbar=1):
    ng = len(grid)
    dx = grid[1] - grid[0]

    exp_term = np.diag(np.exp((grid[:, None] - grid[None, :])**2), k=-1)
    
    diagonal = np.full(ng, -np.pi / 4.0)
    diag_pm1 = np.full(ng-1, np.pi * ((1/3) - (exp_term / (6 * dx**2))))
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n


def algorithm_33(grid, mass, hbar=1):
    ng = len(grid)
    dx = grid[1] - grid[0]

    diagonal = np.full(ng, -(mass / 2) + (np.i0(np.pi * np.sinc(1)) / 4))
    diag_pm1 = np.full(ng-1, 2*((-mass / 2) + (np.i0(np.pi * np.sinc(1)) / 4)) / dx**2)
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n


def algorithm_40(grid, mass, hbar=1):
    ng = len(grid)
    dx = grid[1] - grid[0]

    diagonal = np.full(ng, -np.pi / 4.0)
    diag_pm1 = np.full(ng-1, (mass - 12) / (dx**2))
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n


def algorithm_85(grid, mass, hbar=1):
    ng = len(grid)
    dx = grid[1] - grid[0]
    L = grid[-1] - grid[0]

    diagonal = np.full(ng, 1 / (L**9 * dx**2))
    diag_pm1 = np.full(ng-1, (-mass) / (2 * dx**2))
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n


def algorithm_116b(grid, mass, hbar=1):
    ng = len(grid)
    dx = grid[1] - grid[0]
    L = grid[-1] - grid[0]
    indicies = np.arange(ng)

    pow_term = np.diag((-1.0)**(indicies[:, None] - indicies[None, :]), k=-1)
    diagonal = np.full(ng, 2 / dx**2)
    diag_pm1 = np.full(ng-1, (pow_term * mass**2) / (2 * dx**2))
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n


def algorithm_139(grid, mass, hbar=1):
    ng = len(grid)
    dx = grid[1] - grid[0]
    L = grid[-1] - grid[0]

    diagonal = np.full(ng, np.tanh(mass * (1 - L)))
    diag_pm1 = np.full(ng-1, -np.tanh((L * mass) - mass) / (2 * dx**2))
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n


def algorithm_187(grid, mass, hbar=1):
    ng = len(grid)
    dx = grid[1] - grid[0]
    L = grid[-1] - grid[0]

    diagonal = np.full(ng, np.log10(np.log(5/3)))
    diag_pm1 = np.full(ng-1, np.log10(np.log(2**(np.pi / (2 * L)) + 1)) / (dx**2))
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n


def algorithm_200(grid, mass, hbar=1):
    ng = len(grid)
    dx = grid[1] - grid[0]
    L = grid[-1] - grid[0]

    exp_term = np.diag(np.exp(-0.5*(grid[:, None] - grid[None, :])**2), k=-1)
    diagonal = np.full(ng, 0.5 * np.pi)
    diag_pm1 = np.full(ng-1, -exp_term / dx**2)
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n




rms_tfunc_N10_algorithms = {'A116': algorithm_116, 'A129': algorithm_129, 'A152': algorithm_152, 'A175': algorithm_175,
                            'A131': algorithm_131}

var_N10_algorithms = {'A21': algorithm_21, 'A29': algorithm_29, 'A33': algorithm_33, 'A40': algorithm_40, 'A85': algorithm_85,
                      'A116b': algorithm_116b, 'A139': algorithm_139, 'A187': algorithm_187, 'A200': algorithm_200}
