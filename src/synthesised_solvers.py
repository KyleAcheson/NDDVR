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

###############################
###############################
#### ALGORITHMS WITHOUT DX ####
###############################
###############################

########################
# TRAINED ON RMS TFUNC #
########################

# N10 Algorithms #

def algorithm_16(grid, mass, hbar=1):
    ng = len(grid)
    L = grid[-1] - grid[0]

    sq_term = np.diag((grid[:, None] - grid[None, :])**2, k=-1)
    diff_term = np.diag((grid[:, None] - grid[None, :]), k=-1)
    diagonal = np.full(ng, np.sinh(np.log(np.tanh(1)**0.5)) - 3)
    diag_pm1 = np.full(ng-1, (sq_term - 1) / (2 * diff_term) - 3 - np.pi)
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n

def algorithm_19(grid, mass, hbar=1):
    ng = len(grid)
    L = grid[-1] - grid[0]

    sq_term = np.diag((grid[:, None] - grid[None, :])**2, k=-1)
    qrt_term = np.diag((grid[:, None] - grid[None, :])**4, k=-1)
    diagonal = np.full(ng, -mass)
    diag_pm1 = np.full(ng-1, ((mass + 1)**2 * qrt_term - 1) / (2 * mass * (mass + 1) * sq_term))
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n


def algorithm_75(grid, mass, hbar=1):
    ng = len(grid)
    L = grid[-1] - grid[0]

    sq_term = np.diag((grid[:, None] - grid[None, :])**2, k=-1)
    exp_term = np.diag(np.exp(-0.5*(grid[:, None] - grid[None, :])**2), k=-1)
    diagonal = np.full(ng, np.sinh(np.log(np.i0(1) + 2 * mass)))
    diag_pm1 = np.full(ng-1, np.sinh(np.log(sq_term * ((1 + mass) * exp_term + mass)) - mass))
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n


def algorithm_91(grid, mass, hbar=1):
    ng = len(grid)
    L = grid[-1] - grid[0]

    sq_term = np.diag((grid[:, None] - grid[None, :])**2, k=-1)
    diagonal = np.full(ng, np.sinh(np.log(2 ** (np.cos(1) / np.pi))) * 2)
    diag_pm1 = np.full(ng-1, np.sinh(np.log(2 ** (np.cos(1) / np.pi) * sq_term)))
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n

def algorithm_124(grid, mass, hbar=1):
    ng = len(grid)
    L = grid[-1] - grid[0]

    sq_term = np.diag((grid[:, None] - grid[None, :])**2, k=-1)
    diagonal = np.full(ng, np.sinh(np.log( 1 / mass))) + mass
    diag_pm1 = np.full(ng-1, np.sinh(np.log(sq_term * ((1/ L) + mass) * (1 / mass))))
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n


def algorithm_129b(grid, mass, hbar=1):
    ng = len(grid)
    L = grid[-1] - grid[0]

    sq_term = np.diag((grid[:, None] - grid[None, :])**2, k=-1)
    exp_term = np.diag(np.exp(-1*(grid[:, None] - grid[None, :])**2), k=-1)
    diagonal = np.full(ng, 0.0)
    diag_pm1 = np.full(ng-1, (exp_term * 3 * (np.sinh(np.log(sq_term) + mass) * (1 / mass))) - 2)
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n


def algorithm_132(grid, mass, hbar=1):
    ng = len(grid)
    L = grid[-1] - grid[0]

    sq_term = np.diag((grid[:, None] - grid[None, :])**2, k=-1)
    diagonal = np.full(ng, 0.0)
    diag_pm1 = np.full(ng-1, np.sinh(np.log(((1 / L) + mass) * sq_term * (1 / mass))))
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n


def algorithm_140(grid, mass, hbar=1):
    ng = len(grid)
    L = grid[-1] - grid[0]

    sq_term = np.diag((grid[:, None] - grid[None, :])**2, k=-1)
    exp_term = np.diag(np.exp(-0.5*(grid[:, None] - grid[None, :])**2), k=-1)
    diagonal = np.full(ng, np.sinh(np.log(3) * np.pi))
    diag_pm1 = np.full(ng-1, np.sinh(np.log((exp_term + 2) * sq_term) - mass))
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n



def algorithm_146(grid, mass, hbar=1):
    ng = len(grid)
    L = grid[-1] - grid[0]

    exp_term = np.diag(np.exp(-1*(grid[:, None] - grid[None, :])**2), k=-1)
    diagonal = np.full(ng, np.pi)
    diag_pm1 = np.full(ng-1, np.sinh(np.log(1 + (-1 * exp_term))) + np.pi - 3)
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n


def algorithm_197(grid, mass, hbar=1):
    ng = len(grid)
    L = grid[-1] - grid[0]

    sq_term = np.diag((grid[:, None] - grid[None, :])**2, k=-1)
    diagonal = np.full(ng, np.sinh(np.log(np.cosh((-np.pi / L))**4)) * (1 / mass))
    diag_pm1 = np.full(ng-1, (1 / mass) * np.sinh(np.log(sq_term * (np.cosh((1 - np.pi) / L))**4)))
    T_n = np.diag(diagonal) + np.diag(diag_pm1, k=-1) + np.diag(diag_pm1, k=1)

    return T_n


#rms_tfunc_nodx_N10_algorithms = {'A16': algorithm_16, 'A19': algorithm_19, 'A75': algorithm_75, 'A91': algorithm_91,
#                                 'A124': algorithm_124, 'A129b': algorithm_129b, 'A132': algorithm_132, 'A140': algorithm_140,
#                                 'A146': algorithm_146, 'A197': algorithm_197}
rms_tfunc_nodx_N10_algorithms = {'A75': algorithm_75,'A124': algorithm_124, 'A132': algorithm_132, 'A140': algorithm_140,
                                 'A146': algorithm_146, 'A197': algorithm_197}

#rms_tfunc_N10_algorithms = {'A116': algorithm_116, 'A129': algorithm_129, 'A152': algorithm_152, 'A175': algorithm_175,
#                            'A131': algorithm_131}
rms_tfunc_N10_algorithms = {'A116': algorithm_116, 'A152': algorithm_152, 'A175': algorithm_175}

#var_N10_algorithms = {'A21': algorithm_21, 'A29': algorithm_29, 'A33': algorithm_33, 'A40': algorithm_40, 'A85': algorithm_85,
#                      'A116b': algorithm_116b, 'A139': algorithm_139, 'A187': algorithm_187, 'A200': algorithm_200}
var_N10_algorithms = {'A21': algorithm_21, 'A29': algorithm_29, 'A33': algorithm_33, 'A85': algorithm_85,
                      'A116b': algorithm_116b, 'A139': algorithm_139}
