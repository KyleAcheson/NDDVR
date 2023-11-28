
def harmonic(x, k=1.0):
    return 0.5 * k * x**2

def harmonic_potential_2d(x, y, kx=1.0, ky=1.0, mx=1.0, my=1.0):
    return 0.5 * (kx * x ** 2 / mx + ky * y ** 2 / my)

def harmonic_potential_3d(x, y, z, kx=1.0, ky=1.0, kz=1.0, mx=1.0, my=1.0, mz=1.0):
    return 0.5 * (kx * x**2 / mx + ky * y**2 / my + kz * z**2 / mz)

