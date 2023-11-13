import numpy as np
import matplotlib.pyplot as plt
import timeit
import dvr
import synthesised_algorithms as sa
import potential_functions as potf

def time_algorithms(algorithms, dvr, x, v, mass, neig, nruns):

    timer = timeit.Timer(lambda: dvr(x, v, mass, neig))
    elapsed_dvr = timer.timeit(nruns)
    #print(f'{dvr} average time of {nruns} runs: {elapsed_dvr:.6f} seconds')
    ratios = []
    raw_times = []
    raw_times.append(elapsed_dvr)

    for algorithm in algorithms:
        timer = timeit.Timer(lambda: algorithm(x, v, mass, neig))
        elapsed = timer.timeit(nruns)
        #print(f'{algorithm} average time of {nruns} runs: {elapsed:.6f} seconds')
        raw_times.append(elapsed)
        ratio = elapsed_dvr / elapsed
        ratios.append(ratio)
        #print(f'speed up vs dvr: {ratio:.2f}')

    return ratios, raw_times

def run1d_test():
    neig = 3
    ngrid = 500
    x = np.linspace(-5, 5, ngrid)
    k = 1
    v = potf.harmonic(x, k)
    mass = 1

    wf, energies, H = dvr.cm_dvr(x, v, mass, neig)
    print(energies)
    wf, energies, H = sa.algorithm_100(x, v, mass, neig)
    print(energies)
    wf, energies, H = sa.algorithm_100_sparse(x, v, mass, neig)
    print(energies)

def run_time_tests1d():
    neig = 3
    nruns = 1
    grids = [30, 50, 100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 5000]
    speed_ups = np.zeros((3, len(grids)))
    raw_times = np.zeros((4, len(grids)))
    for i, ngrid in enumerate(grids):
        x = np.linspace(-5, 5, ngrid)
        k = 1
        v = potf.harmonic(x, k)
        mass = 1
        algorithms = [sa.algorithm_100_sparse, sa.algorithm_36_sparse, sa.algorithm_29_sparse]
        ratios, raw = time_algorithms(algorithms, dvr.cm_dvr, x, v, mass, neig, nruns)
        speed_ups[:, i] = np.array(ratios)
        raw_times[:, i] = np.array(raw)

    labels = ['A100', 'A36', 'A29']
    #np.savetxt('ps_speedup.dat', speed_ups, delimiter=', ', newline='\n')
    fig, ax = plt.subplots()
    for i in range(len(algorithms)):
        ax.plot(grids, speed_ups[i, :], '-o', label=labels[i])
    ax.set_xlabel('$N_g$')
    ax.set_ylabel('Speed up')
    ax.legend(frameon=False)
    fig.savefig(f'speedup_nruns{nruns}_SPARSE_npeig.png')

    labels = ['CM-DVR', 'A100', 'A36', 'A29']
    fig, ax = plt.subplots()
    for i in range(len(labels)):
        ax.plot(grids, raw_times[i, :], '-o', label=labels[i])
    ax.set_xlabel('$N_g$')
    ax.set_ylabel('time (s)')
    ax.legend(frameon=False)
    fig.savefig(f'raw_speed_nruns{nruns}_SPARSE_npeig.png')

if __name__ == "__main__":
    run_time_tests1d()