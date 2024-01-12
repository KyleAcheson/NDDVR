import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def plot_wavefunctions_2d(x, y, eigenvectors, eigenvalues, num_to_plot=3, fname=None, show=False):
    for i in range(num_to_plot):
        wf = eigenvectors[:, :, i]

        wf_min, wf_max = np.min(wf[:]), np.max(wf[:])
        if wf_min < 0 and wf_max > 0:
            if wf_max >= wf_min:
                lim = wf_max
            else:
                lim = -1 * wf_min
            cm = 'seismic'
            norm = mpl.colors.TwoSlopeNorm(vmin=-1*lim, vmax=lim, vcenter=0)
        else:
            cm = 'Reds'
            norm = mpl.colors.Normalize(vmin=wf_min, vmax=wf_max)

        # Plot the surface and contour on the same axis
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection='3d')
        X, Y = np.meshgrid(x, y)
        surf = ax.plot_surface(X, Y, wf, cmap=cm, norm=norm, alpha=0.7)
        L = (np.max(wf) - np.min(wf))/2
        contour = ax.contour(x, y, wf, zdir='z', offset=np.min(wf)-L, cmap=cm, norm=norm)
        ax.set_zlim([np.min(wf)-L, np.max(wf)+L])
        ax.set_zticks([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.zaxis.majorTicks = []
        ax.zaxis.minorTicks = []
        #ax.xaxis.majorTicks = []
        #ax.xaxis.minorTicks = []
        #ax.yaxis.majorTicks = []
        #ax.yaxis.minorTicks = []
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_facecolor('white')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.get_zaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.zaxis.line.set_lw(0)
        #ax.xaxis._axinfo["grid"].update({"linewidth": 0})
        #ax.yaxis._axinfo["grid"].update({"linewidth": 0})
        #ax.zaxis._axinfo["grid"].update({"linewidth": 0})
        #fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

        #ax.set_title(f'$\psi {i}$ ($E$ = {eigenvalues[i]:.2f})')

        if fname:
            fout = f'{fname}_neig{i}.png'
            fig.savefig(fout)
        elif show:
            fig.show()


def plot_wavefunctions_3d(x, y, z, eigenvectors, eigenvals, num_to_plot=3, fname=None, show=False):
    for i in range(num_to_plot):
        wf = eigenvectors[:, :, :, i]
        x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z, indexing='ij')
        wf_min, wf_max = np.min(wf[:]), np.max(wf[:])
        if wf_min < 0:
            if wf_max >= wf_min:
                lim = wf_max
            else:
                lim = -1 * wf_min
            cm = 'seismic'
            norm = mpl.colors.TwoSlopeNorm(vmin=-1 * lim, vmax=lim, vcenter=0)
        else:
            cm = 'Reds'
            norm = mpl.colors.Normalize(vmin=0, vmax=wf_max)

        # Plot the wavefunction
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f'$\psi {i}$ ($E$ = {eigenvals[i]:.2f})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        mask = np.abs(wf) > 1E-6
        scat = ax.scatter(x_mesh, y_mesh, z_mesh, c=wf, cmap=cm, norm=norm, alpha=0.3, edgecolor='face', linewidth=0,
                          s=10 * mask)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_facecolor('white')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis._axinfo["grid"].update({"linewidth": 0})
        ax.yaxis._axinfo["grid"].update({"linewidth": 0})
        ax.zaxis._axinfo["grid"].update({"linewidth": 0})
        fig.colorbar(scat, ax=ax, shrink=0.5, aspect=10)

        if fname:
            fout = f'{fname}_neig{i}.png'
            fig.savefig(fout)
        elif show:
            fig.show()
