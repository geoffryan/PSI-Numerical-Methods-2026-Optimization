from pathlib import Path
import sys
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt


def load_the_data_txt(filename, skip):

    m = np.loadtxt(filename, max_rows=1)
    data = np.loadtxt(filename, skiprows=1)

    t = data[::skip, 1]
    e = data[::skip, 2]
    x = data[::skip, 3:]

    n = x.shape[1]
    nt = len(t)

    x = x.reshape(nt, n//6, 6)

    rx = x[:, :, 0]
    ry = x[:, :, 1]
    rz = x[:, :, 2]
    vx = x[:, :, 3]
    vy = x[:, :, 4]
    vz = x[:, :, 5]

    return t, m, rx, ry, rz, vx, vy, vz, e


def load_the_data_h5(filename, skip):

    with h5.File(filename, "r") as f:
        t = f['t'][::skip][...]
        m = f['m'][...]
        e = f['e'][::skip][...]
        x = f['x'][::skip, 0, :, 0][...]
        y = f['x'][::skip, 0, :, 1][...]
        z = f['x'][::skip, 0, :, 2][...]
        vx = f['x'][::skip, 1, :, 0][...]
        vy = f['x'][::skip, 1, :, 1][...]
        vz = f['x'][::skip, 1, :, 2][...]

    return t, m, x, y, z, vx, vy, vz, e


def make_the_plots(filename, skip):

    if filename.suffix == ".h5":
        data = load_the_data_h5(filename, skip)

    else:
        data = load_the_data_txt(filename, skip)

    t, m, x, y, z, vx, vy, vz, e = data

    nb = len(m)

    nt = len(t)

    for i in range(0, nt):
        fig, ax = plt.subplots(2, 2)

        ms = m * nb

        ax[0, 0].scatter(x[i, :], y[i, :], s=ms)
        ax[1, 0].scatter(x[i, :], z[i, :], s=ms)
        ax[1, 1].scatter(y[i, :], z[i, :], s=ms)
        ax[0, 1].plot(t[:], e[:])
        ax[0, 1].plot(t[i], e[i], '.')

        ax[0, 0].set(xlabel=r'$x$', ylabel=r'$y$',
                     xlim=[-150, 150], ylim=[-150, 150])
        ax[1, 0].set(xlabel=r'$x$', ylabel=r'$z$',
                     xlim=[-150, 150], ylim=[-150, 150])
        ax[1, 1].set(xlabel=r'$y$', ylabel=r'$z$',
                     xlim=[-150, 150], ylim=[-150, 150])

        ax[0, 0].set_aspect('equal')
        ax[1, 0].set_aspect('equal')
        ax[1, 1].set_aspect('equal')

        fig.suptitle(r"$t$ = {0:f}".format(t[i]))

        figname = "frame_{0:06d}.png".format(i*skip)
        print("Saving", figname)
        fig.savefig(figname)
        plt.close(fig)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("usage: $ python plot.py ORBIT_FILE [SKIP]")
        sys.exit()

    filename = Path(sys.argv[1])
    skip = int(sys.argv[2]) if len(sys.argv) >= 3 else 1

    make_the_plots(filename, skip)
