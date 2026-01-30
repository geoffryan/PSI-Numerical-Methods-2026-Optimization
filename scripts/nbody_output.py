import sys
import numpy as np
import h5py as h5
from line_profiler import profile


def rk4_step(t, x, h, f, f_kwargs):

    k1 = f(t, x, **f_kwargs)
    k2 = f(t + 0.5*h, x + 0.5*h*k1, **f_kwargs)
    k3 = f(t + 0.5*h, x + 0.5*h*k2, **f_kwargs)
    k4 = f(t + h, x + h*k3, **f_kwargs)

    return x + h * (k1 + 2*k2 + 2*k3 + k4) / 6.0


def evolve(t0, t1, x0, N, f, f_kwargs, filename):

    t = np.linspace(t0, t1, N+1)

    buf_size = 100

    # Set up buffers to hold the solution
    e_buf = np.empty((buf_size,), dtype=float)
    x_buf = np.empty((buf_size,) + x0.shape, dtype=float)
    e_buf[0] = calc_energy(x0, **f_kwargs)
    x_buf[0] = x0
    buf_start = 0
    buf_end = 1

    x = x0.copy()

    for i in range(N):
        # Evolve the solution
        dt = t[i+1] - t[i]
        print(i, t[i], dt)
        x = rk4_step(t, x, dt, f, f_kwargs)

        # Copy to the buffers
        x_buf[i+1 - buf_start] = x
        e_buf[i+1 - buf_start] = calc_energy(x, **f_kwargs)
        buf_end += 1

        # Write out the buffers if full
        if buf_end == buf_start + buf_size:
            with h5.File(filename, "a") as outfile:
                outfile['t'][buf_start:buf_end] = t[buf_start:buf_end]
                outfile['e'][buf_start:buf_end] = e_buf[:buf_size]
                outfile['x'][buf_start:buf_end] = x_buf[:buf_size]
            buf_start = buf_end

    # if the buffer is partially full, write it out.
    if buf_end > buf_start:
        with h5.File(filename, "a") as outfile:
            outfile['t'][buf_start:buf_end] = t[buf_start:buf_end]
            outfile['e'][buf_start:buf_end] = e_buf[:buf_end-buf_start]
            outfile['x'][buf_start:buf_end] = x_buf[:buf_end-buf_start]
        buf_start = buf_end


@profile
def f_nbody(t, x, m=None, eps_soft=None):

    xdot = np.zeros_like(x)

    dr = x[:3, None, :] - x[:3, :, None]

    r2 = (dr**2).sum(axis=0) + eps_soft**2

    r = np.sqrt(r2)

    g = ((m[None, :] / (r**3))[None, :, :] * dr).sum(axis=2)

    xdot[:3, :] = x[3:, :]
    xdot[3:, :] = g

    return xdot


def calc_energy(x, m=None, eps_soft=None):

    kin = 0.0
    pot = 0.0

    r = x[:3, :]
    v = x[3:, :]

    kin = 0.5 * (m[None, :] * v**2).sum()

    dr = r[:, None, :] - r[:, :, None]
    r2 = (dr**2).sum(axis=0) + eps_soft**2

    pot = -0.5*(m[:, None] * m[None, :] / np.sqrt(r2)).sum()

    return kin + pot


def generate_init_disc(N, Rmax, aspect_ratio, Mtotal):

    Hmax = aspect_ratio * Rmax

    z = Hmax * (2*np.random.rand(Nbody) - 1.0)
    r_cyl = Rmax * np.sqrt(np.random.rand(Nbody))
    phi = 2*np.pi * np.random.rand(Nbody)

    M_int = Mtotal * (r_cyl/Rmax)**2
    v_cyl = np.sqrt(M_int / r_cyl)

    M = np.random.rand(Nbody)
    M *= Mtotal / M.sum()

    rx = r_cyl * np.cos(phi)
    ry = r_cyl * np.sin(phi)
    rz = z
    vx = -v_cyl * np.sin(phi)
    vy = v_cyl * np.cos(phi)
    vz = np.zeros_like(z)

    return M, rx, ry, rz, vx, vy, vz


def main(tMax, Nbody, Niter):

    Mtotal = 1.0
    Rmax = 100.0
    eps = 0.01

    M, rx, ry, rz, vx, vy, vz = generate_init_disc(
            Nbody, Rmax, 0.1, Mtotal)

    x0 = np.empty((6, Nbody))

    x0[0, :] = rx
    x0[1, :] = ry
    x0[2, :] = rz
    x0[3, :] = vx
    x0[4, :] = vy
    x0[5, :] = vz

    kwargs = {'m': M, 'eps_soft': eps}

    filename = "orbits.h5"
    with h5.File(filename, "w") as f:
        f.create_dataset("m", data=M)
        f.create_dataset("t", shape=(Niter+1,), dtype=float)
        f.create_dataset("e", shape=(Niter+1,), dtype=float)
        f.create_dataset("x", shape=(Niter+1, 6, Nbody), dtype=float)
    print("Writing to", filename)
    evolve(0.0, tMax, x0, Niter, f_nbody, kwargs, filename)


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("usage: $ python nbody_basic.py TMAX NBODY NSTEPS")
        sys.exit()

    Tmax = float(sys.argv[1])
    Nbody = int(sys.argv[2])
    Niter = int(sys.argv[3])

    main(Tmax, Nbody, Niter)
