import sys
import numpy as np
import h5py as h5
from line_profiler import profile
import cbody


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

    r = x[0]
    g = np.empty_like(r)

    cbody.calc_g(g, r, m, eps_soft)

    xdot = np.zeros_like(x)
    xdot[0] = x[1]
    xdot[1] = g

    return xdot


def calc_energy(x, m=None, eps_soft=None):

    r = x[0]
    v = x[1]

    return cbody.calc_en(r, v, m, eps_soft)


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

    x0 = np.empty((2, Nbody, 3))

    x0[0, :, 0] = rx
    x0[0, :, 1] = ry
    x0[0, :, 2] = rz
    x0[1, :, 0] = vx
    x0[1, :, 1] = vy
    x0[1, :, 2] = vz

    kwargs = {'m': M, 'eps_soft': eps}

    filename = "orbits.h5"
    with h5.File(filename, "w") as f:
        f.create_dataset("m", data=M)
        f.create_dataset("t", shape=(Niter+1,), dtype=float)
        f.create_dataset("e", shape=(Niter+1,), dtype=float)
        f.create_dataset("x", shape=(Niter+1, 2, Nbody, 3), dtype=float)
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
