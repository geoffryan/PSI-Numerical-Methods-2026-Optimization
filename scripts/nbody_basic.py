import sys
import numpy as np


def rk4_step(t, x, h, f, f_kwargs):

    k1 = f(t, x, **f_kwargs)
    k2 = f(t + 0.5*h, x + 0.5*h*k1, **f_kwargs)
    k3 = f(t + 0.5*h, x + 0.5*h*k2, **f_kwargs)
    k4 = f(t + h, x + h*k3, **f_kwargs)

    return x + h * (k1 + 2*k2 + 2*k3 + k4) / 6.0


def evolve(t0, t1, x0, N, f, f_kwargs, filename):

    t = np.linspace(t0, t1, N+1)

    with open(filename, "a") as outfile:
        en = calc_energy(x0, **f_kwargs)
        str_x = " ".join(["{0:e}".format(xi) for xi in x0])
        outfile.write("{0:d} {1:e} {2:e} {3:s}\n".format(0, t0, en, str_x))

    x = x0.copy()

    for i in range(N):
        dt = t[i+1] - t[i]
        print(i, t[i], dt)
        x = rk4_step(t, x, dt, f, f_kwargs)
        with open(filename, "a") as outfile:
            en = calc_energy(x, **f_kwargs)
            str_x = " ".join(["{0:e}".format(xi) for xi in x])
            outfile.write("{0:d} {1:e} {2:e} {3:s}\n"
                          .format(i, t[i+1], en, str_x))


def f_nbody(t, x, m=None, eps_soft=None):

    n = len(x) // 6

    xdot = np.zeros_like(x)

    for i in range(n):
        ri = x[6*i:6*i+3]
        vi = x[6*i+3:6*i+6]

        xdot[6*i:6*i+3] = vi

        for j in range(n):
            if i == j:
                continue

            rj = x[6*j:6*j+3]

            dr = rj - ri
            r2 = dr[0]**2 + dr[1]**2 + dr[2]**2 + eps_soft**2
            r = np.sqrt(r2)

            g = m[j] * dr / (r**3)

            xdot[6*i+3:6*i+6] += g

    return xdot


def calc_energy(x, m=None, eps_soft=None):

    kin = 0.0
    pot = 0.0

    n = len(x) // 6

    for i in range(n):
        ri = x[6*i: 6*i+3]
        vi = x[6*i+3: 6*i+6]

        kin += 0.5*m[i] * (vi**2).sum()

        for j in range(i):

            rj = x[6*j:6*j+3]
            dr = rj - ri
            pot += -m[i]*m[j] / np.sqrt((dr**2).sum() + eps_soft**2)

    return kin + pot


def generate_init_disc(N, Rmax, aspect_ratio, Mmin, Mmax):

    Mavg = 0.5*(Mmin + Mmax)

    Hmax = aspect_ratio * Rmax

    z = Hmax * (2*np.random.rand(Nbody) - 1.0)
    r_cyl = Rmax * np.sqrt(np.random.rand(Nbody))
    phi = 2*np.pi * np.random.rand(Nbody)

    M_int = Mavg * Nbody * (r_cyl/Rmax)**2
    v_cyl = np.sqrt(M_int / r_cyl)

    M = Mmin + (Mmax - Mmin) * np.random.rand(Nbody)

    rx = r_cyl * np.cos(phi)
    ry = r_cyl * np.sin(phi)
    rz = z
    vx = -v_cyl * np.sin(phi)
    vy = v_cyl * np.cos(phi)
    vz = np.zeros_like(z)

    return M, rx, ry, rz, vx, vy, vz


def main(tMax, Nbody, Niter):

    Mmin = 1.0
    Mmax = 10.0
    Rmax = 100.0
    eps = 0.01

    M, rx, ry, rz, vx, vy, vz = generate_init_disc(
            Nbody, Rmax, 0.1, Mmin, Mmax)

    x0 = np.empty(6*Nbody)

    for i in range(Nbody):
        x0[6*i] = rx[i]
        x0[6*i+1] = ry[i]
        x0[6*i+2] = rz[i]
        x0[6*i+3] = vx[i]
        x0[6*i+4] = vy[i]
        x0[6*i+5] = vz[i]

    kwargs = {'m': M, 'eps_soft': eps}

    dt_eps = np.sqrt(eps**3 / Mmax)

    dt = tMax / Niter

    print("dt:", dt, "dt_eps:", dt_eps)

    filename = "orbits.txt"
    with open(filename, "w") as f:
        f.write(" ".join([str(m) for m in M]) + "\n")
    print("Writing to", filename)
    evolve(0.0, tMax, x0, Niter, f_nbody, kwargs, filename)


if __name__ == "__main__":

    Tmax = float(sys.argv[1])
    Nbody = int(sys.argv[2])
    Niter = int(sys.argv[3])

    main(Tmax, Nbody, Niter)
