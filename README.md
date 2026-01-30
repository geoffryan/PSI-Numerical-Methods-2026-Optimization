# Profiling and Optimizing an N-body code

## Run the simulation

Simulation scripts are in `scripts/`. The call signature is: `python nbody_basic.py TFINAL NBODY NSTEPS` where:
- `TMAX` final time to evolve to.
- `NBODY` number of gravitating bodies
- `NSTEPS` number of steps to date.  time step `dt = TMAX / NSTEPS`

Run 64 bodies to t = 1 with 8 steps:
```bash
$ uv run scripts/nbody_basic.py 1.0 64 8
Writing to orbits.txt
0 0.0 0.125
1 0.125 0.125
2 0.25 0.125
3 0.375 0.125
4 0.5 0.125
5 0.625 0.125
6 0.75 0.125
7 0.875 0.125
```

Output is written to `orbits.txt` or `orbits.h5`.

Plot the results with `scripts/plot.py`:
```bash
$ uv run scripts/plot.py orbits.txt
Saving frame_000000.png
Saving frame_000001.png
Saving frame_000002.png
Saving frame_000003.png
Saving frame_000004.png
Saving frame_000005.png
Saving frame_000006.png
Saving frame_000007.png
Saving frame_000008.png
```

## Profiling a code with `cProfile`

cProfile is built in to Python and will profile a whole program.  The signature is: `python -m cProfile -o OUTPUT_FILE SCRIPT_NAME SCRIPT_ARGS...`

Profile the nbody code.
```bash
$ uv run python -m cProfile -o basic.prof scripts/nbody_basic.py 1.0 64 8
```

This produces the `basic.prof` file with profiling information. It can be visualized in a [Flame Graph](https://www.brendangregg.com/flamegraphs.html) with the `flameprof` package:
```bash
$ uv run flameprof basic.prof > basic.svg
```

The .svg file can be viewed in a browser.

## Profiling a function line-by-line with `line-profiler`

`line-profiler` is a Python package to do line-by-line profiling of a Python program.

It must be imported into your program and can be used to instrument specific functions.  Import with `from line-profiler import profile` and add the `@profile` decorator to a function to mark it for profiling.  Perform the profiling with `kernprof` utility included in `line-profiler`.

```bash
$ uv run kernprof -l scripts/nbody_basic.py 1.0 64 8
Writing to orbits.txt
0 0.0 0.125
1 0.125 0.125
2 0.25 0.125
3 0.375 0.125
4 0.5 0.125
5 0.625 0.125
6 0.75 0.125
7 0.875 0.125
Wrote profile results to 'nbody_basic.py.lprof'
Inspect results with:
python -m line_profiler -rmt nbody_basic.py.lprof
```

Nicely, `kernprof` tells you how to display the results:
```bash
$ uv run python -m line_profiler -rmt nbody_basic.py.lprof
Timer unit: 1e-06 s

Total time: 0.504224 s
File: scripts/nbody_basic.py
Function: f_nbody at line 38

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    38                                           @profile
    39                                           def f_nbody(t, x, m=None, eps_soft=None):
    40                                           
    41        32         11.0      0.3      0.0      n = len(x) // 6
    42                                           
    43        32        154.0      4.8      0.0      xdot = np.zeros_like(x)
    44                                           
    45      2080        555.0      0.3      0.1      for i in range(n):
    46      2048        637.0      0.3      0.1          ri = x[6*i:6*i+3]
    47      2048        654.0      0.3      0.1          vi = x[6*i+3:6*i+6]
    48                                           
    49      2048        778.0      0.4      0.2          xdot[6*i:6*i+3] = vi
    50                                           
    51    133120      37543.0      0.3      7.4          for j in range(n):
    52    131072      30624.0      0.2      6.1              if i == j:
    53      2048        437.0      0.2      0.1                  continue
    54                                           
    55    129024      40289.0      0.3      8.0              rj = x[6*j:6*j+3]
    56                                           
    57    129024      59464.0      0.5     11.8              dr = rj - ri
    58    129024      75001.0      0.6     14.9              r2 = dr[0]**2 + dr[1]**2 + dr[2]**2 + eps_soft**2
    59    129024      41191.0      0.3      8.2              r = np.sqrt(r2)
    60                                           
    61    129024     136897.0      1.1     27.2              g = m[j] * dr / (r**3)
    62                                           
    63    129024      79960.0      0.6     15.9              xdot[6*i+3:6*i+6] += g
    64                                           
    65        32         29.0      0.9      0.0      return xdot

  0.50 seconds - scripts/nbody_basic.py:38 - f_nbody
```
