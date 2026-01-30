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

This produces the `basic.prof` file with profiling information. It can be visualized in a (Flame Graph)[https://www.brendangregg.com/flamegraphs.html] with the `flameprof` package:
```bash
$ uv run flameprof basic.prof > basic.svg
```
