#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mycalc.h"

int main(int argc, char *argv[])
{
    if(argc < 4) 
    {
        printf("usage: time_test METHOD N REPEATS\n");
        return 0;
    }

    int aos = 1;
    int par = 0;

    if(strcmp(argv[1], "trans") == 0)
    {
        aos = 0;
        par = 0;
    }

    long N = strtol(argv[2], NULL, 10);
    long repeats = strtol(argv[3], NULL, 10);

    if(N <= 0 || repeats <= 0)
    {
        printf("N and repeats must be positive\n");
        return 0;
    }
    
    srand(314);

    double *r = (double *)malloc(3*N * sizeof(double));
    double *g = (double *)malloc(3*N * sizeof(double));
    double *m = (double *)malloc(N * sizeof(double));

    long i;

    for(i=0; i<N; i++)
    {
        m[i] = 1.0 + i;

        double x = 2 * (((double) rand()) / RAND_MAX) - 1;
        double y = 2 * (((double) rand()) / RAND_MAX) - 1;
        double z = 2 * (((double) rand()) / RAND_MAX) - 1;

        if(aos)
        {
            r[3*i] = x;
            r[3*i+1] = y;
            r[3*i+2] = z;
        }
        else
        {
            r[i] = x;
            r[N + i] = y;
            r[2*N + i] = z;
        }
    }

    double tot = 0.0;

    double start = omp_get_wtime();

    if(aos && !par)
    {
        for(i=0; i<repeats; i++)
        {
            calc_g(g, r, m, N, 0.001);
            tot += g[0] + g[3*N-1];
        }
    }
    else if(!aos && !par)
    {
        for(i=0; i<repeats; i++)
        {
            printf("%ld\n", i);
            calc_g_transpose(g, r, m, N, 0.001);
            tot += g[0] + g[3*N-1];
        }
    }
    else if(aos && par)
    {
        for(i=0; i<repeats; i++)
        {
            calc_g_par(g, r, m, N, 0.001);
            tot += g[0] + g[3*N-1];
        }
    }
    double stop = omp_get_wtime();

    double res = omp_get_wtick();

    double dur = stop - start;

    long num_unique_forces = (N * (N - 1)) / 2;

    double time_per_call = dur / repeats;
    double time_per_force = time_per_call / num_unique_forces;

    printf("value: %g\n", tot);
    printf("duration: %e s\n", dur);
    printf("resolution: %e s\n", res);
    printf("time_per_call: %f us\n", 1.0e6 * time_per_call);
    printf("time_per_force: %f ns\n", 1.0e9 * time_per_force);

    return 0;
}
