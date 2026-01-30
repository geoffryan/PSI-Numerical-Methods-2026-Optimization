#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "mycalc.h"

static double *dr = NULL;
static double *r2 = NULL;
static double *ir = NULL;

void calc_g_transpose(double *g, double *r, double *m, long N, double eps_soft)
{
    long N3 = 3*N;

    double e2 = eps_soft*eps_soft;

    long i, j, k;
    for(i=0; i<N3; i++)
        g[i] = 0.0;

    if(dr == NULL)
        dr = (double *)malloc(N*N*3 * sizeof(double));
    if(r2 == NULL)
        r2 = (double *)malloc(N*N * sizeof(double));
    if(ir == NULL)
        ir = (double *)malloc(N*N * sizeof(double));

    for(k=0; k<3; k++)
    {
        double *drx = dr + N*N*k;
        double *rx = r + N*k;
        for(i=0; i<N; i++)
            for(j=0; j<N; j++)
                drx[N*i + j] = rx[j] - rx[i];
    }

    for(i=0; i < N*N; i++)
        r2[i] = 0.0;

    for(k=0; k<3; k++)
    {
        double *drx = dr + k*N*N;
        for(i=0; i<N*N; i++)
            r2[i] += drx[i] * drx[i];
    }
    
    for(i=0; i<N*N; i++)
        ir[i] = 1.0 / sqrt(r2[i] + e2);

    for(k=0; k<3; k++)
    {
        double *gx = g + N*k;
        double *drx = dr + N*N*k;
        for(i=0; i<N; i++)
        {
            double gi = 0.0;
            double *iri = ir + N*i;
            double *drxi = drx + N*i;
            for(j=0; j<N; j++)
                gi += m[j] * iri[j] * iri[j] * iri[j] * drxi[j];
            gx[i] = gi;
        }
    }
}

void calc_g(double *g, double *r, double *m, long N, double eps_soft)
{
    long N3 = 3*N;

    double e2 = eps_soft*eps_soft;

    long i, j, k;
    for(i=0; i<N3; i++)
        g[i] = 0.0;

    for(i=0; i<N; i++)
    {
        for(j=0; j<i; j++)
        {
            double dx = r[3*j] - r[3*i];
            double dy = r[3*j+1] - r[3*i+1];
            double dz = r[3*j+2] - r[3*i+2];

            double r = sqrt(dx*dx + dy*dy + dz*dz + e2);

            double ir3 = 1.0 / (r*r*r);

            g[3*i] += m[j] * ir3 * dx;
            g[3*i+1] += m[j] * ir3 * dy;
            g[3*i+2] += m[j] * ir3 * dz;
            g[3*j] += -m[i] * ir3 * dx;
            g[3*j+1] += -m[i] * ir3 * dy;
            g[3*j+2] += -m[i] * ir3 * dz;
        }
    }
}

void calc_g_par(double *g, double *r, double *m, long N, double eps_soft)
{
    long N3 = 3*N;

    double e2 = eps_soft*eps_soft;

    long i;
    #pragma omp parallel for simd
    for(i=0; i<N; i++)
    {
        long idx = 3*i;
        g[idx] = 0.0;
        g[idx + 1] = 0.0;
        g[idx + 2] = 0.0;

        long j;
        for(j=0; j<N; j++)
        {
            long jdx = 3*j;
            double dx = r[jdx] - r[idx];
            double dy = r[jdx+1] - r[idx+1];
            double dz = r[jdx+2] - r[idx+2];

            double r = sqrt(dx*dx + dy*dy + dz*dz + e2);

            double ir3 = 1.0 / (r*r*r);

            g[idx] += m[j] * ir3 * dx;
            g[idx+1] += m[j] * ir3 * dy;
            g[idx+2] += m[j] * ir3 * dz;
        }
    }
}
