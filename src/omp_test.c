#include <stdio.h>
#include <math.h>
#include <omp.h>

int main(int argc, char *argv[])
{

    printf("num threads: %d\n", omp_get_num_threads());
    int N = 1024;

    double *x = (double *)malloc(N*sizeof(double));

    long i;
#pragma omp parallel for
    for(i = 0; i < N; i++) {
        printf("num threads: %d\n", omp_get_num_threads());
        x[i] = sin(i) * sqrt(i*i + 1) / (cos(0.1*i) * cos(0.1*i));
    }

    printf("x[17]: %f\n", x[717]);
    printf("num threads: %d\n", omp_get_num_threads());


    return 0;
}
