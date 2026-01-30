#ifndef MYCALC_H
#define MYCALC_H

void calc_g(double *g, double *r, double *m, long N, double eps_soft);
double calc_en(double *r, double *v, double *m, long N, double eps_soft);
void calc_g_transpose(double *g, double *r, double *m, long N, double eps_soft);
void calc_g_par(double *g, double *r, double *m, long N, double eps_soft,
        int num_workers);
double calc_en_par(double *g, double *r, double *m, long N, double eps_soft,
        int num_workers);

#endif
