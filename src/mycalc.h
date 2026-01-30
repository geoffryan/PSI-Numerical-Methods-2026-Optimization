#ifndef MYCALC_H
#define MYCALC_H

void calc_g(double *g, double *r, double *m, long N, double eps_soft);
void calc_g_transpose(double *g, double *r, double *m, long N, double eps_soft);
void calc_g_par(double *g, double *r, double *m, long N, double eps_soft);

#endif
