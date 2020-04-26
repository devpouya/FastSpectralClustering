#ifndef _NORMS_H
#define _NORMS_H

#define SIGMA 1.0

double l2_norm(double *u, double *v, int dim);
double gaussian_similarity(double *u, double *v, int dim);

#endif
