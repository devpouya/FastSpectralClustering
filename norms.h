#ifndef _NORMS_H
#define _NORMS_H

#define SIGMA 1.0

double appx_exp(double x);
double appx_sqrt_babylonian(double x);
double l2_norm(double *u, double *v, int dim);
double appx_l2_norm(double *u, double *v, int dim);
double gaussian_similarity(double *u, double *v, int dim);
double appx_gaussian_similarity(double *u, double *v, int dim);

#endif
