#ifndef _EIG_H
#define _EIG_H

#include <arpack.h>

void smallest_eigenvalues(double *A, a_int N, a_int nev, double *ret_eigenvalues, double *ret_eigenvectors);
void all_eigenvalues(double *A, int n, double *ret_eigenvalues);

#endif