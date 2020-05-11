#ifndef _EIG_H
#define _EIG_H

#include <arpack.h>

void smallest_eigenvalues(float *A, a_int N, a_int nev, float *ret_eigenvalues, float *ret_eigenvectors);
void all_eigenvalues(float *A, int n, float *ret_eigenvalues);

#endif