#ifndef _EIGS_H
#define _EIGS_H

#ifdef __cplusplus
extern "C" {
#endif

void smallest_eigenvalues(double *A, int n, int k, double *ret_eigenvalues, double *ret_eigenvectors);

#ifdef __cplusplus
}
#endif

#endif
