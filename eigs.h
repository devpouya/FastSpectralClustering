#ifndef _EIGS_H
#define _EIGS_H

#ifdef __cplusplus
extern "C" {
#endif

void smallest_eigenvalues(float *A, int n, int k, float *ret_eigenvalues, float *ret_eigenvectors);

#ifdef __cplusplus
}
#endif

#endif