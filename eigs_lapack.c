#include <cblas.h>
#include <arpack.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <lapacke.h>

#include "eigs.h"
#include "util.h"

static void all_eigenvalues(float *A, int n, float *ret_eigenvalues) {
    int lda = n;
    lapack_int info = LAPACKE_ssyev(LAPACK_ROW_MAJOR, 'V', 'U', n, A, lda, ret_eigenvalues);
    /* Check for convergence */
    if (info > 0) {
        fprintf(stderr, "\033[31mWARNING: The algorithm failed to compute eigenvalues.\033[0m\n");
        exit(1);
    }
}

void smallest_eigenvalues(float *A, int n, int k, float *ret_eigenvalues, float *ret_eigenvectors) {
    all_eigenvalues(A, n, ret_eigenvalues);
    copy_submatrix(A, n, n, k, ret_eigenvectors);
}
