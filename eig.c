#include <cblas.h>
#include <arpack.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <lapacke.h>

#include "eig.h"

static void transpose(double *A, int rows, int cols, double *ret) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            ret[i * cols + j] = A[i + j * rows];
        }
    }
}

void smallest_eigenvalues(double *A, a_int N, a_int nev, double *ret_eigenvalues, double *ret_eigenvectors) {
    a_int ido = 0;
    char bmat[] = "I";
    char which[] = "SM";
    double tol = 1e-6;
    double resid[N];
    a_int ncv = 2 * nev + 1;
    double V[ncv * N];
    a_int ldv = N;
    a_int iparam[11];
    a_int ipntr[14];
    double workd[3 * N];
    a_int rvec = 1;
    char howmny[] = "A";
    double* d = ret_eigenvalues;
    a_int select[ncv];
    for (int i = 0; i < ncv; i++) select[i] = 1;
    double *z = malloc(N * nev * sizeof(double));
    a_int ldz = N;
    double sigma = 0;
    memset(workd, 0, sizeof(*workd));
    double workl[3 * (ncv * ncv) + 6 * ncv];
    memset(workl, 0, sizeof(*workl));
    a_int lworkl = 3 * (ncv * ncv) + 6 * ncv;
    a_int info = 0;

    iparam[0] = 1;
    iparam[2] = 100000;
    iparam[3] = 1;
    iparam[4] = 0;  // number of ev found by arpack.
    iparam[6] = 1;

    while (ido != 99) {
        dsaupd_c(&ido, bmat, N, which, nev, tol, resid, ncv, V, ldv, iparam, ipntr,
                    workd, workl, lworkl, &info);

        cblas_dgemv(CblasColMajor, CblasNoTrans, N, N, 1, A, N, &(workd[ipntr[0] - 1]), 1, 0, &(workd[ipntr[1] - 1]), 1);
    }
    if (iparam[4] != nev) {
        fprintf(stderr, "\033[31mWARNING: tolerance of 1e-6 was not reached for some eigenvalues\033[0m\n");
    }

    dseupd_c(rvec, howmny, select, d, z, ldz, sigma, bmat, N, which, nev, tol,
            resid, ncv, V, ldv, iparam, ipntr, workd, workl, lworkl, &info);

    transpose(z, N, nev, ret_eigenvectors);

    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < nev; j++) {
    //         printf("%f, ", ret_eigenvectors[i*nev + j]);
    //     }
    //     printf("\n");
    // }
    free(d);
}

void all_eigenvalues(double *A, int n, double *ret_eigenvalues) {
    int lda = n;
    lapack_int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, A, lda, ret_eigenvalues);
    /* Check for convergence */
    if (info > 0) {
        fprintf(stderr, "\033[31mWARNING: The algorithm failed to compute eigenvalues.\033[0m\n");
    }
}
