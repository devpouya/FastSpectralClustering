#include <cblas.h>
#include <arpack.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <arpack.h>

#include "eigs.h"

static void transpose(double *A, int rows, int cols, double *ret) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            ret[i * cols + j] = A[i + j * rows];
        }
    }
}

void smallest_eigenvalues(double *A, int n, int k, double *ret_eigenvalues, double *ret_eigenvectors) {
    a_int ido = 0;
    char bmat[] = "I";
    char which[] = "SM";
    double tol = 1e-12;
    a_int N = n;
    a_int nev = k;
    double resid[N];
    a_int ncv = 2 * nev + 1;
    double V[ncv * N];
    a_int ldv = N;
    a_int iparam[11];
    a_int ipntr[14];
    double workd[3 * N];
    a_int rvec = 1;
    char howmny[] = "A";
    double *d;
    if (ret_eigenvalues) {
        d = ret_eigenvalues;
    } else {
        d = malloc(k * sizeof(double));
    }
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
        fprintf(stderr, "\033[31mWARNING: tolerance of %lf was not reached for some eigenvalues\033[0m\n", (double) tol);
    }

//    sseupd_c(a_int rvec, char const* howmny, a_int const* select, double* d, double* z, a_int ldz,
//            double sigma, char const* bmat, a_int n, char const* which, a_int nev, double tol,
//            double* resid, a_int ncv, double* v, a_int ldv, a_int* iparam, a_int* ipntr,
//            double* workd, double* workl, a_int lworkl, a_int* info);

    dseupd_c(rvec, howmny, select, d, z, ldz, sigma, bmat, N, which, nev, tol,
            resid, ncv, V, ldv, iparam, ipntr, workd, workl, lworkl, &info);

    transpose(z, N, nev, ret_eigenvectors);

    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < nev; j++) {
    //         printf("%f, ", ret_eigenvectors[i*nev + j]);
    //     }
    //     printf("\n");
    // }
    if (!ret_eigenvalues) free(d);
    free(z);
}
