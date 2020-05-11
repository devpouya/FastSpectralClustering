#include <cblas.h>
#include <arpack.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <arpack.h>

#include "eigs.h"

static void transpose(float *A, int rows, int cols, float *ret) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            ret[i * cols + j] = A[i + j * rows];
        }
    }
}

void smallest_eigenvalues(float *A, int n, int k, float *ret_eigenvalues, float *ret_eigenvectors) {
    a_int ido = 0;
    char bmat[] = "I";
    char which[] = "SM";
    float tol = 1e-12;
    a_int N = n;
    a_int nev = k;
    float resid[N];
    a_int ncv = 2 * nev + 1;
    float V[ncv * N];
    a_int ldv = N;
    a_int iparam[11];
    a_int ipntr[14];
    float workd[3 * N];
    a_int rvec = 1;
    char howmny[] = "A";
    float *d;
    if (ret_eigenvalues) {
        d = ret_eigenvalues;
    } else {
        d = malloc(k * sizeof(float));
    }
    a_int select[ncv];
    for (int i = 0; i < ncv; i++) select[i] = 1;
    float *z = malloc(N * nev * sizeof(float));
    a_int ldz = N;
    float sigma = 0;
    memset(workd, 0, sizeof(*workd));
    float workl[3 * (ncv * ncv) + 6 * ncv];
    memset(workl, 0, sizeof(*workl));
    a_int lworkl = 3 * (ncv * ncv) + 6 * ncv;
    a_int info = 0;

    iparam[0] = 1;
    iparam[2] = 100000;
    iparam[3] = 1;
    iparam[4] = 0;  // number of ev found by arpack.
    iparam[6] = 1;

    while (ido != 99) {
        ssaupd_c(&ido, bmat, N, which, nev, tol, resid, ncv, V, ldv, iparam, ipntr,
                    workd, workl, lworkl, &info);

        cblas_sgemv(CblasColMajor, CblasNoTrans, N, N, 1, A, N, &(workd[ipntr[0] - 1]), 1, 0, &(workd[ipntr[1] - 1]), 1);
    }

    if (iparam[4] != nev) {
        fprintf(stderr, "\033[31mWARNING: tolerance of %lf was not reached for some eigenvalues\033[0m\n", (double) tol);
    }

//    sseupd_c(a_int rvec, char const* howmny, a_int const* select, float* d, float* z, a_int ldz,
//            float sigma, char const* bmat, a_int n, char const* which, a_int nev, float tol,
//            float* resid, a_int ncv, float* v, a_int ldv, a_int* iparam, a_int* ipntr,
//            float* workd, float* workl, a_int lworkl, a_int* info);

    sseupd_c(rvec, howmny, select, d, z, ldz, sigma, bmat, N, which, nev, tol,
            resid, ncv, V, ldv, iparam, ipntr, workd, workl, lworkl, &info);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < nev; j++) {
            printf("%f, ", ret_eigenvectors[i*nev + j]);
        }
        printf("\n");
    }

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
