#include <math.h>
#include "norms.h"
#include "instrumentation.h"
#include <immintrin.h>

/**
 * Exp function
 * @param x
 * @return
 */

double fast_LUT_exp(double x) {
    ENTER_FUNC;
    NUM_ADDS(1);
    NUM_MULS(2);
    uint64_t tmp = (1512775*x+1072632447);
    int index = (int) (tmp>>12) & 0xFF;
    EXIT_FUNC;
    return ((double )(tmp<<32 ))* ADJUSTMENT_LUT[index];
}

double fast_exp(double x) {
    ENTER_FUNC;
    NUM_MULS(2);
    NUM_DIVS(1);
    NUM_ADDS(2);
    int x1 = (long long) (6051102*x+1056478197);
    int x2 = (long long) (1056478197-6051102*x);
    EXIT_FUNC;
    return ((double) x1)/((double) x2);
}

/**
 * Gaussian similarity methods
 * @param u
 * @param v
 * @param dim
 * @return
 */


double gaussian_similarity(double *u, double *v, int dim) {
    ENTER_FUNC;
    NUM_EXPS(1);
    NUM_MULS(1);
    double inner = EXP(-0.5 * l2_norm_squared(u, v, dim));
    EXIT_FUNC;
    return inner;
}


double fast_gaussian_similarity(double *u, double *v, int dim) {
    ENTER_FUNC;
    NUM_MULS(1);
    double inner = EXP(-0.5 * l2_norm_squared(u, v, dim));
    EXIT_FUNC;
    return inner;
}

double fast_gaussian_similarity_lowdim(double *u, double *v, int dim) {
    ENTER_FUNC;
    NUM_MULS(1);
    double inner = exp(-0.5 * l2_norm_squared_lowdim(u, v, dim));

    EXIT_FUNC;
    return inner;
}


/**
 * l2-norm for low dimension = low k ( low # of clusters )
 * @param u
 * @param v
 * @param dim
 * @return
 */

double l2_norm_lowdim_base(double *u, double *v, int dim){
    ENTER_FUNC;
    NUM_ADDS(3*dim);
    NUM_MULS(dim);
    NUM_SQRTS(1);
    double norm = 0;
    for (int i = 0; i < dim; i++){
        norm += (u[i] - v[i]) * (u[i] - v[i]);
    }
    norm = sqrt(norm);
    EXIT_FUNC;
    return norm;
}

double l2_norm_squared_lowdim(double *u, double *v, int dim) {
    ENTER_FUNC;
    NUM_ADDS(3*dim);
    NUM_MULS(dim);
    double norm = 0;
    for (int i = 0; i < dim; i++) {
        norm += (u[i] - v[i]) * (u[i] - v[i]);
    }
    EXIT_FUNC;
    return norm;
}

/**
 * l2 norm for k>8 clusters
 * @param u
 * @param v
 * @param dim
 * @return
 */

double l2_norm_base(double *u, double *v, int dim) {
    ENTER_FUNC;
    NUM_ADDS(3*dim);
    NUM_MULS(dim);
    NUM_SQRTS(1);

    double norm0, norm1, norm2, norm3, norm4, norm5, norm6, norm7;
    norm0 = norm1 = norm2 = norm3 = norm4 = norm5 = norm6 = norm7 = 0;
    int i;
    for (i = 0; i < dim - 7; i = i+8) {
        norm0 += (u[i] - v[i]) * (u[i] - v[i]);
        norm1 += (u[i+1] - v[i+1]) * (u[i+1] - v[i+1]);
        norm2 += (u[i+2] - v[i+2]) * (u[i+2] - v[i+2]);
        norm3 += (u[i+3] - v[i+3]) * (u[i+3] - v[i+3]);
        norm4 += (u[i+4] - v[i+4]) * (u[i+4] - v[i+4]);
        norm5 += (u[i+5] - v[i+5]) * (u[i+5] - v[i+5]);
        norm6 += (u[i+6] - v[i+6]) * (u[i+6] - v[i+6]);
        norm7 += (u[i+7] - v[i+7]) * (u[i+7] - v[i+7]);
    }
    // tail handling
    for (; i < dim; i++){
        norm0 += (u[i] - v[i]) * (u[i] - v[i]);
    }

//    double norm = babylonian_squareRoot(norm0+norm1+norm2+norm3+norm4+norm5+norm6+norm7);
    double norm = sqrt(norm0+norm1+norm2+norm3+norm4+norm5+norm6+norm7);
    EXIT_FUNC;
    return norm;
}


double l2_norm_vec(double *u, double *v, int dim) {
    ENTER_FUNC;
    NUM_ADDS(3*dim);
    NUM_MULS(dim);
    NUM_SQRTS(1);

    double norm = 0;
    double norm2[4];

    __m256d v_u1, v_u2, v_v1, v_v2, v_sub1, v_sub2, zeros, v_norm1, v_norm2;

    zeros = _mm256_setzero_pd();
    v_norm1 = zeros; v_norm2 = zeros;

    int i;
    for (i = 0; i < dim - 7; i+=8) {
        v_u1 = _mm256_loadu_pd(u + i);
        v_v1 = _mm256_loadu_pd(v + i);
        v_u2 = _mm256_loadu_pd(u + i + 4);
        v_v2 = _mm256_loadu_pd(v + i + 4);

        v_sub1 = _mm256_sub_pd(v_u1, v_v1);
        v_sub2 = _mm256_sub_pd(v_u2, v_v2);

        v_norm1 = _mm256_fmadd_pd(v_sub1, v_sub1, v_norm1);
        v_norm2 = _mm256_fmadd_pd(v_sub2, v_sub2, v_norm2);
    }

    v_norm1 = _mm256_add_pd(v_norm1, v_norm2);
    _mm256_storeu_pd(norm2, v_norm1);

    for(int j = 0; j < 4; j++) { norm += norm2[j]; }
    // tail handling
    for (; i < dim; i++) {
        norm += (u[i] - v[i]) * (u[i] - v[i]);
    }
    norm = sqrt(norm);
    EXIT_FUNC;
    return norm;
}

double l2_norm_squared_base(double *u, double *v, int dim) {
    ENTER_FUNC;
    NUM_ADDS(3 * dim);
    NUM_MULS(dim);

    double norm0, norm1, norm2, norm3, norm4, norm5, norm6, norm7;
    norm0 = norm1 = norm2 = norm3 = norm4 = norm5 = norm6 = norm7 = 0;
    int i;
    for (i = 0; i < dim - 7; i = i + 8) {
        norm0 += (u[i] - v[i]) * (u[i] - v[i]);
        norm1 += (u[i+1] - v[i+1]) * (u[i+1] - v[i+1]);
        norm2 += (u[i+2] - v[i+2]) * (u[i+2] - v[i+2]);
        norm3 += (u[i+3] - v[i+3]) * (u[i+3] - v[i+3]);
        norm4 += (u[i+4] - v[i+4]) * (u[i+4] - v[i+4]);
        norm5 += (u[i+5] - v[i+5]) * (u[i+5] - v[i+5]);
        norm6 += (u[i+6] - v[i+6]) * (u[i+6] - v[i+6]);
        norm7 += (u[i+7] - v[i+7]) * (u[i+7] - v[i+7]);
    }
    // tail handling
    for (; i < dim; i++){
        norm0 += (u[i] - v[i]) * (u[i] - v[i]);
    }
    double norm = norm0+norm1+norm2+norm3+norm4+norm5+norm6+norm7;
    EXIT_FUNC;
    return norm;
}

double l2_norm_squared_vec(double *u, double *v, int dim) {
    ENTER_FUNC;
    NUM_ADDS(3*dim);
    NUM_MULS(dim);

    double norm = 0;
    double norm2[4];

    __m256d v_u1, v_u2, v_v1, v_v2, v_sub1, v_sub2, zeros, v_norm1, v_norm2;

    zeros = _mm256_setzero_pd();
    v_norm1 = zeros; v_norm2 = zeros;

    int i;
    for (i = 0; i < dim - 7; i+=8) {
        v_u1 = _mm256_loadu_pd(u + i);
        v_v1 = _mm256_loadu_pd(v + i);
        v_u2 = _mm256_loadu_pd(u + i + 4);
        v_v2 = _mm256_loadu_pd(v + i + 4);

        v_sub1 = _mm256_sub_pd(v_u1, v_v1);
        v_sub2 = _mm256_sub_pd(v_u2, v_v2);

        v_norm1 = _mm256_fmadd_pd(v_sub1, v_sub1, v_norm1);
        v_norm2 = _mm256_fmadd_pd(v_sub2, v_sub2, v_norm2);
    }

    v_norm1 = _mm256_add_pd(v_norm1, v_norm2);
    _mm256_storeu_pd(norm2, v_norm1);

    for(int j = 0; j < 4; j++) { norm += norm2[j]; }
    // tail handling
    for (; i < dim; i++) {
        norm += (u[i] - v[i]) * (u[i] - v[i]);
    }
    EXIT_FUNC;
    return norm;
}

/**
 * Generic Methods for kmeans: change name definition in return {HERE};
 *
 * @param u
 * @param v
 * @param dim
 * @return
 */

double l2_norm(double *u, double *v, int dim) {
    return l2_norm_base(u, v, dim);
}

double l2_norm_squared(double *u, double *v, int dim) {
    return l2_norm_squared_base(u, v, dim);
}

double l2_norm_lowdim(double *u, double *v, int dim){
    return l2_norm_lowdim_base(u, v, dim);
}