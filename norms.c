#include <math.h>
#include "norms.h"
#include "instrumentation.h"
#include <immintrin.h>


/*
 * https://www.tfzx.net/article/918974.html
 */

__m256d exp256_pd(__m256d in)
{
    __m256 x = _mm256_castpd_ps(in);

    __m256 t, f, p, r;
    __m256i i, j;

    const __m256 l2e = _mm256_set1_ps (1.442695041f); /* log2(e) */
    const __m256 l2h = _mm256_set1_ps (-6.93145752e-1f); /* -log(2)_hi */
    const __m256 l2l = _mm256_set1_ps (-1.42860677e-6f); /* -log(2)_lo */
    /* coefficients for core approximation to exp() in [-log(2)/2, log(2)/2] */
    const __m256 c0 =  _mm256_set1_ps (0.041944388f);
    const __m256 c1 =  _mm256_set1_ps (0.168006673f);
    const __m256 c2 =  _mm256_set1_ps (0.499999940f);
    const __m256 c3 =  _mm256_set1_ps (0.999956906f);
    const __m256 c4 =  _mm256_set1_ps (0.999999642f);

    /* exp(x) = 2^i * e^f; i = rint (log2(e) * x), f = x - log(2) * i */
    t = _mm256_mul_ps (x, l2e);      /* t = log2(e) * x */
    r = _mm256_round_ps (t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); /* r = rint (t) */


    f = _mm256_fmadd_ps (r, l2h, x); /* x - log(2)_hi * r */
    f = _mm256_fmadd_ps (r, l2l, f); /* f = x - log(2)_hi * r - log(2)_lo * r */


    i = _mm256_cvtps_epi32(t);       /* i = (int)rint(t) */

    /* p ~= exp (f), -log(2)/2 <= f <= log(2)/2 */
    p = c0;                          /* c0 */

    p = _mm256_fmadd_ps (p, f, c1);  /* c0*f+c1 */
    p = _mm256_fmadd_ps (p, f, c2);  /* (c0*f+c1)*f+c2 */
    p = _mm256_fmadd_ps (p, f, c3);  /* ((c0*f+c1)*f+c2)*f+c3 */
    p = _mm256_fmadd_ps (p, f, c4);  /* (((c0*f+c1)*f+c2)*f+c3)*f+c4 ~= exp(f) */

    /* exp(x) = 2^i * p */
    j = _mm256_slli_epi32 (i, 23); /* i << 23 */
    r = _mm256_castsi256_ps (_mm256_add_epi32 (j, _mm256_castps_si256 (p))); /* r = p * 2^i */
    __m256d out = _mm256_castps_pd(r);

    return out;
}

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

/**
 * Fast gaussian using simd exp
 * @param u [i]
 * @param v [j, j+1, j+2, j+3]
 * @param dim [# of col]
 * @return vector of 4 computation of fast gaussian using simd instr.
 */
__m256d fast_gaussian_similarity_vec(double *u, double *v, int dim) {
    ENTER_FUNC;
    NUM_ADDS(3*dim);
    NUM_MULS(dim);

    double norms[4];
    double norm1[4], norm2[4], norm3[4], norm4[4];

    __m256d v_u1, v_v1, v_v2, v_v3, v_v4, v_sub1, v_sub2, v_sub3, v_sub4;
    __m256d v_norm1, v_norm2, v_norm3, v_norm4, zeros, half, result;

    zeros = _mm256_setzero_pd();
    half = _mm256_set1_pd(-0.5);
    v_norm1 = zeros; v_norm2 = zeros; v_norm3 = zeros; v_norm4 = zeros;

    int i;
    for (i = 0; i < dim - 3; i+=4) {
        v_u1 = _mm256_loadu_pd(u + i);

        v_v1 = _mm256_loadu_pd(v       + i);
        v_v2 = _mm256_loadu_pd(v+dim   + i);
        v_v3 = _mm256_loadu_pd(v+2*dim + i);
        v_v4 = _mm256_loadu_pd(v+3*dim + i);

        v_sub1 = _mm256_sub_pd(v_u1, v_v1);
        v_sub2 = _mm256_sub_pd(v_u1, v_v2);
        v_sub3 = _mm256_sub_pd(v_u1, v_v3);
        v_sub4 = _mm256_sub_pd(v_u1, v_v4);

        v_norm1 = _mm256_fmadd_pd(v_sub1, v_sub1, v_norm1);
        v_norm2 = _mm256_fmadd_pd(v_sub2, v_sub2, v_norm2);
        v_norm3 = _mm256_fmadd_pd(v_sub3, v_sub3, v_norm3);
        v_norm4 = _mm256_fmadd_pd(v_sub4, v_sub4, v_norm4);
    }
    // use doubles
    _mm256_storeu_pd(norm1, v_norm1);
    _mm256_storeu_pd(norm2, v_norm2);
    _mm256_storeu_pd(norm3, v_norm3);
    _mm256_storeu_pd(norm4, v_norm4);
    // sum up entries of array for each one into one double => stored back in a array
    for(int j = 0; j < 4; j++) {
        norms[0] += norm1[j];
        norms[1] += norm2[j];
        norms[2] += norm3[j];
        norms[3] += norm4[j];
    }
    // tail handling
    for (; i < dim; i++) {
        norms[0] += (u[i] - v[i]) * (u[i] - v[i]);
        norms[1] += (u[i+dim] - v[i+dim]) * (u[i+dim] - v[i+dim]);
        norms[2] += (u[i+2*dim] - v[i+2*dim]) * (u[i+2*dim] - v[i+2*dim]);
        norms[3] += (u[i+3*dim] - v[i+3*dim]) * (u[i+3*dim] - v[i+3*dim]);
    }
    result = _mm256_loadu_pd(norms);
    result = _mm256_mul_pd(half, result);
    result = exp256_pd(result);
    EXIT_FUNC;
    return result;
}

double fast_gaussian_similarity_lowdim(double *u, double *v, int dim) {
    ENTER_FUNC;
    NUM_MULS(1);
    double inner = EXP(-0.5 * l2_norm_squared_lowdim(u, v, dim));

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


