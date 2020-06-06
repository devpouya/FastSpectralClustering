#include <math.h>
#include "norms.h"
#include "instrumentation.h"
#include <immintrin.h>
#include <string.h>


/*
 * Inspired from https://www.tfzx.net/article/918974.html
 */

// static void print_m256d(__m256d d) {
//   double *a = (double *) &d;
//   printf("{%lf %lf %lf %lf}\n", a[0], a[1], a[2], a[3]);
// }

// static void print_m128(__m128 d) {
//   float *a = (float *) &d;
//   printf("{%f %f %f %f}\n", a[0], a[1], a[2], a[3]);
// }

// static void print_m128i(__m128i d) {
//   int *a = (int *) &d;
//   printf("{%d %d %d %d}\n", a[0], a[1], a[2], a[3]);
// }

// static void print_m256(__m256 d) {
//   float *a = (float *) &d;
//   printf("{%f %f %f %f %f %f %f %f}\n", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
// }

// static void print_m256i(__m256i d) {
//   int *a = (int *) &d;
//   printf("{%d %d %d %d %d %d %d %d}\n", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
// }

// union test {
//     double d;
//     struct {
//         int i;
//         int j;
//     } n;
// };

#define MAKE_MASK8(i0, i1, i2, i3, i4, i5, i6, i7) (i0 << 7 | i1 << 6 | i2 << 5 | i3 << 4 | i4 << 3 | i5 << 2 | i6 << 1 | i7)

__m256d exp256_pd_fast(__m256d x) {
    // printf("-------------\n");
    NUM_ADDS(4*3);
    NUM_MULS(4);
    // __m256 to_float = _mm256_castpd_ps(x);  // zero latency
    __m256d c1 = _mm256_set1_pd(1512775.3951951856938);
    __m256d c2 = _mm256_set1_pd(1072632447);
    __m256i selector = _mm256_set_epi32(3, 7, 2, 6, 1, 5, 0, 4);

    __m256d temp = _mm256_fmadd_pd(c1, x, c2);  // latency 4
    // print_m256d(temp);
    // printf("%lf\n", C1 * 1 + C2);
    __m128i temp_int = _mm256_cvtpd_epi32(temp);  // latency 7
    // print_m128i(temp_int);
    // printf("%d\n", (int) C1 * 1 + C2);
    // __m128 temp_float_cast = _mm_cvtsi128_ps(temp_int);
    // __m256i temp_int_broadcast = _mm256_broadcastsi128_si256(temp_int);
    __m256i temp_int_extend = _mm256_castsi128_si256(temp_int);  // zero latency
    // print_m256i(temp_int_extend);
    __m256 temp_cast = _mm256_castsi256_ps(temp_int_extend);  // zero latency
    // print_m256(temp_cast);
    // __m256i selector = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256 permute = _mm256_permutevar8x32_ps(temp_cast, selector);  // latency 3
    // print_m256(permute);
    // __m256 float_result = _mm256_blend_ps(to_float, permute, MAKE_MASK8(1, 0, 1, 0, 1, 0, 1, 0));  // latency 1
    // print_m256(float_result);
    // union test test;
    // test.n.j = 1512775.3951951856938*1 +1072632447;
    // printf("%lf\n", test.d);

    // __m256d result = _mm256_castps_pd(float_result);  // latency 0
    __m256d result = _mm256_castps_pd(permute);  // latency 0
    // print_m256d(result);
    // printf("---------sdf-----\n");
    return result;
}

// NOT USED -- NO INSTRUMENTATION
__m256d exp256_pd(__m256d in)
{
    // print_m256d(in);
    __m128 y = _mm256_cvtpd_ps(in);
    // print_m128(y);
    __m256 x = _mm256_castps128_ps256(y);
    // print_m256(x);

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

    // print_m256(r);
    __m128 temp = _mm256_castps256_ps128(r);
    __m256d out = _mm256_cvtps_pd(temp);

    return out;
}

/**
 * Exp function
 * @param x
 * @return
 */

// NOT USED
double fast_LUT_exp(double x) {
    //ENTER_FUNC;
    NUM_ADDS(1);
    NUM_MULS(2);
    uint64_t tmp = (1512775*x+1072632447);
    int index = (int) (tmp>>12) & 0xFF;
    //EXIT_FUNC;
    return ((double )(tmp<<32 ))* ADJUSTMENT_LUT[index];
}

// NOT USED
double fast_exp(double x) {
    //ENTER_FUNC;
    NUM_MULS(2);
    NUM_DIVS(1);
    NUM_ADDS(2);
    int x1 = (long long) (6051102*x+1056478197);
    int x2 = (long long) (1056478197-6051102*x);
    //EXIT_FUNC;
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
    //ENTER_FUNC;
    NUM_ADDS(3);
    NUM_MULS(1+1);
    double inner = exp(-0.5 * l2_norm_squared(u, v, dim));
    //EXIT_FUNC;
    return inner;
}

double gaussian_similarity_lowdim(double *u, double *v, int dim) {
    //ENTER_FUNC;
    NUM_ADDS(3);
    NUM_MULS(1+1);
    double inner = exp(-0.5 * l2_norm_squared_lowdim(u, v, dim));
    //EXIT_FUNC;
    return inner;
}


double fast_gaussian_similarity(double *u, double *v, int dim) {
    ENTER_FUNC;
    NUM_ADDS(3);
    NUM_MULS(1+1);
    double inner = EXP(-0.5 * l2_norm_squared(u, v, dim));
    EXIT_FUNC;
    return inner;
}

// __m256d fast_gaussian_similarity_2d_vec(double *u, double *v) {
//     //ENTER_FUNC;

//     __m256d u1u2 = _mm256_loadu_pd(u);  // 2 points
//     __m256d v1v2 = _mm256_loadu_pd(v);
//     __m256d v3v4 = _mm256_loadu_pd(v + 4);
//     __m256d v5v6 = _mm256_loadu_pd(v + 8);
//     __m256d v7v8 = _mm256_loadu_pd(v + 12);

//     __m256d u1u2_v1v2 = _mm256_sub_pd(u1u2, v1v2);
//     __m256d u1u2_v3v4 = _mm256_sub_pd(u1u2, v3v4);
//     __m256d u1u2_v5v6 = _mm256_sub_pd(u1u2, v5v6);
//     __m256d u1u2_v7v8 = _mm256_sub_pd(u1u2, v7v8);

//     __m256d u1u2_v1v2_2 = _mm256_mul_pd(u1u2_v1v2, u1u2_v1v2);
//     __m256d u1u2_v3v4_2 = _mm256_mul_pd(u1u2_v3v4, u1u2_v3v4);
//     __m256d u1u2_v5v6_2 = _mm256_mul_pd(u1u2_v5v6, u1u2_v5v6);
//     __m256d u1u2_v7v8_2 = _mm256_mul_pd(u1u2_v7v8, u1u2_v7v8);

//     __m256d norm_u1_v1 =
// }

/**
 * Fast gaussian using simd exp
 * @param u [i]
 * @param v [j, j+1, j+2, j+3]
 * @param dim [# of col]
 * @return vector of 4 computation of fast gaussian using simd instr.
 */
__m256d fast_gaussian_similarity_vec(double *u, double *v, int dim) {
    ENTER_FUNC;

    double norms[4] __attribute__((aligned(32)));
    double norm1[4] __attribute__((aligned(32))), norm2[4] __attribute__((aligned(32))), norm3[4] __attribute__((aligned(32))), norm4[4] __attribute__((aligned(32)));

    __m256d v_u1, v_v1, v_v2, v_v3, v_v4, v_sub1, v_sub2, v_sub3, v_sub4;
    __m256d v_norm1, v_norm2, v_norm3, v_norm4, zeros, half, result;

    zeros = _mm256_setzero_pd();
    half = _mm256_set1_pd(-0.5);
    v_norm1 = zeros; v_norm2 = zeros; v_norm3 = zeros; v_norm4 = zeros;
    memset(norms, 0, 4*sizeof(double));

    int i;
    for (i = 0; i < dim - 3; i+=4) {
        v_u1 = _mm256_loadu_pd(u + i);

        v_v1 = _mm256_loadu_pd(v       + i);
        v_v2 = _mm256_loadu_pd(v+dim   + i);
        v_v3 = _mm256_loadu_pd(v+2*dim + i);
        v_v4 = _mm256_loadu_pd(v+3*dim + i);

        NUM_ADDS(4*4);
        v_sub1 = _mm256_sub_pd(v_u1, v_v1);
        v_sub2 = _mm256_sub_pd(v_u1, v_v2);
        v_sub3 = _mm256_sub_pd(v_u1, v_v3);
        v_sub4 = _mm256_sub_pd(v_u1, v_v4);

        NUM_MULS(4*4);
        NUM_ADDS(4*4);
        v_norm1 = _mm256_fmadd_pd(v_sub1, v_sub1, v_norm1);
        v_norm2 = _mm256_fmadd_pd(v_sub2, v_sub2, v_norm2);
        v_norm3 = _mm256_fmadd_pd(v_sub3, v_sub3, v_norm3);
        v_norm4 = _mm256_fmadd_pd(v_sub4, v_sub4, v_norm4);
    }
    // use doubles
    _mm256_store_pd(norm1, v_norm1);
    _mm256_store_pd(norm2, v_norm2);
    _mm256_store_pd(norm3, v_norm3);
    _mm256_store_pd(norm4, v_norm4);
    // sum up entries of array for each one into one double => stored back in a array
    for(int j = 0; j < 4; j++) {
        NUM_ADDS(4);
        norms[0] += norm1[j];
        norms[1] += norm2[j];
        norms[2] += norm3[j];
        norms[3] += norm4[j];
    }
    // tail handling
    for (; i < dim; i++) {
        NUM_ADDS(12);
        NUM_MULS(4);
        norms[0] += (u[i] - v[i]) * (u[i] - v[i]);
        norms[1] += (u[i+dim] - v[i+dim]) * (u[i+dim] - v[i+dim]);
        norms[2] += (u[i+2*dim] - v[i+2*dim]) * (u[i+2*dim] - v[i+2*dim]);
        norms[3] += (u[i+3*dim] - v[i+3*dim]) * (u[i+3*dim] - v[i+3*dim]);
    }
    // printf("norms[0]=%lf norms[1]=%lf norms[2]=%lf norms[3]=%lf\n", norms[0], norms[1], norms[2], norms[3]);
    result = _mm256_load_pd(norms);
    NUM_MULS(4);
    result = _mm256_mul_pd(half, result);
    result = exp256_pd_fast(result);
    // printf("exp = ");
    // print_m256d(result);
    // printf("\n");
    // __m256 test = _mm256_set1_ps(1);
    // test = exp256_ps(test);
    // printf("exp2 = ");
    // print_m256(test);
    // printf("\n");
    // __m256d test2 = _mm256_set1_pd(2);
    // test2 = exp256_pd_test(test2);
    // printf("exp3 = ");
    // print_m256d(test2);
    // printf("\n");
    EXIT_FUNC;
    return result;
}

double fast_gaussian_similarity_lowdim(double *u, double *v, int dim) {
    //ENTER_FUNC;
    NUM_MULS(1+1);
    NUM_ADDS(3);
    double inner = EXP(-0.5 * l2_norm_squared_lowdim(u, v, dim));
    //EXIT_FUNC;
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
    //ENTER_FUNC;
    NUM_ADDS(2*dim);
    NUM_MULS(dim);
    NUM_SQRTS(1);
    double norm = 0;
    double temp;
    for (int i = 0; i < dim; i++){
        temp = u[i] - v[i];
        norm += temp * temp;
    }
    norm = sqrt(norm);
    //EXIT_FUNC;
    return norm;
}

double l2_norm_squared_lowdim(double *u, double *v, int dim) {
    //ENTER_FUNC;
    NUM_ADDS(2*dim);
    NUM_MULS(dim);
    double norm = 0;
    double temp;
    for (int i = 0; i < dim; i++) {
        temp = u[i] - v[i];
        norm += temp * temp;
    }
    //EXIT_FUNC;
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
    //ENTER_FUNC;

    double norm0, norm1, norm2, norm3, norm4, norm5, norm6, norm7;
    norm0 = norm1 = norm2 = norm3 = norm4 = norm5 = norm6 = norm7 = 0;
    int i;
    for (i = 0; i < dim - 7; i = i+8) {
        NUM_ADDS(24);
        NUM_MULS(8);
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
    double temp;
    for (; i < dim; i++){
        NUM_MULS(1);
        NUM_ADDS(2);
        temp = u[i] - v[i];
        norm0 += temp * temp;
    }

    NUM_ADDS(7);
    NUM_SQRTS(1);
//    double norm = babylonian_squareRoot(norm0+norm1+norm2+norm3+norm4+norm5+norm6+norm7);
    double norm = sqrt(norm0+norm1+norm2+norm3+norm4+norm5+norm6+norm7);
    //EXIT_FUNC;
    return norm;
}

// https://stackoverflow.com/questions/49941645/get-sum-of-values-stored-in-m256d-with-sse-avx
// NOT USED
static inline double hsum_double_avx(__m256d v) {
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
    vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
}

double l2_norm_squared_vec(double *u, double *v, int dim) {
    //ENTER_FUNC;

    double norm = 0;

    __m256d v_u1, v_u2, v_v1, v_v2, v_sub1, v_sub2, zeros, v_norm1, v_norm2;

    zeros = _mm256_setzero_pd();
    v_norm1 = zeros; v_norm2 = zeros;

    int i;
    for (i = 0; i < dim - 7; i+=8) {
        v_u1 = _mm256_loadu_pd(u + i);
        v_v1 = _mm256_loadu_pd(v + i);
        v_u2 = _mm256_loadu_pd(u + i + 4);
        v_v2 = _mm256_loadu_pd(v + i + 4);

        NUM_ADDS(16);
        NUM_MULS(8);
        v_sub1 = _mm256_sub_pd(v_u1, v_v1);
        v_sub2 = _mm256_sub_pd(v_u2, v_v2);

        v_norm1 = _mm256_fmadd_pd(v_sub1, v_sub1, v_norm1);
        v_norm2 = _mm256_fmadd_pd(v_sub2, v_sub2, v_norm2);
    }

    NUM_ADDS(4);
    v_norm1 = _mm256_add_pd(v_norm1, v_norm2);
    // norm = hsum_double_avx(v_norm1);

    NUM_ADDS(3);
    norm = (((double *) &v_norm1)[0] + ((double *) &v_norm1)[1]) + (((double *) &v_norm1)[2] + ((double *) &v_norm1)[3]);

    // for(int j = 0; j < 4; j++) { norm += norm2[j]; }
    // tail handling
    double temp;
    for (; i < dim; i++) {
        NUM_ADDS(2);
        NUM_MULS(1);
        temp = u[i] - v[i];
        norm += temp * temp;
    }
    // norm = sqrt(norm);
    //EXIT_FUNC;
    return norm;
}

double l2_norm_vec(double *u, double *v, int dim) {
    //ENTER_FUNC;

    double norm = 0;

    __m256d v_u1, v_u2, v_v1, v_v2, v_sub1, v_sub2, zeros, v_norm1, v_norm2;

    zeros = _mm256_setzero_pd();
    v_norm1 = zeros; v_norm2 = zeros;

    int i;
    for (i = 0; i < dim - 7; i+=8) {
        v_u1 = _mm256_loadu_pd(u + i);
        v_v1 = _mm256_loadu_pd(v + i);
        v_u2 = _mm256_loadu_pd(u + i + 4);
        v_v2 = _mm256_loadu_pd(v + i + 4);

        NUM_ADDS(16);
        NUM_MULS(8);
        v_sub1 = _mm256_sub_pd(v_u1, v_v1);
        v_sub2 = _mm256_sub_pd(v_u2, v_v2);

        v_norm1 = _mm256_fmadd_pd(v_sub1, v_sub1, v_norm1);
        v_norm2 = _mm256_fmadd_pd(v_sub2, v_sub2, v_norm2);
    }

    NUM_ADDS(4);
    v_norm1 = _mm256_add_pd(v_norm1, v_norm2);
    // norm = hsum_double_avx(v_norm1);

    NUM_ADDS(3);
    norm = (((double *) &v_norm1)[0] + ((double *) &v_norm1)[1]) + (((double *) &v_norm1)[2] + ((double *) &v_norm1)[3]);

    // for(int j = 0; j < 4; j++) { norm += norm2[j]; }
    // tail handling
    double temp;
    for (; i < dim; i++) {
        NUM_ADDS(2);
        NUM_MULS(1);
        temp = u[i] - v[i];
        norm += temp * temp;
    }

    NUM_SQRTS(1);
    norm = sqrt(norm);
    //EXIT_FUNC;
    return norm;
}

// OLD -- INSTRUMENTATION OUT OF DATE
double l2_norm_vec_old(double *u, double *v, int dim) {
    //ENTER_FUNC;
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
    //EXIT_FUNC;
    return norm;
}

// NOT USED -- INSTRUMENTATION OUT OF DATE
__m256d l2_norm_4x1_vec(double *u, double *v, int dim) {
    //ENTER_FUNC;
    NUM_ADDS(3*dim);
    NUM_MULS(dim);
    NUM_SQRTS(1);

    double norms[4];
    double norm1[4], norm2[4], norm3[4], norm4[4];

    __m256d v_v1, v_u1, v_u2, v_u3, v_u4, v_sub1, v_sub2, v_sub3, v_sub4;
    __m256d v_norm1, v_norm2, v_norm3, v_norm4, zeros, result;

    zeros = _mm256_setzero_pd();
    v_norm1 = zeros; v_norm2 = zeros; v_norm3 = zeros; v_norm4 = zeros;

    int i;
    for (i = 0; i < dim - 3; i+=4) {

        v_u1 = _mm256_loadu_pd(u       + i);
        v_u2 = _mm256_loadu_pd(u+dim   + i);
        v_u3 = _mm256_loadu_pd(u+2*dim + i);
        v_u4 = _mm256_loadu_pd(u+3*dim + i);

        v_v1 = _mm256_loadu_pd(v + i);

        v_sub1 = _mm256_sub_pd(v_u1, v_v1);
        v_sub2 = _mm256_sub_pd(v_u2, v_v1);
        v_sub3 = _mm256_sub_pd(v_u3, v_v1);
        v_sub4 = _mm256_sub_pd(v_u4, v_v1);

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
    result = _mm256_sqrt_pd(result);
    //EXIT_FUNC;
    return result;
}

double l2_norm_squared_base(double *u, double *v, int dim) {
    //ENTER_FUNC;

    double norm0, norm1, norm2, norm3, norm4, norm5, norm6, norm7;
    norm0 = norm1 = norm2 = norm3 = norm4 = norm5 = norm6 = norm7 = 0;
    int i;
    for (i = 0; i < dim - 7; i = i + 8) {
        NUM_ADDS(24);
        NUM_MULS(8);
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
    double temp;
    for (; i < dim; i++){
        NUM_ADDS(2);
        NUM_MULS(1);
        temp = u[i] - v[i];
        norm0 += temp * temp;
    }
    NUM_ADDS(7);
    double norm = norm0+norm1+norm2+norm3+norm4+norm5+norm6+norm7;
    //EXIT_FUNC;
    return norm;
}

// NOT USED -- INSTRUMENTATION OUT OF DATE
double l2_norm_squared_vec_old(double *u, double *v, int dim) {
    //ENTER_FUNC;
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
    //EXIT_FUNC;
    return norm;
}

// /**
//  * Generic Methods for kmeans: change name definition in return {HERE};
//  *
//  * @param u
//  * @param v
//  * @param dim
//  * @return
//  */

// double l2_norm(double *u, double *v, int dim) {
//     return l2_norm_vec(u, v, dim);
// }

// double l2_norm_squared(double *u, double *v, int dim) {
//     return l2_norm_squared_vec(u, v, dim);
// }

// double l2_norm_lowdim(double *u, double *v, int dim){
//     return l2_norm_lowdim_base(u, v, dim);
// }


