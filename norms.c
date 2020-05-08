#include <math.h>
#include "norms.h"
#include "instrumentation.h"


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

// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.4508&rep=rep1&type=pdf
double fast_schraudolph_exp(double x) {
    ENTER_FUNC;
    NUM_MULS(3);
    NUM_ADDS(1);
    static union {
        double d;
        struct{
#ifdef  LITTLE_ENDIAN
            int i, j;
#else
            int j, i;
#endif

        } n; } eco;

    eco.n.i = 1512775.3951951856938*x +1072632447;
    EXIT_FUNC;
    return eco.d;


}
//
//double l2_norm(double *u, double *v, int dim) {
//    ENTER_FUNC;
//    NUM_ADDS(3*dim);
//    NUM_MULS(dim);
//    NUM_SQRTS(1);
//
//    double norm = 0;
//    int i;
//    for (i = 0; i < dim; i = i+1) {
//        norm += (u[i] - v[i]) * (u[i] - v[i]);
//    }
//    norm = sqrt(norm);
//    EXIT_FUNC;
//    return norm;
//}

//unrolled l2_norm
double l2_norm(double *u, double *v, int dim) {
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

    double norm = sqrt(norm0+norm1+norm2+norm3+norm4+norm5+norm6+norm7);
    EXIT_FUNC;
    return norm;
}



//double l2_norm_squared(double *u, double *v, int dim) {
//    ENTER_FUNC;
//    NUM_ADDS(3*dim);
//    NUM_MULS(dim);
//    double norm = 0;
//    for (int i = 0; i < dim; i++) {
//        norm += (u[i] - v[i]) * (u[i] - v[i]);
//    }
//    EXIT_FUNC;
//    return norm;
//}

double l2_norm_squared(double *u, double *v, int dim) {
    ENTER_FUNC;
    NUM_ADDS(3*dim);
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


double gaussian_similarity(double *u, double *v, int dim) {
    ENTER_FUNC;
    NUM_EXPS(1);
    NUM_MULS(1);
    double inner = exp(-0.5 * l2_norm_squared(u, v, dim));
    EXIT_FUNC;
    return inner;
}

double fast_gaussian_similarity(double *u, double *v, int dim) {
    ENTER_FUNC;
    NUM_MULS(1);
    double inner = fast_schraudolph_exp(-0.5 * l2_norm_squared(u, v, dim));

    EXIT_FUNC;
    return inner;
}
