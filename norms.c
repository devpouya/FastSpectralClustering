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

/*Returns the square root of n. Note that the function */
double babylonian_squareRoot(double n)
{
    /*We are using n itself as initial approximation
   This can definitely be improved */
    double x = n;
    double y = 1;
    double e = 0.000001; /* e decides the accuracy level*/
    while (x - y > e) {
        x = (x + y) / 2;
        y = n / x;
    }
    return x;
}

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

//    double norm = babylonian_squareRoot(norm0+norm1+norm2+norm3+norm4+norm5+norm6+norm7);
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
    double inner = EXP(-0.5 * l2_norm_squared(u, v, dim));

    EXIT_FUNC;
    return inner;
}
