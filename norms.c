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


double l2_norm(double *u, double *v, int dim) {
    ENTER_FUNC;
    NUM_ADDS(3*dim);
    NUM_MULS(dim);
    NUM_SQRTS(1);
    double norm = 0;
    for (int i = 0; i < dim; i++) {
        norm += (u[i] - v[i]) * (u[i] - v[i]);
    }
    norm = sqrt(norm);
    EXIT_FUNC;
    return norm;
}



double l2_norm_squared(double *u, double *v, int dim) {
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
