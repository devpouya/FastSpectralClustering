#include <math.h>
#include "norms.h"
#include "instrumentation.h"

double l2_norm(double *u, double *v, int dim) {
    ENTER_FUNC;
    NUM_ADDS(3*dim);
    NUM_MULS(dim);
    NUM_SQRTS(1);
    double norm = 0;
    for (int i = 0; i < dim; i++) {
        norm += (u[i] - v[i]) * (u[i] - v[i]);
    }
    EXIT_FUNC;
    return sqrt(norm);
}

double gaussian_similarity(double *u, double *v, int dim) {
    ENTER_FUNC;
    NUM_EXPS(1);
    NUM_DIVS(1);
    NUM_MULS(4);
    double norm = l2_norm(u,v,dim);
    double inner = -1.0*norm*norm/(2*SIGMA*SIGMA);
    EXIT_FUNC;
    return exp(inner);
}