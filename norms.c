#include <math.h>
#include "norms.h"
#include "instrumentation.h"


double appx_exp(double x) {
    ENTER_FUNC;
    NUM_ADDS(5);
    NUM_MULS(14);
    EXIT_FUNC;
    return 1 + x + x*x*0.5 + x*x*x*0.166 + x*x*x*x*0.04166 + x*x*x*x*x*0.00833;
}

double appx_sqrt_babylonian(double x) {
    ENTER_FUNC;
    NUM_MULS(100);
    NUM_ADDS(100);
    NUM_DIVS(100);
    double xx = x;
    double y = 1.0;
    // double acc = 0.0001; // find a way to use this and measure flops
    // while (x-y>acc)
    for(int i = 0; i < 100; i++) {
        xx = (xx+y)*0.5;
        y = x/xx;
    }
    EXIT_FUNC;
    return xx;
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
    EXIT_FUNC;
    return sqrt(norm);
}

double appx_l2_norm(double *u, double *v, int dim) {
    ENTER_FUNC;
    NUM_ADDS(3*dim);
    NUM_MULS(dim);
    double norm = 0;
    for (int i = 0; i < dim; i++) {
        norm += (u[i] - v[i]) * (u[i] - v[i]);
    }
    EXIT_FUNC;
    return appx_sqrt_babylonian(norm);
}

/*
double gaussian_similarity(double *u, double *v, int dim) {
    ENTER_FUNC;
    NUM_EXPS(1);
    NUM_DIVS(0);
    NUM_MULS(5);
    double norm = l2_norm(u,v,dim);
    double inner = -1.0*norm*norm*0.5;
    EXIT_FUNC;
    return exp(inner);
}

*/
double gaussian_similarity(double *u, double *v, int dim) {
    ENTER_FUNC;
    NUM_DIVS(0);
    NUM_MULS(5);
    double norm = l2_norm(u,v,dim); // appx_l2_norm(u,v,dim);
    double inner = -1.0*norm*norm*0.5;
    EXIT_FUNC;
    return appx_exp(inner);
}