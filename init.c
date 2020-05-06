#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>

#include "norms.h"
#include "instrumentation.h"

void cumulative_sum(double *probs, int n, int m, double *ret) {
    ENTER_FUNC;
    ret[0] = probs[0];
    for(int i = 1; i < n; i++) {
        ret[i] = ret[i-1]+probs[i];
    }
    EXIT_FUNC;
}

void init_kpp(double *U, int n, int m,int k, double *ret) {
    ENTER_FUNC;
    // add a random initial point to the centers
#ifdef VALIDATION
    srand(30);
#else
    srand(time(0));
#endif
    int ind = ((int)rand()%n);
    printf("ind = %d\n", ind);
    //ret[0] = U[((int)rand()%n)*k];
    for(int j = 0; j < k; j++) {
        ret[j] = U[ind*m+j];
    }
    double sum = 0;
    for (int c = 1; c < k; c++) {

        sum = 0;
        double dists[n];
        for (int i = 0; i < n; i++) {
            //find closest point and add to sum
            double dist = DBL_MAX;
            for(int j = 0; j < c; j++) {
                double tmp = l2_norm(&U[i*m],&ret[j*k],k);
                if (tmp < dist) {
                    dist = tmp;
                }
            }
            sum += dist;
            dists[i] = dist;

        }
        for(int i = 0; i < n; i++) {
            dists[i] /= sum;
        }
        double cumsums[n];
        int index = 0;
        cumulative_sum(dists,n,m, cumsums);
        double r = rand()/((double)RAND_MAX);
        printf("r = %lf\n", r);
        for(int i = 0; i < n; i++) {
            if(r < cumsums[i]) {
                index = i;
                printf("picked index:%d\n",index);
                break;
            }
        }
        for (int i = 0; i < k; i++) {

            for (int j = 0; j < k; j++) {
                ret[c*k + j] = U[index*m+j];
            }
        }
    }
    EXIT_FUNC;
}


