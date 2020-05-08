#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "norms.h"
#include "instrumentation.h"
#include "construct_graph.h"

static double *TheArray;
static int cmp(const void *a, const void *b){
    int ia = *(int *)a;
    int ib = *(int *)b;
    return (TheArray[ia] > TheArray[ib]) - (TheArray[ia] < TheArray[ib]);
}

static void calculate_diagonal_degree_matrix(double * weighted_adj_matrix, int n, double *ret){
    ENTER_FUNC;
    NUM_ADDS(n*n);
    for (int i = 0; i < n; i++) {
        double d_i = 0;
        for (int j = 0; j < n;j++) {
                d_i += weighted_adj_matrix[i * n + j];
        }
        ret[i] = d_i;
    }
    EXIT_FUNC;
}



void construct_fully_connected_matrix(double *points, int lines, int dim, double *ret) {
    ENTER_FUNC;
    for (int i = 0; i < lines-3; i+=4) {
        for (int j = i+1; j < lines; j+=4) {

            ret[i * lines + j] = fast_gaussian_similarity(&points[i * dim], &points[j * dim], dim);
            ret[(i+1) * lines + j] = fast_gaussian_similarity(&points[(i+1) * dim], &points[j * dim], dim);
            ret[(i+2) * lines + j] = fast_gaussian_similarity(&points[(i+2) * dim], &points[j * dim], dim);
            ret[(i+3) * lines + j] = fast_gaussian_similarity(&points[(i+3) * dim], &points[j * dim], dim);

            ret[i * lines + j+1] = fast_gaussian_similarity(&points[i * dim], &points[(j+1) * dim], dim);
            ret[(i+1) * lines + j+1] = fast_gaussian_similarity(&points[(i+1) * dim], &points[(j+1) * dim], dim);
            ret[(i+2) * lines + j+1] = fast_gaussian_similarity(&points[(i+2) * dim], &points[(j+1) * dim], dim);
            ret[(i+3) * lines + j+1] = fast_gaussian_similarity(&points[(i+3) * dim], &points[(j+1) * dim], dim);

            ret[i * lines + j+2] = fast_gaussian_similarity(&points[i * dim], &points[(j+2) * dim], dim);
            ret[(i+1) * lines + j+2] = fast_gaussian_similarity(&points[(i+1) * dim], &points[(j+2) * dim], dim);
            ret[(i+2) * lines + j+2] = fast_gaussian_similarity(&points[(i+2) * dim], &points[(j+2) * dim], dim);
            ret[(i+3) * lines + j+2] = fast_gaussian_similarity(&points[(i+3) * dim], &points[(j+2) * dim], dim);

            ret[i * lines + j+3] = fast_gaussian_similarity(&points[i * dim], &points[(j+3) * dim], dim);
            ret[(i+1) * lines + j+3] = fast_gaussian_similarity(&points[(i+1) * dim], &points[(j+3) * dim], dim);
            ret[(i+2) * lines + j+3] = fast_gaussian_similarity(&points[(i+2) * dim], &points[(j+3) * dim], dim);
            ret[(i+3) * lines + j+3] = fast_gaussian_similarity(&points[(i+3) * dim], &points[(j+3) * dim], dim);

        }
        for (int j = 0; j < i; ++j) {
            ret[i*lines+j] = ret[j*lines+i];
            ret[(i+1)*lines+j] = ret[j*lines+i+1];
            ret[(i+2)*lines+j] = ret[j*lines+i+2];
            ret[(i+3)*lines+j] = ret[j*lines+i+3];
        }
    }
    EXIT_FUNC;
}



void construct_eps_neighborhood_matrix(double *points, int lines, int dim, double *ret) {
    ENTER_FUNC;
    for (int i = 0; i < lines; ++i) {
        for (int j = 0; j < lines; ++j) {
            ret[i*lines + j] = l2_norm(&points[i*dim], &points[j*dim], dim) < EPS;
        }
    }
    EXIT_FUNC;
}

void construct_normalized_laplacian_sym_matrix(double *weighted_adj_matrix, int num_points, double *ret){
    ENTER_FUNC;
    double sqrt_inv_degree_matrix[num_points];  // '1-d' array
    calculate_diagonal_degree_matrix(weighted_adj_matrix, num_points, sqrt_inv_degree_matrix); //load degree_matrix temporarily in sqrt_inv_degree_matrix
    NUM_SQRTS(num_points);
    for (int i =0; i < num_points; i++){
        sqrt_inv_degree_matrix[i] = 1.0/sqrt(sqrt_inv_degree_matrix[i]);
    }
    // compute D^(-1/2) W,  not sure if this code is optimal yet, how to avoid "jumping row"?  process one row each time enccourage spatial locality
    // but with this trick we avoid *0.0
    NUM_MULS(num_points * num_points);
    for (int i = 0; i < num_points; i++){
        for(int j = 0; j < num_points; j++){
            ret[i*num_points + j] = sqrt_inv_degree_matrix[i] * weighted_adj_matrix[i*num_points + j];
        }
    }
    // compute (D^(-1/2)*W)*D^(-1/2)
    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < num_points; j++) {
            if (i == j) {
                NUM_ADDS(1); NUM_MULS(1);
                ret[i*num_points + j] = 1.0 - ret[i*num_points + j] * sqrt_inv_degree_matrix[j];
            } else {
                NUM_MULS(1);
                ret[i*num_points + j] = -ret[i*num_points + j] * sqrt_inv_degree_matrix[j];
            }
        }
    }
    EXIT_FUNC;
}

void construct_normalized_laplacian_rw_matrix(double *weighted_adj_matrix, int num_points, double *ret) {
    ENTER_FUNC;
    double inv_degree_matrix[num_points];
    calculate_diagonal_degree_matrix(weighted_adj_matrix, num_points, inv_degree_matrix); //load degree_matrix temporarily in sqrt_inv_degree_matrix
    NUM_SQRTS(num_points);
    for (int i = 0; i < num_points; i++){
        inv_degree_matrix[i] = 1.0/inv_degree_matrix[i];
    }
    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < num_points; j++) {
            NUM_MULS(1);
            if (i == j) {
                NUM_ADDS(1);
                ret[i*num_points + j] = 1.0 - inv_degree_matrix[i] * weighted_adj_matrix[i*num_points + j];
            } else {
                ret[i*num_points + j] = -inv_degree_matrix[i] * weighted_adj_matrix[i*num_points + j];
            }
        }
    }
    EXIT_FUNC;
}

void construct_unnormalized_laplacian(double *graph, int n, double *ret) {
    ENTER_FUNC;
    // double* degrees = (double *)malloc(n * n * sizeof(double));
    double degrees[n];
    calculate_diagonal_degree_matrix(graph, n, degrees);
    //NUM_ADDS(n*n);
    NUM_MULS(n*n);
    for (int i = 0; i < n; i++) {
        for(int j = 0; j < n; j+=4) {
            //if(i==j) continue; DON'T COMPUTE THE DIAGONAL ELEMENTS (BETTER WHEN VECTORIZING)
            ret[i * n + j] = -1 * graph[i * n + j];

            ret[i * n + j+1] = -1 * graph[i * n + j+1];

            ret[i * n + j+2] = -1 * graph[i * n + j+2];

            ret[i * n + j+3] = -1 * graph[i * n + j+3];


        }

        ret[i*n+i] = degrees[i];

    }

    EXIT_FUNC;
}

void construct_knn_matrix(double *points, int lines, int dim, int k, double *ret) {
    ENTER_FUNC;
    double vals[lines];
    int indices[lines];

    for (int i = 0; i < lines; ++i) {
        for(int ii = 0; ii < lines; ++ii) {
            indices[ii] = ii;
        }
        for (int j = 0; j < lines; ++j) {
            ret[i*lines + j] = 0.0;
            vals[j] = l2_norm(&points[i*dim], &points[j*dim], dim);
        }
        TheArray = vals;
        size_t len = sizeof(vals) / sizeof(*vals);
        qsort(indices, len, sizeof(*indices), cmp);
        for (int s = 0; s < k+1; ++s) {
            ret[i*lines + indices[s]] = (indices[s] != i) ? 1.0 : 0;
        }
    }
    EXIT_FUNC;
}