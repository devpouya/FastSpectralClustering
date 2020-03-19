#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#define EPS 2

static void repeat_str(const char *str, int times, char *ret) {
    int len = strlen(str);
    printf("%d\n", len);
    for (int i = 0; i < times; i++) {
        strncpy(ret + i*len, str, len);
    }
    ret[len*times] = '\0';
}

static double *TheArray;
static int cmp(const void *a, const void *b){
    int ia = *(int *)a;
    int ib = *(int *)b;
    return (TheArray[ia] > TheArray[ib]) - (TheArray[ia] < TheArray[ib]);
}

static double l2_norm(double *u, double *v, int dim) {
    double norm = 0;
    for (int i = 0; i < dim; i++) {
        norm += (u[i] - v[i]) * (u[i] - v[i]);
    }
    return sqrt(norm);
}

static void construct_fully_connected_matrix(double *points, int lines, int dim, double *ret) {
    for (int i = 0; i < lines; ++i) {
        for (int j = 0; j < lines; ++j) {
            ret[i*lines + j] = l2_norm(&points[i*dim], &points[j*dim], dim);
            printf("%lf ", ret[i*lines + j]);
        }
        printf("\n");
    }
}

static void construct_eps_neighborhood_matrix(double *points, int lines, int dim, int *ret) {
    for (int i = 0; i < lines; ++i) {
        for (int j = 0; j < lines; ++j) {
            ret[i*lines + j] = l2_norm(&points[i*dim], &points[j*dim], dim) < EPS;
            printf("%d ", ret[i*lines + j]);
        }
        printf("\n");
    }
}

static void calculate_diagonal_degree_matrix(double * weighted_adj_matrix, int n, double *ret){
    for (int i = 0; i < n; i++) {
        double d_i = 0;
        for (int j = 0; j < n;j++) {
            d_i += weighted_adj_matrix[i*n+ j];
        }
        ret[i] = d_i;
        printf("%lf ", ret[i]);
    }
    printf("\n");
}

static void construct_normalized_laplacian_sym_matrix(double *weighted_adj_matrix, int num_points, double *ret){
    double sqrt_inv_degree_matrix[num_points];  // '1-d' array
    calculate_diagonal_degree_matrix(weighted_adj_matrix, num_points, sqrt_inv_degree_matrix); //load degree_matrix temporarily in sqrt_inv_degree_matrix
    for (int i =0; i < num_points; i++){
        sqrt_inv_degree_matrix[i] = 1.0/sqrt(sqrt_inv_degree_matrix[i]);
    }
    // compute D^(-1/2) W,  not sure if this code is optimal yet, how to avoid "jumping row"?  process one row each time enccourage spatial locality
    // but with this trick we avoid *0.0
    for (int i = 0; i < num_points; i++){
        for(int j = 0; j < num_points; j++){
            ret[i*num_points + j] = sqrt_inv_degree_matrix[i] * weighted_adj_matrix[i*num_points + j];
        }
    }
    // compute (D^(-1/2)*W)*D^(-1/2)
    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < num_points; j++) {
            if (i == j) {
                ret[i*num_points + j] = 1.0 - ret[i*num_points + j] * sqrt_inv_degree_matrix[j];
            } else {
                ret[i*num_points + j] = -ret[i*num_points + j] * sqrt_inv_degree_matrix[j];
            }
            printf("%lf ", ret[i*num_points + j]);
        }
        printf("\n");
    }
}

static void construct_normalized_laplacian_rw_matrix(double *weighted_adj_matrix, int num_points, double *ret) {
    double inv_degree_matrix[num_points];
    calculate_diagonal_degree_matrix(weighted_adj_matrix, num_points, inv_degree_matrix); //load degree_matrix temporarily in sqrt_inv_degree_matrix
    for (int i = 0; i < num_points; i++){
        inv_degree_matrix[i] = 1.0/inv_degree_matrix[i];
    }
    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < num_points; j++) {
            if (i == j) {
                ret[i*num_points + j] = 1.0 - inv_degree_matrix[i] * weighted_adj_matrix[i*num_points + j];
            } else {
                ret[i*num_points + j] = -inv_degree_matrix[i] * weighted_adj_matrix[i*num_points + j];
            }
            printf("%lf ", ret[i*num_points + j]);
        }
        printf("\n");
    }
}

static void construct_unnormalized_laplacian(double *graph, int n, double *ret) {
    // double* degrees = (double *)malloc(n * n * sizeof(double));
    double degrees[n];
    calculate_diagonal_degree_matrix(graph, n, degrees);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            ret[i*n+j] = ((i == j) ? degrees[i] : 0) - graph[i*n+j];
            printf("%f ", ret[i*n+j]);
        }
        printf("\n");
    }
}

static void construct_knn_matrix(double *points, int lines, int dim, int k, int *ret) {
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

    for(int i = 0; i < lines; ++i) {
        for(int j = 0; j < lines; ++j) {
            printf("%d ", ret[i*lines + j]);
        }
        printf("\n");
    }
}



/*
 * The file that the program reads from is stored in the following format, assuming that
 * we are using n d-dimensional datapoints:
 * <d>\n
 * <Dim. 0 of point 0> <Dim. 1 of point 0> <Dim. 2 of point 0> ... <Dim. d of point 0>\n
 * <Dim. 0 of point 1> <Dim. 1 of point 1> <Dim. 2 of point 1> ... <Dim. d of point 1>\n
 *                           ........................
 * <Dim. 0 of point n-1> <Dim. 1 of point n-1> <Dim. 2 of point n-1> ... <Dim. d of point n-1>\n
 */

int main(int argc, char *argv[]) {
    FILE *fp;
    fp = fopen("points.txt", "r");

    // Count the number of lines in the file
    int lines = 0;
    while (EOF != (fscanf(fp, "%*[^\n]"), fscanf(fp,"%*c")))
        ++lines;
    --lines;  // Subtract one because it starts with the dimension
    printf("Read %d lines\n", lines);

    // Find the dimension
    rewind(fp);
    int dim;
    fscanf(fp, "%d\n", &dim);

    // Read the points
    char fmt[4*dim + 1];
    repeat_str("%lf ", dim, fmt);
    fmt[4*dim-1] = '\n';
    fmt[4*dim] = '\0';
    printf("Dimension = %d, fmt = %s", dim, fmt);
    double points[lines][2];
    for (int i = 0; i < lines; ++i) {
        fscanf(fp, fmt, &points[i][0], &points[i][1]);
    }

    // for (int i = 0; i < lines; ++i) {
    //     printf(fmt, points[i][0], points[i][1]);
    // }

    // Construct the matrices and print them
    // fully-connected matrix
    printf("Fully connected matrix:\n");
    double fully_connected[lines][lines];
    construct_fully_connected_matrix((double *) points, lines, dim, (double *) fully_connected);
    // epsilon neighborhood matrix

    printf("\nEps neighborhood matrix:\n");
    int eps_neighborhood[lines][lines];
    construct_eps_neighborhood_matrix((double *) points, lines, dim, (int *) eps_neighborhood);
    // Skip KNN matrix since too annoying to compute

    printf("\nKNN matrix:\n");
    int k = 2;
    int knn_graph[lines][lines];
    construct_knn_matrix((double *) points, lines, dim, k,(int *) knn_graph);

    printf("\nUnnormalized Laplacian:\n");
    // compute unnormalized laplacian
    double laplacian[lines][lines];
    construct_unnormalized_laplacian((double *) fully_connected, lines, (double *) laplacian);

    printf("\nRW Normalized Laplacian\n");
    // compute normalized rw laplacian
    double l_rw[lines][lines];
    construct_normalized_laplacian_rw_matrix((double *) fully_connected, lines, (double *) l_rw);

    printf("\nSymmetric Normalized Laplacian\n");
    // compute normalized rw laplacian
    double l_sym[lines][lines];
    construct_normalized_laplacian_sym_matrix((double *) fully_connected, lines, (double *) l_sym);
    return 0;
}