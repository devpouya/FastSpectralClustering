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

/*
 * compute degrees matrix D
 */
static double compute_degrees(double *graph, double *degrees, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (graph[i*n+j] > 0.0) {
                degrees[i*n+i] += graph[i*n+j];
            } else {
                degrees[i*n+i] += 0;
            }
        }
    }
//    for (int i = 0; i < n; i++) {
//        for (int j = 0; j < n; j++) {
//            printf("%f ", degrees[i*n+ j]);
//        }
//        printf("\n");
//    }
}

/*
 * compute unnormalized graph laplacian
 */
static void construct_unnormalized_laplacian(double *graph, double *laplacian, int n) {
    double* degrees = (double *)malloc(n * n * sizeof(double));
    compute_degrees(graph, degrees, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            laplacian[i*n+j] = degrees[i*n+j] - graph[i*n+j];
            printf("%f ", laplacian[i*n+j]);
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
    int dim ;
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
    double fully_connected[lines][lines];
    construct_fully_connected_matrix((double *) points, lines, dim, (double *) fully_connected);
    // epsilon neighborhood matrix
    int eps_neighborhood[lines][lines];
    construct_eps_neighborhood_matrix((double *) points, lines, dim, (int *) eps_neighborhood);
    // Skip KNN matrix since too annoying to compute

    // compute unnormalized laplacian
    double* laplacian = (double *)malloc(lines * lines * sizeof(double));
    construct_unnormalized_laplacian(fully_connected, laplacian, lines);
    return 0;
}
