#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#define EPS 2

static double *TheArray;
static void repeat_str(const char *str, int times, char *ret) {
    int len = strlen(str);
    printf("%d\n", len);
    for (int i = 0; i < times; i++) {
        strncpy(ret + i*len, str, len);
    }
    ret[len*times] = '\0';
}

int cmp(const void *a, const void *b){
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
        size_t len = sizeof(vals)/ sizeof(*vals);
        qsort(indices, len, sizeof(*indices), cmp);
        for (int s = 0; s < k; ++s) {
            ret[i*lines + indices[s]] = 1.0;
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
    double fully_connected[lines][lines];
    construct_fully_connected_matrix((double *) points, lines, dim, (double *) fully_connected);
    // epsilon neighborhood matrix
    int eps_neighborhood[lines][lines];
    construct_eps_neighborhood_matrix((double *) points, lines, dim, (int *) eps_neighborhood);
    // Skip KNN matrix since too annoying to compute
    int k = 2;
    int knn_graph[lines][lines];
    construct_knn_matrix((double *) points, lines, dim, k,(int *) knn_graph);

    return 0;
}