#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <lapacke.h>

#include "tsc_x86.h"
#include "norms.h"
#include "construct_graph.h"
#include "kmeans.h"
#include "util.h"

/*
 * The file that the program reads from is stored in the following format, assuming that
 * we are using n d-dimensional datapoints:
 * <d>\n
 * <Dim. 0 of point 0> <Dim. 1 of point 0> <Dim. 2 of point 0> ... <Dim. d of point 0>\n
 * <Dim. 0 of point 1> <Dim. 1 of point 1> <Dim. 2 of point 1> ... <Dim. d of point 1>\n
 *                           ........................
 * <Dim. 0 of point n-1> <Dim. 1 of point n-1> <Dim. 2 of point n-1> ... <Dim. d of point n-1>\n
 * arguments: dataset_path, number of clusters (k)
 */
int main(int argc, char *argv[]) {

    if (argc != 4) {
        printf("usage: %s points_file num_clusters output_file\n", argv[0]);
    }

    printf("loading dataset: %s\n", argv[1]);
    printf("number of clusters: %d\n", atoi(argv[2]));
    printf("output path: %s\n", argv[3]);

    struct file f = alloc_load_points_from_file(argv[1]);
    int dim = f.dimension;
    int lines = f.lines;
    double *points = f.points;
    int k = atoi(argv[2]);
    int n = lines;
    int lda = n;

    // Construct the matrices and print them
    // fully-connected matrix
    printf("Constructing fully connected matrix...\n");
    double *fully_connected = malloc(lines * lines * sizeof(double));
    construct_fully_connected_matrix(points, lines, dim, fully_connected);

    printf("Constructing unnormalized Laplacian...\n");
    // compute unnormalized laplacian
    double *laplacian = malloc(lines * lines * sizeof(double));
    myInt64 start = start_tsc();
    construct_unnormalized_laplacian(fully_connected, lines, laplacian);

    //compute the eigendecomposition and take the first k eigenvectors.
    double *w = malloc(lines * sizeof(double));
    lapack_int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, laplacian, lda, w);
    /* Check for convergence */
    if (info > 0) {
        fprintf(stderr, "ERROR: The algorithm failed to compute eigenvalues.\n");
        exit(1);
    }

    // printf("Eigenvalues:\n");
    // for (int i = 0; i < n; i++) {
    //     printf("%lf, ", w[i]);
    // }
    // printf("\n");

    // print_matrix("Eigenvectors (stored columnwise)", n, n, laplacian, lda);

    printf("Performing k-means clustering...\n");
    // U (8x2) is the data in points.txt for now => k = 2
    // number of cluster <=> # columns of U

    // init datastructure
    struct cluster clusters[k];
    for (int i = 0; i < k; i++) {
        clusters[i].mean = malloc(k * sizeof(double)); // k is the "dimension" here
        clusters[i].size = 0;
        clusters[i].indices = malloc(lines * sizeof(int)); // at most
    }
    // try with different max_iter
    // kmeans(points, lines, k, 10, clusters);

    kmeans(laplacian, lines, k, 100, 0.0001, clusters);
    myInt64 runtime = stop_tsc(start);

    print_cluster_indices(clusters, k);
    write_clustering_result(argv[3], clusters, k);

    printf(" => Runtime: %llu cycles\n", runtime);

    free(fully_connected);
    free(laplacian);
    free(w);
    free(f.points);
    return 0;
}
