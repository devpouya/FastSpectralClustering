#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <lapacke.h>
#include <inttypes.h>
#include <sys/time.h>

#include "tsc_x86.h"
#include "instrumentation.h"
#include "norms.h"
#include "construct_graph.h"
#include "kmeans.h"
#include "util.h"
#include "eigs.h"

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
        return 1;
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
    // int lda = n;

    // Construct the matrices and print them
    // fully-connected matrix
    //printf("Constructing fully connected matrix...\n");
    printf("Constructing the Laplacian ONE SHOT...\n");
    myInt64 start1 = start_tsc();
    //double *fully_connected = calloc(lines * lines , sizeof(double));
    //construct_fully_connected_matrix(points, lines, dim, fully_connected);

    printf("Constructing unnormalized Laplacian...\n");
    // compute unnormalized laplacian
    //double *laplacian = malloc(lines * lines * sizeof(double));
    //construct_unnormalized_laplacian(fully_connected, lines, laplacian);
    double *laplacian = calloc(lines*lines, sizeof(double));

    if (dim >=8){
        oneshot_unnormalized_laplacian(points,lines,dim,laplacian);
    }else{
        oneshot_unnormalized_laplacian_lowdim(points,lines,dim,laplacian);
    }
    //compute the eigendecomposition and take the first k eigenvectors.
    myInt64 end1 = stop_tsc(start1);
    printf("Performing eigenvalue decomposition...\n");
    // double *w = malloc(lines * sizeof(double));
    // lapack_int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, laplacian, lda, w);
    // /* Check for convergence */
    // if (info > 0) {
    //     fprintf(stderr, "ERROR: The algorithm failed to compute eigenvalues.\n");
    //     exit(1);
    // }
    myInt64 start2 = start_tsc();

    // printf("Eigenvalues:\n");
    // for (int i = 0; i < 5; i++) {
    //     printf("%lf, ", w[i]);
    // }
    // printf("\n\n");

    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < 5; j++) {
    //         printf("%lf, ", laplacian[i*n + j]);
    //     }
    //     printf("\n");
    // }
    double *eigenvalues = malloc(k * sizeof(double));
    double *eigenvectors = malloc(n * k * sizeof(double));
    smallest_eigenvalues(laplacian, n, k, eigenvalues, eigenvectors);

    // print_matrix("Eigenvectors (stored columnwise)", n, n, laplacian, lda);
    printf("%d, %d", lines, k);

    printf("Performing k-means base_clustering...\n");
    // U (8x2) is the datasets in points.txt for now => k = 2
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
    double timing_start = wtime();
    //kmeans(points, lines, dim, k, 100, 0.0001, clusters); // (for kmeans test purposes)

    if(k>=8){
        elkan_kmeans(eigenvectors, lines, k, 1000, 0.0001, clusters);
    }else{
        elkan_kmeans_lowdim(eigenvectors, lines, k, 1000, 0.0001, clusters);
    }
    double timing = wtime()-timing_start ;
    printf("Timing of kmeans: %f [sec] \n", timing);

    uint64_t runtime = stop_tsc(start2) + end1;

    //print_cluster_indices(clusters, k);

    printf(" => Runtime: %" PRIu64  " cycles; ops: %" PRIu64 " ops\n", runtime, NUM_FLOPS);

    //write result in output file
    write_clustering_result(argv[3], clusters, k);

    // free(fully_connected);
    // free(laplacian);
    // free(eigenvalues);
    // free(eigenvectors);
    // free(f.points);

    PROFILER_LIST();

    // LEAVE THESE PRINTS (for the performance checking script)
    printf("%" PRIu64 "\n", runtime);
    printf("%" PRIu64 "\n", NUM_FLOPS);
    printf("performance: %lf\n", (double)NUM_FLOPS/runtime);

#ifdef VALIDATION
    char *my_argv; // = {"./base_clustering" , argv[1] , argv[2] , "./base_output"};
    my_argv = concat("./base_clustering ", argv[1]);
    my_argv = concat(my_argv, " ");
    my_argv = concat(my_argv, argv[2]);
    my_argv = concat(my_argv, " ./base_output");
    system(my_argv);
    int line, col;
    FILE* fpt1 = fopen("./base_output", "r");
    FILE* fpt2 = fopen(argv[3], "r");
    if (compareFile(fpt1, fpt2, &line, &col) != 0){
        printf("ERROR! optimized version gives different result as base clustering\n");
    }else{
        printf("Result Correct!\n");
    }
#endif

    return 0;
}
