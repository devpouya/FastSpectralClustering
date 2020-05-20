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

int cmpfunc (const void * a, const void * b) {
    return ( *(int*)a - *(int*)b );
}

#ifdef DUMPEV
__attribute__((used))
static void read_ev_from_file(const char *data_path, int n, int k, double *ret_ev) {
    char path[128];
    int len = strlen(data_path);
    int q;
    for (q = 0; q < len-4; q++) {
        path[q] = data_path[q];
    }
    strcpy(path + q, "_ev.txt");
    // path[strlen(data_path) - 4] = '\0';
    printf("path is %s\n", path);
    FILE *fp = fopen(path, "r");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            fscanf(fp, "%lf ", &ret_ev[i*k + j]);
            printf("%lf\n", ret_ev[i*k + j]);
        }
    }
}

__attribute__((used))
static void dump_ev_to_file(const char *data_path, int n, int k, double *ev) {
    char path[128];
    int len = strlen(data_path);
    int q;
    for (q = 0; q < len-4; q++) {
        path[q] = data_path[q];
    }
    strcpy(path + q, "_ev.txt");
    printf("dumping to file: %s\n", path);
    FILE *fp = fopen(path, "w");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            fprintf(fp, "%.17f ", ev[i*k + j]);
        }
        fprintf(fp, "\n");
    }
}
#endif

#define NUM_RUNS 9
int main(int argc, char *argv[]) {

    if (argc != 4) {
        printf("usage: %s points_file num_clusters output_file\n", argv[0]);
        return 1;
    }

//    printf("loading dataset: %s\n", argv[1]);
//    printf("number of clusters: %d\n", atoi(argv[2]));
//    printf("output path: %s\n", argv[3]);

    struct file f = alloc_load_points_from_file(argv[1]);
    int dim = f.dimension;
    int lines = f.lines;
    double *points = f.points;
    int k = atoi(argv[2]);
    int n = lines;


    //printf("Constructing unnormalized Laplacian...\n");
    double *laplacian = calloc(lines*lines, sizeof(double));
    //myInt64 start1 = start_tsc();
    double start1 = wtime();

    if (dim >= 8){
        oneshot_unnormalized_laplacian_vec_blocked(points,lines,dim,laplacian);
    }else{
        oneshot_unnormalized_laplacian_lowdim_vec_blocked(points,lines,dim,laplacian);
    }
    //compute the eigendecomposition and take the first k eigenvectors.
    //myInt64 end1 = stop_tsc(start1);
    double end1 = wtime() - start1;
    //printf("Performing eigenvalue decomposition...\n");
    double *eigenvalues = malloc(k * sizeof(double));
    double *eigenvectors = malloc(n * k * sizeof(double));

    smallest_eigenvalues(laplacian, n, k, eigenvalues, eigenvectors);
#ifdef DUMPEV
    // read_ev_from_file(argv[1], n, k, eigenvectors);
    dump_ev_to_file(argv[1], n, k, eigenvectors);
#endif

    //myInt64 start2 = start_tsc();
    double start2 = wtime();
    // init datastructure
    struct cluster clusters[k];
    for (int i = 0; i < k; i++) {
        clusters[i].mean = malloc(k * sizeof(double)); // k is the "dimension" here
        clusters[i].size = 0;
        clusters[i].indices = malloc(lines * sizeof(int)); // at most
    }

    if(k>=8){
        hamerly_kmeans(eigenvectors, lines, k, 1000, 0.0001, clusters);
    }else{
        hamerly_kmeans_lowdim(eigenvectors, lines, k, 1000, 0.0001, clusters);
    }
    // double timing = wtime()-timing_start ;


    // uint64_t runtime = stop_tsc(start2) + end1;
    double end2 = wtime() - start2;
    //print_cluster_indices(clusters, k);

    //write result in output file
    write_clustering_result(argv[3], clusters, k);

    //PROFILER_LIST();

    free(eigenvectors);
    free(eigenvalues);
    free(laplacian);
    free(f.points);
    double timing = end1 + end2;
    // LEAVE THESE PRINTS (for the performance checking script)
    printf("%f\n", timing);
//    printf("%" PRIu64 "\n", runtime);
//    printf("%" PRIu64 "\n", NUM_FLOPS);


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
