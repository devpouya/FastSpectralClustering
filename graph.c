#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

// #include <mkl.h>
#include <lapacke.h>
#include "instrumentation.h"
#include "tsc_x86.h"
#include "helper.h"

// #include "mkl_lapacke.h"


#define EPS 2
#define DBL_MIN -100000
#define DBL_MAX 100000


static double *TheArray;
static int cmp(const void *a, const void *b){
    int ia = *(int *)a;
    int ib = *(int *)b;
    return (TheArray[ia] > TheArray[ib]) - (TheArray[ia] < TheArray[ib]);
}

static double l2_norm(double *u, double *v, int dim) {
    NUM_ADDS(3*dim);
    NUM_MULS(dim);
    NUM_SQRTS(1);
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
    NUM_ADDS(n*n);
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
            printf("%lf ", ret[i*num_points + j]);
        }
        printf("\n");
    }
}

static void construct_normalized_laplacian_rw_matrix(double *weighted_adj_matrix, int num_points, double *ret) {
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
            printf("%lf ", ret[i*num_points + j]);
        }
        printf("\n");
    }
}

static void construct_unnormalized_laplacian(double *graph, int n, double *ret) {
    // double* degrees = (double *)malloc(n * n * sizeof(double));
    double degrees[n];
    calculate_diagonal_degree_matrix(graph, n, degrees);
    NUM_ADDS(n*n);
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


/* Auxiliary routine: printing a matrix */
/* copied from intel lapack example: https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_cgeev_row.c.htm */
void print_matrix( char* desc, int m, int n, double* a, int lda ) {
    int i, j;
    printf( "\n %s\n", desc );
    for( i = 0; i < m; i++ ) {
        for( j = 0; j < n; j++ )
            printf( " (%6.2f)", a[i*lda+j]);
        printf( "\n" );
    }
}

/*---- K-Means util methods ---------------------------------------------- */


static void init_means(double *U, int n, int k, double *ret) {
    // find min/max bounds for each dimension
    // k is the number of columns
    double bounds[k][2];
    for (int i = 0; i < k; i++) {  // row represents dimension
        bounds[i][0] = DBL_MAX;
        bounds[i][1] = DBL_MIN;
    } // Right you need to set to opposite value !!
    for (int i = 0; i < n; i++) { // each line is a point
        for (int j = 0; j < k; j++) {
            bounds[j][0] = (U[i*n + j] < bounds[j][0]) ? U[i*n + j] : bounds[j][0];
            bounds[j][1] = (U[i*n + j] > bounds[j][1]) ? U[i*n + j] : bounds[j][1];
        }
    }
    srand(time(0));

    // generate k random means stores row-wise
    // ret is k by k
    for (int i = 0; i < k; i++) {
        printf("Center %d: ( ", i);
        for (int j = 0; j < k; j++) {

            ret[i*k + j] = ( ((double )rand() /RAND_MAX)*(bounds[j][1] - bounds[j][0])) + bounds[j][0];
            printf("%lf ", ret[i*k + j]);
        }
        printf(")\n");
    }
}

// mean of each column
// dimension is the column index along which the mean is computed
static double compute_mean_of_one_dimension(double *U, int *indices, int size, int n, int dimension) {
    double sum = 0;
    for (int i = 0; i < size; i++) { // for all points
        sum += U[indices[i]*n+dimension]; // .. select one dimension
    }
    return (size > 0) ? (sum/size) : 0;
}

static void update_means(double *U, struct cluster *clusters, int k, int n, double *ret) {
    for (int i = 0; i < k; i++) { // iterate over cluster i
       printf("Center %d: ( ", i);
        for (int j = 0; j < k; j++) { // j is the dimension here
            ret[i*k + j] = (clusters[j].size > 0) ?
                           compute_mean_of_one_dimension(U, clusters[i].indices, clusters[i].size, n, j) : clusters[i].mean[j];
           printf("%lf ", ret[i*k + j]);
        }
       printf(")\n");
    }
}

static int find_nearest_cluster_index(double *point, double *means, int k) {
    // use l2_norm
    double gap = DBL_MAX;
    int index = 0;
    for (int i = 0; i < k; i++) { // for every cluster check abs distance to point and take the minimal
        double norm = l2_norm(point, &means[i*k], k);
        if(norm < gap) {
            gap = norm;
            index = i;
        }
    }
    return index;
}

static void map_to_nearest_cluster(double *U, int n, int k, double *means, struct cluster *ret) {
    // potentially all points can be in the same cluster
    // find nearest cluster for each point = line
    int index_nn[n];
    for (int j = 0; j < n; j++) {
        index_nn[j] = find_nearest_cluster_index(&U[j * n], means, k); // find nearest mean for this point = line
    }
    for (int i = 0; i < k; i++) { // construct cluster one after another
        int indices[n];
        int cluster_size = 0; // keep tract of cluster size in # of points
        for (int j = 0; j < n; j++) {
            if (index_nn[j] == i) {
                indices[cluster_size] = j; // store index of U => j
                cluster_size++;
            }
        } // done with point j

        for (int j = 0; j < k; j++) {
            ret[i].mean[j] = means[i*k+j];
        }
        for (int j = 0; j < cluster_size; j++) {
            ret[i].indices[j] = indices[j];
        }
        ret[i].size = cluster_size;
    } // done with cluster i

}

static int early_stopping(double *means, struct cluster *clusters, double error, int k) {
    for (int i = 0; i < k; i++) { // iterate over cluster
        for (int j = 0; j < k; j++) { // iterate over each dimension of the mean
            if (fabs(means[i*k+j] - clusters[i].mean[j]) > error) {
                return 0;
            }
        }
    }
    return 1;
}


/*
 * K-Means Algorithm
 *
 *   1. Choose the number of clusters(K) and obtain the data points: Done
 *   2. Place the centroids c_1, c_2, ..... c_k randomly in [min..max]: Done
 *   3. Repeat steps 4 and 5 until convergence or until the end of a fixed number of iterations
 *   4. for each data point x_i:
 *          - find the nearest centroid(c_1, c_2 .. c_k)
 *          - assign the point to that cluster
 *   5. for each cluster j = 1..k
 *          - new centroid = mean of all points assigned to that cluster
 *   6. End
 *
 *   TODO: dynamically allocate and change of size of cluster.points
 *
 */
static void K_means(double *U, int n, int k, int max_iter, double stopping_error, struct cluster *ret) {
    // k is the number of columns in U matrix  U is a n by k matrix (here only!)
    int i = 0;
    // each row represents a cluster each column a dimension
    double means[k*k];
    while (i < max_iter) {
        (i == 0) ? init_means(&U[0], n, k, means) : update_means(U, ret, k, n, means);
        // check if the means are stable, if yes => stop
        if (i > 0) {
            if (early_stopping(means, ret, stopping_error, k)) {
                break;
            }
        }
        // post condition: means is up-to-date
        map_to_nearest_cluster(U, n, k, means, ret);
        i++;
    }
    // print clusters: Cluster i : (1,2) (4,5) etc.
    for (int j = 0; j < k; j++) {
        printf("Cluster %d: ", j);
        for(int e = 0; e < ret[j].size; e++) {
            printf("( ");
            for (int f = 0; f < n; f++) {
                printf("%lf ", U[ret[j].indices[e]*n+f]);
            }
            printf(")  ");
        }
        printf("\n");
    }
}

/*------------------------------------------------------------------------ */


// static struct cluster SpectralClustering() {

// }

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

    printf("loading dataset: %s\n", argv[1]);
    printf("number of clusters: %d\n", atoi(argv[2]));
    printf("output path: %s\n", argv[3]);

    struct file f = load_file(argv[1]);
    int dim = f.dimension;
    int lines = f.lines;
    double *points = f.points;

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
    int k = atoi(argv[2]);
    int knn_graph[lines][lines];
    construct_knn_matrix((double *) points, lines, dim, k,(int *) knn_graph);

    printf("\nUnnormalized Laplacian:\n");
    // compute unnormalized laplacian
    double laplacian[lines][lines];
    myInt64 start = start_tsc();
    construct_unnormalized_laplacian((double *) fully_connected, lines, (double *) laplacian);

    //compute the eigendecomposition and take the first k eigenvectors.
    int n = lines, lda = lines, ldb = lines, ldvl = lines, ldvr = lines, info;
    /* Local arrays */
    double wr[lines], wi[lines], vl[lines*lines], vr[lines*lines];
    double w[lines];
    //info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', n, (double *) laplacian, lda, wr, wi, vl, ldvl, vr, ldvr);
    info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, (double *) laplacian, lda, w);
    /* Check for convergence */
    if( info > 0 ) {
        printf( "The algorithm failed to compute eigenvalues.\n" );
        exit( 1 );
    }

    printf("Eigenvalues:\n");
    for (int i = 0; i < n; i++) {
        //printf("(%lf, %lf) ", wr[i], wi[i]);
        printf("%lf, ", w[i]);
    }
    printf("\n");
    /* Print right eigenvectors */

    //print_matrix( "Right eigenvectors", n, n, vr, ldvr );
    print_matrix( "Eigenvectors (stored columnwise)", n, n, (double *) laplacian, lda );

    printf("\nRW Normalized Laplacian\n");
    // compute normalized rw laplacian
    double l_rw[lines][lines];
    construct_normalized_laplacian_rw_matrix((double *) fully_connected, lines, (double *) l_rw);

    printf("\nSymmetric Normalized Laplacian\n");
    // compute normalized rw laplacian
    double l_sym[lines][lines];
    construct_normalized_laplacian_sym_matrix((double *) fully_connected, lines, (double *) l_sym);

    printf("\nK-means Clustering\n");
    // U (8x2) is the data in points.txt for now => k = 2
    // number of cluster <=> # columns of U

    // init datastructure
    struct cluster clusters[k];
    for (int i = 0; i < k; i++) {
        clusters[i].mean = (double *) malloc(k * sizeof(double)); // k is the "dimension" here
        clusters[i].size = 0;
        clusters[i].indices = (int *) malloc(lines * sizeof(int)); // at most
    }
    // try with different max_iter
    // K_means((double *) points, lines, k, 10, clusters);

    K_means((double *) laplacian, lines, k, 100, 0.0001, clusters);
    myInt64 runtime = stop_tsc(start);

    print_cluster_indices(clusters, k);
    write_clustering_result(argv[3], clusters, k);

    return 0;
}
