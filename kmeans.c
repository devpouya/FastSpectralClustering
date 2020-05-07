#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>

#include "norms.h"
#include "instrumentation.h"
#include "kmeans.h"
#include "init.h"

static void update_means(double *U, int *indices, int k, int n, int m, double *ret) {
    ENTER_FUNC;
    NUM_ADDS(n*k);
    NUM_DIVS(k*k);
    double *tmp_means = calloc(k * k, sizeof(double));
    int *sizes = calloc(k, sizeof(int));;
    for (int i = 0; i < n ; i++) { // iterate over each point
        for (int j = 0; j < k; j++) { // iterate over each indices
            tmp_means[indices[i]*k+j] += U[i*m+j];
        }
        sizes[indices[i]] += 1;
    }
    for (int i = 0; i < k ; i++) { // iterate over cluster
        for (int j = 0; j < k; j++) { // iterate over each sizes
            ret[i*k+j] = tmp_means[i*k+j] / sizes[i];
        }
    }
    EXIT_FUNC;
}

static int find_nearest_cluster_index(double *point, double *means, int k) {
    ENTER_FUNC;
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
    EXIT_FUNC;
    return index;
}

/*
 * K-Means Algorithm
 *   1. Choose the number of clusters(K) and obtain the data points: Done
 *   2. Place the centroids c_1, c_2, ..... c_k randomly in [min..max]: Done
 *   3. Repeat steps 4 and 5 until convergence or until the end of a fixed number of iterations
 *   4. for each data point x_i:
 *          - find the nearest centroid(c_1, c_2 .. c_k)
 *          - assign the point to that cluster
 *   5. for each cluster j = 1..k
 *          - new centroid = mean of all points assigned to that cluster
 */
void kmeans(double *U, int n, int m, int k, int max_iter, double stopping_error, struct cluster *ret) {
    ENTER_FUNC;
    // k is the number of columns in U matrix  U is a n by k matrix (here only!)
    int i = 0;
    // each row represents a cluster each column a dimension
    double means[k*k];
    int indices[n];
    while (i < max_iter) {
        if (i < 1) {
            init_kpp(&U[0], n, m, k, means);
        } else {
            update_means(U, indices, k, n, m, means);
        }
        // for each point find the nearest cluster
        for (int j = 0; j < n; j++) {
            indices[j] = find_nearest_cluster_index(&U[j * m], means, k); // find nearest mean for this point = line
        }
        i++;
    }
    // build the clusters into struct cluster
    int indices_tmp[n];
    for (int i = 0; i < k; i++) { // construct cluster one after another
        int cluster_size = 0; // keep tract of cluster size in # of points
        for (int j = 0; j < n; j++) {
            if (indices[j] == i) {
                indices_tmp[cluster_size] = j; // store index of U => j
                cluster_size++;
            }
        } // done with point j
        for (int j = 0; j < k; j++) {
            ret[i].mean[j] = means[i*k+j];
        }
        for (int j = 0; j < cluster_size; j++) {
            ret[i].indices[j] = indices_tmp[j];
        }
        ret[i].size = cluster_size;
    } // done with cluster i
    EXIT_FUNC;
}

void print_cluster_indices(struct cluster *clusters, int num_clusters){
    printf("Printing clustered point indices:\n");
    for (int j = 0; j < num_clusters; j++) {
        printf("Cluster %d: ", j);
        printf("( ");
        for(int e = 0; e < clusters[j].size; e++) {
            printf("%d ", clusters[j].indices[e]);
        }
        printf(")  ");
        printf("\n");
    }

    printf("CLUSTER SIZES\n");
    for(int i = 0; i < num_clusters; i++) {
        printf("Cluster %d has size: %d\n",i,clusters[i].size);
    }
}

int write_clustering_result(char *file, struct cluster *clusters, int num_clusters){
    FILE *fp;
    fp = fopen(file, "w");
    // write the number of cluster at the beginning of the output
    fprintf(fp, "%d\n", num_clusters);

    // write the sizes of each clusters in the second line
    for (int i = 0; i < num_clusters; i++){
        fprintf(fp, "%d ", clusters[i].size);
    }
    fprintf(fp, "\n");

    //write the indices of points in each cluster, mark the end of the cluster with a new line
    for (int i = 0; i < num_clusters; i++){
        for (int j = 0; j < clusters[i].size; j++){
            fprintf(fp, "%d ", clusters[i].indices[j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    return 0;
}