#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <string.h>

#include "norms.h"
#include "instrumentation.h"
#include "kmeans.h"
#include "init.h"
#include "utils.h"

#define MAX(x, y) ((x > y) ? x : y)

static int find_nearest_cluster_index(float *point, float *means, int k) {
    ENTER_FUNC;
    // use l2_norm
    float gap = DBL_MAX;
    int index = 0;
    for (int i = 0; i < k; i++) { // for every cluster check abs distance to point and take the minimal
        float norm = l2_norm_squared(point, &means[i*k], k);
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
void lloyd_kmeans(float *U, int n, int k, int max_iter, float stopping_error, struct cluster *ret) {
    ENTER_FUNC;
    // k is the number of columns in U matrix  U is a n by k matrix (here only!)
    int i = 0;
    // each row represents a cluster each column a dimension
    float means[k*k];
    int indices[n];
    while (i < max_iter) {
        if (i < 1) {
            init_kpp(&U[0], n, k, means);
        } else {
            update_means(U, indices, k, n, means);
        }
        // for each point find the nearest cluster
        for (int j = 0; j < n; j++) {
            indices[j] = find_nearest_cluster_index(&U[j * k], means, k); // find nearest mean for this point = line
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


/*
 *
 *
 * LOW-DIM VERSION OF LLOYD
 *
 *
 */
static int find_nearest_cluster_index_lowdim(float *point, float *means, int k) {
    ENTER_FUNC;
    // use l2_norm
    float gap = DBL_MAX;
    int index = 0;
    for (int i = 0; i < k; i++) { // for every cluster check abs distance to point and take the minimal
        float norm = l2_norm_squared_lowdim(point, &means[i*k], k);
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
void lloyd_kmeans_lowdim(float *U, int n, int k, int max_iter, float stopping_error, struct cluster *ret) {
    ENTER_FUNC;
    // k is the number of columns in U matrix  U is a n by k matrix (here only!)
    int i = 0;
    // each row represents a cluster each column a dimension
    float means[k*k];
    int indices[n];
    while (i < max_iter) {
        if (i < 1) {
            init_kpp(&U[0], n, k, means);
        } else {
            update_means(U, indices, k, n, means);
        }
        // for each point find the nearest cluster
        for (int j = 0; j < n; j++) {
            indices[j] = find_nearest_cluster_index_lowdim(&U[j * k], means, k); // find nearest mean for this point = line
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
