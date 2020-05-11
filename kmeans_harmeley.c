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
#include "util.h"

#define MAX(x, y) ((x > y) ? x : y)

/*
 * ALGO 2: INITIALIZE ---------------------------------------------------------
 * 1) init DS
 * 2) init kpp
 */
static void initialize(float *clusters_center, float *U, int *clusters_size, float *upper_bounds
        , float *lower_bounds, int *cluster_assignments, int k, int n) {
    ENTER_FUNC;
    clusters_size[0] = n; // first contains all
    for (int i = 1; i < k; i++) {
        clusters_size[i] = 0;
    }
    for (int i = 0; i < n; i++) {
        upper_bounds[i] = FLT_MAX;
        lower_bounds[i] = 0;
        cluster_assignments[i] = 0;
    }
    // perform kpp for init assignments
    init_kpp(U, n, k, clusters_center);
    EXIT_FUNC;
}

/*
 * ALGO 3 - POINT ALL CLUSTER --------------------------------------------------
 * executed on i's iter:
 * 1) find the two closest centers,
 * 2) update the bounds if closest changed, the assignments and the cluster sizes
 */
static void point_all_clusters(float *U, float *clusters_center, int *cluster_assignments
        , float *upper_bounds, float *lower_bounds, int *clusters_size, int k, int i) {
    ENTER_FUNC;
    int closest_center_1 = 0;
    float closest_center_1_dist = FLT_MAX;
    float closest_center_2_dist = FLT_MAX;
    for (int j = 0; j < k; j++) {
        float dist = l2_norm(U + i * k, clusters_center + j * k, k);
        // Find distance between the point and the center.
        if (dist < closest_center_1_dist) {
            closest_center_2_dist = closest_center_1_dist;
            closest_center_1 = j;
            closest_center_1_dist = dist;
        } else if (dist < closest_center_2_dist) {
            closest_center_2_dist = dist;
        }
    }
    // if the closest center changed : ALGO 1 line 12 UPDATE
    if (closest_center_1 != cluster_assignments[i]) {
        // update params
        clusters_size[cluster_assignments[i]] -= 1;
        clusters_size[closest_center_1] += 1;
        upper_bounds[i] = closest_center_1_dist;
        cluster_assignments[i] = closest_center_1;
    }
    // as defined lower bound of 2nd closest
    lower_bounds[i] = closest_center_2_dist;
    EXIT_FUNC;
}

/*
 * ALGO 4 - MOVE CENTERS ---------------------------------------------------------
 * 1) compute the distance moved
 * 2) reassign new centers
 * return maximal dist moved;
 */
static float move_centers(float *new_clusters_centers, int *clusters_size, float *clusters_center
        , float *centers_dist_moved, int k) {
    ENTER_FUNC;
    float dist_moved = 0;
    for (int j = 0; j < k; j++) {
        float dist = 0;
        if (clusters_size[j] > 0) {
            for (int l = 0; l < k; l++) { // update
                if (new_clusters_centers[j*k + l] == clusters_size[j]) {
                    NUM_DIVS(1);
                    new_clusters_centers[j*k + l] = new_clusters_centers[j*k + l] / clusters_size[j];
                } else { // don't update
                    new_clusters_centers[j*k + l] = clusters_center[j*k + l];
                }
            }
            dist = l2_norm(clusters_center + j*k, new_clusters_centers + j*k, k);
        }
        centers_dist_moved[j] = dist;
        if (dist > dist_moved) {
            dist_moved = dist;
        }
    }
    EXIT_FUNC;
    return dist_moved;
}

/*
 * ALGO 5 - UPDATE BOUNDS ---------------------------------------------------------
 * 1) update the new bounds
 */
static void update_bounds(float *upper_bounds, float *lower_bounds, float *centers_dist_moved
        , int *cluster_assignments, float max_dist_moved, int n) {
    ENTER_FUNC;
    for (int i = 0; i < n; i++) {
        NUM_ADDS(2);
        upper_bounds[i] += centers_dist_moved[cluster_assignments[i]];
        lower_bounds[i] -= max_dist_moved;
    }
    EXIT_FUNC;
}

/*
 * ALGO 1 - K-Means Algorithm Hamerly --------------------------------------------
 * Implementation of the following algorithms as presented in the paper:
 *      https://epubs.siam.org/doi/pdf/10.1137/1.9781611972801.12
 */
void hamerly_kmeans(float *U, int n, int k, int max_iter, float stopping_error, struct cluster *ret) {
    ENTER_FUNC;
    // initial centers
    float clusters_center[k*k];
    // tmp for next iteration
    float new_clusters_centers[k*k];
    // cluster sizes
    int clusters_size[k];
    // n upper bounds (of closest center)
    float upper_bounds[n];
    // n lower bounds (of 2nd strict closest center)
    float lower_bounds[n];
    // stores cluster index for all points
    int cluster_assignments[n];
    // Algorithm 2: init + kpp -------------------
    initialize(clusters_center, U, clusters_size, upper_bounds, lower_bounds, cluster_assignments, k, n);
    // Distance to nearest other cluster for each cluster.
    float dist_nearest_cluster[k];
    // distance of centers moved between two iteration
    float centers_dist_moved[k];
    int iteration = 0;
    while (iteration < max_iter) {
        // Initialization after each iteration
        for (int i = 0; i < k*k; i++) {
            new_clusters_centers[i] = 0;
        }
        // min distance between each two centers {update s} --------------------------
        for (int i = 0; i < k; i++) { // for each cluster
            float min_dist = FLT_MAX;
            for (int j = 0; j < k; j++) { // look at the distances to all cluster
                if (i != j) { // is 0
                    float dist = 0;
                    for (int l = 0; l < k; l++) { // iterate over column = dimension
                        NUM_MULS(1);
                        NUM_ADDS(3);
                        dist += (clusters_center[i*k+l] - clusters_center[j*k+l])
                                *(clusters_center[i*k+l] - clusters_center[j*k+l]);
                    }
                    NUM_MULS(1);
                    NUM_SQRTS(1);
                    dist = sqrt(dist) * 0.5;
                    if (dist < min_dist) {
                        min_dist = dist; 
                        dist_nearest_cluster[i] = dist;
                    }
                }
            }
        }
        // ALGO 1: line 5
        for (int i = 0; i < n; i++) {
            // line 6: max_d = max(s(a(i))/2, l(i)) ???
            float max_d = fmax(lower_bounds[i], dist_nearest_cluster[cluster_assignments[i]]);
            // ALGO 1: line7: {first bound test}
            if (upper_bounds[i] > max_d) {
                upper_bounds[i] = l2_norm(U + i * k, clusters_center + cluster_assignments[i] * k, k);
                // ALGO 1: line 9 {second bound test}
                if (upper_bounds[i] > max_d) {
                    // Iterate over all centers and find first and second closest distances and update DS
                    point_all_clusters(U, clusters_center, cluster_assignments, upper_bounds, lower_bounds
                            , clusters_size, k, i);
                }
            }
        }
        // To compute new mean: size calculated in point all clusters, sum now, divide in move!
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                NUM_ADDS(1);
                new_clusters_centers[cluster_assignments[i]*k+j] += U[i*k+j];
            }
        }
        // ALGO 4 - MOVE-CENTERS: check for distance moved then move the centers ---------
        float max_dist_moved = move_centers(new_clusters_centers, clusters_size
                , clusters_center, centers_dist_moved, k);
        // ALGO 5 - Update-bounds : for all U update upper and lower distance bounds ---------------
        update_bounds(upper_bounds, lower_bounds, centers_dist_moved, cluster_assignments, max_dist_moved, n);
        // transfer new state to current
        memcpy(clusters_center, new_clusters_centers, k * k * sizeof(float));
        iteration++;
    }
    // write into convenient data-structure struct cluster
    int indices_tmp[n];
    for (int i = 0; i < k; i++) { // construct cluster one after another
        int cluster_size = 0; // keep tract of cluster size in # of U
        for (int j = 0; j < n; j++) {
            if (cluster_assignments[j] == i) {
                indices_tmp[cluster_size] = j; // store index of U => j
                cluster_size++;
            }
        } // done with point j
        for (int j = 0; j < k; j++) {
            ret[i].mean[j] = clusters_center[i*k+j];
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
 * LOW DIM VERSION OF HARMELEY
 *
 *
 */





/*
 * ALGO 3 - POINT ALL CLUSTER --------------------------------------------------
 * executed on i's iter:
 * 1) find the two closest centers,
 * 2) update the bounds if closest changed, the assignments and the cluster sizes
 */
static void point_all_clusters_lowdim(float *U, float *clusters_center, int *cluster_assignments
        , float *upper_bounds, float *lower_bounds, int *clusters_size, int k, int i) {
    ENTER_FUNC;
    int closest_center_1 = 0;
    float closest_center_1_dist = FLT_MAX;
    float closest_center_2_dist = FLT_MAX;
    for (int j = 0; j < k; j++) {
        float dist = l2_norm_lowdim(U + i * k, clusters_center + j * k, k);
        // Find distance between the point and the center.
        if (dist < closest_center_1_dist) {
            closest_center_2_dist = closest_center_1_dist;
            closest_center_1 = j;
            closest_center_1_dist = dist;
        } else if (dist < closest_center_2_dist) {
            closest_center_2_dist = dist;
        }
    }
    // if the closest center changed : ALGO 1 line 12 UPDATE
    if (closest_center_1 != cluster_assignments[i]) {
        // update params
        clusters_size[cluster_assignments[i]] -= 1;
        clusters_size[closest_center_1] += 1;
        upper_bounds[i] = closest_center_1_dist;
        cluster_assignments[i] = closest_center_1;
    }
    // as defined lower bound of 2nd closest
    lower_bounds[i] = closest_center_2_dist;
    EXIT_FUNC;
}

/*
 * ALGO 4 - MOVE CENTERS ---------------------------------------------------------
 * 1) compute the distance moved
 * 2) reassign new centers
 * return maximal dist moved;
 */
static float move_centers_lowdim(float *new_clusters_centers, int *clusters_size, float *clusters_center
        , float *centers_dist_moved, int k) {
    ENTER_FUNC;
    float dist_moved = 0;
    for (int j = 0; j < k; j++) {
        float dist = 0;
        if (clusters_size[j] > 0) {
            for (int l = 0; l < k; l++) { // update
                if (new_clusters_centers[j*k + l] == clusters_size[j]) {
                    NUM_DIVS(1);
                    new_clusters_centers[j*k + l] = new_clusters_centers[j*k + l] / clusters_size[j];
                } else { // don't update
                    new_clusters_centers[j*k + l] = clusters_center[j*k + l];
                }
            }
            dist = l2_norm_lowdim(clusters_center + j*k, new_clusters_centers + j*k, k);
        }
        centers_dist_moved[j] = dist;
        if (dist > dist_moved) {
            dist_moved = dist;
        }
    }
    EXIT_FUNC;
    return dist_moved;
}


/*
 * ALGO 1 - K-Means Algorithm Hamerly --------------------------------------------
 * Implementation of the following algorithms as presented in the paper:
 *      https://epubs.siam.org/doi/pdf/10.1137/1.9781611972801.12
 */
void hamerly_kmeans_lowdim(float *U, int n, int k, int max_iter, float stopping_error, struct cluster *ret) {
    ENTER_FUNC;
    // initial centers
    float clusters_center[k*k];
    // tmp for next iteration
    float new_clusters_centers[k*k];
    // cluster sizes
    int clusters_size[k];
    // n upper bounds (of closest center)
    float upper_bounds[n];
    // n lower bounds (of 2nd strict closest center)
    float lower_bounds[n];
    // stores cluster index for all points
    int cluster_assignments[n];
    // Algorithm 2: init + kpp -------------------
    initialize(clusters_center, U, clusters_size, upper_bounds, lower_bounds, cluster_assignments, k, n);
    // Distance to nearest other cluster for each cluster.
    float dist_nearest_cluster[k];
    // distance of centers moved between two iteration
    float centers_dist_moved[k];
    int iteration = 0;
    while (iteration < max_iter) {
        // Initialization after each iteration
        for (int i = 0; i < k*k; i++) {
            new_clusters_centers[i] = 0;
        }
        // min distance between each two centers {update s} --------------------------
        for (int i = 0; i < k; i++) { // for each cluster
            float min_dist = FLT_MAX;
            for (int j = 0; j < k; j++) { // look at the distances to all cluster
                if (i != j) { // is 0
                    float dist = 0;
                    for (int l = 0; l < k; l++) { // iterate over column = dimension
                        NUM_MULS(1);
                        NUM_ADDS(3);
                        dist += (clusters_center[i*k+l] - clusters_center[j*k+l])
                                *(clusters_center[i*k+l] - clusters_center[j*k+l]);
                    }
                    NUM_MULS(1);
                    NUM_SQRTS(1);
                    dist = sqrt(dist) * 0.5;
                    if (dist < min_dist) {
                        min_dist = dist;
                        dist_nearest_cluster[i] = dist;
                    }
                }
            }
        }
        // ALGO 1: line 5
        for (int i = 0; i < n; i++) {
            // line 6: max_d = max(s(a(i))/2, l(i)) ???
            float max_d = fmax(lower_bounds[i], dist_nearest_cluster[cluster_assignments[i]]);
            // ALGO 1: line7: {first bound test}
            if (upper_bounds[i] > max_d) {
                upper_bounds[i] = l2_norm_lowdim(U + i * k, clusters_center + cluster_assignments[i] * k, k);
                // ALGO 1: line 9 {second bound test}
                if (upper_bounds[i] > max_d) {
                    // Iterate over all centers and find first and second closest distances and update DS
                    point_all_clusters_lowdim(U, clusters_center, cluster_assignments, upper_bounds, lower_bounds
                            , clusters_size, k, i);
                }
            }
        }
        // To compute new mean: size calculated in point all clusters, sum now, divide in move!
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                NUM_ADDS(1);
                new_clusters_centers[cluster_assignments[i]*k+j] += U[i*k+j];
            }
        }
        // ALGO 4 - MOVE-CENTERS: check for distance moved then move the centers ---------
        float max_dist_moved = move_centers_lowdim(new_clusters_centers, clusters_size
                , clusters_center, centers_dist_moved, k);
        // ALGO 5 - Update-bounds : for all U update upper and lower distance bounds ---------------
        update_bounds(upper_bounds, lower_bounds, centers_dist_moved, cluster_assignments, max_dist_moved, n);
        // transfer new state to current
        memcpy(clusters_center, new_clusters_centers, k * k * sizeof(float));
        iteration++;
    }
    // write into convenient data-structure struct cluster
    int indices_tmp[n];
    for (int i = 0; i < k; i++) { // construct cluster one after another
        int cluster_size = 0; // keep tract of cluster size in # of U
        for (int j = 0; j < n; j++) {
            if (cluster_assignments[j] == i) {
                indices_tmp[cluster_size] = j; // store index of U => j
                cluster_size++;
            }
        } // done with point j
        for (int j = 0; j < k; j++) {
            ret[i].mean[j] = clusters_center[i*k+j];
        }
        for (int j = 0; j < cluster_size; j++) {
            ret[i].indices[j] = indices_tmp[j];
        }
        ret[i].size = cluster_size;
    } // done with cluster i
    EXIT_FUNC;
}

