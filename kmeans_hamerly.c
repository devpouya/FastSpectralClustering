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
static inline void initialize(double *clusters_center, double *U, int *clusters_size, double *upper_bounds
        , double *lower_bounds, int *cluster_assignments, int k, int n) {
    ENTER_FUNC;
    clusters_size[0] = n; // first contains all
    /*
    for (int i = 1; i < k; i++) {
        clusters_size[i] = 0;
    }
    */
    for (int i = 0; i < n; i++) {
        upper_bounds[i] = DBL_MAX;
        //lower_bounds[i] = 0;
        //cluster_assignments[i] = 0;
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
static inline void point_all_clusters(double *U, double *clusters_center, int *cluster_assignments
        , double *upper_bounds, double *lower_bounds, int *clusters_size, int k, int i) {
    ENTER_FUNC;
    int closest_center_1 = 0;
    double closest_center_1_dist = DBL_MAX;
    double closest_center_2_dist = DBL_MAX;
    //inline later? maybe
    for (int j = 0; j < k; j++) {
        double dist = l2_norm(U + i * k, clusters_center + j * k, k);
        // Find distance between the point and the center.
        NUM_ADDS(1);
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
static inline void move_centers(double *new_clusters_centers, int *clusters_size, double *clusters_center
        , double *centers_dist_moved, int k) {
    ENTER_FUNC;
    //double dist_moved = 0;
    //double dist_moved2 = 0;
    for (int j = 0; j < k; j++) {
        double dist = 0;
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
//        NUM_ADDS(1);
        /*
        if (dist > dist_moved) {
            dist_moved2 = dist_moved;
            dist_moved = dist;
        }
         */
    }
    EXIT_FUNC;
    //return dist_moved;
}

/*
 * ALGO 5 - UPDATE BOUNDS ---------------------------------------------------------
 * 1) update the new bounds
 */
static inline void update_bounds(double *upper_bounds, double *lower_bounds, double *centers_dist_moved
        , int *cluster_assignments, int n, int k) {
    ENTER_FUNC;
    double max_moved = 0;
    double second_max_moved = 0;
    for (int i = 0; i < k; i++) {
        if (centers_dist_moved[i] > max_moved) {
            second_max_moved = max_moved;
            max_moved = centers_dist_moved[i];
        }
    }
    for (int i = 0; i < n; i++) {
        NUM_ADDS(2);
        upper_bounds[i] += centers_dist_moved[cluster_assignments[i]];
        if (max_moved == centers_dist_moved[cluster_assignments[i]]){
            lower_bounds[i] -= second_max_moved;
        } else {
            lower_bounds[i] -= max_moved;
        }
    }
    EXIT_FUNC;
}

/*
 * ALGO 1 - K-Means Algorithm Hamerly --------------------------------------------
 * Implementation of the following algorithms as presented in the paper:
 *      https://epubs.siam.org/doi/pdf/10.1137/1.9781611972801.12
 */
void hamerly_kmeans(double *U, int n, int k, int max_iter, double stopping_error, struct cluster *ret) {
    ENTER_FUNC;
    // initial centers
    double clusters_center[k*k];
    // tmp for next iteration
    double new_clusters_centers[k*k];
    // cluster sizes
    //int clusters_size[k];
    int *clusters_size = calloc(k, sizeof(int));
    // n upper bounds (of closest center)
    //double upper_bounds[n];
    // n lower bounds (of 2nd strict closest center)
    //double lower_bounds[n];
    double *lower_bounds = calloc(n, sizeof(double));
    double *upper_bounds = calloc(n, sizeof(double));
    // stores cluster index for all points
    //int cluster_assignments[n];
    int *cluster_assignments = calloc(n, sizeof(int));
    // Algorithm 2: init + kpp -------------------
    initialize(clusters_center, U, clusters_size, upper_bounds, lower_bounds, cluster_assignments, k, n);
    // Distance to nearest other cluster for each cluster.
    double dist_nearest_cluster[k];
    // distance of centers moved between two iteration
    double centers_dist_moved[k];
    int iteration = 0;
    while (iteration < max_iter) {
        // Initialization after each iteration
        for (int i = 0; i < k*k; i++) {
            new_clusters_centers[i] = 0;
        }
        // min distance between each two centers {update s} --------------------------
        for (int i = 0; i < k; i++) { // for each cluster
            double min_dist = DBL_MAX;
            for (int j = 0; j < k; j++) { // look at the distances to all cluster
                if (i != j) { // is 0
                    double dist = 0;
                    for (int l = 0; l < k; l++) { // iterate over column = dimension
                        NUM_MULS(1);
                        NUM_ADDS(3);
                        dist += (clusters_center[i*k+l] - clusters_center[j*k+l])
                                *(clusters_center[i*k+l] - clusters_center[j*k+l]);
                    }
                    NUM_MULS(1);
                    NUM_SQRTS(1);
                    NUM_ADDS(1);
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
            double max_d = fmax(lower_bounds[i], dist_nearest_cluster[cluster_assignments[i]]);
            // ALGO 1: line7: {first bound test}
            NUM_ADDS(1);
            if (upper_bounds[i] > max_d) {
                upper_bounds[i] = l2_norm(U + i * k, clusters_center + cluster_assignments[i] * k, k);
                // ALGO 1: line 9 {second bound test}
                NUM_ADDS(1);
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
        /*
        double max_dist_moved = move_centers(new_clusters_centers, clusters_size
                , clusters_center, centers_dist_moved, k);
        */
        move_centers(new_clusters_centers, clusters_size
                , clusters_center, centers_dist_moved, k);

        // ALGO 5 - Update-bounds : for all U update upper and lower distance bounds ---------------
        update_bounds(upper_bounds, lower_bounds, centers_dist_moved, cluster_assignments, n, k);
        // transfer new state to current
        memcpy(clusters_center, new_clusters_centers, k * k * sizeof(double));
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
static inline void point_all_clusters_lowdim(double *U, double *clusters_center, int *cluster_assignments
        , double *upper_bounds, double *lower_bounds, int *clusters_size, int k, int i) {
    ENTER_FUNC;
    int closest_center_1 = 0;
    double closest_center_1_dist = DBL_MAX;
    double closest_center_2_dist = DBL_MAX;
    //inline later? maybe
    for (int j = 0; j < k; j++) {
        double dist = l2_norm_lowdim(U + i * k, clusters_center + j * k, k);
        // Find distance between the point and the center.
        NUM_ADDS(1);
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

static inline void move_centers_lowdim(double *new_clusters_centers, int *clusters_size, double *clusters_center
        , double *centers_dist_moved, int k) {
    ENTER_FUNC;
    //double dist_moved = 0;
    //double dist_moved2 = 0;
    for (int j = 0; j < k; j++) {
        double dist = 0;
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
//        NUM_ADDS(1);
        /*
        if (dist > dist_moved) {
            dist_moved2 = dist_moved;
            dist_moved = dist;
        }
         */
    }
    EXIT_FUNC;
    //return dist_moved;
}


/*
 * ALGO 1 - K-Means Algorithm Hamerly --------------------------------------------
 * Implementation of the following algorithms as presented in the paper:
 *      https://epubs.siam.org/doi/pdf/10.1137/1.9781611972801.12
 */
void hamerly_kmeans_lowdim(double *U, int n, int k, int max_iter, double stopping_error, struct cluster *ret) {
    ENTER_FUNC;
    // initial centers
    double clusters_center[k*k];
    // tmp for next iteration
    double new_clusters_centers[k*k];
    // cluster sizes
    //int clusters_size[k];
    int *clusters_size = calloc(k, sizeof(int));
    // n upper bounds (of closest center)
    //double upper_bounds[n];
    // n lower bounds (of 2nd strict closest center)
    //double lower_bounds[n];
    double *lower_bounds = calloc(n, sizeof(double));
    double *upper_bounds = calloc(n, sizeof(double));
    // stores cluster index for all points
    //int cluster_assignments[n];
    int *cluster_assignments = calloc(n, sizeof(int));
    // Algorithm 2: init + kpp -------------------
    initialize(clusters_center, U, clusters_size, upper_bounds, lower_bounds, cluster_assignments, k, n);
    // Distance to nearest other cluster for each cluster.
    double dist_nearest_cluster[k];
    // distance of centers moved between two iteration
    double centers_dist_moved[k];
    int iteration = 0;
    while (iteration < max_iter) {
        // Initialization after each iteration
        for (int i = 0; i < k*k; i++) {
            new_clusters_centers[i] = 0;
        }
        // min distance between each two centers {update s} --------------------------
        for (int i = 0; i < k; i++) { // for each cluster
            double min_dist = DBL_MAX;
            for (int j = 0; j < k; j++) { // look at the distances to all cluster
                if (i != j) { // is 0
                    double dist = 0;
                    for (int l = 0; l < k; l++) { // iterate over column = dimension
                        NUM_MULS(1);
                        NUM_ADDS(3);
                        dist += (clusters_center[i*k+l] - clusters_center[j*k+l])
                                *(clusters_center[i*k+l] - clusters_center[j*k+l]);
                    }
                    NUM_MULS(1);
                    NUM_SQRTS(1);
                    NUM_ADDS(1);
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
            double max_d = fmax(lower_bounds[i], dist_nearest_cluster[cluster_assignments[i]]);
            // ALGO 1: line7: {first bound test}
            NUM_ADDS(1);
            if (upper_bounds[i] > max_d) {
                upper_bounds[i] = l2_norm_lowdim(U + i * k, clusters_center + cluster_assignments[i] * k, k);
                // ALGO 1: line 9 {second bound test}
                NUM_ADDS(1);
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
        /*
        double max_dist_moved = move_centers(new_clusters_centers, clusters_size
                , clusters_center, centers_dist_moved, k);
        */
        move_centers(new_clusters_centers, clusters_size
                , clusters_center, centers_dist_moved, k);

        // ALGO 5 - Update-bounds : for all U update upper and lower distance bounds ---------------
        update_bounds(upper_bounds, lower_bounds, centers_dist_moved, cluster_assignments, n, k);
        // transfer new state to current
        memcpy(clusters_center, new_clusters_centers, k * k * sizeof(double));
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

