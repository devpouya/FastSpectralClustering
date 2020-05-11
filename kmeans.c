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

#define MAX(x, y) ((x > y) ? x : y)

static void update_means(double *U, int *indices, int k, int n, double *ret) {
    ENTER_FUNC;
    NUM_ADDS(n*k);
    NUM_DIVS(k*k);
    double *tmp_means = calloc(k * k, sizeof(double));
    int *sizes = calloc(k, sizeof(int));;
    for (int i = 0; i < n ; i++) { // iterate over each point
        for (int j = 0; j < k; j++) { // iterate over each indices
            tmp_means[indices[i]*k+j] += U[i*k+j];
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
        double norm = l2_norm_squared(point, &means[i*k], k);
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
void lloyd_kmeans(double *U, int n, int k, int max_iter, double stopping_error, struct cluster *ret) {
    ENTER_FUNC;
    // k is the number of columns in U matrix  U is a n by k matrix (here only!)
    int i = 0;
    // each row represents a cluster each column a dimension
    double means[k*k];
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

/*
 * ALGO 2: INITIALIZE ---------------------------------------------------------
 * 1) init DS
 * 2) init kpp
 */
static void initialize(double *clusters_center, double *U, int *clusters_size, double *upper_bounds
        , double *lower_bounds, int *cluster_assignments, int k, int n) {
    ENTER_FUNC;
    clusters_size[0] = n; // first contains all
    for (int i = 1; i < k; i++) {
        clusters_size[i] = 0;
    }
    for (int i = 0; i < n; i++) {
        upper_bounds[i] = DBL_MAX;
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
static void point_all_clusters(double *U, double *clusters_center, int *cluster_assignments
        , double *upper_bounds, double *lower_bounds, int *clusters_size, int k, int i) {
    ENTER_FUNC;
    int closest_center_1 = 0;
    double closest_center_1_dist = DBL_MAX;
    double closest_center_2_dist = DBL_MAX;
    for (int j = 0; j < k; j++) {
        double dist = l2_norm(U + i * k, clusters_center + j * k, k);
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
static double move_centers(double *new_clusters_centers, int *clusters_size, double *clusters_center
        , double *centers_dist_moved, int k) {
    ENTER_FUNC;
    double dist_moved = 0;
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
static void update_bounds(double *upper_bounds, double *lower_bounds, double *centers_dist_moved
        , int *cluster_assignments, double max_dist_moved, int n) {
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
void hamerly_kmeans(double *U, int n, int k, int max_iter, double stopping_error, struct cluster *ret) {
    ENTER_FUNC;
    // initial centers
    double clusters_center[k*k];
    // tmp for next iteration
    double new_clusters_centers[k*k];
    // cluster sizes
    int clusters_size[k];
    // n upper bounds (of closest center)
    double upper_bounds[n];
    // n lower bounds (of 2nd strict closest center)
    double lower_bounds[n];
    // stores cluster index for all points
    int cluster_assignments[n];
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
        double max_dist_moved = move_centers(new_clusters_centers, clusters_size
                , clusters_center, centers_dist_moved, k);
        // ALGO 5 - Update-bounds : for all U update upper and lower distance bounds ---------------
        update_bounds(upper_bounds, lower_bounds, centers_dist_moved, cluster_assignments, max_dist_moved, n);
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

static void comp_distance_between_centers(double *means, int k, double *ret_dist_centers){
    ENTER_FUNC;
    for(int i = 0; i < k; i++){
        for(int j = 0; j <= i; j++){
            //symmetric matrix, only store in the lower matrix
            //TODO  better way to store (might be not worth it, seems computing the index result in worse execution)
            double dist = l2_norm(&means[i*k], &means[j*k], k);
            ret_dist_centers[i*k+j] = dist;
        }
    }
    EXIT_FUNC;
}

static double get_dist_centers(int i, int j, double *dist_centers, int k){
    if(i>j){
        return dist_centers[i*k+j];
    }else{
        return dist_centers[j*k+i];
    }
}

static void comp_array_s(double *dist_centers, int k, double *ret_s){
    for(int i =0; i < k; i++){
        double max = DBL_MAX;
        for(int j = 0; j < k; j++){
            if(j!=i){
                NUM_MULS(1);
                double half_dist = 0.5*dist_centers[i*k+j];
                if(half_dist < max){
                    max = half_dist;
                }
            }
        }
        ret_s[i] = max;
    }
}


static void init_elkan(double *U, int n, int k, double *means, double *lb, double *ub, int *indices){
    ENTER_FUNC;

    double dist_centers[k*k];
    //initialization
    for(int i = 0; i<k; i++){
        for(int j = 0; j<k; j++){
            dist_centers[i*k+j] = 0;
        }
    }
    //get the distances between means, store in dist_centers as lower triangle matrix
    comp_distance_between_centers(means, k, dist_centers);

    for (int i = 0; i < n; i++){
        //in initialization we always first 'assign' a point to the first cluster
        //then use lemma 1 in https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf
        //to avoid redundant computation
        int cur_cluster_assigned = 0;
        double dist_p_c = l2_norm(&U[i * k], &means[0*k], k);
        lb[0*k+0] = 0;
        ub[i] = dist_p_c;

        for(int j = 1; j < k; j++) { //traverse through the rest of clusters
            lb[i * k + j] = 0;
            NUM_MULS(1);
            if (0.5 * dist_centers[cur_cluster_assigned * k + j] < dist_p_c) {
                //first compute distance between current point and cluster j's mean
                double dist_p_c_j = l2_norm(&U[i * k], &means[j * k], k);
                lb[i * k + j] = dist_p_c_j;
                if (dist_p_c_j < dist_p_c) {
                    //found a closer cluster
                    cur_cluster_assigned = j;
                    dist_p_c = dist_p_c_j;
                    ub[i] = dist_p_c_j;
                }
            } else {
                // no need to update
            }
        }
        indices[i] = cur_cluster_assigned;
    }

    EXIT_FUNC;
}

/*
 * ELKAN
 */

void elkan_kmeans(double *U, int n, int k, int max_iter, double stopping_error, struct cluster *ret) {
    // printf("entering kmeans\n");
    ENTER_FUNC;
    // k is the number of columns in U matrix  U is a n by k matrix (here only!)

    // each row represents a cluster each column a dimension
    double *means = malloc(k*k*sizeof(double));
    double *new_means = malloc(k*k*sizeof(double));
    // each row represents the lower bound of the same point x and different centers (total k centers)
    double *lb = malloc(n*k*sizeof(double));
    double *ub = malloc(n*sizeof(double));
    int *indices = malloc(n*sizeof(int));
//    int *r = malloc(n*sizeof(int));
    int r;
    double *dist_centers = malloc(k*k*sizeof(double));
    double *s_dist_centers = malloc(k*sizeof(double));
    double delta[k];
//    for(int i = 0; i < n ; i++){r[i] = 1;}

    init_kpp(&U[0], n, k, means);
    int init_elkan_flag = 1;
    int step = 1;

    while (step < max_iter) {

        if(init_elkan_flag){
            init_elkan(&U[0], n, k, means, lb, ub, indices);
            init_elkan_flag = 0;
        }

        //step 1.
        comp_distance_between_centers(means, k, dist_centers);
        comp_array_s(dist_centers, k, s_dist_centers);

        //step 2 and 3
        for (int i = 0; i < n; i++){
            r = 1;
            double dist_p_c = 0;
            if(ub[i] <= s_dist_centers[indices[i]]){
                //do nothing?
            }else{
                for (int j = 0; j < k; j++){
                    NUM_MULS(1);
                    if((indices[i]!=j)&&
                       (ub[i] > lb[i*k+j])&&
                       (ub[i] > 0.5*dist_centers[indices[i]*k+j])){
                            if(r==1){
                                dist_p_c = l2_norm(&U[i * k], &means[indices[i]*k], k);
                                r = 0;
                            }else{
                                dist_p_c = ub[i];
                            }
                            NUM_MULS(1);
                            if((dist_p_c > lb[i*k+j]) || (dist_p_c > 0.5*get_dist_centers(indices[i], j, dist_centers, k))){
                                double dist_p_j = l2_norm(&U[i * k], &means[j*k], k);
                                if(dist_p_j < dist_p_c){
                                    indices[i] = j;
                                }
                            }
                    }
                }
            }
        }

        // step 4.
        update_means(U, indices, k, n, new_means);

        for(int i=0; i<k; i++){
            delta[i] = l2_norm(&means[i * k], &new_means[i * k], k);
        }

        //step 5.
        for(int i = 0; i<n; i++){
            for(int j = 0; j<k;j++) {
//                    todo double check this c, m(c)
                double temp = lb[i * k + j];
//                double norm_temp =l2_norm(&means[j * k], &new_means[j * k], k);
//                /*double norm_temp =*/ l2_norm((double[]){1.0, 1.0, 987, 234, 123, 34},(double[]){123,123,123,123,123,123} , 6);
                lb[i * k + j] = MAX(0, temp - delta[j]);
//                      lb[i*k+j] = 0;
            }
        }


        //step 6.
        NUM_ADDS(n);
        for(int i = 0; i < n; i++){
            //todo check this notation..
//            ub[i] = ub[i] + l2_norm(&new_means[indices[i]*k], &means[indices[i]*k], k);
            ub[i] = ub[i] + delta[indices[i]];
//            r[i] = 1;
        }

        //step 7.
        for(int i = 0; i < k*k; i++){
            means[i] = new_means[i];
        }
        step++;
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
