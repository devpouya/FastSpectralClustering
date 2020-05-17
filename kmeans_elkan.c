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

static double get_dist_centers(int i, int j, double *dist_centers, int k){
    if(i>j){
        return dist_centers[i*k+j];
    }else{
        return dist_centers[j*k+i];
    }
}

static void comp_array_s(double *dist_centers, int k, double *ret_s){
    for(int i =0; i < k; i++){
        double max = FLT_MAX;
        for(int j = 0; j < k; j++){
            if(j!=i){
                NUM_MULS(1);
                double half_dist = 0.5*dist_centers[i*k+j];
                NUM_ADDS(1);
                if(half_dist < max){
                    max = half_dist;
                }
            }
        }
        ret_s[i] = max;
    }
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
            NUM_ADDS(1);
            if (0.5 * dist_centers[cur_cluster_assigned * k + j] < dist_p_c) {
                //first compute distance between current point and cluster j's mean
                double dist_p_c_j = l2_norm(&U[i * k], &means[j * k], k);
                lb[i * k + j] = dist_p_c_j;
                NUM_ADDS(1);
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
            NUM_ADDS(1);
            if(ub[i] <= s_dist_centers[indices[i]]){
                //do nothing?
            }else{
                for (int j = 0; j < k; j++){
                    NUM_MULS(1);
                    NUM_ADDS(2);
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
                            NUM_ADDS(2);
                            if((dist_p_c > lb[i*k+j]) || (dist_p_c > 0.5*get_dist_centers(indices[i], j, dist_centers, k))){
                                double dist_p_j = l2_norm(&U[i * k], &means[j*k], k);
                                NUM_ADDS(1);
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
                NUM_ADDS(1);
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


/*
 *
 *
 * LOW DIM VERSION OF ELKAN
 *
 *
 */


static void comp_distance_between_centers_lowdim(double *means, int k, double *ret_dist_centers){
    ENTER_FUNC;
    for(int i = 0; i < k; i++){
        for(int j = 0; j <= i; j++){
            //symmetric matrix, only store in the lower matrix
            //TODO  better way to store (might be not worth it, seems computing the index result in worse execution)
            double dist = l2_norm_lowdim(&means[i*k], &means[j*k], k);
            ret_dist_centers[i*k+j] = dist;
        }
    }
    EXIT_FUNC;
}

static void init_elkan_lowdim(double *U, int n, int k, double *means, double *lb, double *ub, int *indices){
    ENTER_FUNC;

    double dist_centers[k*k];
    //initialization
    for(int i = 0; i<k; i++){
        for(int j = 0; j<k; j++){
            dist_centers[i*k+j] = 0;
        }
    }
    //get the distances between means, store in dist_centers as lower triangle matrix
    comp_distance_between_centers_lowdim(means, k, dist_centers);

    for (int i = 0; i < n; i++){
        //in initialization we always first 'assign' a point to the first cluster
        //then use lemma 1 in https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf
        //to avoid redundant computation
        int cur_cluster_assigned = 0;
        double dist_p_c = l2_norm_lowdim(&U[i * k], &means[0*k], k);
        lb[0*k+0] = 0;
        ub[i] = dist_p_c;

        for(int j = 1; j < k; j++) { //traverse through the rest of clusters
            lb[i * k + j] = 0;
            NUM_MULS(1);
            NUM_ADDS(1);
            if (0.5 * dist_centers[cur_cluster_assigned * k + j] < dist_p_c) {
                //first compute distance between current point and cluster j's mean
                double dist_p_c_j = l2_norm_lowdim(&U[i * k], &means[j * k], k);
                lb[i * k + j] = dist_p_c_j;
                NUM_ADDS(1);
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

void elkan_kmeans_lowdim(double *U, int n, int k, int max_iter, double stopping_error, struct cluster *ret) {
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
            init_elkan_lowdim(&U[0], n, k, means, lb, ub, indices);
            init_elkan_flag = 0;
        }

        //step 1.
        comp_distance_between_centers_lowdim(means, k, dist_centers);
        comp_array_s(dist_centers, k, s_dist_centers);

        //step 2 and 3
        for (int i = 0; i < n; i++){
            r = 1;
            double dist_p_c = 0;
            NUM_ADDS(1);
            if(ub[i] <= s_dist_centers[indices[i]]){
                //do nothing?
            }else{
                for (int j = 0; j < k; j++){
                    NUM_MULS(1);
                    NUM_ADDS(2);
                    if((indices[i]!=j)&&
                       (ub[i] > lb[i*k+j])&&
                       (ub[i] > 0.5*dist_centers[indices[i]*k+j])){
                        if(r==1){
                            dist_p_c = l2_norm_lowdim(&U[i * k], &means[indices[i]*k], k);
                            r = 0;
                        }else{
                            dist_p_c = ub[i];
                        }
                        NUM_MULS(1);
                        NUM_ADDS(2);
                        if((dist_p_c > lb[i*k+j]) || (dist_p_c > 0.5*get_dist_centers(indices[i], j, dist_centers, k))){
                            double dist_p_j = l2_norm_lowdim(&U[i * k], &means[j*k], k);
                            NUM_ADDS(1);
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
            delta[i] = l2_norm_lowdim(&means[i * k], &new_means[i * k], k);
        }

        //step 5.
        for(int i = 0; i<n; i++){
            for(int j = 0; j<k;j++) {
//                    todo double check this c, m(c)
                double temp = lb[i * k + j];
//                double norm_temp =l2_norm(&means[j * k], &new_means[j * k], k);
//                /*double norm_temp =*/ l2_norm((double[]){1.0, 1.0, 987, 234, 123, 34},(double[]){123,123,123,123,123,123} , 6);
                NUM_ADDS(1);
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

