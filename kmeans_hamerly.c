#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <string.h>

#include "norms.h"
#include "instrumentation.h"
#include "kmeans.h"
//#include "init.h"
#include "util.h"
#include <immintrin.h>

#define MAKE_MASK(i0, i1, i2, i3) (i3 << 3 | i2 << 2 | i1 << 1 | i0)
#define MAX(x, y) ((x > y) ? x : y)

static inline void cumulative_sum(double *probs, int n, double *ret) {
    ENTER_FUNC;
    NUM_ADDS(n);
    ret[0] = probs[0];
    double ret_tmp = ret[0];
    for(int i = 1; i < n; i++) {
        ret[i] += ret_tmp;//+probs[i];
        ret_tmp = ret[i];
    }
    EXIT_FUNC;
}

static inline void init_kpp(double *U, int n, int k, double *ret) {
    ENTER_FUNC;
    // add a random initial point to the centers
#ifdef SEED
    srand(SEED);
#else
    srand(time(0));
#endif
    int ind = ((int)rand()%n);
//    printf("ind = %d\n", ind);
    //ret[0] = U[((int)rand()%n)*k];
    for(int j = 0; j < k; j++) {
        ret[j] = U[ind*k+j];
    }
    double sum = 0;
    for (int c = 1; c < k; c++) {

        sum = 0;
        double dists[n];
        for (int i = 0; i < n; i++) {
            //find closest point and add to sum
            double dist = DBL_MAX;
            for(int j = 0; j < c; j++) {
                double tmp = l2_norm(&U[i*k],&ret[j*k],k);
                if (tmp < dist) {
                    dist = tmp;
                }
            }
            sum += dist;
            dists[i] = dist;

        }
        double inv_sum = 1/sum;
        for(int i = 0; i < n; i++) {
            dists[i] *= inv_sum;
        }
        double cumsums[n];

        int index = 0;
        cumulative_sum(dists, n, cumsums);
        double r = rand()/((double)RAND_MAX);
//        printf("r = %lf\n", r);
        for(int i = 0; i < n; i++) {
            if(r < cumsums[i]) {
                index = i;
//                printf("picked index:%d\n",index);
                break;
            }
        }
        for (int i = 0; i < k; i++) {

            for (int j = 0; j < k; j++) {
                ret[c*k + j] = U[index*k+j];
            }
        }
    }
    EXIT_FUNC;
}


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
        NUM_ADDS(3);
        double tmp = centers_dist_moved[cluster_assignments[i]];
        upper_bounds[i] += tmp;
        if (max_moved == tmp){
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

static inline void init_kpp_lowdim(double *U, int n, int k, double *ret) {
    ENTER_FUNC;
    // add a random initial point to the centers
#ifdef SEED
    srand(SEED);
#else
    srand(time(0));
#endif
    int ind = ((int)rand()%n);

    for(int j = 0; j < k; j++) {
        ret[j] = U[ind*k+j];
    }

    double sum = 0;
    double *dist_to_cluster = malloc(k*n* sizeof(double));

    for (int c = 1; c < k; c++) {
        sum = 0;
        double *dists = malloc(n* sizeof(double));

        int i;
        //__m256d red81 = _mm256_setzero_pd();
        //__m256d red82 = _mm256_setzero_pd();
        //__m256d zero_vec = _mm256_setzero_pd();


        for(i = 0; i < n-7; i+=8) {
            //double dist = DBL_MAX;
            __m256d dist_vec = _mm256_set1_pd(DBL_MAX);
            __m256d dist_vec2 = _mm256_set1_pd(DBL_MAX);
            double tmp = l2_norm_lowdim(&U[i*k],&ret[(c-1)*k],k);
            double tmp1 = l2_norm_lowdim(&U[(i+1)*k],&ret[(c-1)*k],k);
            double tmp2 = l2_norm_lowdim(&U[(i+2)*k],&ret[(c-1)*k],k);
            double tmp3 = l2_norm_lowdim(&U[(i+3)*k],&ret[(c-1)*k],k);

            double tmp4 = l2_norm_lowdim(&U[(i+4)*k],&ret[(c-1)*k],k);
            double tmp5 = l2_norm_lowdim(&U[(i+5)*k],&ret[(c-1)*k],k);
            double tmp6 = l2_norm_lowdim(&U[(i+6)*k],&ret[(c-1)*k],k);
            double tmp7 = l2_norm_lowdim(&U[(i+7)*k],&ret[(c-1)*k],k);

            dist_to_cluster[(c-1)*n+i] = tmp;
            dist_to_cluster[(c-1)*n+i+1] = tmp1;
            dist_to_cluster[(c-1)*n+i+2] = tmp2;
            dist_to_cluster[(c-1)*n+i+3] = tmp3;
            dist_to_cluster[(c-1)*n+i+4] = tmp4;
            dist_to_cluster[(c-1)*n+i+5] = tmp5;
            dist_to_cluster[(c-1)*n+i+6] = tmp6;
            dist_to_cluster[(c-1)*n+i+7] = tmp7;

            __m256d comp03, comp47;
            for(int j = 0; j < c; j++) {
                comp03 = _mm256_load_pd(dist_to_cluster+j*n+i);
                comp47 = _mm256_load_pd(dist_to_cluster+j*n+i+4);

                dist_vec = _mm256_min_pd(comp03,dist_vec);
                dist_vec2 = _mm256_min_pd(comp47,dist_vec2);

            }
            /*
            __m256d red1, red2, red3;
            red1 = _mm256_permute_pd(dist_vec,0x05);
            red2 = _mm256_add_pd(dist_vec,red1);
            red3 = _mm256_permute2f128_pd(red2,red2,0x01);
            red81 = _mm256_add_pd(red2,red3);
            red1 = _mm256_permute_pd(dist_vec,0x05);
            red2 = _mm256_add_pd(dist_vec,red1);
            red3 = _mm256_permute2f128_pd(red2,red2,0x01);
            red82 = _mm256_add_pd(red2,red3);
            */
            _mm256_store_pd(dists+i,dist_vec);
            _mm256_store_pd(dists+i+4,dist_vec2);

            sum += dists[i]+dists[i+1]+dists[i+2]+dists[i+3]+dists[i+4]+dists[i+5]+dists[i+6]+dists[i+7];

        }
        /*
        double sum_out1[4] = {0.0,0.0,0.0,0.0};
        _mm256_storeu_pd(sum_out1,red81);
        double sum_out2[4] = {0.0,0.0,0.0,0.0};
        _mm256_storeu_pd(sum_out2,red82);
        sum += sum_out1[0]+sum_out2[0];
         */
        for(; i < n-3; i+=4) {
            __m256d dist_vec = _mm256_set1_pd(DBL_MAX);
            double tmp = l2_norm_lowdim(&U[i*k],&ret[(c-1)*k],k);
            double tmp1 = l2_norm_lowdim(&U[(i+1)*k],&ret[(c-1)*k],k);
            double tmp2 = l2_norm_lowdim(&U[(i+2)*k],&ret[(c-1)*k],k);
            double tmp3 = l2_norm_lowdim(&U[(i+3)*k],&ret[(c-1)*k],k);


            dist_to_cluster[(c-1)*n+i] = tmp;
            dist_to_cluster[(c-1)*n+i+1] = tmp1;
            dist_to_cluster[(c-1)*n+i+2] = tmp2;
            dist_to_cluster[(c-1)*n+i+3] = tmp3;
            __m256d comp03;
            for(int j = 0; j < c; j++) {
                comp03 = _mm256_load_pd(dist_to_cluster+j*n+i);
                dist_vec = _mm256_min_pd(comp03,dist_vec);
            }
            /*
            __m256d red1, red2, red3;
            red1 = _mm256_permute_pd(dist_vec,0x05);
            red2 = _mm256_add_pd(dist_vec,red1);
            red3 = _mm256_permute2f128_pd(red2,red2,0x01);
            red4 = _mm256_add_pd(red2,red3);
             */
            _mm256_store_pd(dists+i,dist_vec);
            sum += dists[i]+dists[i+1]+dists[i+2]+dists[i+3];

        }
        //double sum_out3[4] = {0.0,0.0,0.0,0.0};
        //_mm256_storeu_pd(sum_out3,red4);
        //sum += sum_out3[0];
        for(;i<n;i++) {
            double dist = DBL_MAX;
            double tmp = l2_norm_lowdim(&U[i*k],&ret[(c-1)*k],k);
            dist_to_cluster[(c-1)*n+i] = tmp;
            for(int j = 0; j < c; j++) {
                double tmp22 = dist_to_cluster[(j)*n+i];
                if (tmp22 < dist) {
                    dist = tmp22;
                }
            }
            sum += dist;
            dists[i] = dist;
        }


        double inv_sum = 1/sum;

        __m256d inv_vec = _mm256_set1_pd(inv_sum);

        __m256d dists_vec, dists_vec2;

        for(i = 0; i < n-7; i+=8) {
            dists_vec = _mm256_load_pd(dists+i);
            dists_vec2 = _mm256_load_pd(dists+i+4);

            dists_vec = _mm256_mul_pd(dists_vec,inv_vec);
            dists_vec2 = _mm256_mul_pd(dists_vec2,inv_vec);

            _mm256_store_pd(dists+i,dists_vec);
            _mm256_store_pd(dists+i+4,dists_vec2);

        }

        for(; i < n-3; i+=4) {
            dists_vec = _mm256_load_pd(dists+i);
            dists_vec = _mm256_mul_pd(dists_vec,inv_vec);
            _mm256_store_pd(dists+i,dists_vec);
        }
        for(;i<n;i++) {
            dists[i] *= inv_sum;
        }

        int index = 0;
        /*
        __m256d offset = _mm256_setzero_pd();
        for(i = 0; i< n-7; i+=8) {
            __m256d x = _mm256_loadu_pd(dists+i);
            __m256d t0,t1, out, tmp;
            t0 = _mm256_permute_pd(x,_MM_SHUFFLE(2,1,0,3));
            t1 = _mm256_permute2f128_pd(t0,t0,41);
            tmp = _mm256_blend_pd(t0,t1,MAKE_MASK(1,0,1,0));
            x = _mm256_add_pd(x,tmp);

            t0 = _mm256_permute_pd(x,_MM_SHUFFLE(1,0,3,2));
            t1 = _mm256_permute2f128_pd(t0,t0,41);
            tmp = _mm256_blend_pd(t0,t1,MAKE_MASK(1,0,1,0));
            x = _mm256_add_pd(x,tmp);

            out = _mm256_add_pd(x,_mm256_permute2f128_pd(x,x,41));
            out = _mm256_add_pd(out,offset);
            _mm256_storeu_pd(dists+i,out);
            __m256d t3 = _mm256_permute2f128_pd(out,out,17);
            offset = _mm256_permute_pd(t3,255);
        }
         */
        double tmp = dists[0];
        for(int i = 0; i < n; i++) {
            dists[i] += tmp;
            tmp = dists[i];
        }


        double r = rand()/((double)RAND_MAX);
//        printf("r = %lf\n", r);
        for(int i = 0; i < n; i++) {
            if(r < dists[i]) {
                index = i;
//                printf("picked index:%d\n",index);
                break;
            }
        }

        for(int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                ret[c*k+j] = U[index*k+j];
            }
        }

    }



    EXIT_FUNC;
}



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
    clusters_size[0] = n;
    for (int i = 0; i < n; i++) {
        upper_bounds[i] = DBL_MAX;
    }
    init_kpp_lowdim(U, n, k, clusters_center);
    // Distance to nearest other cluster for each cluster.
    double dist_nearest_cluster[k];
    // distance of centers moved between two iteration
    double centers_dist_moved[k];
    int iteration = 0;

    double *mask = calloc(n, sizeof(double));
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
//        // ALGO 1: line 5
//        for (int i = 0; i < n; i++) {
//            // line 6: max_d = max(s(a(i))/2, l(i)) ???
//            double max_d = fmax(lower_bounds[i], dist_nearest_cluster[cluster_assignments[i]]);
//            // ALGO 1: line7: {first bound test}
//            NUM_ADDS(1);
//            if (upper_bounds[i] > max_d) {
//                upper_bounds[i] = l2_norm_lowdim(U + i * k, clusters_center + cluster_assignments[i] * k, k);
//                // ALGO 1: line 9 {second bound test}
//                NUM_ADDS(1);
//                if (upper_bounds[i] > max_d) {
//                    // Iterate over all centers and find first and second closest distances and update DS
//                    point_all_clusters(U, clusters_center, cluster_assignments, upper_bounds, lower_bounds
//                            , clusters_size, k, i);
//                }
//            }
//        }

////    vectorizing mask computation
        // ALGO 1: line 5
        double dist_nearest_cluster_seq[n];
        for (int i = 0; i < n; i++){
            dist_nearest_cluster_seq[i] = dist_nearest_cluster[cluster_assignments[i]];
        }
//        _mm256_load_pd

        __m256d lb_vec, ub_vec,  dist_nearest_cluster_seq_vec;
        __m256d cmp_max_vec, cmp_max_vec1;
        __m256d lb_vec1, ub_vec1, dist_nearest_cluster_seq_vec1;
        __m256d mask_vec, mask_vec1;
        double max_d_arr[n];
        int j;
        for (j = 0; j < n-7; j+=8) {
            lb_vec = _mm256_load_pd(lower_bounds+j);
            lb_vec1 = _mm256_load_pd(lower_bounds+j+4);
            dist_nearest_cluster_seq_vec = _mm256_load_pd(dist_nearest_cluster_seq+j);
            dist_nearest_cluster_seq_vec1 = _mm256_load_pd(dist_nearest_cluster_seq+j+4);

            cmp_max_vec = _mm256_max_pd(lb_vec, dist_nearest_cluster_seq_vec);
            cmp_max_vec1 = _mm256_max_pd(lb_vec1, dist_nearest_cluster_seq_vec1);

            mask_vec = _mm256_cmp_pd(ub_vec, cmp_max_vec, _CMP_GT_OQ);
            mask_vec1 = _mm256_cmp_pd(ub_vec1, cmp_max_vec1, _CMP_GT_OQ);

            _mm256_store_pd(max_d_arr+j, cmp_max_vec);
            _mm256_store_pd(max_d_arr+j+4, cmp_max_vec1);
            _mm256_store_pd(mask+j,mask_vec);
            _mm256_store_pd(mask+j+4,mask_vec1);
        }

        for (; j<n; j++){
            NUM_ADDS(2);
            max_d_arr[j] = MAX(lower_bounds[j], dist_nearest_cluster[cluster_assignments[j]]);
            mask[j] = upper_bounds[j] > max_d_arr[j];
        }

        for (int i = 0; i < n; i++){
            if (mask[i]!=0) {
                upper_bounds[i] = l2_norm_lowdim(U + i * k, clusters_center + cluster_assignments[i] * k, k);
                // ALGO 1: line 9 {second bound test}
                NUM_ADDS(1);
                if (upper_bounds[i] > max_d_arr[i]) {
                    // Iterate over all centers and find first and second closest distances and update DS
                    point_all_clusters(U, clusters_center, cluster_assignments, upper_bounds, lower_bounds
                            , clusters_size, k, i);
                }
            }
        }


        memset(mask, 0, n*sizeof(int));

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

