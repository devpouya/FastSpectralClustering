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

/*
static void print_m256d(__m256d d) {
    double *a = (double *) &d;
    printf("{%lf %lf %lf %lf}\n", a[0], a[1], a[2], a[3]);
}
*/

static inline __m256d LoadArbitrary(double*p0, double*p1, double*p2, double*p3) {
    __m256d a, b, c, d, e, f;
    a = _mm256_loadu_pd(p0);
    b = _mm256_loadu_pd(p1);
    c = _mm256_loadu_pd(p2-2);
    d = _mm256_loadu_pd(p3-2);
    e = _mm256_unpacklo_pd(a, b);
    f = _mm256_unpacklo_pd(c, d);
    return _mm256_blend_pd(e, f, 0b1100);
}

//static inline __m256d gatherArbitrary(int idx_offset, double *arr_address, double *mem){
//    __m256i vindeces = _mm256_loadu_pd(arr_address + idx_offset);
//
//    return _mm256_i64gather_pd(mem,,8);
//}

static inline void cumulative_sum(double *probs, int n, double *ret) {
    ENTER_FUNC;
    ret[0] = probs[0];
    for(int i = 1; i < n; i++) {
        NUM_ADDS(1);
        ret[i] = ret[i-1]+probs[i];
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

    for(int j = 0; j < k; j++) {
        ret[j] = U[ind*k+j];
    }

    double sum = 0;
    //double *dist_to_cluster = malloc(k*n* sizeof(double));
    double dist_to_cluster[k][n] __attribute__((aligned(32)));

    for (int c = 1; c < k; c++) {
        sum = 0;
//        double *dists = malloc(n* sizeof(double));
        double dists[n] __attribute__((aligned(32)));

        int i;
        //__m256d red81 = _mm256_setzero_pd();
        //__m256d red82 = _mm256_setzero_pd();
        //__m256d zero_vec = _mm256_setzero_pd();

//        double sum1, sum2;
        for(i = 0; i < n-7; i+=8) {
            //double dist = DBL_MAX;
            __m256d dist_vec = _mm256_set1_pd(DBL_MAX);
            __m256d dist_vec2 = _mm256_set1_pd(DBL_MAX);
            double tmp = l2_norm_vec(&U[i*k],&ret[(c-1)*k],k);
            double tmp1 = l2_norm_vec(&U[(i+1)*k],&ret[(c-1)*k],k);
            double tmp2 = l2_norm_vec(&U[(i+2)*k],&ret[(c-1)*k],k);
            double tmp3 = l2_norm_vec(&U[(i+3)*k],&ret[(c-1)*k],k);

            double tmp4 = l2_norm_vec(&U[(i+4)*k],&ret[(c-1)*k],k);
            double tmp5 = l2_norm_vec(&U[(i+5)*k],&ret[(c-1)*k],k);
            double tmp6 = l2_norm_vec(&U[(i+6)*k],&ret[(c-1)*k],k);
            double tmp7 = l2_norm_vec(&U[(i+7)*k],&ret[(c-1)*k],k);

            dist_to_cluster[(c-1)][i] = tmp;
            dist_to_cluster[(c-1)][i+1] = tmp1;
            dist_to_cluster[(c-1)][i+2] = tmp2;
            dist_to_cluster[(c-1)][i+3] = tmp3;
            dist_to_cluster[(c-1)][i+4] = tmp4;
            dist_to_cluster[(c-1)][i+5] = tmp5;
            dist_to_cluster[(c-1)][i+6] = tmp6;
            dist_to_cluster[(c-1)][i+7] = tmp7;

            __m256d comp03, comp47;
            for(int j = 0; j < c; j++) {
                comp03 = _mm256_loadu_pd(&dist_to_cluster[j][i]);
                comp47 = _mm256_loadu_pd(&dist_to_cluster[j][i+4]);
                NUM_ADDS(8);
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
            NUM_ADDS(8);
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
            double tmp = l2_norm_vec(&U[i*k],&ret[(c-1)*k],k);
            double tmp1 = l2_norm_vec(&U[(i+1)*k],&ret[(c-1)*k],k);
            double tmp2 = l2_norm_vec(&U[(i+2)*k],&ret[(c-1)*k],k);
            double tmp3 = l2_norm_vec(&U[(i+3)*k],&ret[(c-1)*k],k);

            dist_to_cluster[(c-1)][i] = tmp;
            dist_to_cluster[(c-1)][i+1] = tmp1;
            dist_to_cluster[(c-1)][i+2] = tmp2;
            dist_to_cluster[(c-1)][i+3] = tmp3;
            __m256d comp03;
            for(int j = 0; j < c; j++) {
                comp03 = _mm256_loadu_pd(&dist_to_cluster[j][i]);
                NUM_ADDS(4);
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
            NUM_ADDS(4);
            sum += dists[i]+dists[i+1]+dists[i+2]+dists[i+3];
        }
        //double sum_out3[4] = {0.0,0.0,0.0,0.0};
        //_mm256_storeu_pd(sum_out3,red4);
        //sum += sum_out3[0];
        for(;i<n;i++) {
            double dist = DBL_MAX;
            double tmp = l2_norm_vec(&U[i*k],&ret[(c-1)*k],k);
            dist_to_cluster[(c-1)][i] = tmp;
            for(int j = 0; j < c-1; j++) {
                //double tmp22 = dist_to_cluster[j][i];
                NUM_ADDS(1);
                if (tmp < dist) {
                    dist = tmp;
                }
                tmp = dist_to_cluster[j][i];

            }
            NUM_ADDS(1);
            sum += dist;
            dists[i] = dist;
        }

        NUM_DIVS(1);
        double inv_sum = 1/sum;

        __m256d inv_vec = _mm256_set1_pd(inv_sum);

        __m256d dists_vec, dists_vec2;

        for(i = 0; i < n-7; i+=8) {
            dists_vec = _mm256_load_pd(dists+i);
            dists_vec2 = _mm256_load_pd(dists+i+4);
            NUM_MULS(8);
            dists_vec = _mm256_mul_pd(dists_vec,inv_vec);
            dists_vec2 = _mm256_mul_pd(dists_vec2,inv_vec);

            _mm256_store_pd(dists+i,dists_vec);
            _mm256_store_pd(dists+i+4,dists_vec2);

        }

        for(; i < n-3; i+=4) {
            dists_vec = _mm256_load_pd(dists+i);
            NUM_MULS(4);
            dists_vec = _mm256_mul_pd(dists_vec,inv_vec);
            _mm256_store_pd(dists+i,dists_vec);
        }
        for(;i<n;i++) {
            NUM_MULS(1);
            dists[i] *= inv_sum;
        }

        __m256d offset = _mm256_setzero_pd();
        __m256i mask0111_int = _mm256_set_epi64x(-1, -1, -1, 0);
        __m256d mask0111 = _mm256_castsi256_pd(mask0111_int);
        __m256i mask0011_int = _mm256_set_epi64x(-1, -1 , 0 , 0);
        __m256d mask0011 = _mm256_castsi256_pd(mask0011_int);
        __m256i mask0001_int = _mm256_set_epi64x(-1, 0 , 0 , 0);
        __m256d mask0001 = _mm256_castsi256_pd(mask0001_int);
        //__m256i mask1000_int = _mm256_set_epi64x(0, 0 , 0 , -1);
        //__m256d mask1000 = _mm256_castsi256_pd(mask1000_int);
        for(i = 0; i< n-3; i+=4) {
            __m256d x = _mm256_load_pd(dists+i);
            //printf("X IS:\n");
            //print_m256d(x);
            NUM_ADDS(4);
            x = _mm256_add_pd(x, offset);
            //printf("AFTER OFFSET X IS\n");
            //print_m256d(x);

            __m256d t0 = _mm256_permute4x64_pd(x, _MM_SHUFFLE(2,1,0,3));
            __m256d t1 = _mm256_and_pd(t0, mask0111);
            //printf("T1 is:\n");
            //print_m256d(t1);
            __m256d t2 = _mm256_permute4x64_pd(x, _MM_SHUFFLE(1,0,2,3));
            __m256d t3 = _mm256_and_pd(t2, mask0011);
            //printf("T3 is:\n");
            //print_m256d(t3);

            __m256d t4 = _mm256_permute4x64_pd(x,_MM_SHUFFLE(0,2,1,3));
            __m256d t5 = _mm256_and_pd(t4, mask0001);
            NUM_ADDS(12);
            x = _mm256_add_pd(x,t1);
            //printf("X AFTER FIRST ADD\n");
            //print_m256d(x);
            x = _mm256_add_pd(x,t3);
            //printf("X Second AFTER ADD is:\n");
            //print_m256d(x);

            x = _mm256_add_pd(x,t5);
            //printf("X Second AFTER ADD is:\n");
            //print_m256d(x);

            _mm256_store_pd(dists+i, x);

            offset = _mm256_and_pd(x, mask0001);
            offset = _mm256_permute4x64_pd(offset,_MM_SHUFFLE(0,2,1,3));
            //printf("OFFSET\n");
            //print_m256d(offset);
        }


        double tmp = dists[i-1];
        for(; i < n; i++) {
            NUM_ADDS(1);
            dists[i] += tmp;
            tmp = dists[i];
        }



        int index = 0;
        NUM_DIVS(1);
        double r = rand()/((double)RAND_MAX);
//        printf("r = %lf\n", r);
        for(int i = 0; i < n; i++) {
            NUM_ADDS(1);
            if(r < dists[i]) {
                index = i;
//                printf("picked index:%d\n",index);
                break;
            }
        }

        // for(int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                ret[c*k+j] = U[index*k+j];
            }
        // }
    }
    EXIT_FUNC;
}

/*
 * ALGO 2: INITIALIZE ---------------------------------------------------------
 * 1) init DS
 * 2) init kpp
 */

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
        double dist = l2_norm_vec(U + i * k, clusters_center + j * k, k);
        // Find distance between the point and the center.
        if (dist < closest_center_1_dist) {
            NUM_ADDS(1);
            closest_center_2_dist = closest_center_1_dist;
            closest_center_1 = j;
            closest_center_1_dist = dist;
        } else if (dist < closest_center_2_dist) {
            NUM_ADDS(1);
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
    for (int j = 0; j < k; j++) {
        double dist = 0;
        if (clusters_size[j] > 0) {
            for (int l = 0; l < k; l++) { // update
                NUM_DIVS(1);
                new_clusters_centers[j * k + l] = new_clusters_centers[j * k + l] / clusters_size[j];
                dist = l2_norm_vec(clusters_center + j * k, new_clusters_centers + j * k, k);
            }
            centers_dist_moved[j] = dist;
        }

    }
    EXIT_FUNC;
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
        NUM_ADDS(1);
        if (centers_dist_moved[i] > max_moved) {
            second_max_moved = max_moved;
            max_moved = centers_dist_moved[i];
        }
    }

    for (int i = 0; i < n; i++) {
        NUM_ADDS(3);
        double tmp = centers_dist_moved[cluster_assignments[i]];
        NUM_ADDS(1);
        upper_bounds[i] += tmp;
        NUM_ADDS(1);
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
    double clusters_center[k*k] __attribute__((aligned(32)));
    // tmp for next iteration
    double new_clusters_centers[k*k] __attribute((aligned(32)));
    // cluster sizes
    // int *clusters_size = calloc(k, sizeof(int));
    int clusters_size[k] __attribute__((aligned(32)));
    memset(clusters_size, 0, k * sizeof(int));
    // n upper bounds (of closest center)
    // n lower bounds (of 2nd strict closest center)
    // double *lower_bounds = calloc(n, sizeof(double));
    double lower_bounds[n] __attribute__((aligned(32)));
    memset(lower_bounds, 0, n * sizeof(double));
    // double *upper_bounds = calloc(n, sizeof(double));
    double upper_bounds[n] __attribute__((aligned(32)));
    // stores cluster index for all points
    // int *cluster_assignments = calloc(n, sizeof(int));
    int cluster_assignments[n] __attribute__((aligned(32)));
    memset(cluster_assignments, 0, n * sizeof(int));
    // Algorithm 2: init + kpp -------------------
    clusters_size[0] = n;
    for (int i = 0; i < n; i++) {
        upper_bounds[i] = DBL_MAX;
    }
//    printf("start init\n");
    init_kpp(U, n, k, clusters_center);
//    printf("finished init\n");
    // Distance to nearest other cluster for each cluster.
    double dist_nearest_cluster[k] __attribute__((aligned(32)));
    // distance of centers moved between two iteration
    double centers_dist_moved[k] __attribute__((aligned(32)));
    int iteration = 0;
    while (iteration < max_iter) {
        // Initialization after each iteration
        for (int i = 0; i < k*k; i++) {
            new_clusters_centers[i] = 0;
        }
        // min distance between each two centers {update s} --------------------------
        /*
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
        */
        for (int i = 0; i < k; i++) { // for each cluster
            double min_dist = DBL_MAX;
            for (int j = 0; j < k; j++) { // look at the distances to all cluster

                    double dist = 0;
                    int l;
                    __m256d dist_vec = _mm256_setzero_pd();
                    for (l = 0; l < k-3; l+=4) { // iterate over column = dimension
                        __m256d cent1 = _mm256_loadu_pd(&clusters_center[i*k+l]);
                        __m256d cent2 = _mm256_loadu_pd(&clusters_center[j*k+l]);
                        NUM_ADDS(12);
                        __m256d tmp = _mm256_sub_pd(cent1,cent2);
                        NUM_MULS(4);
                        NUM_ADDS(4);
                        dist_vec = _mm256_fmadd_pd(tmp,tmp,dist_vec);

                    }
                    for(;l<k;l++) {
                        NUM_ADDS(2);
                        NUM_MULS(1);
                        double tmp = clusters_center[i*k+l] - clusters_center[j*k+l];
                        dist += tmp*tmp;

                    }
                    double out[4] __attribute__((aligned(32)));
                    _mm256_store_pd(out,dist_vec);
                    NUM_ADDS(3);
                    double tmp1 = out[0]+out[1]+out[2]+out[3];
                    NUM_ADDS(1);
                    dist = tmp1 + dist;
                    NUM_MULS(1);
                    NUM_SQRTS(1);
                    dist = sqrt(dist) * 0.5;
                    NUM_ADDS(1);
                    if (dist < min_dist) {
                        min_dist = dist;
                        dist_nearest_cluster[i] = dist;
                    }
                }
        }

        // ALGO 1: line 5
        __m256d lb_vec; __m256d lb_vec1;
        __m256d dist_nearest_cluster_seq_vec;
        __m256d cmp_max_vec, cmp_max_vec1, dist_nearest_cluster_seq_vec1;
        double max_d_arr[n] __attribute__((aligned(32)));
        int j;
        for (j = 0; j < n-7; j+=8) {
            lb_vec = _mm256_load_pd(lower_bounds+j);
            lb_vec1 = _mm256_load_pd(lower_bounds+j+4);

            dist_nearest_cluster_seq_vec = LoadArbitrary(dist_nearest_cluster+cluster_assignments[j],
                                                         dist_nearest_cluster+cluster_assignments[j+1],
                                                         dist_nearest_cluster+cluster_assignments[j+2],
                                                         dist_nearest_cluster+cluster_assignments[j+3]);
            dist_nearest_cluster_seq_vec1 = LoadArbitrary(dist_nearest_cluster+cluster_assignments[j+4],
                                                         dist_nearest_cluster+cluster_assignments[j+5],
                                                         dist_nearest_cluster+cluster_assignments[j+6],
                                                         dist_nearest_cluster+cluster_assignments[j+7]);
            NUM_ADDS(8);
            cmp_max_vec = _mm256_max_pd(lb_vec, dist_nearest_cluster_seq_vec);
            cmp_max_vec1 = _mm256_max_pd(lb_vec1, dist_nearest_cluster_seq_vec1);
            _mm256_store_pd(max_d_arr+j, cmp_max_vec);
            _mm256_store_pd(max_d_arr+j+4, cmp_max_vec1);
        }
        for (; j<n; j++){
            NUM_ADDS(1);
            max_d_arr[j] = MAX(lower_bounds[j], dist_nearest_cluster[cluster_assignments[j]]);
        }
        for (int i = 0; i < n; i++){
            NUM_ADDS(1);
            if (upper_bounds[i] > max_d_arr[i]) {
                upper_bounds[i] = l2_norm_vec(U + i * k, clusters_center + cluster_assignments[i] * k, k);
                // ALGO 1: line 9 {second bound test}
                NUM_ADDS(1);
                if (upper_bounds[i] > max_d_arr[i]) {
                    // Iterate over all centers and find first and second closest distances and update DS
                    point_all_clusters(U, clusters_center, cluster_assignments, upper_bounds, lower_bounds
                            , clusters_size, k, i);
                }
            }
        }

        // To compute new mean: size calculated in point all clusters, sum now, divide in move!
        /*
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                NUM_ADDS(1);
                new_clusters_centers[cluster_assignments[i]*k+j] += U[i*k+j];
            }
        }
        */

        for(int i = 0; i < n; i++) {
            int j;
            for(j = 0; j < k-3; j+=4) {
                __m256d sumvec = _mm256_loadu_pd(&new_clusters_centers[cluster_assignments[i]*k+j]);
                __m256d uvec = _mm256_loadu_pd(&U[i*k+j]);
                NUM_ADDS(4);
                sumvec = _mm256_add_pd(uvec,sumvec);
                _mm256_storeu_pd(&new_clusters_centers[cluster_assignments[i]*k+j],sumvec);
            }
            for(;j<k;j++){
                NUM_ADDS(1);
                new_clusters_centers[cluster_assignments[i]*k+j] += U[i*k+j];
            }
        }



        // ALGO 4 - MOVE-CENTERS: check for distance moved then move the centers ---------
//        move_centers(new_clusters_centers, clusters_size
//                , clusters_center, centers_dist_moved, k);
        /*
        for (int j = 0; j < k; j++) {
            double dist = 0;
            if (clusters_size[j] > 0) {
                for (int l = 0; l < k; l++) { // update
                    NUM_DIVS(1);
                    new_clusters_centers[j * k + l] = new_clusters_centers[j * k + l] / clusters_size[j];
                    dist = l2_norm(clusters_center + j * k, new_clusters_centers + j * k, k);
                }
                centers_dist_moved[j] = dist;
            }
        }
        */
        for(int j = 0; j < k; j++) {
            double dist = 0;
            int l;
            NUM_DIVS(1);
            double inv = (double) 1/clusters_size[j];
            __m256d inv_vec = _mm256_set1_pd(inv);
            for(l = 0; l < k-3; l+=4) {
                __m256d clust_vec = _mm256_loadu_pd(&new_clusters_centers[j*k+l]);
                NUM_MULS(4);
                clust_vec = _mm256_mul_pd(clust_vec,inv_vec);
                _mm256_storeu_pd(&new_clusters_centers[j*k+l],clust_vec);
            }
            for(;l<k;l++) {
                NUM_MULS(1);
                new_clusters_centers[j*k+l] *= inv;
            }
            centers_dist_moved[j] = dist;
        }
        /*
        __m256d max_all = _mm256_setzero_pd();
        __m256d second_max = _mm256_setzero_pd();
        int p;
        for(p = 0; p <k-3; p+=4) {
            __m256d curr = _mm256_loadu_pd(centers_dist_moved+p);
            second_max = max_all;
            max_all = _mm256_max_pd(curr,max_all);
        }
        __m256d y = _mm256_permute2f128_pd(max_all,max_all,1);
        __m256d m1 = _mm256_max_pd(max_all,y);
        __m256d m2 = _mm256_permute_pd(m1,5);
        __m256d m = _mm256_max_pd(m1,m2);
        double out1[4];
        double out2[4];
        _mm256_storeu_pd(out1,m);
        _mm256_storeu_pd(out2,second_max);

        double max_moved = out1[0];
        double second_max_moved=0;
        double tmp_max = out2[0];
        for(int i = 0; i < 4; i++) {
            double curr = out2[i];
            if (curr > tmp_max) {
                second_max_moved = tmp_max;
                tmp_max = curr;
            }
        }
        for(; p <k; p++) {
            double curr = centers_dist_moved[p];
            if (curr > max_moved){
                second_max_moved = max_moved;
                max_moved = curr;

            }
        }
        */
        // ALGO 5 - Update-bounds : for all U update upper and lower distance bounds ---------------
//        update_bounds(upper_bounds, lower_bounds, centers_dist_moved, cluster_assignments, n, k);

        double max_moved = 0;
        double second_max_moved = 0;
        for (int i = 0; i < k; i++) {
            NUM_ADDS(1);
            if (centers_dist_moved[i] > max_moved) {
                second_max_moved = max_moved;
                max_moved = centers_dist_moved[i];
            }
        }


//        double centers_dist_moved_seq[n];
//        for(i = 0; i < n; i++){
//            centers_dist_moved_seq[i] = centers_dist_moved[cluster_assignments[i]];
//        }
        __m256d tmp_vec, ub_vec; //, lb_vec;
        __m256d max_moved_tmp_equal_mask, max_moved_tmp_inequal_mask;
        __m256d tmp_vec1, ub_vec1; //, lb_vec1;
        __m256d max_moved_tmp_equal_mask1, max_moved_tmp_inequal_mask1;
        __m256d zero_vec = _mm256_setzero_pd();
        __m256d max_moved_vec = _mm256_set1_pd(max_moved);
        __m256d second_max_moved_vec = _mm256_set1_pd(second_max_moved);
        int i;
        for(i = 0; i < n-7; i+=8){
//            tmp_vec = _mm256_loadu_pd(centers_dist_moved_seq+i);
            tmp_vec = LoadArbitrary(centers_dist_moved+cluster_assignments[i],
                                    centers_dist_moved+cluster_assignments[i+1],
                                    centers_dist_moved+cluster_assignments[i+2],
                                    centers_dist_moved+cluster_assignments[i+3]);
            tmp_vec1 = LoadArbitrary(centers_dist_moved+cluster_assignments[i+4],
                                    centers_dist_moved+cluster_assignments[i+5],
                                    centers_dist_moved+cluster_assignments[i+6],
                                    centers_dist_moved+cluster_assignments[i+7]);
            ub_vec = _mm256_load_pd(upper_bounds+i);
            lb_vec = _mm256_load_pd(lower_bounds+i);
            ub_vec1 = _mm256_load_pd(upper_bounds+i+4);
            lb_vec1 = _mm256_load_pd(lower_bounds+i+4);
            NUM_ADDS(8);
            ub_vec = _mm256_add_pd(ub_vec, tmp_vec);
            ub_vec1 = _mm256_add_pd(ub_vec1, tmp_vec1);
            NUM_ADDS(8);
            max_moved_tmp_equal_mask = _mm256_cmp_pd(max_moved_vec,tmp_vec,_CMP_EQ_OQ);
            max_moved_tmp_inequal_mask = _mm256_xor_pd(zero_vec, max_moved_tmp_equal_mask);
            max_moved_tmp_equal_mask1 = _mm256_cmp_pd(max_moved_vec,tmp_vec1,_CMP_EQ_OQ);
            max_moved_tmp_inequal_mask1 = _mm256_xor_pd(zero_vec, max_moved_tmp_equal_mask1);
            NUM_ADDS(16);
            lb_vec = _mm256_sub_pd(lb_vec, _mm256_and_pd(max_moved_tmp_equal_mask, second_max_moved_vec));
            lb_vec = _mm256_sub_pd(lb_vec, _mm256_and_pd(max_moved_tmp_inequal_mask, max_moved_vec));
            lb_vec1 = _mm256_sub_pd(lb_vec1, _mm256_and_pd(max_moved_tmp_equal_mask1, second_max_moved_vec));
            lb_vec1 = _mm256_sub_pd(lb_vec1, _mm256_and_pd(max_moved_tmp_inequal_mask1, max_moved_vec));

            _mm256_store_pd(upper_bounds+i, ub_vec);
            _mm256_store_pd(lower_bounds+i, lb_vec);
            _mm256_store_pd(upper_bounds+i+4, ub_vec1);
            _mm256_store_pd(lower_bounds+i+4, lb_vec1);
        }
        for (; i < n; i++) {
            double tmp = centers_dist_moved[cluster_assignments[i]];
            NUM_ADDS(1);
            upper_bounds[i] += tmp;
            NUM_ADDS(2);
            if (max_moved == tmp){
                lower_bounds[i] -= second_max_moved;
            } else {
                lower_bounds[i] -= max_moved;
            }
        }

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
    //double *dist_to_cluster = malloc(k*n* sizeof(double));
    double dist_to_cluster[k][n] __attribute__((aligned(32)));

    for (int c = 1; c < k; c++) {
        sum = 0;
//        double *dists = malloc(n* sizeof(double));
        double dists[n] __attribute__((aligned(32)));

        int i;
        //__m256d red81 = _mm256_setzero_pd();
        //__m256d red82 = _mm256_setzero_pd();
        //__m256d zero_vec = _mm256_setzero_pd();

//        double sum1, sum2;
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

            dist_to_cluster[(c-1)][i] = tmp;
            dist_to_cluster[(c-1)][i+1] = tmp1;
            dist_to_cluster[(c-1)][i+2] = tmp2;
            dist_to_cluster[(c-1)][i+3] = tmp3;
            dist_to_cluster[(c-1)][i+4] = tmp4;
            dist_to_cluster[(c-1)][i+5] = tmp5;
            dist_to_cluster[(c-1)][i+6] = tmp6;
            dist_to_cluster[(c-1)][i+7] = tmp7;

            __m256d comp03, comp47;
            for(int j = 0; j < c; j++) {
                comp03 = _mm256_loadu_pd(&dist_to_cluster[j][i]);
                comp47 = _mm256_loadu_pd(&dist_to_cluster[j][i+4]);
                NUM_ADDS(8);
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
            NUM_ADDS(8);
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

            dist_to_cluster[(c-1)][i] = tmp;
            dist_to_cluster[(c-1)][i+1] = tmp1;
            dist_to_cluster[(c-1)][i+2] = tmp2;
            dist_to_cluster[(c-1)][i+3] = tmp3;
            __m256d comp03;
            for(int j = 0; j < c; j++) {
                comp03 = _mm256_loadu_pd(&dist_to_cluster[j][i]);
                NUM_ADDS(4);
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
            NUM_ADDS(4);
            sum += dists[i]+dists[i+1]+dists[i+2]+dists[i+3];

        }
        //double sum_out3[4] = {0.0,0.0,0.0,0.0};
        //_mm256_storeu_pd(sum_out3,red4);
        //sum += sum_out3[0];
        for(;i<n;i++) {
            double dist = DBL_MAX;
            double tmp = l2_norm_lowdim(&U[i*k],&ret[(c-1)*k],k);
            dist_to_cluster[(c-1)][i] = tmp;
            for(int j = 0; j < c; j++) {
                double tmp22 = dist_to_cluster[j][i];
                NUM_ADDS(1);
                if (tmp22 < dist) {
                    dist = tmp22;
                }
            }
            NUM_ADDS(1);
            sum += dist;
            dists[i] = dist;
        }

        NUM_DIVS(1);
        double inv_sum = 1/sum;

        __m256d inv_vec = _mm256_set1_pd(inv_sum);

        __m256d dists_vec, dists_vec2;

        for(i = 0; i < n-7; i+=8) {
            dists_vec = _mm256_load_pd(dists+i);
            dists_vec2 = _mm256_load_pd(dists+i+4);
            NUM_MULS(8);
            dists_vec = _mm256_mul_pd(dists_vec,inv_vec);
            dists_vec2 = _mm256_mul_pd(dists_vec2,inv_vec);
            _mm256_store_pd(dists+i,dists_vec);
            _mm256_store_pd(dists+i+4,dists_vec2);
        }

        for(; i < n-3; i+=4) {
            dists_vec = _mm256_load_pd(dists+i);
            NUM_MULS(4);
            dists_vec = _mm256_mul_pd(dists_vec,inv_vec);
            _mm256_store_pd(dists+i,dists_vec);
        }
        for(;i<n;i++) {
            NUM_MULS(1);
            dists[i] *= inv_sum;
        }

        __m256d offset = _mm256_setzero_pd();
        __m256i mask0111_int = _mm256_set_epi64x(-1, -1, -1, 0);
        __m256d mask0111 = _mm256_castsi256_pd(mask0111_int);
        __m256i mask0011_int = _mm256_set_epi64x(-1, -1 , 0 , 0);
        __m256d mask0011 = _mm256_castsi256_pd(mask0011_int);
        __m256i mask0001_int = _mm256_set_epi64x(-1, 0 , 0 , 0);
        __m256d mask0001 = _mm256_castsi256_pd(mask0001_int);
        for(i = 0; i< n-3; i+=4) {
            __m256d x = _mm256_load_pd(dists+i);
            NUM_ADDS(4);
            x = _mm256_add_pd(x, offset);
            __m256d t0 = _mm256_permute4x64_pd(x, _MM_SHUFFLE(2,1,0,3));
            __m256d t1 = _mm256_and_pd(t0, mask0111);
            __m256d t2 = _mm256_permute4x64_pd(x, _MM_SHUFFLE(1,0,2,3));
            __m256d t3 = _mm256_and_pd(t2, mask0011);
            __m256d t4 = _mm256_permute4x64_pd(x,_MM_SHUFFLE(0,2,1,3));
            __m256d t5 = _mm256_and_pd(t4, mask0001);
            NUM_ADDS(12);
            x = _mm256_add_pd(x,t1);
            x = _mm256_add_pd(x,t3);
            x = _mm256_add_pd(x,t5);
            _mm256_store_pd(dists+i, x);
            offset = _mm256_and_pd(x, mask0001);
            offset = _mm256_permute4x64_pd(offset,_MM_SHUFFLE(0,2,1,3));
        }

        double tmp = dists[i-1];
        for(; i < n; i++) {
            NUM_ADDS(1);
            dists[i] += tmp;
            tmp = dists[i];
        }
        int index = 0;
        NUM_DIVS(1);
        double r = rand()/((double)RAND_MAX);
        for(int i = 0; i < n; i++) {
            NUM_ADDS(1);
            if(r < dists[i]) {
                index = i;
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
    for (int j = 0; j < k; j++) {
        double dist = 0;
        NUM_ADDS(1);
        if (clusters_size[j] > 0) {
            for (int l = 0; l < k; l++) { // update
                NUM_DIVS(1);
                new_clusters_centers[j*k + l] = new_clusters_centers[j*k + l] / clusters_size[j];
            }
            dist = l2_norm_lowdim(clusters_center + j*k, new_clusters_centers + j*k, k);
        }
        centers_dist_moved[j] = dist;
    }
    EXIT_FUNC;
}


/*
 * ALGO 1 - K-Means Algorithm Hamerly --------------------------------------------
 * Implementation of the following algorithms as presented in the paper:
 *      https://epubs.siam.org/doi/pdf/10.1137/1.9781611972801.12
 */
void hamerly_kmeans_lowdim(double *U, int n, int k, int max_iter, double stopping_error, struct cluster *ret) {
    ENTER_FUNC;
    // initial centers
    double clusters_center[k*k] __attribute__((aligned(32)));
    // tmp for next iteration
    double new_clusters_centers[k*k] __attribute((aligned(32)));
    // cluster sizes
    // int *clusters_size = calloc(k, sizeof(int));
    int clusters_size[k] __attribute__((aligned(32)));
    memset(clusters_size, 0, k * sizeof(int));
    // n upper bounds (of closest center)
    // n lower bounds (of 2nd strict closest center)
    // double *lower_bounds = calloc(n, sizeof(double));
    double lower_bounds[n] __attribute__((aligned(32)));
    memset(lower_bounds, 0, n * sizeof(double));
    // double *upper_bounds = calloc(n, sizeof(double));
    double upper_bounds[n] __attribute__((aligned(32)));
    // stores cluster index for all points
    // int *cluster_assignments = calloc(n, sizeof(int));
    int cluster_assignments[n] __attribute__((aligned(32)));
    memset(cluster_assignments, 0, n * sizeof(int));
    // Algorithm 2: init + kpp -------------------
    clusters_size[0] = n;
    for (int i = 0; i < n; i++) {
        upper_bounds[i] = DBL_MAX;
    }

    init_kpp_lowdim(U, n, k, clusters_center);
    // Distance to nearest other cluster for each cluster.
    double dist_nearest_cluster[k] __attribute__((aligned(32)));
    // distance of centers moved between two iteration
    double centers_dist_moved[k] __attribute__((aligned(32)));
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
                        NUM_ADDS(2);
                        double tmp = clusters_center[i*k+l] - clusters_center[j*k+l];
                        dist += tmp*tmp;

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
        __m256d lb_vec; __m256d lb_vec1;
        __m256d dist_nearest_cluster_seq_vec;
        __m256d cmp_max_vec, cmp_max_vec1, dist_nearest_cluster_seq_vec1;
        double max_d_arr[n] __attribute__((aligned(32)));
        int j;
        for (j = 0; j < n-7; j+=8) {
            lb_vec = _mm256_load_pd(lower_bounds+j);
            lb_vec1 = _mm256_load_pd(lower_bounds+j+4);
            dist_nearest_cluster_seq_vec = LoadArbitrary(dist_nearest_cluster+cluster_assignments[j],
                                                         dist_nearest_cluster+cluster_assignments[j+1],
                                                         dist_nearest_cluster+cluster_assignments[j+2],
                                                         dist_nearest_cluster+cluster_assignments[j+3]);
            dist_nearest_cluster_seq_vec1 = LoadArbitrary(dist_nearest_cluster+cluster_assignments[j+4],
                                                          dist_nearest_cluster+cluster_assignments[j+5],
                                                          dist_nearest_cluster+cluster_assignments[j+6],
                                                          dist_nearest_cluster+cluster_assignments[j+7]);
            NUM_ADDS(8);
            cmp_max_vec = _mm256_max_pd(lb_vec, dist_nearest_cluster_seq_vec);
            cmp_max_vec1 = _mm256_max_pd(lb_vec1, dist_nearest_cluster_seq_vec1);
            _mm256_store_pd(max_d_arr+j, cmp_max_vec);
            _mm256_store_pd(max_d_arr+j+4, cmp_max_vec1);
        }
        for (; j<n; j++){
            NUM_ADDS(2);
            max_d_arr[j] = MAX(lower_bounds[j], dist_nearest_cluster[cluster_assignments[j]]);
        }
        for (int i = 0; i < n; i++){
            NUM_ADDS(1);
            if (upper_bounds[i] > max_d_arr[i]) {
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



        // To compute new mean: size calculated in point all clusters, sum now, divide in move!
        /*
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                NUM_ADDS(1);
                new_clusters_centers[cluster_assignments[i]*k+j] += U[i*k+j];
            }
        }
        */

        for(int i = 0; i < n; i++) {
            int j;
            for(j = 0; j < k-3; j+=4) {
                __m256d sumvec = _mm256_loadu_pd(&new_clusters_centers[cluster_assignments[i]*k+j]);
                __m256d uvec = _mm256_loadu_pd(&U[i*k+j]);
                NUM_ADDS(4);
                sumvec = _mm256_add_pd(uvec,sumvec);
                _mm256_storeu_pd(&new_clusters_centers[cluster_assignments[i]*k+j],sumvec);
            }
            for(;j<k;j++){
                NUM_ADDS(1);
                new_clusters_centers[cluster_assignments[i]*k+j] += U[i*k+j];
            }
        }
        // ALGO 4 - MOVE-CENTERS: check for distance moved then move the centers ---------
        //move_centers_lowdim(new_clusters_centers, clusters_size
        //        , clusters_center, centers_dist_moved, k);
        for(int j = 0; j < k; j++) {
            double dist = 0;
            int l;
            NUM_DIVS(1);
            double inv = (double) 1/clusters_size[j];
            __m256d inv_vec = _mm256_set1_pd(inv);
            for(l = 0; l < k-3; l+=4) {
                __m256d clust_vec = _mm256_loadu_pd(&new_clusters_centers[j*k+l]);
                NUM_MULS(4);
                clust_vec = _mm256_mul_pd(clust_vec,inv_vec);
                _mm256_storeu_pd(&new_clusters_centers[j*k+l],clust_vec);
            }
            for(;l<k;l++) {
                NUM_MULS(1);
                new_clusters_centers[j*k+l] *= inv;
            }
            centers_dist_moved[j] = dist;
        }

        // ALGO 5 - Update-bounds : for all U update upper and lower distance bounds ---------------
        double max_moved = 0;
        double second_max_moved = 0;
        for (int i = 0; i < k; i++) {
            NUM_ADDS(1);
            if (centers_dist_moved[i] > max_moved) {
                second_max_moved = max_moved;
                max_moved = centers_dist_moved[i];
            }
        }

        int i;
        __m256d tmp_vec, ub_vec; //, lb_vec;
        __m256d max_moved_tmp_equal_mask, max_moved_tmp_inequal_mask;
        __m256d tmp_vec1, ub_vec1; //, lb_vec1;
        __m256d max_moved_tmp_equal_mask1, max_moved_tmp_inequal_mask1;
        __m256d zero_vec = _mm256_setzero_pd();
        __m256d max_moved_vec = _mm256_set1_pd(max_moved);
        __m256d second_max_moved_vec = _mm256_set1_pd(second_max_moved);
        for(i = 0; i < n-7; i+=8){
            tmp_vec = LoadArbitrary(centers_dist_moved+cluster_assignments[i],
                                    centers_dist_moved+cluster_assignments[i+1],
                                    centers_dist_moved+cluster_assignments[i+2],
                                    centers_dist_moved+cluster_assignments[i+3]);
            tmp_vec1 = LoadArbitrary(centers_dist_moved+cluster_assignments[i+4],
                                     centers_dist_moved+cluster_assignments[i+5],
                                     centers_dist_moved+cluster_assignments[i+6],
                                     centers_dist_moved+cluster_assignments[i+7]);
            ub_vec = _mm256_load_pd(upper_bounds+i);
            lb_vec = _mm256_load_pd(lower_bounds+i);
            ub_vec1 = _mm256_load_pd(upper_bounds+i+4);
            lb_vec1 = _mm256_load_pd(lower_bounds+i+4);
            NUM_ADDS(8);
            ub_vec = _mm256_add_pd(ub_vec, tmp_vec);
            ub_vec1 = _mm256_add_pd(ub_vec1, tmp_vec1);
            NUM_ADDS(8);
            max_moved_tmp_equal_mask = _mm256_cmp_pd(max_moved_vec,tmp_vec,_CMP_EQ_OQ);
            max_moved_tmp_inequal_mask = _mm256_xor_pd(zero_vec, max_moved_tmp_equal_mask);
            max_moved_tmp_equal_mask1 = _mm256_cmp_pd(max_moved_vec,tmp_vec1,_CMP_EQ_OQ);
            max_moved_tmp_inequal_mask1 = _mm256_xor_pd(zero_vec, max_moved_tmp_equal_mask1);
            NUM_ADDS(16);
            lb_vec = _mm256_sub_pd(lb_vec, _mm256_and_pd(max_moved_tmp_equal_mask, second_max_moved_vec));
            lb_vec = _mm256_sub_pd(lb_vec, _mm256_and_pd(max_moved_tmp_inequal_mask, max_moved_vec));
            lb_vec1 = _mm256_sub_pd(lb_vec1, _mm256_and_pd(max_moved_tmp_equal_mask1, second_max_moved_vec));
            lb_vec1 = _mm256_sub_pd(lb_vec1, _mm256_and_pd(max_moved_tmp_inequal_mask1, max_moved_vec));

            _mm256_store_pd(upper_bounds+i, ub_vec);
            _mm256_store_pd(lower_bounds+i, lb_vec);
            _mm256_store_pd(upper_bounds+i+4, ub_vec1);
            _mm256_store_pd(lower_bounds+i+4, lb_vec1);
        }
        for (; i < n; i++) {
            NUM_ADDS(3);
            double tmp = centers_dist_moved[cluster_assignments[i]];
            NUM_ADDS(1);
            upper_bounds[i] += tmp;
            NUM_ADDS(2);
            if (max_moved == tmp){
                lower_bounds[i] -= second_max_moved;
            } else {
                lower_bounds[i] -= max_moved;
            }
        }
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

