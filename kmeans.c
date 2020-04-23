#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>

#include "norms.h"
#include "instrumentation.h"
#include "kmeans.h"

static void cumulative_sum(double *probs, int n, double *ret) {
    ret[0] = probs[0];
    for(int i = 1; i < n; i++) {
        ret[i] = ret[i-1]+probs[i];
    }
}
static void init_kpp(double *U, int n, int k, double *ret) {
    // add a random initial point to the centers
    srand(time(0));
    int ind = ((int)rand()%n);
    //ret[0] = U[((int)rand()%n)*k];
    for(int j = 0; j < k; j++) {
        ret[j] = U[ind*n+j];
    }
    double sum = 0;
    for (int c = 1; c < k; c++) {

        sum = 0;
        double dists[n];
        for (int i = 0; i < n; i++) {
            //find closest point and add to sum
            double dist = DBL_MAX;
            for(int j = 0; j < c; j++) {
                double tmp = l2_norm(&U[i*n],&ret[j*k],k);
                if (tmp < dist) {
                    dist = tmp;
                }
            }
            sum += dist;
            dists[i] = dist;

        }
        for(int i = 0; i < n; i++) {
            dists[i] /= sum;
        }
        double cumsums[n];
        int index = 0;
        cumulative_sum(dists,n,cumsums);
        double r = rand()/((double)RAND_MAX);
        for(int i = 0; i < n; i++) {
            if(r < cumsums[i]) {
                index = i;
                printf("picked index:%d\n",index);
                break;
            }
        }
        for (int i = 0; i < k; i++) {

            for (int j = 0; j < k; j++) {
                ret[c*k + j] = U[index*n+j];
            }
        }
        /*
        double ransom = sum*rand() / (RAND_MAX-1);

        for (int i = 0; i < k; i++) {
            if(ransom-dists[i]>0){
                printf("TRAP i %d c %d\n",i,c);

                continue;
            }
            printf("Center %d: ( ", i);
            for (int j = 0; j < k; j++) {
                ret[c*k + j] = U[i*n+j];
                 printf("%lf ", ret[i*k + j]);
            }
            printf(")\n");
        }
         */
    }




}


/*
static void init_rand(double *U, int n, int k, double *ret) {
    srand(time(0));
    // knuth algorithm for distinct random values in range
   int rem, havs;
    rem = 0;
    int inds[k];
    for(havs = 0; havs < k && rem < k; ++havs) {
        int rh = k-havs;
        int rm = k-rem;
        if(rand()%rh<rm) {
            inds[rem++] = havs+1;
        }
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            NUM_DIVS(1);
            NUM_MULS(1);
            NUM_ADDS(2);
            ret[i*k + j] = U[inds[i]*n+j];
        }
    }
}
*/
/*static void init_means(double *U, int n, int k, double *ret) {
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
        // printf("Center %d: ( ", i);
        for (int j = 0; j < k; j++) {
            NUM_DIVS(1);
            NUM_MULS(1);
            NUM_ADDS(2);
            ret[i*k + j] = ( ((double )rand() /RAND_MAX)*(bounds[j][1] - bounds[j][0])) + bounds[j][0];
            // printf("%lf ", ret[i*k + j]);
        }
        // printf(")\n");
    }
}*/


// mean of each column
// dimension is the column index along which the mean is computed
static double compute_mean_of_one_dimension(double *U, int *indices, int size, int n, int dimension) {
    double sum = 0;
    NUM_ADDS(size);
    NUM_DIVS(1);
    for (int i = 0; i < size; i++) { // for all points
        sum += U[indices[i]*n+dimension]; // .. select one dimension
    }
    return (size > 0) ? (sum/size) : 0;
}

static void update_means(double *U, struct cluster *clusters, int k, int n, double *ret) {
    for (int i = 0; i < k; i++) { // iterate over cluster i
    //    printf("Center %d: ( ", i);
        for (int j = 0; j < k; j++) { // j is the dimension here
            ret[i*k + j] = (clusters[i].size > 0) ?
                           compute_mean_of_one_dimension(U, clusters[i].indices, clusters[i].size, n, j) : clusters[i].mean[j];
        //    printf("%lf ", ret[i*k + j]);
        }
    //    printf(")\n");
    }
}

static void copy_means(struct cluster *clusters, int k, double *means) {
    for (int i = 0; i < k; i++) { // iterate over cluster i
        //    printf("Center %d: ( ", i);
        for (int j = 0; j < k; j++) { // j is the dimension here
            means[i*k + j] = clusters[i].mean[j];
            //    printf("%lf ", ret[i*k + j]);
        }
        //    printf(")\n");
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
//
//static int early_stopping(double *means, struct cluster *clusters, double error, int k) {
//    NUM_ADDS(k*k);
//    for (int i = 0; i < k; i++) { // iterate over cluster
//        for (int j = 0; j < k; j++) { // iterate over each dimension of the mean
//            if (fabs(means[i*k+j] - clusters[i].mean[j]) > error) {
//                return 0;
//            }
//        }
//    }
//    return 1;
//}


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
void kmeans(double *U, int n, int k, int max_iter, double stopping_error, struct cluster *ret) {
    // k is the number of columns in U matrix  U is a n by k matrix (here only!)
    int i = 0;
    // each row represents a cluster each column a dimension
    double means[k*k];
    while (i < max_iter) {
        (i == 0) ? init_kpp(&U[0], n, k, means) : update_means(U, ret, k, n, means);
        // check if the means are stable, if yes => stop
        if (i > 0) {
//            if (early_stopping(means, ret, stopping_error, k)) {
//                break;
//            }
            // update means double array (aims to store previous means)
            copy_means(ret, k, means);
        }

        // post condition: means is up-to-date
        map_to_nearest_cluster(U, n, k, means, ret);

        i++;
    }
    // print clusters: Cluster i : (1,2) (4,5) etc.
    for (int j = 0; j < k; j++) {
        // printf("Cluster %d: ", j);
        for(int e = 0; e < ret[j].size; e++) {
            // printf("( ");
            for (int f = 0; f < n; f++) {
                // printf("%lf ", U[ret[j].indices[e]*n+f]);
            }
            // printf(")  ");
        }
        // printf("\n");
    }
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