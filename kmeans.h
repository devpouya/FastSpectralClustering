#ifndef _KMEANS_H
#define _KMEANS_H

struct cluster {
    double *mean; // center of the cluster
    int size; // size of cluster points
    int *indices; // stores the indices of the points of U (stored row-ise)
};

void lloyd_kmeans(double *U, int n, int k, int max_iter, double stopping_error, struct cluster *ret);
void hamerly_kmeans(double *U, int n, int k, int max_iter, double stopping_error, struct cluster *ret);
void elkan_kmeans(double *U, int n, int k, int max_iter, double stopping_error, struct cluster *ret);

void lloyd_kmeans_lowdim(double *U, int n, int k, int max_iter, double stopping_error, struct cluster *ret);
void hamerly_kmeans_lowdim(double *U, int n, int k, int max_iter, double stopping_error, struct cluster *ret);
void elkan_kmeans_lowdim(double *U, int n, int k, int max_iter, double stopping_error, struct cluster *ret);

void print_cluster_indices(struct cluster *clusters, int num_clusters);


#endif