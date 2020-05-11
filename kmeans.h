#ifndef _KMEANS_H
#define _KMEANS_H

struct cluster {
    float *mean; // center of the cluster
    int size; // size of cluster points
    int *indices; // stores the indices of the points of U (stored row-ise)
};

void lloyd_kmeans(float *U, int n, int k, int max_iter, float stopping_error, struct cluster *ret);
void hamerly_kmeans(float *U, int n, int k, int max_iter, float stopping_error, struct cluster *ret);
void elkan_kmeans(float *U, int n, int k, int max_iter, float stopping_error, struct cluster *ret);

void lloyd_kmeans_lowdim(float *U, int n, int k, int max_iter, float stopping_error, struct cluster *ret);
void hamerly_kmeans_lowdim(float *U, int n, int k, int max_iter, float stopping_error, struct cluster *ret);
void elkan_kmeans_lowdim(float *U, int n, int k, int max_iter, float stopping_error, struct cluster *ret);

void print_cluster_indices(struct cluster *clusters, int num_clusters);


#endif