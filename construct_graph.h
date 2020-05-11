#ifndef _CONSTRUCT_GRAPH_H
#define _CONSTRUCT_GRAPH_H

#define EPS 2

void construct_fully_connected_matrix(float *points, int lines, int dim, float *ret);
void construct_eps_neighborhood_matrix(float *points, int lines, int dim, float *ret);
void construct_normalized_laplacian_sym_matrix(float *weighted_adj_matrix, int num_points, float *ret);
void construct_normalized_laplacian_rw_matrix(float *weighted_adj_matrix, int num_points, float *ret);
void construct_unnormalized_laplacian(float *graph, int n, float *ret);
void construct_knn_matrix(float *points, int lines, int dim, int k, float *ret);
void oneshot_unnormalized_laplacian(float *points, int n, int dim, float *ret);
#endif
