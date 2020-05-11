#ifndef _CONSTRUCT_GRAPH_H
#define _CONSTRUCT_GRAPH_H

#define EPS 2

void construct_fully_connected_matrix(double *points, int lines, int dim, double *ret);
void construct_eps_neighborhood_matrix(double *points, int lines, int dim, double *ret);
void construct_normalized_laplacian_sym_matrix(double *weighted_adj_matrix, int num_points, double *ret);
void construct_normalized_laplacian_rw_matrix(double *weighted_adj_matrix, int num_points, double *ret);
void construct_unnormalized_laplacian(double *graph, int n, double *ret);
void construct_knn_matrix(double *points, int lines, int dim, int k, double *ret);
void oneshot_unnormalized_laplacian(double *points, int n, int dim, double *ret);
void oneshot_unnormalized_laplacian_lowdim(double *points, int n, int dim, double *ret);
#endif
