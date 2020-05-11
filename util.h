#ifndef _UTIL_H
#define _UTIL_H

struct file {
    double *points;
    int lines;
    int dimension;
};

struct file alloc_load_points_from_file(char *file);
void copy_submatrix(double *srcmat, int nrows, int ncols, int dest_cols, double *ret);
void print_matrix(char* desc, int m, int n, double* a, int lda);
void copy_submatrix(double *srcmat, int nrows, int ncols, int dest_cols, double *ret);
double  wtime(void);
char* concat(const char *s1, const char *s2);
int compareFile(FILE * fPtr1, FILE * fPtr2, int * line, int * col);

#endif