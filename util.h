#ifndef _UTIL_H
#define _UTIL_H

struct file {
    float *points;
    int lines;
    int dimension;
};

struct file alloc_load_points_from_file(char *file);
void copy_submatrix(float *srcmat, int nrows, int ncols, int dest_cols, float *ret);
void print_matrix(char* desc, int m, int n, float* a, int lda);
void copy_submatrix(float *srcmat, int nrows, int ncols, int dest_cols, float *ret);
double  wtime(void);
char* concat(const char *s1, const char *s2);
int compareFile(FILE * fPtr1, FILE * fPtr2, int * line, int * col);

#endif