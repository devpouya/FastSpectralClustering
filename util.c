#include <stdio.h>
#include <stdlib.h>

#include "util.h"

struct file alloc_load_points_from_file(char *file) {
    FILE *fp;
    fp = fopen(file, "r");
    // Count the number of lines in the file
    int lines = 0;
    while (EOF != (fscanf(fp, "%*[^\n]"), fscanf(fp,"%*c")))
        ++lines;
    --lines;  // Subtract one because it starts with the dimension
    printf("Read %d lines\n", lines);
    // Find the dimension
    rewind(fp);
    int dim;
    fscanf(fp, "%d\n", &dim);
    printf("Dimension = %d \n" , dim);
    double *points = malloc(lines * dim * sizeof(double));
    for (int i = 0; i < lines; ++i) {
        for (int j = 0; j < dim; ++j){
            fscanf(fp, "%lf", &points[i*dim + j]);
        }
    }
    struct file f;
    f.points = points;
    f.dimension = dim;
    f.lines = lines;
    return f;
}

/* Auxiliary routine: printing a matrix */
/* copied from intel lapack example: https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_cgeev_row.c.htm */
void print_matrix(char* desc, int m, int n, double* a, int lda) {
    int i, j;
    printf( "\n %s\n", desc );
    for( i = 0; i < m; i++ ) {
        for( j = 0; j < n; j++ )
            printf( " (%6.2f)", a[i*lda+j]);
        printf( "\n" );
    }
}