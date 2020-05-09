#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

/***
 * Inspired by http://www.mymathlib.com
 * */
void copy_submatrix(double *srcmat, int nrows, int ncols, int dest_cols, double *ret) {
    int numb_bytes = sizeof(double) * dest_cols;

    for(srcmat+=0; nrows > 0; srcmat += ncols, ret+=dest_cols, nrows--) {
        memcpy(ret,srcmat,numb_bytes);
    }
}
char* concat(const char *s1, const char *s2)
{
    char *result = malloc(strlen(s1) + strlen(s2) + 1); // +1 for the null-terminator
    // in real code you would check for errors in malloc here
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

/**
 * function from https://codeforwin.org/2018/02/c-program-compare-two-files.html
 * Function to compare two files.
 * Returns 0 if both files are equivalent, otherwise returns
 * -1 and sets line and col where both file differ.
 */
int compareFile(FILE * fPtr1, FILE * fPtr2, int * line, int * col)
{
    char ch1, ch2;
    *line = 1;
    *col  = 0;
    do
    {
        // Input character from both files
        ch1 = fgetc(fPtr1);
        ch2 = fgetc(fPtr2);
        // Increment line
        if (ch1 == '\n')
        {
            *line += 1;
            *col = 0;
        }
        // If characters are not same then return -1
        if (ch1 != ch2)
            return -1;
        *col  += 1;
    } while (ch1 != EOF && ch2 != EOF);
    /* If both files have reached end */
    if (ch1 == EOF && ch2 == EOF)
        return 0;
    else
        return -1;
}