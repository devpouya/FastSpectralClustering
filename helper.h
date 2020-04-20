struct file {
    double *points;
    int lines;
    int dimension;
};

void repeat_str(const char *str, int times, char *ret) {
    int len = strlen(str);
    printf("%d\n", len);
    for (int i = 0; i < times; i++) {
        strncpy(ret + i*len, str, len);
    }
    ret[len*times] = '\0';
}

struct file load_file(char *file) {
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
    // char fmt[4*dim + 1];
    // repeat_str("%lf ", dim, fmt);
    // fmt[4*dim-1] = '\n';
    // fmt[4*dim] = '\0';
    printf("Dimension = %d \n" fmt);
    double points[lines][dim];
    for (int i = 0; i < lines; ++i) {
        for (int j = 0; j < dim; ++j){
            fscanf(fp, "%lf", &points[i][j])
        }
    }
    struct file f;
    f.points = *points;
    f.dimension = dim;
    f.lines = lines;
    return f;
}
