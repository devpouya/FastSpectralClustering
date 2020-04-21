struct file {
    double *points;
    int lines;
    int dimension;
};

struct cluster {
    double *mean; // center of the cluster
    int size; // size of cluster points
    int *indices; // stores the indices of the points of U (stored row-ise)
};

void repeat_str(const char *str, int times, char *ret) {
    int len = strlen(str);
    printf("%d\n", len);
    for (int i = 0; i < times; i++) {
        strncpy(ret + i*len, str, len);
    }
    ret[len*times] = '\0';
}

void print_cluster_indices(struct cluster *clusters, int num_clusters){
    printf("Print clustered point indices\n");
        for (int j = 0; j < num_clusters; j++) {
            printf("Cluster %d: ", j);
                        printf("( ");
            for(int e = 0; e < clusters[j].size; e++) {
                    printf("%d ", clusters[j].indices[e]);
            }
            printf(")  ");
        printf("\n");
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
