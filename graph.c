#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

int main(int argc, char *argv[]) {
    FILE *fp;
    fp = fopen("points.txt", "r");
    int lines = 0;
    while (EOF != (fscanf(fp, "%*[^\n]"), fscanf(fp,"%*c")))
        ++lines;
    printf("lines: %d\n", lines);
    rewind(fp);
    double points[lines][2];
    for (int i = 0; i < lines; ++i) {
        fscanf(fp, "%lf %lf", &points[i][0], &points[i][1]);
    }
    for (int i = 0; i < lines; ++i) {
        printf("%lf %lf\n", points[i][0], points[i][1]);
    }
    // fully-connected matrix
    double fully_connected[lines][lines];
    for (int i = 0; i < lines; ++i) {
        for (int j = 0; j < lines; ++j) {
            fully_connected[i][j] = sqrt((points[i][0] - points[j][0])*(points[i][0] - points[j][0]) + (points[i][1] - points[j][1])*(points[i][1] - points[j][1]));
            printf("%lf ", fully_connected[i][j]);
        }
        printf("\n");
    }
    // epsilon neighborhood matrix
#define EPS 2
    int eps_neighborhood[lines][lines];
    for (int i = 0; i < lines; i++) {
        for (int j = 0; j < lines; j++) {
            eps_neighborhood[i][j] = sqrt((points[i][0] - points[j][0])*(points[i][0] - points[j][0]) + (points[i][1] - points[j][1])*(points[i][1] - points[j][1])) < EPS;
            printf("%d ", eps_neighborhood[i][j]);
        }
        printf("\n");
    }
    // Skip KNN matrix since too annoying to compute
    return 0;
}