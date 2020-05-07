#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <sys/time.h>
#include <stdio.h>


#include "instrumentation.h"

// Timer from
double wtime(void){
    double now_time;
    struct timeval  etstart;
    struct timezone tzp;

    if (gettimeofday(&etstart, &tzp) == -1)
        perror("Error: calling gettimeofday() not successful.\n");

    now_time = ((double)etstart.tv_sec) +              /* in seconds */
               ((double)etstart.tv_usec) / 1000000.0;  /* in microseconds */
    return now_time;
}

#ifdef INSTRUMENTATION
__attribute__((used)) static uint64_t __num_adds = 0;
__attribute__((used)) static uint64_t __num_muls = 0;
__attribute__((used)) static uint64_t __num_divs = 0;
__attribute__((used)) static uint64_t __num_sqrts = 0;
__attribute__((used)) static uint64_t __num_exps = 0;

void num_adds(int n) { __num_adds += n; }
void num_muls(int n) { __num_muls += n; }
void num_divs(int n) { __num_divs += n; }
void num_sqrts(int n) { __num_sqrts += n; }
void num_exps(int n) { __num_exps += n; }
uint64_t num_flops(void) { return __num_adds + __num_muls + __num_divs + __num_sqrts + __num_exps; }
#endif

#ifdef PROFILING
const char *__func_names[PROFILING_NUM_FUNCS] = {0};
uint64_t __func_times[PROFILING_NUM_FUNCS] = {0};
int __curr_idx = 0;

static uint64_t *TheArray;
static int cmp(const void *a, const void *b) {
    int ia = *(int *)a;
    int ib = *(int *)b;
    return (TheArray[ia] < TheArray[ib]) - (TheArray[ia] > TheArray[ib]);
}

void __profiler_list(void) {
    char *sorted_names[PROFILING_NUM_FUNCS];
    uint64_t sorted_times[PROFILING_NUM_FUNCS];
    int indices[PROFILING_NUM_FUNCS];
    for (int i = 0; i < PROFILING_NUM_FUNCS; i++) {
        indices[i] = i;
    }
    memcpy(sorted_names, __func_names, PROFILING_NUM_FUNCS * sizeof(*sorted_names));
    memcpy(sorted_times, __func_times, PROFILING_NUM_FUNCS * sizeof(*sorted_times));
    TheArray = sorted_times;
    qsort(indices, PROFILING_NUM_FUNCS, sizeof(*indices), cmp);
    printf("\n+-------------- List of function execution times --------------+\n");
    for (int i = 0; i < PROFILING_NUM_FUNCS; ++i) {
        int index = indices[i];
        if (sorted_names[index] == NULL) {
            continue;
        }
        printf("| %-41s %11" PRIu64 " cycles |\n", sorted_names[index], sorted_times[index]);
    }
    printf("+--------------------------------------------------------------+\n\n");
}


#endif
