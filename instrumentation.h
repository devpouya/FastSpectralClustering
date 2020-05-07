#ifndef _INSTRUMENTATION_H
#define _INSTRUMENTATION_H

#include <stdio.h>
#include <stdint.h>
#include "tsc_x86.h"

#ifdef INSTRUMENTATION

#define NUM_ADDS(n) num_adds(n)
#define NUM_MULS(n) num_muls(n)
#define NUM_DIVS(n) num_divs(n)
#define NUM_SQRTS(n) num_sqrts(n)
#define NUM_EXPS(n) num_exps(n)
#define NUM_FLOPS num_flops()

void num_adds(int n);
void num_muls(int n);
void num_divs(int n);
void num_sqrts(int n);
void num_exps(int n);
uint64_t num_flops(void);

#else
#define NUM_ADDS(n) ((void) 0)
#define NUM_MULS(n) ((void) 0)
#define NUM_DIVS(n) ((void) 0)
#define NUM_SQRTS(n) ((void) 0)
#define NUM_EXPS(n) ((void) 0)
#define NUM_FLOPS ((uint64_t) 0)
#endif

#ifdef PROFILING

#ifndef PROFILING_NUM_FUNCS
#define PROFILING_NUM_FUNCS 128
#endif

extern const char *__func_names[PROFILING_NUM_FUNCS];
extern uint64_t __func_times[PROFILING_NUM_FUNCS];
extern int __curr_idx;

#define ENTER_FUNC \
static int __profiler_ready = 0; \
static int __func_index; \
if (!__profiler_ready) { \
    __func_index = ++__curr_idx; \
    __func_names[__func_index] = __func__; \
    __profiler_ready = 1; \
} \
uint64_t __func_start = start_tsc()

#define EXIT_FUNC __func_times[__func_index] += stop_tsc(__func_start)
#define PROFILER_LIST() __profiler_list()

void __profiler_list(void);

#else

#define ENTER_FUNC ((void) 0)
#define EXIT_FUNC ((void) 0)
#define PROFILER_LIST() ((void) 0)

#endif

#endif