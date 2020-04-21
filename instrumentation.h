#ifndef _INSTRUMENTATION_H
#define _INSTRUMENTATION_H

#include <stdio.h>

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
int num_flops(void);

#else
#define NUM_ADDS(n) ((void) 0)
#define NUM_MULS(n) ((void) 0)
#define NUM_DIVS(n) ((void) 0)
#define NUM_SQRTS(n) ((void) 0)
#define NUM_EXPS(n) ((void) 0)
#define NUM_FLOPS 0
#endif

#endif