#include "instrumentation.h"

__attribute__((used)) static int __num_adds = 0;
__attribute__((used)) static int __num_muls = 0;
__attribute__((used)) static int __num_divs = 0;
__attribute__((used)) static int __num_sqrts = 0;
__attribute__((used)) static int __num_exps = 0;

void num_adds(int n) { __num_adds += n; }
void num_muls(int n) { __num_muls += n; }
void num_divs(int n) { __num_divs += n; }
void num_sqrts(int n) { __num_sqrts += n; }
void num_exps(int n) { __num_exps += n; }
int num_flops(void) { return __num_adds + __num_muls + __num_divs + __num_sqrts + __num_exps; }
