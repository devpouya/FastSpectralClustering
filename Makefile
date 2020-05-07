UNAME := $(shell uname)

CFLAGS := -O3 -ffast-math -Wall -Werror -Wno-unused-result

ifeq ($(UNAME), Linux)
	CC := gcc
	CINCLUDES :=
	CLIBS := -L/usr/lib/x86_64-linux-gnu/lib -llapacke -lm
endif
ifeq ($(UNAME), Darwin)
	CC := gcc-9
	CINCLUDES := -I/usr/local/Cellar/openblas/0.3.9/include
	CLIBS := -L/usr/local/Cellar/openblas/0.3.9/lib -lopenblas
endif

all: main.c norms.c construct_graph.c kmeans.c util.c instrumentation.c
	$(CC) $(CFLAGS) -o clustering main.c init.c norms.c construct_graph.c kmeans.c util.c instrumentation.c $(CINCLUDES) $(CLIBS)

base: main.c norms.c construct_graph.c kmeans.c util.c instrumentation.c
	$(CC) $(CFLAGS) -DSEED=30 -o base_clustering main.c init.c norms.c construct_graph.c kmeans.c util.c instrumentation.c $(CINCLUDES) $(CLIBS)

validation: main.c norms.c construct_graph.c kmeans.c util.c instrumentation.c
	$(CC) $(CFLAGS) -DSEED=30 -DVALIDATION -o clustering main.c init.c norms.c construct_graph.c kmeans.c util.c instrumentation.c $(CINCLUDES) $(CLIBS)

profiling: main.c norms.c construct_graph.c kmeans.c util.c instrumentation.c
	$(CC) $(CFLAGS) -DPROFILING -DINSTRUMENTATION -o clustering main.c init.c norms.c construct_graph.c kmeans.c util.c instrumentation.c $(CINCLUDES) $(CLIBS)

.PHONY clean:
	rm -rf clustering