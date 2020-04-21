UNAME := $(shell uname)

CFLAGS := -O0 -Wall -Werror -DINSTRUMENTATION

ifeq ($(UNAME), Linux)
	CC := gcc
	CINCLUDES :=
	CLIBS := -L/usr/lib/x86_64-linux-gnu -llapacke -lm
endif
ifeq ($(UNAME), Darwin)
	CC := clang
	CINCLUDES := -I/usr/local/Cellar/openblas/0.3.9/include
	CLIBS := -L/usr/local/Cellar/openblas/0.3.9/lib -lopenblas
endif

all: main.c norms.c construct_graph.c kmeans.c util.c
	$(CC) $(CFLAGS) -o clustering main.c norms.c construct_graph.c kmeans.c util.c instrumentation.c $(CINCLUDES) $(CLIBS) 

.PHONY clean:
	rm -rf clustering