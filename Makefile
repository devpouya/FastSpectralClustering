UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
CC := gcc
CINCLUDES :=
CLIBS := -L/usr/lib/x86_64-linux-gnu -llapacke -lm
CFLAGS := -O0
endif
ifeq ($(UNAME), Darwin)
CC := clang
CINCLUDES := -I/usr/local/Cellar/openblas/0.3.6_1/include
CLIBS := -L/usr/local/Cellar/openblas/0.3.6_1/lib -lopenblas
CFLAGS := -O0
endif

all: graph.c
	$(CC) $(CFLAGS) $(CINCLUDES) $(CLIBS) -o graph graph.c

.PHONY clean:
	rm -rf graph