UNAME := $(shell uname)
PWD := $(shell pwd)

CFLAGS := -O3 -ffast-math -Wall -Werror -Wno-unused-result
CINCLUDES := -I$(PWD)/arpack-ng/ICB

ifeq ($(UNAME), Linux)
	CC := gcc
	CLIBS := -L/usr/lib/x86_64-linux-gnu/lib -lopenblas -llapacke -lgfortran -lm
endif
ifeq ($(UNAME), Darwin)
	CC := gcc-9
	CINCLUDES += -I/usr/local/Cellar/openblas/0.3.9/include
	CLIBS := -L/usr/local/Cellar/openblas/0.3.9/lib -lopenblas -lgfortran
endif

SRC := main.c init.c norms.c construct_graph.c kmeans.c util.c instrumentation.c eig.c

all: clustering

clustering: $(SRC)
	$(CC) $(CFLAGS) -o clustering $(SRC) arpack-ng/libarpack.a $(CINCLUDES) $(CLIBS)

base_clustering: $(SRC)
	$(CC) $(CFLAGS) -DSEED=30 -o base_clustering $(SRC) arpack-ng/libarpack.a $(CINCLUDES) $(CLIBS)

validation: $(SRC)
	$(CC) $(CFLAGS) -DSEED=30 -DVALIDATION -o validation $(SRC) arpack-ng/libarpack.a $(CINCLUDES) $(CLIBS)

profiling: $(SRC)
	$(CC) $(CFLAGS) -DPROFILING -DINSTRUMENTATION -o profiling $(SRC) arpack-ng/libarpack.a $(CINCLUDES) $(CLIBS)

countops: $(SRC)
	$(CC) $(CFLAGS) -DINSTRUMENTATION -o countops $(SRC) arpack-ng/libarpack.a $(CINCLUDES) $(CLIBS)

.PHONY: bootstrap-arpack
bootstrap-arpack:
	git submodule init && git submodule update && cd arpack-ng && sh bootstrap && cd ..

.PHONY: configure-arpack
configure-arpack:
	cd arpack-ng && ./configure --enable-icb && cd ..

.PHONY: arpack
arpack:
	cd arpack-ng && cmake -DICB=ON && make && cp arpackdef.h ICB/arpackdef.h && cd ..

.PHONY: clean
clean:
	rm -rf clustering validation profiling