UNAME := $(shell uname)
PWD := $(shell pwd)

CFLAGS := -O3 -ffast-math -march=skylake -mfma -Wall -Werror -Wno-unused-result
CXXFLAGS := -O3 -ffast-math -march=skylake -mfma -Wall -Werror -Wno-unused-result
CINCLUDES := -I$(PWD)/arpack-ng/ICB
CXXINCLUDES := -I$(PWD)/spectra/include

ifeq ($(UNAME), Linux)
	CC := gcc
	CXX := g++
	CLIBS := -L/usr/lib/x86_64-linux-gnu/lib -lopenblas -llapacke -lgfortran -lm -lstdc++
	CXXINCLUDES += -I /usr/include/eigen3
endif
ifeq ($(UNAME), Darwin)
	CC := gcc-9
	CXX := g++-9
	CINCLUDES += -I/usr/local/Cellar/openblas/0.3.9/include
	CLIBS := -L/usr/local/Cellar/openblas/0.3.9/lib -lopenblas -lgfortran -lstdc++
	CXXINCLUDES +=  -I/usr/local/Cellar/eigen/3.3.7/include/eigen3
endif

SRC := main.c init.c norms.c construct_graph.c kmeans_elkan.c kmeans_hamerly.c kmeans_lloyd.c util.c instrumentation.c
ifeq ($(EIGS_SOLVER), arpack)
	EIGS := eigs_arpack.c arpack-ng/libarpack.a
else ifeq ($(EIGS_SOLVER), lapack)
	EIGS := eigs_lapack.c
else
	EIGS := eigs_spectra.o
endif

all: clustering

clustering: $(SRC) $(EIGS)
	$(CC) $(CFLAGS) -o clustering $(SRC) $(EIGS) $(CINCLUDES) $(CLIBS)

base_clustering: $(SRC) $(EIGS)
	$(CC) $(CFLAGS) -DSEED=30 -o base_clustering $(SRC) $(EIGS) $(CINCLUDES) $(CLIBS)

validation: $(SRC) $(EIGS)
	$(CC) $(CFLAGS) -DSEED=30 -DVALIDATION -o validation $(SRC) $(EIGS) $(CINCLUDES) $(CLIBS)

profiling: $(SRC) $(EIGS)
	$(CC) $(CFLAGS) -DPROFILING -DINSTRUMENTATION -o profiling $(SRC) $(EIGS) $(CINCLUDES) $(CLIBS)

countops: $(SRC) $(EIGS)
	$(CC) $(CFLAGS) -DINSTRUMENTATION -o countops $(SRC) $(EIGS) $(CINCLUDES) $(CLIBS)

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
	rm -rf clustering validation profiling countops eigs_spectra.o

.PHONY: init-spectra
init-spectra:
	git submodule init && git submodule update

eigs_spectra.o: eigs_spectra.cpp
	$(CXX) $(CXXFLAGS) -o eigs_spectra.o -c eigs_spectra.cpp $(CXXINCLUDES) 
