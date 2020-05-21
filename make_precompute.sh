#!/bin/bash

make dump_ev

for file in `ls benchmarks/datasets/72c_300d_growing_n_normalized | sort -n`; do
    # k=${file%.*}
    ./dump_ev "benchmarks/datasets/72c_300d_growing_n_normalized/${file}" 8 /dev/null
done

