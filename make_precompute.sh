#!/bin/bash

make

for file in `ls datasets/perf_blobs/growing_k | sort -n`; do
    k=${file%.*}
    ./dump_ev "datasets/perf_blobs/growing_k/${file}" $k /dev/null
done

