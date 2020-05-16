#!/bin/bash

rm -r datasets/perf_data/*

i=200
while [[ $i -lt 3000 ]]; do
    echo $i
    python3 ./scripts/generate_gaussian.py 2 $i "./datasets/perf_data/${i}.txt"; i=`expr $i + 200`
done
