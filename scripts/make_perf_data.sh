#!/bin/bash

rm -r datasets/perf_data/
mkdir datasets/perf_data

i=50
while [[ $i -lt 2000 ]]; do
    echo $i
    python3 ./scripts/generate_gaussian.py 2 $i "./datasets/perf_data/${i}.txt"; i=`expr $i + 100`
done
