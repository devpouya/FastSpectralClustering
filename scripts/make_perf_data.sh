#!/bin/bash

rm -r datasets/perf_data/
mkdir datasets/perf_data

i=5
while [[ $i -lt 100 ]]; do
    echo $i
    python3 ./scripts/generate_gaussian.py 2 $i "./datasets/perf_data/${i}.txt"; i=`expr $i + 5`
done
