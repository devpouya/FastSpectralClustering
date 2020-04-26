#!/bin/bash

i=0
while [[ $i -lt 10000 ]]; do
    echo $i
    python3 generate_gaussian.py 2 $i "perf_data/${i}.txt"; i=`expr $i + 50`
done
