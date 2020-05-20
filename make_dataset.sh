#!/bin/bash

out="${1}"

for ((n=100;n<=6000;n+=100)); do
    for ((dim=300;dim<=300;dim++)); do
        for ((k=72;k<=72;k++)); do
            python3 scripts/generate_gaussian.py $k $n "${out}/${n}.txt" $dim
        done
    done
done