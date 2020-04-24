#!/bin/bash

out="$1"
rm -f $out

for file in `ls perf_data | sort -n`; do
    i=${file%????}
    echo $i
    info=`./clustering "perf_data/${file}" 2 /dev/null | tail -2`
    info_array=(${info//:/ })
    runtime=${info_array[0]}
    flops=${info_array[1]}
    perf=`python3 -c "print($flops/$runtime)"`
    echo "$i ${perf}" >> $out
done
