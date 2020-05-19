#!/bin/bash

out="$1"
rm -f $out

for file in `ls datasets/6c_5000n | sort -n`; do
    i=${file%????}
    echo $i
    info=`./profiling "./datasets/6c_5000n/${file}" 6 /dev/null | tail -2`
    info_array=(${info//:/ })
    runtime=${info_array[0]}
    #flops=${info_array[1]}
    perf=`python3 -c "print($runtime)"`
    echo "$i ${perf}" >> "./output/measurements/"$out
done
