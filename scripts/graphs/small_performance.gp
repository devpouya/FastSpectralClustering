reset

set terminal pngcairo enhanced
set output 'output/graphs/'.@ARG1
set title "Spectral Clustering (2 clusters), on i7-8550U CPU @ 1.80GHz"
set ylabel "Performance [flops/cycle]"
set xlabel "n (number of datapoints)"
set grid ytics
set xrange [0:200]
set yrange [0:0.25]

plot './output/measurements/'.@ARG2 using 1:2 w lp title 'base performance, no optimization' ps 1, \

