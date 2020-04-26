reset

set term postscript eps enhanced 14 color
set output 'output/graphs/'.@ARG1
set title "Spectral Clustering on i7-8550U CPU @ 1.80GHz, Ubuntu 18.04"
set ylabel "Performance [flops/cycle]"
set xlabel "n"
set grid ytics
set xrange [0:1000]
set yrange [0:0.02]

plot './output/measurements/'.@ARG2 using 1:2 w lp title 'base performance, no optimization' ps 1, \

