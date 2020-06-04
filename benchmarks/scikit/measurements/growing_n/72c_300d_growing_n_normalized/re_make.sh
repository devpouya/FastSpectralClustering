#!/bin/sh
python3 fastcodeTemplate.py --ylabel seconds --t "Runtime comparison for kmeans" data_kmeans
python3 fastcodeTemplate.py --ylabel seconds --t "Runtime comparison for graph and Laplacian construction" data_graph
