#!/bin/bash
# datasets=("hilb" "rando" "sampling" "squared" "cubed")
# # instance_sizes=(10 20 50 100 1000 2000 5000 10000 20000)
# instance_sizes=(10 20 50 100 1000 2000 5000)
# restarts=10
# gemmrestarts=25

# for debug:
datasets=("rohess" "frank" "golub" "hilb" "rando" "sampling" "squared" "cubed" "triw" "magic" "lotkin" "parter" "grcar")
instance_sizes=(5000)
restarts=2
gemmrestarts=1

nvcc -std=c++17 -O3 -lcublas -lcudart -lcusolver -Wno-deprecated-gpu-targets -o benchmark benchmark.cu || exit 1

julia data/generate.jl --datasets "${datasets[@]}" --instance_sizes "${instance_sizes[@]}" || exit 1

./benchmark --datasets "${datasets[@]}" --instance_sizes "${instance_sizes[@]}" --restarts "$restarts" --gemmrestarts "$gemmrestarts"