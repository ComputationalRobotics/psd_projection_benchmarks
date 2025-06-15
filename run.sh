#!/bin/bash
datasets=("hilb" "rando")
instance_sizes=(10 20)
restarts=10
gemmrestarts=25

nvcc -std=c++17 -O3 -lcublas -lcudart -lcusolver -Wno-deprecated-gpu-targets -o benchmark benchmark.cu || exit 1

julia data/generate.jl --datasets "${datasets[@]}" --instance_sizes "${instance_sizes[@]}" || exit 1
./benchmark --datasets "${datasets[@]}" --instance_sizes "${instance_sizes[@]}" --restarts "$restarts" --gemmrestarts "$gemmrestarts"