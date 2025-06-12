#!/bin/bash
datasets=("hilb" "rando")
instance_sizes=(10 15)
restarts=10

nvcc -std=c++17 -O3 -lcublas -lcudart -lcusolver -Wno-deprecated-gpu-targets -o benchmark benchmark.cu

julia data/generate.jl --datasets "${datasets[@]}" --instance_sizes "${instance_sizes[@]}"
./benchmark --datasets "${datasets[@]}" --instance_sizes "${instance_sizes[@]}" --restarts "$restarts"