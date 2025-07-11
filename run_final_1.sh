#!/bin/bash

# first half of the datasets
datasets=("cauchy" "chebspec" "chow" "circul" "clement" "companion" "dingdong" "fiedler" "forsythe" "frank" "golub" "grcar" "hankel" "hilb" "kahan" "kms")

instance_sizes=(5000 10000 20000)

restarts=5
gemmrestarts=1

# build the benchmark executable
nvcc -std=c++17 -O3 -lcublas -lcudart -lcusolver -Wno-deprecated-gpu-targets -o benchmark benchmark_full.cu || exit 1

# run the benchmark
./benchmark --datasets "${datasets[@]}" --instance_sizes "${instance_sizes[@]}" --restarts "$restarts" --gemmrestarts "$gemmrestarts" --csv_name "./results/psd_final_1.csv" > output_final_1.txt 2>&1