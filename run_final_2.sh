#!/bin/bash

# second half of the datasets
datasets=("lehmer" "lotkin" "magic" "minij" "moler" "oscillate" "parter" "pei" "prolate" "randcorr" "rando" "rohess" "sampling" "toeplitz" "tridiag" "triw" "wilkinson") # without the removed datasets

instance_sizes=(5000 10000 20000)

restarts=5
gemmrestarts=1

# build the benchmark executable
nvcc -std=c++17 -O3 -lcublas -lcudart -lcusolver -Wno-deprecated-gpu-targets -o benchmark benchmark_full.cu || exit 1

# run the benchmark
./benchmark --datasets "${datasets[@]}" --instance_sizes "${instance_sizes[@]}" --restarts "$restarts" --gemmrestarts "$gemmrestarts" --csv_name "./results/psd_final_2.csv" > output_final_2.txt 2>&1