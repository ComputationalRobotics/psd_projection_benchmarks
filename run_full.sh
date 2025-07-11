#!/bin/bash

datasets=("cauchy" "chebspec" "chow" "circul" "clement" "companion" "dingdong" "fiedler" "forsythe" "frank" "golub" "grcar" "hankel" "hilb" "kahan" "kms" "lehmer" "lotkin" "magic" "minij" "moler" "oscillate" "parter" "pei" "prolate" "randcorr" "rando" "rohess" "sampling" "toeplitz" "tridiag" "triw" "wilkinson") # without the removed datasets

instance_sizes=(5000 10000 20000)

restarts=1
gemmrestarts=1

# build the benchmark executable
nvcc -std=c++17 -O3 -lcublas -lcudart -lcusolver -Wno-deprecated-gpu-targets -o benchmark benchmark_full.cu || exit 1

# run the benchmark
./benchmark --datasets "${datasets[@]}" --instance_sizes "${instance_sizes[@]}" --restarts "$restarts" --gemmrestarts "$gemmrestarts" --csv_name "./results/psd_results.csv" > output.txt 2>&1