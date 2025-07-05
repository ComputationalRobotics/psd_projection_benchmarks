#!/bin/bash
# datasets=("hilb" "rando" "sampling" "squared" "cubed")
# datasets=("sampling")
# instance_sizes=(50 100 1000 2000 5000 10000 20000)
# instance_sizes=(50 100 1000)

# datasets=("rohess" "frank" "golub" "hilb" "rando" "sampling" "squared" "cubed" "triw" "magic" "lotkin" "parter" "grcar")

### full tests
# datasets=("binomial" "cauchy" "chebspec" "chow" "circul" "clement" "companion" "dingdong" "fiedler" "forsythe" "frank" "golub" "grcar" "hadamard" "hankel" "hilb" "invhilb" "invol" "kahan" "kms" "lehmer" "lotkin" "magic" "minij" "moler" "neumann" "oscillate" "parter" "pascal" "pei" "poisson" "prolate" "randcorr" "rando" "randsvd" "rohess" "rosser" "sampling" "toeplitz" "tridiag" "triw" "vand" "wathen" "wilkinson") # all datasets

# datasets=("cauchy" "chebspec" "chow" "circul" "clement" "companion" "dingdong" "fiedler" "forsythe" "frank" "golub" "grcar" "hankel" "hilb" "kahan" "kms" "lehmer" "lotkin" "magic" "minij" "moler" "oscillate" "parter" "pei" "prolate" "randcorr" "rando" "rohess" "sampling" "toeplitz" "tridiag" "triw" "wilkinson") # without the removed datasets

# removed: binomial (overflow), hadamard (power of 2), invhilb (overflow), neumann (sparse), poisson (sparse), rosser (power of 2), wathen (sparse), invol (overflow), pascal (overflow), vand (overflow), randsvd (timeout)

datasets=("kahan") 

# datasets=("cauchy" "chebspec" "chow" "circul" "clement" "companion" "dingdong" "fiedler" "forsythe")

# instance_sizes=(5000 10000)
instance_sizes=(5000)

restarts=20
gemmrestarts=1

# build the psd_projection library
# export LD_LIBRARY_PATH=psd_projection/build:$LD_LIBRARY_PATH
# cd psd_projection && cmake -S . -B build && cmake --build build && cd .. || exit 1

# build the benchmark executable
nvcc -std=c++17 -O3 -lcublas -lcudart -lcusolver -Wno-deprecated-gpu-targets -o benchmark benchmark_full.cu || exit 1

# generate the datasets
# julia data/generate.jl --datasets "${datasets[@]}" --instance_sizes "${instance_sizes[@]}" || exit 1

# run the benchmark
./benchmark --datasets "${datasets[@]}" --instance_sizes "${instance_sizes[@]}" --restarts "$restarts" --gemmrestarts "$gemmrestarts" --csv_name "./results/psd_results.csv" > output.txt 2>&1