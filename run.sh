#!/bin/bash

# datasets=("binomial" "cauchy" "chebspec" "chow" "circul" "clement" "companion" "dingdong" "fiedler" "forsythe" "frank" "golub" "grcar" "hadamard" "hankel" "hilb" "invhilb" "invol" "kahan" "kms" "lehmer" "lotkin" "magic" "minij" "moler" "neumann" "oscillate" "parter" "pascal" "pei" "poisson" "prolate" "randcorr" "rando" "randsvd" "rohess" "rosser" "sampling" "toeplitz" "tridiag" "triw" "vand" "wathen" "wilkinson") # all datasets

datasets=("cauchy" "chebspec" "chow" "circul" "clement" "companion" "dingdong" "fiedler" "forsythe" "frank" "golub" "grcar" "hankel" "hilb" "kahan" "kms" "lehmer" "lotkin" "magic" "minij" "moler" "oscillate" "parter" "pei" "prolate" "randcorr" "rando" "rohess" "sampling" "toeplitz" "tridiag" "triw" "wilkinson") # without the removed datasets

# removed: binomial (overflow), hadamard (power of 2), invhilb (overflow), neumann (sparse), poisson (sparse), rosser (power of 2), wathen (sparse), invol (overflow), pascal (overflow), vand (overflow), randsvd (timeout)

instance_sizes=(10000)

restarts=1
gemmrestarts=1

# build the psd_projection library
export LD_LIBRARY_PATH=psd_projection/build:$LD_LIBRARY_PATH
cd psd_projection && cmake -S . -B build && cmake --build build && cd .. || exit 1

# build the benchmark executable
nvcc -std=c++17 -O3 -L. -Lpsd_projection/build -lpsd_lib -Ipsd_projection/include -lcublas -lcudart -lcusolver -Wno-deprecated-gpu-targets -o benchmark benchmark.cu || exit 1

# generate the datasets (optional)
# julia data/generate.jl --datasets "${datasets[@]}" --instance_sizes "${instance_sizes[@]}" || exit 1
# note that the datasets were generated in parallel, 
# and you may not get the same results if you generate them yourself

# run the benchmark
./benchmark --datasets "${datasets[@]}" --instance_sizes "${instance_sizes[@]}" --restarts "$restarts" --gemmrestarts "$gemmrestarts"