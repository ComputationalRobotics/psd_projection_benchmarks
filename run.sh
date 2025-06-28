#!/bin/bash
# datasets=("hilb" "rando" "sampling" "squared" "cubed")
# datasets=("sampling")
# instance_sizes=(50 100 1000 2000 5000 10000 20000)
# instance_sizes=(50 100 1000)
# instance_sizes=(100)

# datasets=("rohess" "frank" "golub" "hilb" "rando" "sampling" "squared" "cubed" "triw" "magic" "lotkin" "parter" "grcar")
# datasets=("rohess" "frank" "golub" "hilb" "rando" "sampling" "squared" "cubed" "triw" "magic" "lotkin" "parter" "grcar")
# instance_sizes=(5000)

# full tests
# datasets=("binomial" "cauchy" "chebspec" "chow" "circul" "clement" "companion" "dingdong" "fiedler" "forsythe" "frank" "golub" "grcar" "hadamard" "hankel" "hilb" "invhilb" "invol" "kahan" "kms" "lehmer" "lotkin" "magic" "minij" "moler" "neumann" "oscillate" "parter" "pascal" "pei" "poisson" "prolate" "randcorr" "rando" "randsvd" "rohess" "rosser" "sampling" "toeplitz" "tridiag" "triw" "vand" "wathen" "wilkinson")
# datasets=("cauchy" "chebspec" "chow" "circul" "clement" "companion" "dingdong" "fiedler" "forsythe" "frank" "golub" "grcar" "hankel" "hilb" "kahan" "kms" "lehmer" "lotkin" "magic" "minij" "moler" "oscillate" "parter" "pei" "prolate" "randcorr" "rando" "randsvd" "rohess" "sampling" "toeplitz" "tridiag" "triw" "wilkinson")

datasets=("circul" "clement" "companion" "dingdong" "fiedler" "forsythe" "frank" "golub" "grcar" "hankel" "hilb" "kahan" "kms" "lehmer" "lotkin" "magic" "minij" "moler" "oscillate" "parter" "pei" "prolate" "randcorr" "rando" "randsvd" "rohess" "sampling" "toeplitz" "tridiag" "triw" "wilkinson" "cauchy" "chebspec" "chow")

# removed: binomial (overflow), hadamard (power of 2), invhilb (overflow), neumann (sparse), poisson (sparse), rosser (power of 2), wathen (sparse), invol (overflow), pascal (overflow), vand (overflow)

# # for debug
# datasets=("parter")

instance_sizes=(1000 2000 5000)

restarts=1
gemmrestarts=1

nvcc -std=c++17 -O3 -lcublas -lcudart -lcusolver -Wno-deprecated-gpu-targets -o benchmark benchmark.cu || exit 1

# julia data/generate.jl --datasets "${datasets[@]}" --instance_sizes "${instance_sizes[@]}" || exit 1

./benchmark --datasets "${datasets[@]}" --instance_sizes "${instance_sizes[@]}" --restarts "$restarts" --gemmrestarts "$gemmrestarts"