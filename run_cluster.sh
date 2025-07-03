#!/bin/bash

# install gdown to download files from Google Drive
python -m pip install gdown

# datasets=("cauchy" "chebspec" "chow" "circul" "clement" "companion" "dingdong" "fiedler" "forsythe" "frank" "golub" "grcar" "hankel" "hilb" "kahan" "kms" "lehmer" "lotkin" "magic" "minij" "moler" "oscillate" "parter" "pei" "prolate" "randcorr" "rando" "rohess" "sampling" "toeplitz" "tridiag" "triw" "wilkinson")
# instance_sizes=(5000 10000 20000)
datasets=("cauchy" "chebspec")
instance_sizes=(50 100)

restarts=1
gemmrestarts=1

# build the psd_projection library
export LD_LIBRARY_PATH=psd_projection/build:$LD_LIBRARY_PATH
cd psd_projection && cmake -S . -B build && cmake --build build && cd .. || exit 1

nvcc -std=c++17 -O3 -L. -Lpsd_projection/build -lpsd_lib -Ipsd_projection/include -lcublas -lcudart -lcusolver -Wno-deprecated-gpu-targets -o benchmark benchmark.cu || exit 1

for id in "${datasets[@]}"; do
    gdown "https://drive.google.com/uc?id=${id}" -O "gdata/${id}.txt"
    ./benchmark --datasets "${id}" --instance_sizes "${instance_sizes[@]}" --restarts "$restarts" --gemmrestarts "$gemmrestarts"
done

