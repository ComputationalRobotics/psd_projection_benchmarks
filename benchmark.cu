#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <chrono>

void load_matrix(const std::string& filename, std::vector<double>& data, const int64_t instance_size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file\n";
        throw std::runtime_error("Cannot open file");
    }

    data.resize(instance_size * instance_size);
    file.read(reinterpret_cast<char*>(data.data()), instance_size * instance_size * sizeof(double));
}

// Check if the function returns a CUDA error
#define CHECK_CUDA(func)                                                       \
do {                                                                           \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at %s:%d with error: %s (%d)",                 \
               __FILE__, __LINE__, cudaGetErrorString(status), status);        \
        std::cout << std::endl;                                                \
    }                                                                          \
} while (0) // wrap it in a do-while loop to be called with a semicolon

// Check if the function returns a cuBLAS error
#define CHECK_CUBLAS(func)                                                     \
do {                                                                           \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("cuBLAS error %d at %s:%d", status, __FILE__, __LINE__);        \
        std::cout << std::endl;                                                \
    }                                                                          \
} while (0)

// Check if the function returns a cuSPARSE error
#define CHECK_CUSOLVER(func)                                                   \
do {                                                                           \
    cusolverStatus_t status = (func);                                          \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
        printf("cuSOLVER error %d at %s:%d", status, __FILE__, __LINE__);      \
        std::cout << std::endl;                                                \
    }                                                                          \
} while (0)

// Check if the function returns a cuSPARSE error
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("cuSPARSE error %s (%d) at %s:%d",                              \
              cusparseGetErrorString(status), status, __FILE__, __LINE__);     \
        std::cout << std::endl;                                                \
    }                                                                          \
}

std::chrono::duration<double> cusolver_FP64_psd( cusolverDnHandle_t solverH, cublasHandle_t cublasH, double* dA, double* dA_psd, size_t n) {
    int *devInfo; CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));
    size_t nn = n * n;
    double one_d = 1.0;
    double zero_d = 0.0;

    auto start = std::chrono::high_resolution_clock::now();
    double *dW; CHECK_CUDA(cudaMalloc(&dW, n*sizeof(double)));
    int lwork_ev = 0;
    CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(
        solverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
        n, dA, n, dW, &lwork_ev));
    double *dWork_ev; CHECK_CUDA(cudaMalloc(&dWork_ev, lwork_ev*sizeof(double)));
    CHECK_CUSOLVER(cusolverDnDsyevd(
        solverH,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        n, dA, n, dW,
        dWork_ev, lwork_ev, devInfo));
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<double> W_h(n);
    CHECK_CUDA(cudaMemcpy(W_h.data(), dW, n*sizeof(double), cudaMemcpyDeviceToHost));
    for(int i=0;i<n;i++) if(W_h[i]<0) W_h[i]=0;

    // Copy eigenvectors from dA to dV
    double *dV; CHECK_CUDA(cudaMalloc(&dV, nn*sizeof(double)));
    CHECK_CUDA(cudaMemcpy(dV, dA, nn*sizeof(double), cudaMemcpyDeviceToDevice));

    // Scale columns of dV by W_h
    for(int i=0;i<n;i++){
        CHECK_CUBLAS(cublasDscal(cublasH, n, &W_h[i], dV + i*n, 1));
    }

    // Reconstruct A_psd = V * V^T
    double *dTmp; CHECK_CUDA(cudaMalloc(&dTmp, nn*sizeof(double)));
    CHECK_CUBLAS(cublasGemmEx(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
        n, n, n,
        &one_d,
        dV, CUDA_R_64F, n,
        dA, CUDA_R_64F, n,
        &zero_d,
        dTmp, CUDA_R_64F, n,
        CUDA_R_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaMemcpy(dA_psd, dTmp, nn*sizeof(double), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaFree(dTmp));
    CHECK_CUDA(cudaFree(dV));
    auto end = std::chrono::high_resolution_clock::now();
    return end - start;
}

int main(int argc, char* argv[]) {
    std::vector<std::string> datasets;
    std::vector<size_t> instance_sizes;
    int restarts = 1;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--datasets") {
            ++i;
            while (i < argc && argv[i][0] != '-') {
                datasets.push_back(argv[i]);
                ++i;
            }
            --i;
        } else if (arg == "--instance_sizes") {
            ++i;
            while (i < argc && argv[i][0] != '-') {
                instance_sizes.push_back(std::stoul(argv[i]));
                ++i;
            }
            --i;
        } else if (arg == "--restarts") {
            if (i + 1 < argc) {
                restarts = std::stoi(argv[++i]);
            }
        }
    }

    /* Initialize data and handles */
    std::vector<double> data;
    cusolverDnHandle_t solverH;
    CHECK_CUSOLVER(cusolverDnCreate(&solverH));
    cublasHandle_t cublasH;
    CHECK_CUBLAS(cublasCreate(&cublasH));

    /* Warmup the GPU */
    std::cout << "Warming up the GPU...";
    for (int i = 0; i < restarts; ++i) {
        double *dA, *dB, *dC;
        size_t n = 1024; // Use a fixed size for warmup
        double one_d = 1.0;
        double zero_d = 0.0;
        CHECK_CUDA(cudaMalloc(&dA, n * n * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&dB, n * n * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&dC, n * n * sizeof(double)));
        // Initialize dA and dB with ones
        CHECK_CUDA(cudaMemset(dA, 1, n * n * sizeof(double)));
        CHECK_CUDA(cudaMemset(dB, 1, n * n * sizeof(double)));
        CHECK_CUBLAS(cublasDgemm(
            cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one_d, dA, n, dB, n,
            &zero_d, dC, n));
        CHECK_CUDA(cudaFree(dA));
        CHECK_CUDA(cudaFree(dB));
        CHECK_CUDA(cudaFree(dC));
    }
    std::cout << " done." << std::endl << std::endl;

    /* Main benchmarking loop */
    for (const auto& dataset : datasets) {
        for (const auto& n : instance_sizes) {
            /* 0) Generate the matrix */
            std::cout << "DATASET '" << dataset << "' WITH INSTANCE SIZE " << n << std::endl;

            // load the matrix from the generated binary file
            std::string filename = "data/bin/" + dataset + "-" + std::to_string(n) + ".bin";
            load_matrix(filename, data, n);

            // copy the matrix to the device
            double *A, *A_psd, *A_psd_ref;
            CHECK_CUDA(cudaMalloc(&A,         n * n * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&A_psd,     n * n * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&A_psd_ref, n * n * sizeof(double)));
            CHECK_CUDA(cudaMemcpy(A, data.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

            /* 1) Pure GEMM and EIG times */
            std::cout << "\t Pure EIG and GEMM times" << std::endl;
            // cuSOLVER FP32
            // cuSOLVER FP64 eig
            // TF16
            // TF32
            // FP32
            // FP64

            /* 2) PSD cone projection */
            std::cout << "\t PSD cone projection" << std::endl;

            // cuSOLVER FP64
            std::chrono::duration<double> duration(0.0);
            for (int i = 0; i < restarts; ++i) {
                duration += cusolver_FP64_psd(solverH, cublasH, A, A_psd_ref, n);
            }
            duration /= restarts;
            std::cout << "\t\t cuSOLVER FP64 -- Time: " << std::scientific << duration.count() << " s" << std::endl;
            std::cout << "\t\t        Relative error: 0.0" << std::endl;

            // cuSOLVER FP32
            // composite TF16
            // composite TF32
            // composite FP32
            // composite FP64
        }
    }

    /* Clean up */
    CHECK_CUSOLVER(cusolverDnDestroy(solverH));

    return 0;
}