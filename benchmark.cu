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
#include <curand_kernel.h>
#include <algorithm>

#define D2H cudaMemcpyDeviceToHost
#define H2D cudaMemcpyHostToDevice
#define D2D cudaMemcpyDeviceToDevice

#define RUN_PURE_TESTS false

#define K_DEFLATE 30 // must be greater than 0, otherwise use non-deflate versions

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

// Kernel: convert double array to float array
__global__ void convert_double_to_float(const double* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<float>(in[idx]);
    }
}

__global__ void convert_float_to_double(const float* in, double* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<double>(in[idx]);
    }
}

void launch_convert_double_to_float(const double* d_in, float* d_out, int n) {
    const int threadsPerBlock = 1024;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    convert_double_to_float<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, n);
    cudaDeviceSynchronize();  // Optional: ensures kernel finishes before returning
}

void launch_convert_float_to_double(const float* d_in, double* d_out, int n) {
    const int threadsPerBlock = 1024;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    convert_float_to_double<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, n);
    cudaDeviceSynchronize();  // Optional: wait for completion
}

std::chrono::duration<double> cusolver_FP64_eig(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dW, double* dA, size_t n) {
    auto start = std::chrono::high_resolution_clock::now();
    int *devInfo; CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));

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

    // cleanup
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(dWork_ev));
    CHECK_CUDA(cudaFree(devInfo));

    auto end = std::chrono::high_resolution_clock::now();
    return end - start;
}

std::chrono::duration<double> cusolver_FP32_eig(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dW, double* dA, size_t n) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t nn = n * n;
    
    int *devInfo; CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));
    // convert dA from double to float
    float *sA;
    CHECK_CUDA(cudaMalloc(&sA, nn*sizeof(float)));
    
    std::vector<double> A_h_d(nn);
    CHECK_CUDA(cudaMemcpy(A_h_d.data(), dA, nn*sizeof(double), D2H));
    std::vector<float> A_h(nn);
    for (size_t i = 0; i < nn; i++) {
        A_h[i] = static_cast<float>(A_h_d[i]);
    }
    CHECK_CUDA(cudaMemcpy(sA, A_h.data(), nn*sizeof(float), H2D));

    float *sW; CHECK_CUDA(cudaMalloc(&sW, n*sizeof(float)));
    int lwork_ev = 0;
    CHECK_CUSOLVER(cusolverDnSsyevd_bufferSize(
        solverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
        n, sA, n, sW, &lwork_ev));
    float *sWork_ev; CHECK_CUDA(cudaMalloc(&sWork_ev, lwork_ev*sizeof(float)));
    CHECK_CUSOLVER(cusolverDnSsyevd(
        solverH,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        n, sA, n, sW,
        sWork_ev, lwork_ev, devInfo));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Convert sA back to double
    std::vector<double> dA_h(nn);
    std::vector<float> sA_h(nn);
    CHECK_CUDA(cudaMemcpy(sA_h.data(), sA, nn*sizeof(float), D2H));
    for (size_t i = 0; i < nn; i++) {
        dA_h[i] = static_cast<double>(sA_h[i]);
    }
    CHECK_CUDA(cudaMemcpy(dA, dA_h.data(), nn*sizeof(double), H2D));

    // Cleanup
    CHECK_CUDA(cudaFree(sWork_ev));
    CHECK_CUDA(cudaFree(sW));
    CHECK_CUDA(cudaFree(sA));
    CHECK_CUDA(cudaFree(devInfo));
    CHECK_CUDA(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    return end - start;
}


std::chrono::duration<double> cusolver_FP64_psd(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dA_psd, size_t n) {
    auto start = std::chrono::high_resolution_clock::now();
    int *devInfo; CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));
    size_t nn = n * n;
    double one_d = 1.0;
    double zero_d = 0.0;

    double *dA; CHECK_CUDA(cudaMalloc(&dA, nn*sizeof(double)));
    CHECK_CUDA(cudaMemcpy(dA, dA_orig, nn*sizeof(double), cudaMemcpyDeviceToDevice));
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
    CHECK_CUDA(cudaFree(dWork_ev));
    CHECK_CUDA(cudaFree(dW));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(devInfo));
    CHECK_CUDA(cudaDeviceSynchronize());

    return std::chrono::high_resolution_clock::now() - start;
}

#include <iomanip>
#include <assert.h>
// Print an n×n matrix of doubles
inline void printMatrixDouble(const double* dM, int n, int m = -1) {
    // If m is not specified, use n for a square matrix
    if (m == -1)
        m = n;

    size_t N = size_t(n) * m;
    std::vector<double> hM(N);
    CHECK_CUDA(cudaMemcpy(hM.data(), dM, N*sizeof(double), cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < m; ++j) {
            std::cout << std::fixed << std::setprecision(6)
                      << hM[j*n + i] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

inline void printMatrixFloat(const float* dM, int n) {
    size_t N = size_t(n)*n;
    std::vector<float> hM(N);
    CHECK_CUDA(cudaMemcpy(hM.data(), dM, N*sizeof(float), cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            std::cout << std::fixed << std::setprecision(6)
                      << hM[i*n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

std::chrono::duration<double> cusolver_FP32_psd(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA, double* dA_psd, size_t n) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t nn = n * n;
    float one_s = 1.0;
    float zero_s = 0.0;
    
    int *devInfo; CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));
    // convert dA from double to float
    float *sA, *sA_psd;
    CHECK_CUDA(cudaMalloc(&sA, nn*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sA_psd, nn*sizeof(float)));
    
    std::vector<double> A_h_d(nn);
    CHECK_CUDA(cudaMemcpy(A_h_d.data(), dA, nn*sizeof(double), D2H));
    std::vector<float> A_h(nn);
    for (size_t i = 0; i < nn; i++) {
        A_h[i] = static_cast<float>(A_h_d[i]);
    }
    CHECK_CUDA(cudaMemcpy(sA, A_h.data(), nn*sizeof(float), H2D));

    float *sW; CHECK_CUDA(cudaMalloc(&sW, n*sizeof(float)));
    int lwork_ev = 0;
    CHECK_CUSOLVER(cusolverDnSsyevd_bufferSize(
        solverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
        n, sA, n, sW, &lwork_ev));
    float *sWork_ev; CHECK_CUDA(cudaMalloc(&sWork_ev, lwork_ev*sizeof(float)));
    CHECK_CUSOLVER(cusolverDnSsyevd(
        solverH,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        n, sA, n, sW,
        sWork_ev, lwork_ev, devInfo));
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> W_h(n);
    CHECK_CUDA(cudaMemcpy(W_h.data(), sW, n*sizeof(float), cudaMemcpyDeviceToHost));
    for(int i=0;i<n;i++) if(W_h[i]<0) W_h[i]=0;

    // Copy eigenvectors from dA to dV
    float *sV; CHECK_CUDA(cudaMalloc(&sV, nn*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(sV, sA, nn*sizeof(float), cudaMemcpyDeviceToDevice));

    // Scale columns of dV by W_h
    for(int i=0;i<n;i++){
        CHECK_CUBLAS(cublasSscal(cublasH, n, &W_h[i], sV + i*n, 1));
    }

    // Reconstruct A_psd = V * V^T
    float *sTmp; CHECK_CUDA(cudaMalloc(&sTmp, nn*sizeof(float)));
    CHECK_CUBLAS(cublasGemmEx(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
        n, n, n,
        &one_s,
        sV, CUDA_R_32F, n,
        sA, CUDA_R_32F, n,
        &zero_s,
        sTmp, CUDA_R_32F, n,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaMemcpy(sA_psd, sTmp, nn*sizeof(float), cudaMemcpyDeviceToDevice));

    // Convert sA_psd back to double
    std::vector<double> dA_psd_h(nn);
    std::vector<float> sA_psd_h(nn);
    CHECK_CUDA(cudaMemcpy(sA_psd_h.data(), sA_psd, nn*sizeof(float), D2H));
    for (size_t i = 0; i < nn; i++) {
        dA_psd_h[i] = static_cast<double>(sA_psd_h[i]);
    }
    CHECK_CUDA(cudaMemcpy(dA_psd, dA_psd_h.data(), nn*sizeof(double), H2D));

    // Cleanup
    CHECK_CUDA(cudaFree(sWork_ev));
    CHECK_CUDA(cudaFree(sW));
    CHECK_CUDA(cudaFree(sA));
    CHECK_CUDA(cudaFree(sA_psd));
    CHECK_CUDA(cudaFree(sTmp));
    CHECK_CUDA(cudaFree(sV));
    CHECK_CUDA(cudaFree(devInfo));
    CHECK_CUDA(cudaDeviceSynchronize());

    return std::chrono::high_resolution_clock::now() - start;
}

inline void symmetrizeFloat(
    cublasHandle_t cublasH, float* M, int n, float* workspace
) {
    const float one = 1.0, half = 0.5, zero = 0.0;

    // workspace = M^T
    CHECK_CUBLAS(cublasSgeam(
        cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
        n, n,
        &one, M, n,
        &zero, M, n,
        workspace, n
    ));

    // M = M + workspace (which is M^T)
    CHECK_CUBLAS(cublasSgeam(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        n, n,
        &one, M, n,
        &one, workspace, n,
        M, n
    ));

    // M = 0.5 * M
    CHECK_CUBLAS(cublasSscal(cublasH, n * n, &half, M, 1));
}

inline void symmetrizeDouble(
    cublasHandle_t cublasH, double* M, int n, double* workspace
) {
    const double one = 1.0, half = 0.5, zero = 0.0;

    // workspace = M^T
    CHECK_CUBLAS(cublasDgeam(
        cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
        n, n,
        &one, M, n,
        &zero, M, n,
        workspace, n
    ));

    // M = M + workspace (which is M^T)
    CHECK_CUBLAS(cublasDgeam(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        n, n,
        &one, M, n,
        &one, workspace, n,
        M, n
    ));

    // M = 0.5 * M
    CHECK_CUBLAS(cublasDscal(cublasH, n * n, &half, M, 1));
}

__global__ void build_identity_kernel(float* mat, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n)
        mat[idx] = (idx / n == idx % n) ? 1.0f : 0.0f;
}

void build_identity(
    cublasHandle_t cublasH,
    float* mat,
    int n
) {
    const int threadsPerBlock = 1024;
    const int blocksPerGrid = (n * n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel to build identity matrix
    build_identity_kernel<<<blocksPerGrid, threadsPerBlock>>>(mat, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

void express_FP32(
    cublasHandle_t cublasH,
    double* mat,
    const int n,
    const int mat_offset
) {
    const int nn = n * n;

    /* Allocations */
    // device memory
    float *A, *A2, *A3;
    CHECK_CUDA( cudaMalloc(&A,  nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&A2, nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&A3, nn * sizeof(float)) );

    // useful constants
    const float half       =  0.5f;
    const float minus_half = -0.5f;
    const float one        =  1.0f;
    const float one_n_half =  1.5f;
    const float zero       =  0.0f;

    /* Convert the initial matrix*/
    launch_convert_double_to_float(mat + mat_offset, A, nn);

    /* Coefficients */
    // std::vector<std::vector<float>> coeff = {
    //     {8.4724206924, -24.5001735687, 17.7268180847},
    //     {4.2052841187, -3.0549299717, 0.5567536354},
    //     {4.0443077087, -2.9473149776, 0.5449726582},
    //     {3.5078327656, -2.5842490196, 0.5067413449},
    //     {2.5075511932, -1.8485442400, 0.4358045161}
    // };
    // const std::vector<std::vector<float>> coeff = { 
    //     { 8.3885353390, -23.7796270883, 16.8664591580 }, 
    //     { 4.1636476423, -2.9650849331, 0.5297319805 }, 
    //     { 4.0042650581, -2.8606348801, 0.5185227850 }, 
    //     { 3.4731017481, -2.5082466382, 0.4821470022 }, 
    //     { 2.4827239537, -1.7941788274, 0.4146530436 }, 
    // }; const size_t smoothing_steps = 3;
    const std::vector<std::vector<float>> coeff = { 
        { 8.5018632351, -24.6330845767, 17.8466614026 },
        { 4.2394319792, -3.0803745982, 0.5596805290 },
        { 4.2371780379, -3.0779047407, 0.5594995022 },
        { 4.1553447421, -3.0255808203, 0.5534594007 },
        { 3.8719053120, -2.8289969308, 0.5331377564 },
        { 3.0503282930, -2.2392300982, 0.4703818765 },
        { 2.1450160790, -1.4976204044, 0.3936105784 }
    }; const size_t smoothing_steps = 4;
    

    /* Approximation of the step function */
    for (int i = 0; i < coeff.size(); i++) {
        const float a = coeff[i][0];
        const float b = coeff[i][1];
        const float c = coeff[i][2];

        /* Compute the powers of A*/
        // A2 = A * A
        CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A, n, A, n, &zero, A2, n) );

        // A3 = A2 * A
        CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A2, n, A, n, &zero, A3, n) );

        // A = a * A
        CHECK_CUBLAS( cublasSscal(cublasH, nn, &a, A, 1) );
        // A = b * A3 + A
        CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &b, A3, 1, A, 1) );
        // at this point, A = a * A + b * A3

        /* Symmetrize A3, A5 */
        // symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace
        // symmetrizeFloat(cublasH, A5, n, A2); // we use A2 as a workspace

        // A = c * A3 * A2 + A
        CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &c, A3, n, A2, n, &one, A, n) );

        /* Symmetrize A */
        symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace
    }

    /* Smoothing function */
    for (int i = 0; i < smoothing_steps; i++) {
        // A2 = A * A
        CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A, n, A, n, &zero, A2, n) );

        // A3 = A2 * A
        CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A2, n, A, n, &zero, A3, n) );

        /* Symmetrize A3 */
        symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace

        /* Compute A = 1.5 * A - 0.5 * A3 */
        // A = 1.5 * A
        CHECK_CUBLAS( cublasSscal(cublasH, nn, &one_n_half, A, 1) );
        // A = -0.5 * A3 + A
        CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &minus_half, A3, 1, A, 1) );

        /* Symmetrize A */
        symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace
    }

    /* Compute A = (I + A)/2 */
    // build I on device and store it in A2
    build_identity(cublasH, A2, n);

    // A = 1 * I + A
    CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &one, A2, 1, A, 1) );
    // A = 0.5 * A
    CHECK_CUBLAS( cublasSscal(cublasH, nn, &half, A, 1) );

    /* Symmetrize A */
    symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace

    /* Multiply the original matrix by A */
    // W = A_origin * A
    launch_convert_double_to_float(mat + mat_offset, A2, nn);
    CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A2, n, A, n, &zero, A3, n) );

    /* Symmetrize W */
    symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace

    /* Copy the result back to mat */
    launch_convert_float_to_double(A3, mat + mat_offset, nn);
    CHECK_CUDA( cudaDeviceSynchronize() );

    /* Free device memory */
    CHECK_CUDA( cudaFree(A) );
    CHECK_CUDA( cudaFree(A2) );
    CHECK_CUDA( cudaFree(A3) );
}

// void express_FP64(
//     cublasHandle_t cublasH,
//     double* mat,
//     const int n,
//     const int mat_offset = 0
// ) {
//     const int nn = n * n;

//     /* Allocations */
//     // device memory
//     double *A, *A2, *A3, *A5, *I, *Wout;
//     CHECK_CUDA( cudaMalloc(&A,  nn * sizeof(double)) );
//     CHECK_CUDA( cudaMalloc(&A2, nn * sizeof(double)) );
//     CHECK_CUDA( cudaMalloc(&A3, nn * sizeof(double)) );
//     CHECK_CUDA( cudaMalloc(&A5, nn * sizeof(double)) );
//     CHECK_CUDA( cudaMalloc(&I,  nn * sizeof(double)) );
//     CHECK_CUDA( cudaMalloc(&Wout,  nn * sizeof(double)) );

//     // useful constants
//     const double half       =  0.5;
//     const double minus_half = -0.5;
//     const double one        =  1.0;
//     const double one_n_half =  1.5;
//     const double zero       =  0.0;

//     // build identity I on device
//     std::vector<double> I_h(nn, 0.0);
//     for (int i = 0; i < n; i++) I_h[i*n + i] = 1.0;
//     CHECK_CUDA( cudaMemcpy(I, I_h.data(), nn * sizeof(double), H2D) );

//     CHECK_CUDA( cudaMemcpy(A, mat + mat_offset, nn * sizeof(double), D2D) );

//     /* Coefficients */
//     std::vector<std::vector<double>> coeff = {
//         {8.4724206924, -24.5001735687, 17.7268180847},
//         {4.2052841187, -3.0549299717, 0.5567536354},
//         {4.0443077087, -2.9473149776, 0.5449726582},
//         {3.5078327656, -2.5842490196, 0.5067413449},
//         {2.5075511932, -1.8485442400, 0.4358045161}
//     };

//     /* Approximation of the step function */
//     for (int i = 0; i < coeff.size(); i++) {
//         const double a = coeff[i][0];
//         const double b = coeff[i][1];
//         const double c = coeff[i][2];

//         /* Compute the powers of A*/
//         // A2 = A * A
//         CHECK_CUBLAS( cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A, n, A, n, &zero, A2, n) );

//         // A3 = A2 * A
//         CHECK_CUBLAS( cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A2, n, A, n, &zero, A3, n) );

//         // A5 = A3 * A2
//         CHECK_CUBLAS( cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A3, n, A2, n, &zero, A5, n) );

//         /* Symmetrize A3, A5 */
//         symmetrizeDouble(cublasH, A3, n, A2); // we use A2 as a workspace
//         symmetrizeDouble(cublasH, A5, n, A2); // we use A2 as a workspace

//         /* Compute A = a * A + b * A3 + c * A5 */
//         // A = a * A
//         CHECK_CUBLAS( cublasDscal(cublasH, nn, &a, A, 1) );
//         // A = b * A3 + A
//         CHECK_CUBLAS( cublasDaxpy(cublasH, nn, &b, A3, 1, A, 1) );
//         // A = c * A5 + A
//         CHECK_CUBLAS( cublasDaxpy(cublasH, nn, &c, A5, 1, A, 1) );

//         /* Symmetrize A */
//         symmetrizeDouble(cublasH, A, n, A2); // we use A2 as a workspace
//     }

//     /* Smoothing function */
//     for (int i =0; i < 3; i++) {
//         // A2 = A * A
//         CHECK_CUBLAS( cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A, n, A, n, &zero, A2, n) );

//         // A3 = A2 * A
//         CHECK_CUBLAS( cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A2, n, A, n, &zero, A3, n) );

//         /* Symmetrize A3 */
//         symmetrizeDouble(cublasH, A3, n, A2); // we use A2 as a workspace

//         /* Compute A = 1.5 * A - 0.5 * A3 */
//         // A = 1.5 * A
//         CHECK_CUBLAS( cublasDscal(cublasH, nn, &one_n_half, A, 1) );
//         // A = -0.5 * A3 + A
//         CHECK_CUBLAS( cublasDaxpy(cublasH, nn, &minus_half, A3, 1, A, 1) );

//         /* Symmetrize A */
//         symmetrizeDouble(cublasH, A, n, A2); // we use A2 as a workspace
//     }

//     /* Compute A = (I + A)/2 */
//     // A = 1 * I + A
//     CHECK_CUBLAS( cublasDaxpy(cublasH, nn, &one, I, 1, A, 1) );
//     // A = 0.5 * A
//     CHECK_CUBLAS( cublasDscal(cublasH, nn, &half, A, 1) );

//     /* Symmetrize A */
//     symmetrizeDouble(cublasH, A, n, A2); // we use A2 as a workspace

//     /* Multiply the original matrix by A */
//     // Wout = mat * A
//     CHECK_CUBLAS( cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, mat, n, A, n, &zero, Wout, n) );

//     /* Symmetrize W */
//     symmetrizeDouble(cublasH, Wout, n, A2); // we use A2 as a workspace

//     /* Copy the result back to mat */
//     CHECK_CUDA( cudaMemcpy(mat + mat_offset, Wout, nn * sizeof(double), D2D) );
//     CHECK_CUDA( cudaDeviceSynchronize() );

//     /* Free device memory */
//     CHECK_CUDA( cudaFree(A) );
//     CHECK_CUDA( cudaFree(A2) );
//     CHECK_CUDA( cudaFree(A3) );
//     CHECK_CUDA( cudaFree(A5) );
//     CHECK_CUDA( cudaFree(I) );
//     CHECK_CUDA( cudaFree(Wout) );
// }

__global__ void fill_random_kernel(double* vec, int n, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        vec[idx] = curand_uniform_double(&state); // random double in (0,1]
    }
}

void fill_random(double* vec, int n, unsigned long seed = 0, const int threadsPerBlock = 1024) {
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    // printf("blocks: %d, threads per block: %d \n", blocks, threadsPerBlock);
    fill_random_kernel<<<blocks, threadsPerBlock>>>(vec, n, seed);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in fill_random: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void approximate_two_norm(
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH,
    const double* A, size_t n,
    double* lo, double* up,
    size_t max_iter=10, double tol=1e-3
) {
    /* Allocations */
    // constants
    const double zero = 0.0;
    const double one = 1.0;
    
    // storage
    double *V, *V_old, *alpha, *q, *w;
    max_iter = max(max_iter, n);
    CHECK_CUDA(cudaMalloc(&V,     n * max_iter * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&V_old,            n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&alpha,     max_iter * sizeof(double))); // TODO: on host
    CHECK_CUDA(cudaMalloc(&q,                n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&w,                n * sizeof(double)));

    std::vector<double> beta(max_iter, 0.0);

    double minus_alpha, minus_beta_old = 0.0;

    /* Initial vector */
    // q = randn(n, 1)
    fill_random(q, n, 0);

    // q = q / norm(q)
    double norm_q;
    CHECK_CUBLAS(cublasDnrm2(cublasH, n, q, 1, &norm_q));
    if (norm_q != 0.0) {
        double norm_q_inv = 1.0 / norm_q;
        CHECK_CUBLAS(cublasDscal(cublasH, n, &norm_q_inv, q, 1));
    }

    // V(:, 1) = q
    CHECK_CUBLAS(cublasDcopy(cublasH, n, q, 1, V, 1));
    // fill V_old with zeros
    CHECK_CUDA(cudaMemset(V_old, 0, n * sizeof(double)));

    /* Lanczos loop */
    int nb_iter = 0;
    for (int k = 0; k < max_iter; k++) {
        // w = A * q
        CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_N, n, n,
                                 &one, A, n, q, 1,
                                 &zero, w, 1));
        // w = At * w
        CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_T, n, n,
                                 &one, A, n, w, 1,
                                 &zero, w, 1));
        // hence w = A^T * A * q

        // alpha(k) = q^T * w
        CHECK_CUBLAS(cublasDdot(cublasH, n, q, 1, w, 1, &alpha[k]));

        // minus_alpha = -alpha[k]
        CHECK_CUDA(cudaMemcpy(&minus_alpha, &alpha[k], sizeof(double), D2H));
        minus_alpha = -minus_alpha;
        
        // w = w - alpha(k) * q - beta_old * V_old
        CHECK_CUBLAS(cublasDaxpy(cublasH, n, &minus_alpha, q, 1, w, 1));
        CHECK_CUBLAS(cublasDaxpy(cublasH, n, &minus_beta_old, V_old, 1, w, 1));
        
        // beta(k) = norm(w)
        CHECK_CUBLAS(cublasDnrm2(cublasH, n, w, 1, &beta[k]));

        if (beta[k] <= tol * (-minus_alpha) && k > 1)
            break;
            
        // V_old = q
        CHECK_CUBLAS(cublasDcopy(cublasH, n, q, 1, V_old, 1));
        // q = w / beta(k)
        CHECK_CUDA(cudaMemcpy(q, w, n * sizeof(double), cudaMemcpyDeviceToDevice));
        if (beta[k] != 0.0) {
            double beta_inv = 1.0 / beta[k];
            CHECK_CUBLAS(cublasDscal(cublasH, n, &beta_inv, q, 1));
        } else {
            // If beta is zero, we cannot proceed further
            // fprintf(stderr, "Lanczos iteration %d: beta is zero, stopping early.\n", k);
            break;
        }

        if (k < max_iter - 1) {
            // V(:, k+1) = q
            CHECK_CUBLAS(cublasDcopy(cublasH, n, q, 1, V + (k + 1) * n, 1));
        }
        // minus_beta_old = -beta[k]
        minus_beta_old = -beta[k];

        nb_iter++;
    }

    // printf("nb_iter: %d \n", nb_iter);
    if (nb_iter == 0) {
        // in this case, the matrix is an all-zero matrix
        *lo = 0.0;
        *up = 1.0;

        CHECK_CUDA(cudaFree(V));
        CHECK_CUDA(cudaFree(V_old));
        CHECK_CUDA(cudaFree(alpha));
        CHECK_CUDA(cudaFree(q));
        CHECK_CUDA(cudaFree(w));
        CHECK_CUDA(cudaDeviceSynchronize());

        return;
    }

    /* Tridiagonal T */
    // T = diag(alpha) + diag(beta(2:end),1) + diag(beta(2:end),-1);
    double *T;
    CHECK_CUDA(cudaMalloc(&T, nb_iter * nb_iter * sizeof(double)));
    std::vector<double> alpha_host(nb_iter, 0.0);
    CHECK_CUDA(cudaMemcpy(alpha_host.data(), alpha, nb_iter * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> T_host(nb_iter * nb_iter, 0.0);
    for (int i = 0; i < nb_iter; i++) {
        T_host[i * nb_iter + i] = alpha_host[i]; // diagonal
        if (i < nb_iter - 1) {
            T_host[i * nb_iter + (i + 1)] = beta[i + 1]; // upper diagonal
            T_host[(i + 1) * nb_iter + i] = beta[i + 1]; // lower diagonal
        }
    }
    CHECK_CUDA(cudaMemcpy(T, T_host.data(), nb_iter * nb_iter * sizeof(double), H2D));

    /* Largest Ritz pair */
    // allocate memory for eigenvalues
    double *d_eigenvalues;
    CHECK_CUDA(cudaMalloc(&d_eigenvalues, nb_iter * sizeof(double)));

    // allocate workspace for eigenvalue decomposition
    int lwork_eig, *devInfo;
    double *d_work_eig;
    CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                               nb_iter, T, nb_iter, d_eigenvalues, &lwork_eig));
    CHECK_CUDA(cudaMalloc(&d_work_eig, lwork_eig * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));

    // compute eigenvalues and eigenvectors
    CHECK_CUSOLVER(cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                    nb_iter, T, nb_iter, d_eigenvalues,
                                    d_work_eig, lwork_eig, devInfo));

    // retrieve the max eigenvalue and corresponding eigenvector
    int idx_max;
    CHECK_CUBLAS(cublasIdamax(cublasH, nb_iter, d_eigenvalues, 1, &idx_max));
    idx_max--; // convert to 0-based index

    double theta;
    CHECK_CUDA(cudaMemcpy(&theta, d_eigenvalues + idx_max, sizeof(double), D2H));

    double *uk, *y;
    CHECK_CUDA(cudaMalloc(&uk, nb_iter * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y,        n * sizeof(double)));
    // uk = T(:, idx_max)
    CHECK_CUBLAS(cublasDcopy(cublasH, nb_iter, T + idx_max * nb_iter, 1, uk, 1));
    // y = V(:,1:nb_iter) * uk
    CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_N, n, nb_iter, &one, V, n, uk, 1, &zero, y, 1));

    double *ry;
    CHECK_CUDA(cudaMalloc(&ry, n * sizeof(double)));
    // ry = A * q
    CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_N, n, n,
                                &one, A, n, q, 1,
                                &zero, ry, 1));
    // ry = At * ry
    CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_T, n, n,
                                &one, A, n, ry, 1,
                                &zero, ry, 1));
    // hence ry = A^T * A * q

    // ry = ry - theta * y
    double minus_theta = -theta;
    CHECK_CUBLAS(cublasDaxpy(cublasH, n, &minus_theta, y, 1, ry, 1));

    /* Output */
    // lo = sqrt(theta)
    *lo = std::sqrt(theta);

    // up = sqrt(theta + norm(ry))
    double norm_ry;
    CHECK_CUBLAS(cublasDnrm2(cublasH, n, ry, 1, &norm_ry));
    *up = std::sqrt(theta + norm_ry);

    /* Free memory */
    CHECK_CUDA(cudaFree(V));
    CHECK_CUDA(cudaFree(V_old));
    CHECK_CUDA(cudaFree(alpha));
    CHECK_CUDA(cudaFree(q));
    CHECK_CUDA(cudaFree(w));
    CHECK_CUDA(cudaFree(T));
    CHECK_CUDA(cudaFree(d_eigenvalues));
    CHECK_CUDA(cudaFree(d_work_eig));
    CHECK_CUDA(cudaFree(devInfo));
    CHECK_CUDA(cudaFree(uk));
    CHECK_CUDA(cudaFree(y));
    CHECK_CUDA(cudaFree(ry));
    CHECK_CUDA(cudaDeviceSynchronize());

    return;
}

__global__ void compute_res_all_kernel(const double* Z, double beta_m, double* res_all, int m) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < m) {
        // Z[j*m] is the 1st row, j-th column (column-major)
        res_all[j] = fabs(beta_m * Z[j * m]);
    }
}

void compute_res_all(
    const double* Z, size_t n, size_t m,
    double beta_m, double* res_all
) {
    // Allocate device memory for res_all
    CHECK_CUDA(cudaMemset(res_all, 0, m * sizeof(double)));

    // Launch kernel to compute residuals for all Ritz pairs
    int blockSize = 1024;
    int numBlocks = (m + blockSize - 1) / blockSize;
    compute_res_all_kernel<<<numBlocks, blockSize>>>(Z, beta_m, res_all, m);
    
    // Check for errors in kernel launch
    CHECK_CUDA(cudaGetLastError());
}

__global__ void fill_tridiagonal_kernel(double* T, const double* alpha, const double* beta, int nb_iter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nb_iter) {
        // Set diagonal
        T[i * nb_iter + i] = alpha[i];
        // Set upper diagonal
        if (i < nb_iter - 1) {
            T[i * nb_iter + (i + 1)] = beta[i];
            T[(i + 1) * nb_iter + i] = beta[i];
        }
    }
}

void fill_tridiagonal(
    double* T, const double *d_alpha, const double *d_beta, int nb_iter, const int threadsPerBlock = 1024
) {
    // Launch kernel to fill the tridiagonal matrix T
    int blocks = (nb_iter + threadsPerBlock - 1) / threadsPerBlock;
    fill_tridiagonal_kernel<<<blocks, threadsPerBlock>>>(T, d_alpha, d_beta, nb_iter);
    
    // Check for errors in kernel launch
    CHECK_CUDA(cudaGetLastError());
}

double compute_eigenpairs(
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH,
    const double* A, size_t n,
    const size_t k,
    size_t *r,
    double* eigenvalues, double* eigenvectors,
    const bool upper_bound_only = false,
    size_t max_iter = 0, const double tol = 1e-10, const double ortho_tol = 1e-5,
    const bool verbose = false
) {
    if (max_iter == 0)
        max_iter = n;

    /* Allocation */
    double *v0, *Q, *w;
    CHECK_CUDA(cudaMalloc(&v0,                     n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Q,     n * (max_iter + 1) * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&w,                      n * sizeof(double)));

    double zero = 0.0;
    double one = 1.0;

    double minus_beta = 0.0;
    std::vector<double> alpha, beta;
    double minus_alpha;

    /* Initialize */
    // v0 = randn(n, 1)

    fill_random(v0, n);
    CHECK_CUDA( cudaDeviceSynchronize() );

    // v0 = v0 / norm(v0)
    double norm_v0;
    CHECK_CUBLAS(cublasDnrm2(cublasH, n, v0, 1, &norm_v0));
    if (norm_v0 != 0.0) {
        double norm_v0_inv = 1.0 / norm_v0;
        CHECK_CUBLAS(cublasDscal(cublasH, n, &norm_v0_inv, v0, 1));
    }

    // Q(:, 1) = v0
    CHECK_CUBLAS(cublasDcopy(cublasH, n, v0, 1, Q, 1));

    int nb_iter = 0;
    /* Lanczos reccurence */
    for (int m = 0; m < max_iter; m++) {
        nb_iter++;
        // w = A * Q(:, m)
        CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_N, n, n,
                                 &one, A, n, Q + m * n, 1,
                                 &zero, w, 1));
        
        // w = w - beta(m-1) * Q(:, m-1)
        CHECK_CUBLAS(cublasDaxpy(cublasH, n, &minus_beta, Q + (m - 1) * n, 1, w, 1));

        // alpha = Q(:, m)^T * w
        CHECK_CUBLAS(cublasDdot(cublasH, n, Q + m * n, 1, w, 1, &minus_alpha));
        alpha.push_back(minus_alpha);
        minus_alpha = -minus_alpha;
        
        // w = w - alpha * Q(:, m)
        CHECK_CUBLAS(cublasDaxpy(cublasH, n, &minus_alpha, Q + m * n, 1, w, 1));

        // beta(m) = norm(w)
        CHECK_CUBLAS(cublasDnrm2(cublasH, n, w, 1, &minus_beta));
        beta.push_back(minus_beta);
        minus_beta = -minus_beta;

        if (-minus_beta <= std::numeric_limits<double>::epsilon())
            break;
        
        // Q(:, m+1) = w / beta(m)
        double beta_inv = -1.0 / minus_beta;
        CHECK_CUBLAS(cublasDscal(cublasH, n, &beta_inv, w, 1));
        CHECK_CUBLAS(cublasDcopy(cublasH, n, w, 1, Q + (m + 1) * n, 1));
    }

    /* Tridiagonal T */
    double *T, *d_alpha, *d_beta;
    CHECK_CUDA(cudaMalloc(&T,       nb_iter * nb_iter * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_alpha,           nb_iter * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_beta,      (nb_iter - 1) * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_alpha, alpha.data(), nb_iter * sizeof(double), H2D));
    CHECK_CUDA(cudaMemcpy(d_beta,  beta.data(),  (nb_iter - 1) * sizeof(double), H2D));
    fill_tridiagonal(
        T, d_alpha, d_beta, nb_iter
    );

    /* Largest Ritz pair */
    // allocate memory for eigenvalues
    double *d_eigenvalues;
    CHECK_CUDA(cudaMalloc(&d_eigenvalues, nb_iter * sizeof(double)));

    // allocate workspace for eigenvalue decomposition
    int lwork_eig, *devInfo;
    double *d_work_eig;
    CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                               nb_iter, T, nb_iter, d_eigenvalues, &lwork_eig));
    CHECK_CUDA(cudaMalloc(&d_work_eig, lwork_eig * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));

    // compute eigenvalues and eigenvectors
    CHECK_CUSOLVER(cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                    nb_iter, T, nb_iter, d_eigenvalues,
                                    d_work_eig, lwork_eig, devInfo));
    // note that the eigenvalues are sorted in ascending order

    // compute the residuals for all Ritz pairs
    double *res_all;
    CHECK_CUDA(cudaMalloc(&res_all, nb_iter * sizeof(double)));
    double beta_m = minus_beta; // last beta value
    compute_res_all(T, nb_iter, nb_iter, beta_m, res_all);

    /* Compute the decreasing absolute value order */
    std::vector<int> idx(nb_iter);
    for (int i = 0; i < nb_iter; i++) {
        idx[i] = i;
    }
    std::vector<double> eigenvalues_host(nb_iter);
    CHECK_CUDA(cudaMemcpy(eigenvalues_host.data(), d_eigenvalues, nb_iter * sizeof(double), cudaMemcpyDeviceToHost));
    std::sort(idx.begin(), idx.end(), [&](size_t i1, size_t i2) {
        return std::abs(eigenvalues_host[i1]) > std::abs(eigenvalues_host[i2]);
    });

    size_t sel_count = 0;
    double *x_candidate, *overlap, *X_basis, *sel;
    if (!upper_bound_only) {
        /* Choose the first k */
        CHECK_CUDA(cudaMalloc(&x_candidate, n * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&overlap,     k * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&X_basis, k * n * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&sel,         k * sizeof(double)));
        bool accept;
        int j;

        for (int id = 0; id < nb_iter; id++) {
            j = idx[id]; // get the index of the j-th Ritz pair

            // Step 1 cleaning: residual filter
            double res_j;
            CHECK_CUDA(cudaMemcpy(&res_j, res_all + j, sizeof(double), D2H));
            if (res_j < ortho_tol) {
                // Step 2 cleaning: ghost orthogonality filter
                // x_candidate = Q(:,1:m) * Z(:,j)
                CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_N, n, nb_iter,
                                        &one, Q, n, T + j * nb_iter, 1,
                                        &zero, x_candidate, 1));            
                if (sel_count == 0) {
                    accept = true;
                } else {
                    // overlap = X_basis' * x_candidate
                    CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_T, n, sel_count,
                                            &one, X_basis, n, x_candidate, 1,
                                            &zero, overlap, 1));

                    // get overlap back to host
                    std::vector<double> overlap_host(sel_count);
                    CHECK_CUDA(cudaMemcpy(overlap_host.data(), overlap, sel_count * sizeof(double), cudaMemcpyDeviceToHost));

                    // check if overlap is below the tolerance
                    accept = true;
                    for (size_t i = 0; i < sel_count; i++) {
                        if (abs(overlap_host[i]) > ortho_tol) {
                            accept = false;
                            break;
                        }
                    }
                }

                if (accept) {
                    // add the new eigenvector to the basis
                    CHECK_CUBLAS(cublasDcopy(cublasH, n, x_candidate, 1, X_basis + sel_count * n, 1));
                    // store the eigenvalue
                    CHECK_CUDA(cudaMemcpy(sel + sel_count, d_eigenvalues + j, sizeof(double), cudaMemcpyDeviceToHost));
                    
                    sel_count++;

                    if (sel_count == k) {
                        // if we have enough eigenpairs, stop
                        break;
                    }
                }
            }
        }

        if (sel_count < k && verbose) {
            fprintf(stderr, "Warning: only %zu eigenpairs found, requested %zu.\n", sel_count, k);
        }
        *r = sel_count;
    }

    /* Spectral norm upper bound */
    // retrieve the max eigenvalue and corresponding eigenvector
    int idx_max = idx[0];

    double theta;
    CHECK_CUDA(cudaMemcpy(&theta, d_eigenvalues + idx_max, sizeof(double), D2H));

    double norm_upper;
    CHECK_CUDA(cudaMemcpy(&norm_upper, res_all + idx_max, sizeof(double), D2H));
    norm_upper += fabs(theta);

    /* Output */
    if (!upper_bound_only) {
        // eigenvectors = X_basis
        CHECK_CUBLAS(cublasDcopy(cublasH, sel_count * n, X_basis, 1, eigenvectors, 1));
        // eigenvalues = sel
        CHECK_CUDA(cudaMemcpy(eigenvalues, sel, sel_count * sizeof(double), D2D));

        CHECK_CUDA(cudaFree(x_candidate));
        CHECK_CUDA(cudaFree(overlap));
        CHECK_CUDA(cudaFree(X_basis));
        CHECK_CUDA(cudaFree(sel));
    }

    /* Free memory */
    CHECK_CUDA(cudaFree(v0));
    CHECK_CUDA(cudaFree(Q));
    CHECK_CUDA(cudaFree(w));
    CHECK_CUDA(cudaFree(T));
    CHECK_CUDA(cudaFree(d_alpha));
    CHECK_CUDA(cudaFree(d_beta));
    CHECK_CUDA(cudaFree(d_eigenvalues));
    CHECK_CUDA(cudaFree(d_work_eig));
    CHECK_CUDA(cudaFree(devInfo));
    CHECK_CUDA(cudaFree(res_all));

    CHECK_CUDA(cudaDeviceSynchronize());

    return norm_upper + 1e-3;
}

std::chrono::duration<double> composite_FP32_psd(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dA_psd, size_t n) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t nn = n * n;
    
    CHECK_CUDA(cudaMemcpy(dA_psd, dA_orig, nn * sizeof(double), cudaMemcpyDeviceToDevice));

    double lo, up;
    approximate_two_norm(
        cublasH, solverH, dA_psd, n, &lo, &up
    ); // TODO: sometimes we use a TF32 handle here but the computations are done in FP64

    // scale to have eigenvalues in [-1, 1]
    const double scale = up > 0.0 ? up : 1.0;
    const double inv_scale = 1.0/scale;
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, dA_psd, 1) );

    express_FP32(
        cublasH, dA_psd, n, 0
    );

    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

    CHECK_CUDA(cudaDeviceSynchronize());

    return std::chrono::high_resolution_clock::now() - start;
}

std::chrono::duration<double> composite_FP32_psd_deflate(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dA_psd, size_t n) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t nn = n * n;
    size_t k = K_DEFLATE;
    assert(n > k);
    
    CHECK_CUDA(cudaMemcpy(dA_psd, dA_orig, nn * sizeof(double), cudaMemcpyDeviceToDevice));

    /* Step 1: compute the largest eigenpairs of the matrix */
    size_t r;
    double *eigenvalues, *eigenvectors;
    CHECK_CUDA( cudaMalloc(&eigenvalues,      k * sizeof(double)) );
    CHECK_CUDA( cudaMalloc(&eigenvectors, n * k * sizeof(double)) );

    double _ = compute_eigenpairs(
        cublasH, solverH, dA_psd, n, k, &r, eigenvalues, eigenvectors, false, 0
    );

    std::vector<double> eigenvalues_host(r);
    CHECK_CUDA( cudaMemcpy(eigenvalues_host.data(), eigenvalues, r * sizeof(double), D2H) );

    /* Step 2: remove the largest eigenvalues from the matrix */
    for (int i = 0; i < r; i++) {
        // X <- X - \lambda_i * v_i v_i^T
        double lambda = -eigenvalues_host[i];
        double *v_i = eigenvectors + i * n;
        CHECK_CUBLAS( cublasDger(cublasH, n, n, &lambda, v_i, 1, v_i, 1, dA_psd, n) );
    }

    /* Step 3: scale the deflated matrix */
    double up = compute_eigenpairs(
        cublasH, solverH, dA_psd, n, 0, nullptr, nullptr, nullptr, true, 100
    );

    // scale to have eigenvalues in [-1, 1]
    const double scale = up > 0.0 ? up : 1.0;
    const double inv_scale = 1.0/scale;
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, dA_psd, 1) );

    express_FP32(
        cublasH, dA_psd, n, 0
    );

    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

    for (int i = 0; i < r; i++) {
        // X <- X + \lambda_i * v_i v_i^T
        double lambda = eigenvalues_host[i];
        if (lambda > 0.0) { // only add positive eigenvalues
            double *v_i = eigenvectors + i * n;
            CHECK_CUBLAS( cublasDger(cublasH, n, n, &lambda, v_i, 1, v_i, 1, dA_psd, n) );
        }
    }

    CHECK_CUDA( cudaFree(eigenvalues) );
    CHECK_CUDA( cudaFree(eigenvectors) );
    CHECK_CUDA(cudaDeviceSynchronize());

    return std::chrono::high_resolution_clock::now() - start;
}

// std::chrono::duration<double> composite_FP64_psd(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dA_psd, size_t n) {
//     auto start = std::chrono::high_resolution_clock::now();
//     size_t nn = n * n;
    
//     CHECK_CUDA(cudaMemcpy(dA_psd, dA_orig, nn * sizeof(double), cudaMemcpyDeviceToDevice));

//     double lo, up;
//     approximate_two_norm(
//         cublasH, solverH, dA_psd, n, &lo, &up
//     );

//     // scale to have eigenvalues in [-1, 1]
//     const double scale = up > 0.0 ? 1.1 * up : 1.0; // TODO: fix
//     const double inv_scale = 1.0/scale;
//     CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, dA_psd, 1) );

//     express_FP64(
//         cublasH, dA_psd, n, 0
//     );

//     CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

//     CHECK_CUDA(cudaDeviceSynchronize());

//     return std::chrono::high_resolution_clock::now() - start;
// }

std::chrono::duration<double> FP64_gemm(cublasHandle_t cublasH, const double* dA_orig, double* dA2, size_t n, int gemm_restarts) {
    auto start = std::chrono::high_resolution_clock::now();
    double one = 1.0;
    double zero = 0.0;

    for (int i = 0; i < gemm_restarts; i++) {
        CHECK_CUBLAS( cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, dA_orig, n, dA_orig, n, &zero, dA2, n) );
    }
    CHECK_CUDA( cudaDeviceSynchronize() );
    
    return (std::chrono::high_resolution_clock::now() - start) / static_cast<double>(gemm_restarts);
}

std::chrono::duration<double> FP32_gemm(cublasHandle_t cublasH, const double* dA_orig, double* dA2, size_t n, int gemm_restarts) {
    auto start = std::chrono::high_resolution_clock::now();
    float one = 1.0;
    float zero = 0.0;
    size_t nn = n*n;

    float *sA, *sA2;
    std::vector<double> A_h_d(nn);
    std::vector<float> A_h(nn);
    CHECK_CUDA( cudaMalloc(&sA,  nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&sA2, nn * sizeof(float)) );

    CHECK_CUDA( cudaMemcpy(A_h_d.data(), dA_orig, nn * sizeof(double), D2H) );
    for (int i = 0; i < nn; i++)
        A_h[i] = static_cast<float>(A_h_d[i]);
    CHECK_CUDA( cudaMemcpy(sA, A_h.data(), nn * sizeof(float), H2D) );

    for (int i = 0; i < gemm_restarts; i++) {
        CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, sA, n, sA, n, &zero, sA2, n) );
    }
    CHECK_CUDA( cudaDeviceSynchronize() );
    
    CHECK_CUDA( cudaMemcpy(A_h.data(), sA2, nn * sizeof(float), D2H) );
    for (int i = 0; i < nn; i++)
        A_h_d[i] = static_cast<double>(A_h[i]);
    CHECK_CUDA( cudaMemcpy(dA2, A_h_d.data(), nn * sizeof(double), H2D) );
    
    CHECK_CUDA( cudaFree(sA) );
    CHECK_CUDA( cudaFree(sA2) );

    CHECK_CUDA( cudaDeviceSynchronize() );

    return (std::chrono::high_resolution_clock::now() - start) / static_cast<double>(gemm_restarts);
}

__global__ void float4_to_half_kernel(
    const float4* __restrict__ A4,
    __half2 * __restrict__ B2,
    size_t N4
) {
    size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= N4) return;

    // load 4 floats
    float4 v = A4[idx];

    // pack low two floats into half2
    B2[2*idx + 0] = __float22half2_rn(make_float2(v.x, v.y));
    // pack high two floats into half2
    B2[2*idx + 1] = __float22half2_rn(make_float2(v.z, v.w));
}

void convertFloatToHalf4(const float* dA, __half* dB, size_t N) {
    size_t N4 = (N + 3)/4;  // how many float4’s
    auto A4 = reinterpret_cast<const float4*>(dA);
    auto B2 = reinterpret_cast<__half2*>(dB);

    const int blk = 1024;
    int grid = (N4 + blk - 1)/blk;
    float4_to_half_kernel<<<grid,blk>>>(A4, B2, N4);
    // cudaDeviceSynchronize();
}

std::chrono::duration<double> TF16_gemm(cublasHandle_t cublasH, const double* dA_orig, double* dA2, size_t n, int gemm_restarts) {
    auto start = std::chrono::high_resolution_clock::now();
    float one = 1.0;
    float zero = 0.0;
    size_t nn = n*n;

    float *sA, *sA2;
    std::vector<double> A_h_d(nn);
    std::vector<float> A_h(nn);
    CHECK_CUDA( cudaMalloc(&sA,  nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&sA2, nn * sizeof(float)) );

    CHECK_CUDA( cudaMemcpy(A_h_d.data(), dA_orig, nn * sizeof(double), D2H) );
    for (int i = 0; i < nn; i++)
        A_h[i] = static_cast<float>(A_h_d[i]);
    CHECK_CUDA( cudaMemcpy(sA, A_h.data(), nn * sizeof(float), H2D) );

    __half *hA; 
    CHECK_CUDA(cudaMalloc(&hA, nn*sizeof(__half)));
    convertFloatToHalf4(sA, hA, nn);

    for (int i = 0; i < gemm_restarts; i++) {
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            hA, CUDA_R_16F, n,
            hA, CUDA_R_16F, n,
            &zero,
            sA2,    CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA( cudaDeviceSynchronize() );
    
    CHECK_CUDA( cudaMemcpy(A_h.data(), sA2, nn * sizeof(float), D2H) );
    for (int i = 0; i < nn; i++)
        A_h_d[i] = static_cast<double>(A_h[i]);
    CHECK_CUDA( cudaMemcpy(dA2, A_h_d.data(), nn * sizeof(double), H2D) );
    
    CHECK_CUDA( cudaDeviceSynchronize() );
    CHECK_CUDA( cudaFree(sA) );
    CHECK_CUDA( cudaFree(sA2) );
    CHECK_CUDA( cudaFree(hA) );

    return (std::chrono::high_resolution_clock::now() - start) / static_cast<double>(gemm_restarts);
}

// void express_TF16(
//     cublasHandle_t cublasH,
//     double* mat,
//     const int n,
//     const int mat_offset
// ) {
//     const int nn = n * n;

//     /* Allocations */
//     // device memory
//     float *A, *A2, *A3, *A5, *I, *W, *Wout;
//     CHECK_CUDA( cudaMalloc(&A,    nn * sizeof(float)) );
//     CHECK_CUDA( cudaMalloc(&A2,   nn * sizeof(float)) );
//     CHECK_CUDA( cudaMalloc(&A3,   nn * sizeof(float)) );
//     CHECK_CUDA( cudaMalloc(&A5,   nn * sizeof(float)) );
//     CHECK_CUDA( cudaMalloc(&I,    nn * sizeof(float)) );
//     CHECK_CUDA( cudaMalloc(&W,    nn * sizeof(float)) );
//     CHECK_CUDA( cudaMalloc(&Wout, nn * sizeof(float)) );

//     __half *hA, *hA2, *hA3, *hW;
//     CHECK_CUDA(cudaMalloc(&hA,  nn * sizeof(__half)));
//     CHECK_CUDA(cudaMalloc(&hA2, nn * sizeof(__half)));
//     CHECK_CUDA(cudaMalloc(&hA3, nn * sizeof(__half)));
//     CHECK_CUDA(cudaMalloc(&hW,  nn * sizeof(__half)));

//     // useful constants
//     const float half       =  0.5f;
//     const float minus_half = -0.5f;
//     const float one        =  1.0f;
//     const float one_n_half =  1.5f;
//     const float zero       =  0.0f;

//     // build identity I on device
//     std::vector<float> I_h(nn, 0.0f);
//     for (int i = 0; i < n; i++) I_h[i*n + i] = 1.0f;
//     CHECK_CUDA( cudaMemcpy(I, I_h.data(), nn * sizeof(float), H2D) );

//     /* Convert the initial matrix*/
//     // copy the double matrix back to the host
//     std::vector<double> A_h_d(nn);
//     CHECK_CUDA( cudaMemcpy(A_h_d.data(), mat + mat_offset, nn * sizeof(double), D2H) );

//     // convert the host matrix to float
//     std::vector<float> A_h(nn);
//     for (int i = 0; i < nn; i++)
//         A_h[i] = static_cast<float>(A_h_d[i]);

//     // copy the float host matrix to the device
//     CHECK_CUDA( cudaMemcpy(A, A_h.data(), nn * sizeof(float), H2D) );
//     CHECK_CUDA( cudaMemcpy(W, A_h.data(), nn * sizeof(float), H2D) );

//     /* Coefficients */
//     // std::vector<std::vector<float>> coeff = {
//     //     {8.4724206924, -24.5001735687, 17.7268180847},
//     //     {4.2052841187, -3.0549299717, 0.5567536354},
//     //     {4.0443077087, -2.9473149776, 0.5449726582},
//     //     {3.5078327656, -2.5842490196, 0.5067413449},
//     //     {2.5075511932, -1.8485442400, 0.4358045161}
//     // };
//     std::vector<std::vector<float>> coeff = { 
//         { 8.3885353390, -23.7796270883, 16.8664591580 }, 
//         { 4.1636476423, -2.9650849331, 0.5297319805 }, 
//         { 4.0042650581, -2.8606348801, 0.5185227850 }, 
//         { 3.4731017481, -2.5082466382, 0.4821470022 }, 
//         { 2.4827239537, -1.7941788274, 0.4146530436 }, 
//     };

//     /* Approximation of the step function */
//     for (int i = 0; i < coeff.size(); i++) {
//         const float a = coeff[i][0];
//         const float b = coeff[i][1];
//         const float c = coeff[i][2];

//         /* Compute the powers of A*/
//         // A2 = A * A
//         convertFloatToHalf4(A, hA, nn);
//         CHECK_CUBLAS(cublasGemmEx(
//             cublasH,
//             CUBLAS_OP_N, CUBLAS_OP_N,
//             n, n, n,
//             &one,
//             hA, CUDA_R_16F, n,
//             hA, CUDA_R_16F, n,
//             &zero,
//             A2,    CUDA_R_32F, n,
//             CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

//         // A3 = A2 * A
//         convertFloatToHalf4(A2, hA2, nn);
//         CHECK_CUBLAS(cublasGemmEx(
//             cublasH,
//             CUBLAS_OP_N, CUBLAS_OP_N,
//             n, n, n,
//             &one,
//             hA, CUDA_R_16F, n,
//             hA2, CUDA_R_16F, n,
//             &zero,
//             A3,    CUDA_R_32F, n,
//             CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

//         // A5 = A3 * A2
//         convertFloatToHalf4(A3, hA3, nn);
//         CHECK_CUBLAS(cublasGemmEx(
//             cublasH,
//             CUBLAS_OP_N, CUBLAS_OP_N,
//             n, n, n,
//             &one,
//             hA2, CUDA_R_16F, n,
//             hA3, CUDA_R_16F, n,
//             &zero,
//             A5,    CUDA_R_32F, n,
//             CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

//         /* Symmetrize A3, A5 */
//         symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace
//         symmetrizeFloat(cublasH, A5, n, A2); // we use A2 as a workspace

//         /* Compute A = a * A + b * A3 + c * A5 */
//         // A = a * A
//         CHECK_CUBLAS( cublasSscal(cublasH, nn, &a, A, 1) );
//         // A = b * A3 + A
//         CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &b, A3, 1, A, 1) );
//         // A = c * A5 + A
//         CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &c, A5, 1, A, 1) );

//         /* Symmetrize A */
//         symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace
//     }

//     /* Smoothing function */
//     for (int i =0; i < 3; i++) {
//         // A2 = A * A
//         convertFloatToHalf4(A, hA, nn);
//         CHECK_CUBLAS(cublasGemmEx(
//             cublasH,
//             CUBLAS_OP_N, CUBLAS_OP_N,
//             n, n, n,
//             &one,
//             hA, CUDA_R_16F, n,
//             hA, CUDA_R_16F, n,
//             &zero,
//             A2,    CUDA_R_32F, n,
//             CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

//         // // ---------------------------
//         // // A3 = I - A2
//         // CHECK_CUBLAS(cublasSgeam(
//         //     cublasH,
//         //     CUBLAS_OP_N, CUBLAS_OP_N,
//         //     n, n,
//         //     &one,  I,  n,
//         //     &neg1, A2, n,
//         //     A3,       n));

//         // // A2 = I + 0.5 * A3
//         // CHECK_CUBLAS(cublasSgeam(
//         //     cublasH,
//         //     CUBLAS_OP_N, CUBLAS_OP_N,
//         //     n, n,
//         //     &one,  I,  n,
//         //     &half, A3, n,
//         //     A2,       n));

//         // // A3 = A * A2
//         // convertFloatToHalf4(A2, hA2, nn);
//         // CHECK_CUBLAS(cublasGemmEx(
//         //     cublasH,
//         //     CUBLAS_OP_N, CUBLAS_OP_N,
//         //     n, n, n,
//         //     &one,
//         //     hA, CUDA_R_16F, n,
//         //     hA2, CUDA_R_16F, n,
//         //     &zero,
//         //     A3,    CUDA_R_32F, n,
//         //     CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

//         // // A = A3
//         // CHECK_CUDA(cudaMemcpy(A, A3, nn*sizeof(float), D2D));
//         // // ---------------------------



//         // A3 = A2 * A
//         convertFloatToHalf4(A2, hA2, nn);
//         CHECK_CUBLAS(cublasGemmEx(
//             cublasH,
//             CUBLAS_OP_N, CUBLAS_OP_N,
//             n, n, n,
//             &one,
//             hA, CUDA_R_16F, n,
//             hA2, CUDA_R_16F, n,
//             &zero,
//             A3,    CUDA_R_32F, n,
//             CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

//         /* Symmetrize A3 */
//         symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace

//         /* Compute A = 1.5 * A - 0.5 * A3 */
//         // A = 1.5 * A
//         CHECK_CUBLAS( cublasSscal(cublasH, nn, &one_n_half, A, 1) );
//         // A = -0.5 * A3 + A
//         CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &minus_half, A3, 1, A, 1) );



//         /* Symmetrize A */
//         symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace

//         // Compute Frobenius norm ||A_our||_F
//         float fro_err = 0.0f;
//         CHECK_CUBLAS(cublasSnrm2(cublasH, nn, A, 1, &fro_err));

//         // printf("Iter: %d | Fro norm = %.10f \n", i, fro_err);
//     }

//     /* Compute A = (I + A)/2 */
//     // A = 1 * I + A
//     CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &one, I, 1, A, 1) );
//     // A = 0.5 * A
//     CHECK_CUBLAS( cublasSscal(cublasH, nn, &half, A, 1) );

//     /* Symmetrize A */
//     symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace

//     /* Multiply the original matrix by A */
//     // Wout = W * A
//     convertFloatToHalf4(W, hW, nn);
//     convertFloatToHalf4(A, hA, nn);
//     CHECK_CUBLAS(cublasGemmEx(
//         cublasH,
//         CUBLAS_OP_N, CUBLAS_OP_N,
//         n, n, n,
//         &one,
//         hW, CUDA_R_16F, n,
//         hA, CUDA_R_16F, n,
//         &zero,
//         Wout,    CUDA_R_32F, n,
//         CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

//     /* Symmetrize W */
//     symmetrizeFloat(cublasH, Wout, n, A2); // we use A2 as a workspace

//     /* Copy the result back to mat */
//     std::vector<float> A_h_f(nn);
//     CHECK_CUDA( cudaMemcpy(A_h_f.data(), Wout, nn * sizeof(float), D2H) );
//     for (size_t i = 0; i < nn; i++) {
//         A_h_d[i] = static_cast<double>(A_h_f[i]);
//     }
//     CHECK_CUDA( cudaMemcpy(mat + mat_offset, A_h_d.data(), nn * sizeof(double), H2D) );
//     CHECK_CUDA( cudaDeviceSynchronize() );

//     /* Free device memory */
//     CHECK_CUDA( cudaFree(A) );
//     CHECK_CUDA( cudaFree(A2) );
//     CHECK_CUDA( cudaFree(A3) );
//     CHECK_CUDA( cudaFree(A5) );
//     CHECK_CUDA( cudaFree(I) );
//     CHECK_CUDA( cudaFree(W) );
//     CHECK_CUDA( cudaFree(Wout) );
//     CHECK_CUDA( cudaFree(hA) );
//     CHECK_CUDA( cudaFree(hA2) );
//     CHECK_CUDA( cudaFree(hA3) );
//     CHECK_CUDA( cudaFree(hW) );
//     CHECK_CUDA( cudaDeviceSynchronize() );
// }

// std::chrono::duration<double> composite_TF16_psd(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dA_psd, size_t n) {
//     auto start = std::chrono::high_resolution_clock::now();
//     size_t nn = n * n;
    
//     CHECK_CUDA(cudaMemcpy(dA_psd, dA_orig, nn * sizeof(double), cudaMemcpyDeviceToDevice));

//     double lo, up;
//     approximate_two_norm(
//         cublasH, solverH, dA_psd, n, &lo, &up
//     ); // TODO: we use a TF16 handle here but the computations are done in FP64

//     // scale to have eigenvalues in [-1, 1]
//     const double scale = up > 0.0 ? up : 1.0;
//     const double inv_scale = 1.0/scale;
//     CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, dA_psd, 1) );

//     express_TF16(
//         cublasH, dA_psd, n, 0
//     );

//     CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

//     CHECK_CUDA(cudaDeviceSynchronize());

//     return std::chrono::high_resolution_clock::now() - start;
// }

void express_TF16(
    cublasHandle_t cublasH,
    float* mat,
    const int n,
    const int mat_offset
) {
    const int nn = n * n;

    /* Allocations */
    // device memory
    float *A, *A2, *A3, *A5, *I, *W, *Wout;
    CHECK_CUDA( cudaMalloc(&A,    nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&A2,   nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&A3,   nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&A5,   nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&I,    nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&W,    nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&Wout, nn * sizeof(float)) );

    __half *hA, *hA2, *hA3, *hW;
    CHECK_CUDA(cudaMalloc(&hA,  nn * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&hA2, nn * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&hA3, nn * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&hW,  nn * sizeof(__half)));

    // useful constants
    const float half       =  0.5f;
    const float one        =  1.0f;
    const float neg1       = -1.0f;
    const float zero       =  0.0f;

    // build identity I on device
    std::vector<float> I_h(nn, 0.0f);
    for (int i = 0; i < n; i++) I_h[i*n + i] = 1.0f;
    CHECK_CUDA( cudaMemcpy(I, I_h.data(), nn * sizeof(float), H2D) );

    /* Convert the initial matrix*/
    // copy the float matrix back to the host

    // convert the host matrix to float
    // std::vector<float> A_h(nn);
    // for (int i = 0; i < nn; i++)
    //     A_h[i] = static_cast<float>(A_h_d[i]);

    // copy the float host matrix to the device
    CHECK_CUDA( cudaMemcpy(A, mat, nn * sizeof(float), D2D) );
    CHECK_CUDA( cudaMemcpy(W, mat, nn * sizeof(float), D2D) );

    /* Coefficients */
    // std::vector<std::vector<float>> coeff = {
    //     {8.4724206924, -24.5001735687, 17.7268180847},
    //     {4.2052841187, -3.0549299717, 0.5567536354},
    //     {4.0443077087, -2.9473149776, 0.5449726582},
    //     {3.5078327656, -2.5842490196, 0.5067413449},
    //     {2.5075511932, -1.8485442400, 0.4358045161}
    // };

    std::vector<std::vector<float>> coeff = { 
        { 8.3885353390, -23.7796270883, 16.8664591580 }, 
        { 4.1636476423, -2.9650849331, 0.5297319805 }, 
        { 4.0042650581, -2.8606348801, 0.5185227850 }, 
        { 3.4731017481, -2.5082466382, 0.4821470022 }, 
        { 2.4827239537, -1.7941788274, 0.4146530436 }, 
    };

    /* Approximation of the step function */
    for (int i = 0; i < coeff.size(); i++) {
        const float a = coeff[i][0];
        const float b = coeff[i][1];
        const float c = coeff[i][2];

        /* Compute the powers of A*/
        // A2 = A * A
        convertFloatToHalf4(A, hA, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            hA, CUDA_R_16F, n,
            hA, CUDA_R_16F, n,
            &zero,
            A2,    CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // A3 = A2 * A
        convertFloatToHalf4(A2, hA2, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            hA, CUDA_R_16F, n,
            hA2, CUDA_R_16F, n,
            &zero,
            A3,    CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // A5 = A3 * A2
        convertFloatToHalf4(A3, hA3, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            hA2, CUDA_R_16F, n,
            hA3, CUDA_R_16F, n,
            &zero,
            A5,    CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        /* Symmetrize A3, A5 */
        symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace
        symmetrizeFloat(cublasH, A5, n, A2); // we use A2 as a workspace

        /* Compute A = a * A + b * A3 + c * A5 */
        // A = a * A
        CHECK_CUBLAS( cublasSscal(cublasH, nn, &a, A, 1) );
        // A = b * A3 + A
        CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &b, A3, 1, A, 1) );
        // A = c * A5 + A
        CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &c, A5, 1, A, 1) );

        /* Symmetrize A */
        symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace
    }

    /* Smoothing function */
    for (int i =1; i <= 3; i++) {
        // A2 = A * A
        convertFloatToHalf4(A, hA, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            hA, CUDA_R_16F, n,
            hA, CUDA_R_16F, n,
            &zero,
            A2,    CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        
        // ---------------------------
        // A3 = I - A2
        CHECK_CUBLAS(cublasSgeam(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n,
            &one,  I,  n,
            &neg1, A2, n,
            A3,       n));

        // A2 = I + 0.5 * A3
        CHECK_CUBLAS(cublasSgeam(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n,
            &one,  I,  n,
            &half, A3, n,
            A2,       n));

        // A3 = A * A2
        convertFloatToHalf4(A2, hA2, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            hA, CUDA_R_16F, n,
            hA2, CUDA_R_16F, n,
            &zero,
            A3,    CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // A = A3
        CHECK_CUDA(cudaMemcpy(A, A3, nn*sizeof(float), D2D));
        // ---------------------------



        // // A3 = A2 * A
        // convertFloatToHalf4(A2, hA2, nn);
        // CHECK_CUBLAS(cublasGemmEx(
        //     cublasH,
        //     CUBLAS_OP_N, CUBLAS_OP_N,
        //     n, n, n,
        //     &one,
        //     hA, CUDA_R_16F, n,
        //     hA2, CUDA_R_16F, n,
        //     &zero,
        //     A3,    CUDA_R_32F, n,
        //     CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // /* Symmetrize A3 */
        // // symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace

        // /* Compute A = 1.5 * A - 0.5 * A3 */
        // // A = 1.5 * A
        // CHECK_CUBLAS( cublasSscal(cublasH, nn, &one_n_half, A, 1) );
        // // A = -0.5 * A3 + A
        // CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &minus_half, A3, 1, A, 1) );



        /* Symmetrize A */
        symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace

        // Compute Frobenius norm ||A_our||_F
        float fro_err = 0.0f;
        CHECK_CUBLAS(cublasSnrm2(cublasH, nn, A, 1, &fro_err));

        // printf("Iter: %d | Fro norm = %.10f \n", i, fro_err);

        // // save the matrix for debugging
        // std::string filename = "/home/jordan/antoine/psd_projection_benchmarks/ksc/data/express/" + std::to_string(i) + ".bin";
        // saveMatrixForMatlab(A, n, filename);
    }

    /* Compute A = (I + A)/2 */
    // A = 1 * I + A
    CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &one, I, 1, A, 1) );
    // A = 0.5 * A
    CHECK_CUBLAS( cublasSscal(cublasH, nn, &half, A, 1) );

    /* Symmetrize A */
    symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace

    /* Multiply the original matrix by A */
    // Wout = W * A
    convertFloatToHalf4(A, hA, nn);
    convertFloatToHalf4(W, hW, nn);
    CHECK_CUBLAS(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &one,
        hW, CUDA_R_16F, n,
        hA, CUDA_R_16F, n,
        &zero,
        Wout,    CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    /* Symmetrize W */
    symmetrizeFloat(cublasH, Wout, n, A2); // we use A2 as a workspace

    /* Copy the result back to mat */
    CHECK_CUDA( cudaMemcpy(mat, Wout, nn * sizeof(float), D2D) );
    CHECK_CUDA( cudaDeviceSynchronize() );

    /* Free device memory */
    CHECK_CUDA( cudaFree(A) );
    CHECK_CUDA( cudaFree(A2) );
    CHECK_CUDA( cudaFree(A3) );
    CHECK_CUDA( cudaFree(A5) );
    CHECK_CUDA( cudaFree(I) );
    CHECK_CUDA( cudaFree(W) );
    CHECK_CUDA( cudaFree(Wout) );
    CHECK_CUDA( cudaFree(hA) );
    CHECK_CUDA( cudaFree(hA2) );
    CHECK_CUDA( cudaFree(hA3) );
    CHECK_CUDA( cudaFree(hW) );
    CHECK_CUDA( cudaDeviceSynchronize() );

    return;
}

std::chrono::duration<double> composite_TF16_psd(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dA_psd, size_t n) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t nn = n * n;
    
    float *dA_f;
    CHECK_CUDA(cudaMemcpy(dA_psd, dA_orig, nn * sizeof(double), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMalloc(&dA_f, nn * sizeof(float)));

    double lo, up;
    approximate_two_norm(
        cublasH, solverH, dA_psd, n, &lo, &up
    ); // TODO: we use a TF16 handle here but the computations are done in FP64

    // scale to have eigenvalues in [-1, 1]
    const double scale = up > 0.0 ? up : 1.0;
    const double inv_scale = 1.0/scale;
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, dA_psd, 1) );

    launch_convert_double_to_float(
        dA_psd, dA_f, n
    );

    express_TF16(
        cublasH, dA_f, n, 0
    );

    launch_convert_float_to_double(
        dA_f, dA_psd, n
    );


    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

    CHECK_CUDA(cudaDeviceSynchronize());

    return std::chrono::high_resolution_clock::now() - start;
}

void haoyu_TF16(
    cublasHandle_t cublasH,
    float* mat,
    const int n,
    const int mat_offset
) {
    const int nn = n * n;

    // 3) Allocate device buffers
    float *dA_our, *dTmp, *dI, *dT1, *dT2, *dF, *dDiff;
    // CHECK_CUDA(cudaMalloc(&dA_orig, nn*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dA_our,  nn*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dTmp,    nn*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dI,      nn*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dT1,     nn*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dT2,     nn*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dF,      nn*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dDiff,   nn*sizeof(float)));

    

    // 4) Copy host to device
    CHECK_CUDA(cudaMemcpy(dA_our, mat, nn*sizeof(float), D2D));
    // 5) Build identity I on device
    std::vector<float> I_h(nn, 0.0f);
    for (int i = 0; i < n; i++) I_h[i*n + i] = 1.0f;
    CHECK_CUDA(cudaMemcpy(dI, I_h.data(), nn*sizeof(float), cudaMemcpyHostToDevice));

    // half buffers
    __half *dT3_half, *dT4_half; 
    CHECK_CUDA(cudaMalloc(&dT3_half, nn*sizeof(__half))); 
    CHECK_CUDA(cudaMalloc(&dT4_half, nn*sizeof(__half)));

    const float one = 1.0f, zero = 0.0f, neg1 = -1.0f;
    float half = 0.5f;
    // 6) Iterative algorithm in float, printing after each iter

    // size_t threads=1024, blocks=(nn+threads-1)/threads;

    // printf("Start solving haoyu! \n");
    // float2half_kernel<<<blocks,threads>>>(dA_our, dT3_half, nn);
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 1; iter <= 7; iter++) {
        // T1 = A_our * A_our
        // float2half_kernel<<<blocks,threads>>>(dA_our, dT3_half, nn);
        convertFloatToHalf4(dA_our, dT3_half, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            dT3_half, CUDA_R_16F, n,
            dT3_half, CUDA_R_16F, n,
            &zero,
            dT1,    CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printMatrixHalf(dT3_half, n);
        // printMatrixFloat(dT1, n);

        // T2 = I - T1
        CHECK_CUBLAS(cublasSgeam(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n,
            &one,  dI,  n,
            &neg1, dT1, n,
            dT2,       n));
        // T1 = T2 * T2
        // float2half_kernel<<<blocks,threads>>>(dT2, dT3_half, nn);
        convertFloatToHalf4(dT2, dT3_half, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            dT3_half, CUDA_R_16F, n,
            dT3_half, CUDA_R_16F, n,
            &zero,
            dT1, CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printMatrixHalf(dT3_half, n);
        // printMatrixFloat(dT1, n);

        // F = I + log(iter+10)*T1
        float logv = std::log(iter + 10.0f);
        CHECK_CUBLAS(cublasSgeam(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n,
            &one,  dI,  n,
            &logv, dT1, n,
            dF,      n));
        
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printMatrixFloat(dF, n);

        // A_our = A_our * F (to dTmp, then copy back)
        // float2half_kernel<<<blocks,threads>>>(dA_our, dT3_half, nn);
        // float2half_kernel<<<blocks,threads>>>(dF, dT4_half, nn);
        convertFloatToHalf4(dA_our, dT3_half, nn);
        convertFloatToHalf4(dF, dT4_half, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            dT3_half, CUDA_R_16F, n,
            dT4_half, CUDA_R_16F, n,
            &zero,
            dTmp,   CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printMatrixHalf(dT3_half, n);
        // printMatrixHalf(dT4_half, n);
        // printMatrixFloat(dTmp, n);
        
        CHECK_CUDA(cudaMemcpy(dA_our, dTmp, nn*sizeof(float), cudaMemcpyDeviceToDevice));
        // T1 = A_our^2, T2 = I - T1
        // float2half_kernel<<<blocks,threads>>>(dA_our, dT3_half, nn);
        convertFloatToHalf4(dA_our, dT3_half, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            dT3_half, CUDA_R_16F, n,
            dT3_half, CUDA_R_16F, n,
            &zero,
            dT1,    CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printMatrixHalf(dT3_half, n);
        // printMatrixFloat(dT1, n);

        CHECK_CUBLAS(cublasSgeam(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n,
            &one,  dI,  n,
            &neg1, dT1, n,
            dT2,       n));
        // F = I + (1/sqrt(iter))*T2
        float invs = 1.0f / std::sqrt((float)iter);
        CHECK_CUBLAS(cublasSgeam(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n,
            &one,  dI,  n,
            &invs, dT2, n,
            dF,      n));
        // A_our = A_our * F (to dTmp)
        // float2half_kernel<<<blocks,threads>>>(dA_our, dT3_half, nn);
        // float2half_kernel<<<blocks,threads>>>(dF, dT4_half, nn);
        convertFloatToHalf4(dA_our, dT3_half, nn);
        convertFloatToHalf4(dF, dT4_half, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            dT3_half, CUDA_R_16F, n,
            dT4_half, CUDA_R_16F, n,
            &zero,
            dTmp,   CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printMatrixHalf(dT3_half, n);
        // printMatrixHalf(dT4_half, n);
        // printMatrixFloat(dTmp, n);

        // A_our <-- Tmp
        CHECK_CUDA(cudaMemcpy(dA_our, dTmp, nn*sizeof(float), cudaMemcpyDeviceToDevice));

        // force symmetry: A_our <-- 0.5 * (A_our + Tmp')
        CHECK_CUBLAS(cublasSgeam(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_T,
            n, n,
            &half, dA_our, n,
            &half, dTmp, n,
            dA_our, n));

        // Compute Frobenius norm ||A_our||_F
        float fro_err = 0.0f;
        CHECK_CUBLAS(cublasSnrm2(cublasH, nn, dA_our, 1, &fro_err));

        // printf("Iter: %d | Fro norm = %.10f \n", iter, fro_err);

        // std::cout << std::fixed << std::setprecision(10);
        // std::cout << "Iter " << iter
        //           << " | Fro norm = " << fro_err << "\n";
    }
    // 7) Final combine: mat = mat * (A_our + I) / 2
    CHECK_CUBLAS(cublasSgeam(
        cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n,
        &one, dA_our, n,
        &one, dI,     n,
        dF,            n));
    // float2half_kernel<<<blocks,threads>>>(dA_orig, dT3_half, nn);
    // float2half_kernel<<<blocks,threads>>>(dF, dT4_half, nn);
    convertFloatToHalf4(mat, dT3_half, nn);
    convertFloatToHalf4(dF, dT4_half, nn);
    CHECK_CUBLAS(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &one,
        dT3_half, CUDA_R_16F, n,
        dT4_half, CUDA_R_16F, n,
        &zero,
        dTmp,    CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaMemcpy(mat, dTmp, nn*sizeof(float), D2D));
    CHECK_CUBLAS(cublasSscal(cublasH, nn, &half, mat, 1));
    
    CHECK_CUDA(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end - start;
    // std::cout << "Total time: " << elapsed.count() << " seconds\n";
    return;
}

std::chrono::duration<double> haoyu_TF16_psd(
    cusolverDnHandle_t solverH, cublasHandle_t cublasH, 
    const double* dA_orig, double* dA_psd, size_t n
) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t nn = n * n;
    
    CHECK_CUDA(cudaMemcpy(dA_psd, dA_orig, nn * sizeof(double), cudaMemcpyDeviceToDevice));

    double lo, up;
    approximate_two_norm(
        cublasH, solverH, dA_psd, n, &lo, &up
    ); // TODO: we use a TF16 handle here but the computations are done in FP64
    // printf("lower bound: %5.4e, upper bound: %5.4e \n", lo, up);

    // scale to have eigenvalues in [-1, 1]
    const double scale = up > 0.0 ? up : 1.0;
    // const double scale = 1.0f;
    // printf("scale: %5.4e \n", scale);results/
    const double inv_scale = 1.0/scale;
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, dA_psd, 1) );

    float *dA_psd_float;
    CHECK_CUDA(cudaMalloc(&dA_psd_float, nn*sizeof(float)));
    launch_convert_double_to_float(dA_psd, dA_psd_float, nn);

    haoyu_TF16(
        cublasH, dA_psd_float, n, 0
    );

    launch_convert_float_to_double(dA_psd_float, dA_psd, nn);

    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

    CHECK_CUDA(cudaDeviceSynchronize());

    return std::chrono::high_resolution_clock::now() - start;
}

std::chrono::duration<double> haoyu_TF16_psd_deflate(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dA_psd, size_t n) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t nn = n * n;
    size_t k = K_DEFLATE;
    assert(n > k);
    
    CHECK_CUDA(cudaMemcpy(dA_psd, dA_orig, nn * sizeof(double), cudaMemcpyDeviceToDevice));

    /* Step 1: compute the largest eigenpairs of the matrix */
    size_t r;
    double *eigenvalues, *eigenvectors;
    CHECK_CUDA( cudaMalloc(&eigenvalues,      k * sizeof(double)) );
    CHECK_CUDA( cudaMalloc(&eigenvectors, n * k * sizeof(double)) );

    double _ = compute_eigenpairs(
        cublasH, solverH, dA_psd, n, k, &r, eigenvalues, eigenvectors, false, 0
    );

    std::vector<double> eigenvalues_host(r);
    CHECK_CUDA( cudaMemcpy(eigenvalues_host.data(), eigenvalues, r * sizeof(double), D2H) );

    // double lo, up;
    // approximate_two_norm(
    //     cublasH, solverH, dA_psd, n, &lo, &up
    // ); // TODO: sometimes we use a TF32 handle here but the computations are done in FP64

    /* Step 2: remove the largest eigenvalues from the matrix */
    for (int i = 0; i < r; i++) {
        // X <- X - \lambda_i * v_i v_i^T
        double lambda = -eigenvalues_host[i];
        double *v_i = eigenvectors + i * n;
        CHECK_CUBLAS( cublasDger(cublasH, n, n, &lambda, v_i, 1, v_i, 1, dA_psd, n) );
    }

    /* Step 3: scale the deflated matrix */
    double up = compute_eigenpairs(
        cublasH, solverH, dA_psd, n, 0, nullptr, nullptr, nullptr, true, 100
    );

    // scale to have eigenvalues in [-1, 1]
    const double scale = up > 0.0 ? up : 1.0;
    const double inv_scale = 1.0/scale;
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, dA_psd, 1) );

    float *dA_psd_float;
    CHECK_CUDA(cudaMalloc(&dA_psd_float, nn*sizeof(float)));
    launch_convert_double_to_float(dA_psd, dA_psd_float, nn);

    haoyu_TF16(
        cublasH, dA_psd_float, n, 0
    );

    launch_convert_float_to_double(dA_psd_float, dA_psd, nn);

    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

    for (int i = 0; i < r; i++) {
        // X <- X + \lambda_i * v_i v_i^T
        double lambda = eigenvalues_host[i];
        if (lambda > 0.0) { // only add positive eigenvalues
            double *v_i = eigenvectors + i * n;
            CHECK_CUBLAS( cublasDger(cublasH, n, n, &lambda, v_i, 1, v_i, 1, dA_psd, n) );
        }
    }

    CHECK_CUDA( cudaFree(eigenvalues) );
    CHECK_CUDA( cudaFree(eigenvectors) );
    CHECK_CUDA( cudaFree(dA_psd_float) );
    CHECK_CUDA(cudaDeviceSynchronize());

    return std::chrono::high_resolution_clock::now() - start;
}

std::chrono::duration<double> composite_TF16_psd_deflate(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dA_psd, size_t n) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t nn = n * n;
    size_t k = K_DEFLATE;
    assert(n > k);
    
    CHECK_CUDA(cudaMemcpy(dA_psd, dA_orig, nn * sizeof(double), cudaMemcpyDeviceToDevice));

    /* Step 1: compute the largest eigenpairs of the matrix */
    size_t r;
    double *eigenvalues, *eigenvectors;
    CHECK_CUDA( cudaMalloc(&eigenvalues,      k * sizeof(double)) );
    CHECK_CUDA( cudaMalloc(&eigenvectors, n * k * sizeof(double)) );

    double _ = compute_eigenpairs(
        cublasH, solverH, dA_psd, n, k, &r, eigenvalues, eigenvectors, false, 0
    );

    std::vector<double> eigenvalues_host(r);
    CHECK_CUDA( cudaMemcpy(eigenvalues_host.data(), eigenvalues, r * sizeof(double), D2H) );

    // double lo, up;
    // approximate_two_norm(
    //     cublasH, solverH, dA_psd, n, &lo, &up
    // ); // TODO: sometimes we use a TF32 handle here but the computations are done in FP64

    /* Step 2: remove the largest eigenvalues from the matrix */
    for (int i = 0; i < r; i++) {
        // X <- X - \lambda_i * v_i v_i^T
        double lambda = -eigenvalues_host[i];
        double *v_i = eigenvectors + i * n;
        CHECK_CUBLAS( cublasDger(cublasH, n, n, &lambda, v_i, 1, v_i, 1, dA_psd, n) );
    }

    /* Step 3: scale the deflated matrix */
    double up = compute_eigenpairs(
        cublasH, solverH, dA_psd, n, 0, nullptr, nullptr, nullptr, true, 100
    );

    // scale to have eigenvalues in [-1, 1]
    const double scale = up > 0.0 ? up : 1.0;
    const double inv_scale = 1.0/scale;
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, dA_psd, 1) );

    float *dA_f;
    CHECK_CUDA(cudaMalloc(&dA_f, nn * sizeof(float)));

    launch_convert_double_to_float(
        dA_psd, dA_f, n
    );

    express_TF16(
        cublasH, dA_f, n, 0
    );

    launch_convert_float_to_double(
        dA_f, dA_psd, n
    );

    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

    for (int i = 0; i < r; i++) {
        // X <- X + \lambda_i * v_i v_i^T
        double lambda = eigenvalues_host[i];
        if (lambda > 0.0) { // only add positive eigenvalues
            double *v_i = eigenvectors + i * n;
            CHECK_CUBLAS( cublasDger(cublasH, n, n, &lambda, v_i, 1, v_i, 1, dA_psd, n) );
        }
    }

    CHECK_CUDA( cudaFree(eigenvalues) );
    CHECK_CUDA( cudaFree(eigenvectors) );
    CHECK_CUDA(cudaDeviceSynchronize());

    return std::chrono::high_resolution_clock::now() - start;
}

void append_csv(
    const std::string& filename,
    const std::string& method,
    const std::string& dataset,
    size_t n,
    const std::chrono::duration<double>& duration,
    double relative_error
) {
    std::ofstream file(filename, std::ios_base::app);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    file << dataset << ","
         << n << ","
         << method << ","
         << std::fixed << std::setprecision(17)
         << duration.count() << ","
         << relative_error << "\n";
    file.close();
}

int main(int argc, char* argv[]) {
    std::vector<std::string> datasets;
    std::vector<size_t> instance_sizes;
    int restarts = 1;
    int gemm_restarts = 1;
    std::string gemm_output_file = "results/gemm_results.csv";
    std::string psd_output_file = "results/psd_results.csv";

    // check if the files are empty
    if (RUN_PURE_TESTS) {
        std::ofstream gemm_file(gemm_output_file, std::ios_base::app);
        if (gemm_file.tellp() == 0) {
            gemm_file << "dataset,n,method,time,relative_error\n";
        } else {
            std::cerr << "ERROR: " << gemm_output_file << " already exists and is not empty." << std::endl;
            return 1;
        }
        gemm_file.close();
    }
    std::ofstream psd_file(psd_output_file, std::ios_base::app);
    if (psd_file.tellp() == 0) {
        psd_file << "dataset,n,method,time,relative_error\n";
    } else {
        std::cerr << "ERROR: " << psd_output_file << " already exists and is not empty." << std::endl;
        return 1;
    }
    psd_file.close();

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
        } else if (arg == "--gemmrestarts") {
            if (i + 1 < argc) {
                gemm_restarts = std::stoi(argv[++i]);
            }
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    /* Initialize data and handles */
    std::vector<double> data;
    std::chrono::duration<double> duration(0.0);
    double one = 1.0, neg1 = -1.0;
    double error = 0.0, ref_norm = 0.0, final_err = 0.0;

    cusolverDnHandle_t solverH;
    CHECK_CUSOLVER(cusolverDnCreate(&solverH));
    cublasHandle_t cublasH;
    CHECK_CUBLAS(cublasCreate(&cublasH));
    cublasHandle_t cublasH_TF32;
    CHECK_CUBLAS(cublasCreate(&cublasH_TF32));
    cublasSetMathMode(cublasH_TF32, CUBLAS_TF32_TENSOR_OP_MATH);
    cublasHandle_t cublasH_TF16;
    CHECK_CUBLAS(cublasCreate(&cublasH_TF16));
    CHECK_CUBLAS(cublasSetMathMode(cublasH_TF16, CUBLAS_TENSOR_OP_MATH));

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
            cublasH_TF32, CUBLAS_OP_N, CUBLAS_OP_N,
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
            size_t nn = n * n;
            /* 0) Generate the matrix */
            std::cout << "DATASET '" << dataset << "' WITH INSTANCE SIZE " << n << std::endl;

            // load the matrix from the generated binary file
            std::string filename = "data/bin/" + dataset + "-" + std::to_string(n) + ".bin";
            load_matrix(filename, data, n);

            // copy the matrix to the device
            double *A, *A_psd, *A_psd_ref, *A_diff, *W, *W_ref, *A_eig, *A_eig_ref;
            CHECK_CUDA(cudaMalloc(&A,         n * n * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&A_psd,     n * n * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&A_psd_ref, n * n * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&A_eig,     n * n * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&A_eig_ref, n * n * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&W,             n * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&W_ref,         n * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&A_diff,    n * n * sizeof(double)));
            CHECK_CUDA(cudaMemcpy(A, data.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));


            // /* 1) Pure GEMM and EIG */
            if (RUN_PURE_TESTS) {
                std::cout << "\t Pure EIG and GEMM" << std::endl;

                // cuSOLVER FP64 eig
                duration = std::chrono::duration<double>(0.0);
                error = 0.0;
                for (int i = 0; i < restarts; ++i) {
                    CHECK_CUDA(cudaMemset(A_eig, 0, nn * sizeof(double)));
                    CHECK_CUDA(cudaMemset(W,     0,  n * sizeof(double)));
                    duration += cusolver_FP64_eig(solverH, cublasH, A, W, A_eig, n);

                    if (i == 0) {
                        // copy the reference eigenvectors matrix
                        CHECK_CUDA(cudaMemcpy(A_eig_ref, A_eig, n * n * sizeof(double), cudaMemcpyDeviceToDevice));
                        CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A_eig_ref, 1, &ref_norm));
                    }

                    // compute error
                    CHECK_CUBLAS(cublasDgeam(
                        cublasH,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        n, n,
                        &one,  A_eig_ref, n,
                        &neg1, A_eig, n,
                        A_diff,       n));
                    CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A_diff, 1, &final_err));
                    
                    error += final_err / ref_norm;
                }
                duration /= restarts;
                error /= restarts;
                std::cout << "\t\t cuSOLVER FP64 -- Time: " << std::scientific << duration.count() << " s" << std::endl;
                std::cout << "\t\t        Relative error: " << std::scientific << error << std::endl;
                append_csv(gemm_output_file, "cuSOLVER FP64", dataset, n, duration, error);

                // cuSOLVER FP32 eig
                duration = std::chrono::duration<double>(0.0);
                error = 0.0;
                for (int i = 0; i < restarts; ++i) {
                    CHECK_CUDA(cudaMemset(A_eig, 0, nn * sizeof(double)));
                    CHECK_CUDA(cudaMemset(W,     0,  n * sizeof(double)));
                    duration += cusolver_FP32_eig(solverH, cublasH, A, W, A_eig, n);
                    CHECK_CUDA(cudaDeviceSynchronize());

                    // compute error
                    CHECK_CUBLAS(cublasDgeam(
                        cublasH,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        n, n,
                        &one,  A_eig_ref, n,
                        &neg1, A_eig, n,
                        A_diff,       n));
                    CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A_diff, 1, &final_err));
                    error += final_err / ref_norm;
                }
                duration /= restarts;
                error /= restarts;
                std::cout << "\t\t cuSOLVER FP32 -- Time: " << std::scientific << duration.count() << " s" << std::endl;
                std::cout << "\t\t        Relative error: " << std::scientific << error << std::endl;
                append_csv(gemm_output_file, "cuSOLVER FP32", dataset, n, duration, error);
                
                // FP64
                duration = std::chrono::duration<double>(0.0);
                error = 0.0;
                for (int i = 0; i < restarts; ++i) {
                    CHECK_CUDA(cudaMemset(A_eig, 0, nn * sizeof(double)));
                    duration += FP64_gemm(cublasH, A, A_eig, n, gemm_restarts);
                    CHECK_CUDA(cudaDeviceSynchronize());

                    if (i == 0) {
                        // copy the reference A2 matrix
                        CHECK_CUDA(cudaMemcpy(A_eig_ref, A_eig, n * n * sizeof(double), cudaMemcpyDeviceToDevice));
                        CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A_eig_ref, 1, &ref_norm));
                    }

                    // compute error
                    CHECK_CUBLAS(cublasDgeam(
                        cublasH,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        n, n,
                        &one,  A_eig_ref, n,
                        &neg1, A_eig, n,
                        A_diff,       n));
                    CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A_diff, 1, &final_err));
                    error += final_err;
                }
                duration /= restarts;
                error /= restarts;
                std::cout << "\t\t     GEMM FP64 -- Time: " << std::scientific << duration.count() << " s" << std::endl;
                std::cout << "\t\t           Total error: " << std::scientific << error << std::endl;
                std::cout << "\t\t        Relative error: " << std::scientific << error / ref_norm << std::endl;
                append_csv(gemm_output_file, "GEMM FP64", dataset, n, duration, error);

                // FP32
                duration = std::chrono::duration<double>(0.0);
                error = 0.0;
                for (int i = 0; i < restarts; ++i) {
                    CHECK_CUDA(cudaMemset(A_eig, 0, nn * sizeof(double)));
                    duration += FP32_gemm(cublasH, A, A_eig, n, gemm_restarts);
                    CHECK_CUDA(cudaDeviceSynchronize());

                    // compute error
                    CHECK_CUBLAS(cublasDgeam(
                        cublasH,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        n, n,
                        &one,  A_eig_ref, n,
                        &neg1, A_eig, n,
                        A_diff,       n));
                    CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A_diff, 1, &final_err));
                    error += final_err;
                }
                duration /= restarts;
                error /= restarts;
                std::cout << "\t\t     GEMM FP32 -- Time: " << std::scientific << duration.count() << " s" << std::endl;
                std::cout << "\t\t           Total error: " << std::scientific << error << std::endl;
                std::cout << "\t\t        Relative error: " << std::scientific << error / ref_norm << std::endl;
                append_csv(gemm_output_file, "GEMM FP32", dataset, n, duration, error);

                // TF32
                duration = std::chrono::duration<double>(0.0);
                error = 0.0;
                for (int i = 0; i < restarts; ++i) {
                    CHECK_CUDA(cudaMemset(A_eig, 0, nn * sizeof(double)));
                    duration += FP32_gemm(cublasH, A, A_eig, n, gemm_restarts);
                    CHECK_CUDA(cudaDeviceSynchronize());

                    // compute error
                    CHECK_CUBLAS(cublasDgeam(
                        cublasH_TF32,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        n, n,
                        &one,  A_eig_ref, n,
                        &neg1, A_eig, n,
                        A_diff,       n));
                    CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A_diff, 1, &final_err));
                    error += final_err;
                }
                duration /= restarts;
                error /= restarts;
                std::cout << "\t\t     GEMM TF32 -- Time: " << std::scientific << duration.count() << " s" << std::endl;
                std::cout << "\t\t           Total error: " << std::scientific << error << std::endl;
                std::cout << "\t\t        Relative error: " << std::scientific << error / ref_norm << std::endl;
                append_csv(gemm_output_file, "GEMM TF32", dataset, n, duration, error);

                // TF16
                duration = std::chrono::duration<double>(0.0);
                error = 0.0;
                for (int i = 0; i < restarts; ++i) {
                    CHECK_CUDA(cudaMemset(A_eig, 0, nn * sizeof(double)));
                    duration += TF16_gemm(cublasH, A, A_eig, n, gemm_restarts);
                    CHECK_CUDA(cudaDeviceSynchronize());

                    // compute error
                    CHECK_CUBLAS(cublasDgeam(
                        cublasH,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        n, n,
                        &one,  A_eig_ref, n,
                        &neg1, A_eig, n,
                        A_diff,       n));
                    CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A_diff, 1, &final_err));
                    error += final_err;
                }
                duration /= restarts;
                error /= restarts;
                std::cout << "\t\t     GEMM TF16 -- Time: " << std::scientific << duration.count() << " s" << std::endl;
                std::cout << "\t\t           Total error: " << std::scientific << error << std::endl;
                std::cout << "\t\t        Relative error: " << std::scientific << error / ref_norm << std::endl;
                append_csv(gemm_output_file, "GEMM TF16", dataset, n, duration, error);
            }

            /* 2) PSD cone projection */
            std::cout << "\t PSD cone projection" << std::endl;

            // cuSOLVER FP64
            duration = std::chrono::duration<double>(0.0);
            error = 0.0;
            for (int i = 0; i < restarts; ++i) {
                CHECK_CUDA(cudaMemset(A_psd, 0, nn * sizeof(double)));
                duration += cusolver_FP64_psd(solverH, cublasH, A, A_psd, n);

                if (i == 0) {
                    // copy the reference PSD matrix
                    CHECK_CUDA(cudaMemcpy(A_psd_ref, A_psd, n * n * sizeof(double), cudaMemcpyDeviceToDevice));
                    CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A_psd_ref, 1, &ref_norm));
                }

                // compute error
                CHECK_CUBLAS(cublasDgeam(
                    cublasH,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, n,
                    &one,  A_psd_ref, n,
                    &neg1, A_psd, n,
                    A_diff,       n));
                CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A_diff, 1, &final_err));
                
                error += final_err / ref_norm;
            }
            duration /= restarts;
            error /= restarts;
            std::cout << "\t\t cuSOLVER FP64 -- Time: " << std::scientific << duration.count() << " s" << std::endl;
            std::cout << "\t\t        Relative error: " << std::scientific << error << std::endl;
            append_csv(psd_output_file, "cuSOLVER FP64", dataset, n, duration, error);

            // cuSOLVER FP32
            duration = std::chrono::duration<double>(0.0);
            error = 0.0;
            for (int i = 0; i < restarts; ++i) {
                CHECK_CUDA(cudaMemset(A_psd, 0, nn * sizeof(double)));
                duration += cusolver_FP32_psd(solverH, cublasH, A, A_psd, n);
                CHECK_CUDA(cudaDeviceSynchronize());

                // compute error
                CHECK_CUBLAS(cublasDgeam(
                    cublasH,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, n,
                    &one,  A_psd_ref, n,
                    &neg1, A_psd, n,
                    A_diff,       n));
                CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A_diff, 1, &final_err));
                error += final_err / ref_norm;
            }
            duration /= restarts;
            error /= restarts;
            std::cout << "\t\t cuSOLVER FP32 -- Time: " << std::scientific << duration.count() << " s" << std::endl;
            std::cout << "\t\t        Relative error: " << std::scientific << error << std::endl;
            append_csv(psd_output_file, "cuSOLVER FP32", dataset, n, duration, error);

            // // composite FP64
            // duration = std::chrono::duration<double>(0.0);
            // error = 0.0;
            // for (int i = 0; i < restarts; ++i) {
            //     CHECK_CUDA(cudaMemset(A_psd, 0, nn * sizeof(double)));
            //     duration += composite_FP64_psd(solverH, cublasH, A, A_psd, n);
            //     CHECK_CUDA(cudaDeviceSynchronize());

            //     // compute error
            //     CHECK_CUBLAS(cublasDgeam(
            //         cublasH,
            //         CUBLAS_OP_N, CUBLAS_OP_N,
            //         n, n,
            //         &one,  A_psd_ref, n,
            //         &neg1, A_psd, n,
            //         A_diff,       n));
            //     CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A_diff, 1, &final_err));
            //     error += final_err / ref_norm;
            // }
            // duration /= restarts;
            // error /= restarts;
            // std::cout << "\t\tcomposite FP64 -- Time: " << std::scientific << duration.count() << " s" << std::endl;
            // std::cout << "\t\t        Relative error: " << std::scientific << error << std::endl;
            // append_csv(psd_output_file, "composite FP64", dataset, n, duration, error);

            // composite FP32
            duration = std::chrono::duration<double>(0.0);
            error = 0.0;
            for (int i = 0; i < restarts; ++i) {
                CHECK_CUDA(cudaMemset(A_psd, 0, nn * sizeof(double)));
                duration += composite_FP32_psd(solverH, cublasH, A, A_psd, n);
                CHECK_CUDA(cudaDeviceSynchronize());

                // compute error
                CHECK_CUBLAS(cublasDgeam(
                    cublasH,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, n,
                    &one,  A_psd_ref, n,
                    &neg1, A_psd, n,
                    A_diff,       n));
                CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A_diff, 1, &final_err));
                error += final_err / ref_norm;
            }
            duration /= restarts;
            error /= restarts;
            std::cout << "\t\tcomposite FP32 -- Time: " << std::scientific << duration.count() << " s" << std::endl;
            std::cout << "\t\t        Relative error: " << std::scientific << error << std::endl;
            append_csv(psd_output_file, "composite FP32", dataset, n, duration, error);

            // // composite TF32
            // duration = std::chrono::duration<double>(0.0);
            // error = 0.0;
            // for (int i = 0; i < restarts; ++i) {
            //     CHECK_CUDA(cudaMemset(A_psd, 0, nn * sizeof(double)));
            //     duration += composite_FP32_psd(solverH, cublasH_TF32, A, A_psd, n); // same function, different cuBLAS handle
            //     CHECK_CUDA(cudaDeviceSynchronize());

            //     // compute error
            //     CHECK_CUBLAS(cublasDgeam(
            //         cublasH,
            //         CUBLAS_OP_N, CUBLAS_OP_N,
            //         n, n,
            //         &one,  A_psd_ref, n,
            //         &neg1, A_psd, n,
            //         A_diff,       n));
            //     CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A_diff, 1, &final_err));
            //     error += final_err / ref_norm;
            // }
            // duration /= restarts;
            // error /= restarts;
            // std::cout << "\t\tcomposite TF32 -- Time: " << std::scientific << duration.count() << " s" << std::endl;
            // std::cout << "\t\t        Relative error: " << std::scientific << error << std::endl;
            // append_csv(psd_output_file, "composite TF32", dataset, n, duration, error);

            // composite TF16
            duration = std::chrono::duration<double>(0.0);
            error = 0.0;
            for (int i = 0; i < restarts; ++i) {
                CHECK_CUDA(cudaMemset(A_psd, 0, nn * sizeof(double)));
                duration += composite_TF16_psd(solverH, cublasH_TF16, A, A_psd, n);
                CHECK_CUDA(cudaDeviceSynchronize());

                // compute error
                CHECK_CUBLAS(cublasDgeam(
                    cublasH,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, n,
                    &one,  A_psd_ref, n,
                    &neg1, A_psd, n,
                    A_diff,       n));
                CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A_diff, 1, &final_err));
                error += final_err / ref_norm;
            }
            duration /= restarts;
            error /= restarts;
            std::cout << "\t\tcomposite TF16 -- Time: " << std::scientific << duration.count() << " s" << std::endl;
            std::cout << "\t\t        Relative error: " << std::scientific << error << std::endl;
            append_csv(psd_output_file, "composite TF16", dataset, n, duration, error);

            // haoyu TF16
            duration = std::chrono::duration<double>(0.0);
            error = 0.0;
            for (int i = 0; i < restarts; ++i) {
                CHECK_CUDA(cudaMemset(A_psd, 0, nn * sizeof(double)));
                duration += haoyu_TF16_psd(solverH, cublasH_TF16, A, A_psd, n);
                CHECK_CUDA(cudaDeviceSynchronize());

                // compute error
                CHECK_CUBLAS(cublasDgeam(
                    cublasH,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, n,
                    &one,  A_psd_ref, n,
                    &neg1, A_psd, n,
                    A_diff,       n));
                CHECK_CUBLAS(cublasDnrm2(cublasH, nn, A_diff, 1, &final_err));
                error += final_err / ref_norm;
            }
            duration /= restarts;
            error /= restarts;
            std::cout << "\t\t    haoyu TF16 -- Time: " << std::scientific << duration.count() << " s" << std::endl;
            std::cout << "\t\t        Relative error: " << std::scientific << error << std::endl;
            append_csv(psd_output_file, "haoyu TF16", dataset, n, duration, error);

            /* Clean up */
            CHECK_CUDA(cudaFree(A));
            CHECK_CUDA(cudaFree(A_psd));
            CHECK_CUDA(cudaFree(A_psd_ref));
            CHECK_CUDA(cudaFree(A_diff));
            CHECK_CUDA(cudaFree(A_eig));
            CHECK_CUDA(cudaFree(A_eig_ref));
            CHECK_CUDA(cudaFree(W));
            CHECK_CUDA(cudaFree(W_ref));
            CHECK_CUDA(cudaDeviceSynchronize());
            std::cout << std::endl;
        }
    }

    /* Clean up */
    CHECK_CUSOLVER(cusolverDnDestroy(solverH));

    return 0;
}