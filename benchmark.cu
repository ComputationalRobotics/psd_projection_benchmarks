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
#include <iomanip>
#include <assert.h>

#include "psd_projection/check.h"
#include "psd_projection/utils.h"
#include "psd_projection/lanczos.h"
#include "psd_projection/composite_FP32.h"
#include "psd_projection/composite_FP32_emulated.h"
#include "psd_projection/composite_TF16.h"
#include "psd_projection/haoyu_TF16.h"
#include "psd_projection/haoyu_FP32.h"

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

    convert_double_to_float(dA_orig, sA, nn);

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
   convert_float_to_double(sA, dA, nn);

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

std::chrono::duration<double> cusolver_FP32_psd(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA, double* dA_psd, size_t n) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t nn = n * n;
    float one_s = 1.0;
    float zero_s = 0.0;
    
    int *devInfo; CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));
    float *sA, *sA_psd;
    CHECK_CUDA(cudaMalloc(&sA, nn*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sA_psd, nn*sizeof(float)));
    
    // convert dA from double to float
    convert_double_to_float(dA, sA, nn);

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
    CHECK_CUDA(cudaMemcpy(W_h.data(), sW, n*sizeof(float), D2H));
    for(int i=0;i<n;i++) if(W_h[i]<0) W_h[i]=0;

    // Copy eigenvectors from dA to dV
    float *sV; CHECK_CUDA(cudaMalloc(&sV, nn*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(sV, sA, nn*sizeof(float), D2D));

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
    CHECK_CUDA(cudaMemcpy(sA_psd, sTmp, nn*sizeof(float), D2D));

    convert_float_to_double(sA_psd, dA_psd, nn);

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


// void composite_FP64(
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

std::chrono::duration<double> composite_FP32_psd(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dA_psd, size_t n) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t nn = n * n;
    
    CHECK_CUDA(cudaMemcpy(dA_psd, dA_orig, nn * sizeof(double), cudaMemcpyDeviceToDevice));

    double lo, up;
    approximate_two_norm(
        cublasH, solverH, dA_psd, n, &lo, &up
    );

    // scale to have eigenvalues in [-1, 1]
    const double scale = up > 0.0 ? up : 1.0;
    const double inv_scale = 1.0/scale;
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, dA_psd, 1) );

    composite_FP32(
        cublasH, dA_psd, n, 0
    );

    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

    CHECK_CUDA(cudaDeviceSynchronize());

    return std::chrono::high_resolution_clock::now() - start;
}

std::chrono::duration<double> composite_FP32_emulated_psd(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dA_psd, size_t n) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t nn = n * n;
    
    CHECK_CUDA(cudaMemcpy(dA_psd, dA_orig, nn * sizeof(double), cudaMemcpyDeviceToDevice));

    double lo, up;
    approximate_two_norm(
        cublasH, solverH, dA_psd, n, &lo, &up
    );

    // scale to have eigenvalues in [-1, 1]
    const double scale = up > 0.0 ? up : 1.0;
    const double inv_scale = 1.0/scale;
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, dA_psd, 1) );

    composite_FP32_emulated(
        cublasH, dA_psd, n, 0
    );

    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

    CHECK_CUDA(cudaDeviceSynchronize());

    return std::chrono::high_resolution_clock::now() - start;
}

// std::chrono::duration<double> composite_FP32_psd_deflate(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dA_psd, size_t n) {
//     auto start = std::chrono::high_resolution_clock::now();
//     size_t nn = n * n;
//     size_t k = K_DEFLATE;
//     assert(n > k);
    
//     CHECK_CUDA(cudaMemcpy(dA_psd, dA_orig, nn * sizeof(double), cudaMemcpyDeviceToDevice));

//     /* Step 1: compute the largest eigenpairs of the matrix */
//     size_t r;
//     double *eigenvalues, *eigenvectors;
//     CHECK_CUDA( cudaMalloc(&eigenvalues,      k * sizeof(double)) );
//     CHECK_CUDA( cudaMalloc(&eigenvectors, n * k * sizeof(double)) );

//     double _ = compute_eigenpairs(
//         cublasH, solverH, dA_psd, n, k, &r, eigenvalues, eigenvectors, false, 0
//     );

//     std::vector<double> eigenvalues_host(r);
//     CHECK_CUDA( cudaMemcpy(eigenvalues_host.data(), eigenvalues, r * sizeof(double), D2H) );

//     /* Step 2: remove the largest eigenvalues from the matrix */
//     for (int i = 0; i < r; i++) {
//         // X <- X - \lambda_i * v_i v_i^T
//         double lambda = -eigenvalues_host[i];
//         double *v_i = eigenvectors + i * n;
//         CHECK_CUBLAS( cublasDger(cublasH, n, n, &lambda, v_i, 1, v_i, 1, dA_psd, n) );
//     }

//     /* Step 3: scale the deflated matrix */
//     double up = compute_eigenpairs(
//         cublasH, solverH, dA_psd, n, 0, nullptr, nullptr, nullptr, true, 100
//     );

//     // scale to have eigenvalues in [-1, 1]
//     const double scale = up > 0.0 ? up : 1.0;
//     const double inv_scale = 1.0/scale;
//     CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, dA_psd, 1) );

//     composite_FP32(
//         cublasH, dA_psd, n, 0
//     );

//     CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

//     for (int i = 0; i < r; i++) {
//         // X <- X + \lambda_i * v_i v_i^T
//         double lambda = eigenvalues_host[i];
//         if (lambda > 0.0) { // only add positive eigenvalues
//             double *v_i = eigenvectors + i * n;
//             CHECK_CUBLAS( cublasDger(cublasH, n, n, &lambda, v_i, 1, v_i, 1, dA_psd, n) );
//         }
//     }

//     CHECK_CUDA( cudaFree(eigenvalues) );
//     CHECK_CUDA( cudaFree(eigenvectors) );
//     CHECK_CUDA(cudaDeviceSynchronize());

//     return std::chrono::high_resolution_clock::now() - start;
// }

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

//     composite_FP64(
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
    CHECK_CUDA( cudaMalloc(&sA,  nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&sA2, nn * sizeof(float)) );

    convert_double_to_float(dA_orig, sA, nn);

    for (int i = 0; i < gemm_restarts; i++) {
        CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, sA, n, sA, n, &zero, sA2, n) );
    }
    CHECK_CUDA( cudaDeviceSynchronize() );
    
    convert_float_to_double(sA2, dA2, nn);
    
    CHECK_CUDA( cudaFree(sA) );
    CHECK_CUDA( cudaFree(sA2) );

    CHECK_CUDA( cudaDeviceSynchronize() );

    return (std::chrono::high_resolution_clock::now() - start) / static_cast<double>(gemm_restarts);
}

std::chrono::duration<double> TF16_gemm(cublasHandle_t cublasH, const double* dA_orig, double* dA2, size_t n, int gemm_restarts) {
    auto start = std::chrono::high_resolution_clock::now();
    float one = 1.0;
    float zero = 0.0;
    size_t nn = n*n;

    float *sA, *sA2;
    CHECK_CUDA( cudaMalloc(&sA,  nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&sA2, nn * sizeof(float)) );

    convert_double_to_float(dA_orig, sA, nn);

    __half *hA; 
    CHECK_CUDA(cudaMalloc(&hA, nn*sizeof(__half)));
    convert_float_to_half4(sA, hA, nn);

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
    
    convert_float_to_double(sA2, dA2, nn);
    
    CHECK_CUDA( cudaDeviceSynchronize() );
    CHECK_CUDA( cudaFree(sA) );
    CHECK_CUDA( cudaFree(sA2) );
    CHECK_CUDA( cudaFree(hA) );

    return (std::chrono::high_resolution_clock::now() - start) / static_cast<double>(gemm_restarts);
}

std::chrono::duration<double> composite_TF16_psd(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dA_psd, size_t n) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t nn = n * n;
    
    CHECK_CUDA(cudaMemcpy(dA_psd, dA_orig, nn * sizeof(double), cudaMemcpyDeviceToDevice));

    double lo, up;
    approximate_two_norm(
        cublasH, solverH, dA_psd, n, &lo, &up
    ); // TODO: we use a TF16 handle here but the computations are done in FP64

    // scale to have eigenvalues in [-1, 1]
    const double scale = up > 0.0 ? up : 1.0;
    const double inv_scale = 1.0/scale;
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, dA_psd, 1) );

    composite_TF16(cublasH, dA_psd, n);

    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

    CHECK_CUDA(cudaDeviceSynchronize());

    return std::chrono::high_resolution_clock::now() - start;
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
    convert_double_to_float(dA_psd, dA_psd_float, nn);

    haoyu_TF16(cublasH, dA_psd_float, n);

    convert_float_to_double(dA_psd_float, dA_psd, nn);

    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(dA_psd_float));

    return std::chrono::high_resolution_clock::now() - start;
}

std::chrono::duration<double> haoyu_FP32_psd(
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

    // scale to have eigenvalues in [-1, 1]
    const double scale = up > 0.0 ? up : 1.0;
    const double inv_scale = 1.0/scale;
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, dA_psd, 1) );

    float *dA_psd_float;
    CHECK_CUDA(cudaMalloc(&dA_psd_float, nn*sizeof(float)));
    convert_double_to_float(dA_psd, dA_psd_float, nn);

    haoyu_FP32(cublasH, dA_psd_float, n);

    convert_float_to_double(dA_psd_float, dA_psd, nn);

    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(dA_psd_float));

    return std::chrono::high_resolution_clock::now() - start;
}

// std::chrono::duration<double> haoyu_TF16_psd_deflate(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dA_psd, size_t n) {
//     auto start = std::chrono::high_resolution_clock::now();
//     size_t nn = n * n;
//     size_t k = K_DEFLATE;
//     assert(n > k);
    
//     CHECK_CUDA(cudaMemcpy(dA_psd, dA_orig, nn * sizeof(double), cudaMemcpyDeviceToDevice));

//     /* Step 1: compute the largest eigenpairs of the matrix */
//     size_t r;
//     double *eigenvalues, *eigenvectors;
//     CHECK_CUDA( cudaMalloc(&eigenvalues,      k * sizeof(double)) );
//     CHECK_CUDA( cudaMalloc(&eigenvectors, n * k * sizeof(double)) );

//     double _ = compute_eigenpairs(
//         cublasH, solverH, dA_psd, n, k, &r, eigenvalues, eigenvectors, false, 0
//     );

//     std::vector<double> eigenvalues_host(r);
//     CHECK_CUDA( cudaMemcpy(eigenvalues_host.data(), eigenvalues, r * sizeof(double), D2H) );

//     /* Step 2: remove the largest eigenvalues from the matrix */
//     for (int i = 0; i < r; i++) {
//         // X <- X - \lambda_i * v_i v_i^T
//         double lambda = -eigenvalues_host[i];
//         double *v_i = eigenvectors + i * n;
//         CHECK_CUBLAS( cublasDger(cublasH, n, n, &lambda, v_i, 1, v_i, 1, dA_psd, n) );
//     }

//     /* Step 3: scale the deflated matrix */
//     double up = compute_eigenpairs(
//         cublasH, solverH, dA_psd, n, 0, nullptr, nullptr, nullptr, true, 100
//     );

//     // scale to have eigenvalues in [-1, 1]
//     const double scale = up > 0.0 ? up : 1.0;
//     const double inv_scale = 1.0/scale;
//     CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, dA_psd, 1) );

//     float *dA_psd_float;
//     CHECK_CUDA(cudaMalloc(&dA_psd_float, nn*sizeof(float)));
//     convert_double_to_float(dA_psd, dA_psd_float, nn);

//     haoyu_TF16(cublasH, dA_psd_float, n);

//     convert_float_to_double(dA_psd_float, dA_psd, nn);

//     CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

//     for (int i = 0; i < r; i++) {
//         // X <- X + \lambda_i * v_i v_i^T
//         double lambda = eigenvalues_host[i];
//         if (lambda > 0.0) { // only add positive eigenvalues
//             double *v_i = eigenvectors + i * n;
//             CHECK_CUBLAS( cublasDger(cublasH, n, n, &lambda, v_i, 1, v_i, 1, dA_psd, n) );
//         }
//     }

//     CHECK_CUDA( cudaFree(eigenvalues) );
//     CHECK_CUDA( cudaFree(eigenvectors) );
//     CHECK_CUDA( cudaFree(dA_psd_float) );
//     CHECK_CUDA(cudaDeviceSynchronize());

//     return std::chrono::high_resolution_clock::now() - start;
// }

// std::chrono::duration<double> composite_TF16_psd_deflate(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dA_psd, size_t n) {
//     auto start = std::chrono::high_resolution_clock::now();
//     size_t nn = n * n;
//     size_t k = K_DEFLATE;
//     assert(n > k);
    
//     CHECK_CUDA(cudaMemcpy(dA_psd, dA_orig, nn * sizeof(double), cudaMemcpyDeviceToDevice));

//     /* Step 1: compute the largest eigenpairs of the matrix */
//     size_t r;
//     double *eigenvalues, *eigenvectors;
//     CHECK_CUDA( cudaMalloc(&eigenvalues,      k * sizeof(double)) );
//     CHECK_CUDA( cudaMalloc(&eigenvectors, n * k * sizeof(double)) );

//     double _ = compute_eigenpairs(
//         cublasH, solverH, dA_psd, n, k, &r, eigenvalues, eigenvectors, false, 0
//     );

//     std::vector<double> eigenvalues_host(r);
//     CHECK_CUDA( cudaMemcpy(eigenvalues_host.data(), eigenvalues, r * sizeof(double), D2H) );

//     /* Step 2: remove the largest eigenvalues from the matrix */
//     for (int i = 0; i < r; i++) {
//         // X <- X - \lambda_i * v_i v_i^T
//         double lambda = -eigenvalues_host[i];
//         double *v_i = eigenvectors + i * n;
//         CHECK_CUBLAS( cublasDger(cublasH, n, n, &lambda, v_i, 1, v_i, 1, dA_psd, n) );
//     }

//     /* Step 3: scale the deflated matrix */
//     double up = compute_eigenpairs(
//         cublasH, solverH, dA_psd, n, 0, nullptr, nullptr, nullptr, true, 100
//     );

//     // scale to have eigenvalues in [-1, 1]
//     const double scale = up > 0.0 ? up : 1.0;
//     const double inv_scale = 1.0/scale;
//     CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, dA_psd, 1) );

//     float *dA_f;
//     CHECK_CUDA(cudaMalloc(&dA_f, nn * sizeof(float)));

//     convert_double_to_float(
//         dA_psd, dA_f, n
//     );

//     composite_TF16(
//         cublasH, dA_f, n, 0
//     );

//     convert_float_to_double(
//         dA_f, dA_psd, n
//     );

//     CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

//     for (int i = 0; i < r; i++) {
//         // X <- X + \lambda_i * v_i v_i^T
//         double lambda = eigenvalues_host[i];
//         if (lambda > 0.0) { // only add positive eigenvalues
//             double *v_i = eigenvectors + i * n;
//             CHECK_CUBLAS( cublasDger(cublasH, n, n, &lambda, v_i, 1, v_i, 1, dA_psd, n) );
//         }
//     }

//     CHECK_CUDA( cudaFree(eigenvalues) );
//     CHECK_CUDA( cudaFree(eigenvectors) );
//     CHECK_CUDA(cudaDeviceSynchronize());

//     return std::chrono::high_resolution_clock::now() - start;
// }

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
        // return 1;
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

    cublasHandle_t cublasH_emulated;
    CHECK_CUBLAS(cublasCreate(&cublasH_emulated));
    CHECK_CUBLAS(cublasSetMathMode(cublasH_emulated, CUBLAS_TENSOR_OP_MATH));
    CHECK_CUBLAS(cublasSetEmulationStrategy(cublasH_emulated, CUBLAS_EMULATION_STRATEGY_EAGER));


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
            cublasH_TF16, CUBLAS_OP_N, CUBLAS_OP_N,
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

            // composite FP32 emulated
            duration = std::chrono::duration<double>(0.0);
            error = 0.0;
            for (int i = 0; i < restarts; ++i) {
                CHECK_CUDA(cudaMemset(A_psd, 0, nn * sizeof(double)));
                duration += composite_FP32_emulated_psd(solverH, cublasH_emulated, A, A_psd, n);
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
            std::cout << "\t\tcomposite FP32 emulated -- Time: " << std::scientific << duration.count() << " s" << std::endl;
            std::cout << "\t\t        Relative error: " << std::scientific << error << std::endl;
            append_csv(psd_output_file, "composite FP32 emulated", dataset, n, duration, error);

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

            // // haoyu FP32
            // duration = std::chrono::duration<double>(0.0);
            // error = 0.0;
            // for (int i = 0; i < restarts; ++i) {
            //     CHECK_CUDA(cudaMemset(A_psd, 0, nn * sizeof(double)));
            //     duration += haoyu_FP32_psd(solverH, cublasH, A, A_psd, n);
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
            // std::cout << "\t\t    haoyu FP32 -- Time: " << std::scientific << duration.count() << " s" << std::endl;
            // std::cout << "\t\t        Relative error: " << std::scientific << error << std::endl;
            // append_csv(psd_output_file, "haoyu FP32", dataset, n, duration, error);

            // // haoyu TF16
            // duration = std::chrono::duration<double>(0.0);
            // error = 0.0;
            // for (int i = 0; i < restarts; ++i) {
            //     CHECK_CUDA(cudaMemset(A_psd, 0, nn * sizeof(double)));
            //     duration += haoyu_TF16_psd(solverH, cublasH_TF16, A, A_psd, n);
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
            // std::cout << "\t\t    haoyu TF16 -- Time: " << std::scientific << duration.count() << " s" << std::endl;
            // std::cout << "\t\t        Relative error: " << std::scientific << error << std::endl;
            // append_csv(psd_output_file, "haoyu TF16", dataset, n, duration, error);

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
    CHECK_CUBLAS(cublasDestroy(cublasH));
    CHECK_CUBLAS(cublasDestroy(cublasH_TF16));
    CHECK_CUBLAS(cublasDestroy(cublasH_TF32));
    return 0;
}