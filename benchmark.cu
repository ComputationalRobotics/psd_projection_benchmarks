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

std::chrono::duration<double> cusolver_FP64_eig(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dW, double* dA, size_t n) {
    auto start = std::chrono::high_resolution_clock::now();
    int *devInfo; CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));
    size_t nn = n * n;

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

    auto end = std::chrono::high_resolution_clock::now();
    return end - start;
}

std::chrono::duration<double> cusolver_FP32_eig(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dW, double* dA, size_t n) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t nn = n * n;
    float one_s = 1.0;
    float zero_s = 0.0;
    
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

    auto end = std::chrono::high_resolution_clock::now();
    return end - start;
}


std::chrono::duration<double> cusolver_FP64_psd( cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dA_psd, size_t n) {
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
    auto end = std::chrono::high_resolution_clock::now();
    return end - start;
}

#define D2H cudaMemcpyDeviceToHost
#define H2D cudaMemcpyHostToDevice
#define D2D cudaMemcpyDeviceToDevice

#include <iomanip>
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

std::chrono::duration<double> cusolver_FP32_psd( cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA, double* dA_psd, size_t n) {
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

    auto end = std::chrono::high_resolution_clock::now();
    return end - start;
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

void express_FP32(
    cublasHandle_t cublasH,
    double* mat,
    const int n,
    const int mat_offset
) {
    const int nn = n * n;

    /* Allocations */
    // device memory
    float *A, *A2, *A3, *A5, *I, *W, *Wout;
    CHECK_CUDA( cudaMalloc(&A,  nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&A2, nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&A3, nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&A5, nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&I,  nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&W,  nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&Wout,  nn * sizeof(float)) );

    // useful constants
    const float half       =  0.5f;
    const float minus_half = -0.5f;
    const float one        =  1.0f;
    const float one_n_half =  1.5f;
    const float zero       =  0.0f;

    // build identity I on device
    std::vector<float> I_h(nn, 0.0f);
    for (int i = 0; i < n; i++) I_h[i*n + i] = 1.0f;
    CHECK_CUDA( cudaMemcpy(I, I_h.data(), nn * sizeof(float), H2D) );

    /* Convert the initial matrix*/
    // copy the double matrix back to the host
    std::vector<double> A_h_d(nn);
    CHECK_CUDA( cudaMemcpy(A_h_d.data(), mat + mat_offset, nn * sizeof(double), D2H) );

    // convert the host matrix to float
    std::vector<float> A_h(nn);
    for (int i = 0; i < nn; i++)
        A_h[i] = static_cast<float>(A_h_d[i]);

    // copy the float host matrix to the device
    CHECK_CUDA( cudaMemcpy(A, A_h.data(), nn * sizeof(float), H2D) );
    CHECK_CUDA( cudaMemcpy(W, A_h.data(), nn * sizeof(float), H2D) );

    /* Coefficients */
    std::vector<std::vector<float>> coeff = {
        {8.4724206924, -24.5001735687, 17.7268180847},
        {4.2052841187, -3.0549299717, 0.5567536354},
        {4.0443077087, -2.9473149776, 0.5449726582},
        {3.5078327656, -2.5842490196, 0.5067413449},
        {2.5075511932, -1.8485442400, 0.4358045161}
    };

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

        // A5 = A3 * A2
        CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A3, n, A2, n, &zero, A5, n) );

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
    for (int i =0; i < 3; i++) {
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
    // A = 1 * I + A
    CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &one, I, 1, A, 1) );
    // A = 0.5 * A
    CHECK_CUBLAS( cublasSscal(cublasH, nn, &half, A, 1) );

    /* Symmetrize A */
    symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace

    /* Multiply the original matrix by A */
    // Wout = W * A
    CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, W, n, A, n, &zero, Wout, n) );

    /* Symmetrize W */
    symmetrizeFloat(cublasH, Wout, n, A2); // we use A2 as a workspace

    /* Copy the result back to mat */
    std::vector<float> A_h_f(nn);
    CHECK_CUDA( cudaMemcpy(A_h_f.data(), Wout, nn * sizeof(float), D2H) );
    for (size_t i = 0; i < nn; i++) {
        A_h_d[i] = static_cast<double>(A_h_f[i]);
    }
    CHECK_CUDA( cudaMemcpy(mat + mat_offset, A_h_d.data(), nn * sizeof(double), H2D) );
    CHECK_CUDA( cudaDeviceSynchronize() );

    /* Free device memory */
    CHECK_CUDA( cudaFree(A) );
    CHECK_CUDA( cudaFree(A2) );
    CHECK_CUDA( cudaFree(A3) );
    CHECK_CUDA( cudaFree(A5) );
    CHECK_CUDA( cudaFree(I) );
    CHECK_CUDA( cudaFree(W) );
    CHECK_CUDA( cudaFree(Wout) );
}

void express_FP64(
    cublasHandle_t cublasH,
    double* mat,
    const int n,
    const int mat_offset = 0
) {
    const int nn = n * n;

    /* Allocations */
    // device memory
    double *A, *A2, *A3, *A5, *I, *W, *Wout;
    CHECK_CUDA( cudaMalloc(&A,  nn * sizeof(double)) );
    CHECK_CUDA( cudaMalloc(&A2, nn * sizeof(double)) );
    CHECK_CUDA( cudaMalloc(&A3, nn * sizeof(double)) );
    CHECK_CUDA( cudaMalloc(&A5, nn * sizeof(double)) );
    CHECK_CUDA( cudaMalloc(&I,  nn * sizeof(double)) );
    CHECK_CUDA( cudaMalloc(&W,  nn * sizeof(double)) );
    CHECK_CUDA( cudaMalloc(&Wout,  nn * sizeof(double)) );

    // useful constants
    const double half       =  0.5;
    const double minus_half = -0.5;
    const double one        =  1.0;
    const double one_n_half =  1.5;
    const double zero       =  0.0;

    // build identity I on device
    std::vector<double> I_h(nn, 0.0f);
    for (int i = 0; i < n; i++) I_h[i*n + i] = 1.0f;
    CHECK_CUDA( cudaMemcpy(I, I_h.data(), nn * sizeof(double), H2D) );

    CHECK_CUDA( cudaMemcpy(A, mat + offset, nn * sizeof(double), D2D) );

    /* Coefficients */
    std::vector<std::vector<double>> coeff = {
        {8.4724206924, -24.5001735687, 17.7268180847},
        {4.2052841187, -3.0549299717, 0.5567536354},
        {4.0443077087, -2.9473149776, 0.5449726582},
        {3.5078327656, -2.5842490196, 0.5067413449},
        {2.5075511932, -1.8485442400, 0.4358045161}
    };

    /* Approximation of the step function */
    for (int i = 0; i < coeff.size(); i++) {
        const double a = coeff[i][0];
        const double b = coeff[i][1];
        const double c = coeff[i][2];

        /* Compute the powers of A*/
        // A2 = A * A
        CHECK_CUBLAS( cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A, n, A, n, &zero, A2, n) );

        // A3 = A2 * A
        CHECK_CUBLAS( cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A2, n, A, n, &zero, A3, n) );

        // A5 = A3 * A2
        CHECK_CUBLAS( cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A3, n, A2, n, &zero, A5, n) );

        /* Symmetrize A3, A5 */ // TODO:
        symmetrizeDouble(cublasH, A3, n, A2); // we use A2 as a workspace
        symmetrizeDouble(cublasH, A5, n, A2); // we use A2 as a workspace

        /* Compute A = a * A + b * A3 + c * A5 */
        // A = a * A
        CHECK_CUBLAS( cublasDscal(cublasH, nn, &a, A, 1) );
        // A = b * A3 + A
        CHECK_CUBLAS( cublasDaxpy(cublasH, nn, &b, A3, 1, A, 1) );
        // A = c * A5 + A
        CHECK_CUBLAS( cublasDaxpy(cublasH, nn, &c, A5, 1, A, 1) );

        /* Symmetrize A */
        symmetrizeDouble(cublasH, A, n, A2); // we use A2 as a workspace
    }

    /* Smoothing function */
    for (int i =0; i < 3; i++) {
        // A2 = A * A
        CHECK_CUBLAS( cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A, n, A, n, &zero, A2, n) );

        // A3 = A2 * A
        CHECK_CUBLAS( cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A2, n, A, n, &zero, A3, n) );

        /* Symmetrize A3 */
        symmetrizeDouble(cublasH, A3, n, A2); // we use A2 as a workspace

        /* Compute A = 1.5 * A - 0.5 * A3 */
        // A = 1.5 * A
        CHECK_CUBLAS( cublasDscal(cublasH, nn, &one_n_half, A, 1) );
        // A = -0.5 * A3 + A
        CHECK_CUBLAS( cublasDaxpy(cublasH, nn, &minus_half, A3, 1, A, 1) );

        /* Symmetrize A */
        symmetrizeDouble(cublasH, A, n, A2); // we use A2 as a workspace
    }

    /* Compute A = (I + A)/2 */
    // A = 1 * I + A
    CHECK_CUBLAS( cublasDaxpy(cublasH, nn, &one, I, 1, A, 1) );
    // A = 0.5 * A
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &half, A, 1) );

    /* Symmetrize A */
    symmetrizeDouble(cublasH, A, n, A2); // we use A2 as a workspace

    /* Multiply the original matrix by A */
    // Wout = W * A
    CHECK_CUBLAS( cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, W, n, A, n, &zero, Wout, n) );

    /* Symmetrize W */
    symmetrizeDouble(cublasH, Wout, n, A2); // we use A2 as a workspace

    /* Copy the result back to mat */
    CHECK_CUDA( cudaMemcpy(mat + mat_offset, A, nn * sizeof(double), D2D) );
    CHECK_CUDA( cudaDeviceSynchronize() );

    /* Free device memory */
    CHECK_CUDA( cudaFree(A) );
    CHECK_CUDA( cudaFree(A2) );
    CHECK_CUDA( cudaFree(A3) );
    CHECK_CUDA( cudaFree(A5) );
    CHECK_CUDA( cudaFree(I) );
    CHECK_CUDA( cudaFree(W) );
    CHECK_CUDA( cudaFree(Wout) );
}

void approximate_two_norm(
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH,
    const double* A, size_t n,
    double* lo, double* up,
    size_t max_iter=25, double tol=1e-4
) {
    /* Allocations */
    // constants
    const double zero = 0.0;
    const double one = 1.0;
    
    // storage
    double *V, *V_old, *alpha, *q, *w, *AtA;
    CHECK_CUDA(cudaMalloc(&V,     n * max_iter * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&V_old,            n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&alpha,     max_iter * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&q,                n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&w,                n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&AtA,          n * n * sizeof(double)));

    std::vector<double> beta(max_iter, 0.0);

    // precompute A^T * A
    CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
                             &one, A, n, A, n,
                             &zero, AtA, n));

    double minus_alpha, minus_beta_old;

    /* Initial vector */
    // q = randn(n, 1)
    std::vector<double> q_host(n);
    for (size_t i = 0; i < n; ++i) {
        q_host[i] = 2.0 * (static_cast<double>(rand()) / RAND_MAX) - 1.0; // Random values in [-1, 1)
    }
    CHECK_CUDA(cudaMemcpy(q, q_host.data(), n * sizeof(double), cudaMemcpyHostToDevice));

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
    CHECK_CUBLAS(cublasDscal(cublasH, n, &zero, V_old, 1));

    /* Lanczos loop */
    int nb_iter = 0;
    for (int k = 0; k < max_iter; k++) {
        // w = A^T * A * q
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, n,
                                 &one, AtA, n, q, n,
                                 &zero, w, n));

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
            fprintf(stderr, "Lanczos iteration %d: beta is zero, stopping early.\n", k);
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
    // uk = V(:, idx_max)
    CHECK_CUBLAS(cublasDcopy(cublasH, nb_iter, V + idx_max * n, 1, uk, 1));
    // y = V(:,1:nb_iter) * uk
    CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_N, n, nb_iter, &one, V, n, uk, 1, &zero, y, 1));

    // ry = A^T * A * y
    double *ry;
    CHECK_CUDA(cudaMalloc(&ry, n * sizeof(double)));
    CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_T, n, n, &one, AtA, n, y, 1, &zero, ry, 1));
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
    CHECK_CUDA(cudaFree(AtA));
    CHECK_CUDA(cudaFree(uk));
    CHECK_CUDA(cudaFree(y));
    CHECK_CUDA(cudaFree(ry));
}

std::chrono::duration<double> composite_FP32_psd(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dA_psd, size_t n) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t nn = n * n;
    
    CHECK_CUDA(cudaMemcpy(dA_psd, dA_orig, nn * sizeof(double), cudaMemcpyDeviceToDevice));

    double lo, up;
    approximate_two_norm(
        cublasH, solverH, dA_psd, n, &lo, &up
    );

    // scale to have eigenvalues in [-1, 1]
    const double scale = up > 0.0f ? up : 1.0f;
    // const double scale = 1.0f;
    const double inv_scale = 1.0f/scale;
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, dA_psd, 1) );

    express_FP32(
        cublasH, dA_psd, n, 0
    );

    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

    CHECK_CUDA(cudaDeviceSynchronize());

    return std::chrono::high_resolution_clock::now() - start;
}

std::chrono::duration<double> composite_FP64_psd(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dA_psd, size_t n) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t nn = n * n;
    
    CHECK_CUDA(cudaMemcpy(dA_psd, dA_orig, nn * sizeof(double), cudaMemcpyDeviceToDevice));

    double lo, up;
    approximate_two_norm(
        cublasH, solverH, dA_psd, n, &lo, &up
    );

    // scale to have eigenvalues in [-1, 1]
    const double scale = up > 0.0f ? up : 1.0f;
    // const double scale = 1.0f;
    const double inv_scale = 1.0f/scale;
    CHECK_CUBLAS( cublasDscal(cublasH, nn, &inv_scale, dA_psd, 1) );

    express_FP64(
        cublasH, dA_psd, n, 0
    );

    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

    CHECK_CUDA(cudaDeviceSynchronize());

    return std::chrono::high_resolution_clock::now() - start;
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
    std::chrono::duration<double> duration(0.0);
    double one = 1.0, neg1 = -1.0;
    double error = 0.0, ref_norm = 0.0, final_err = 0.0;

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


            /* 1) Pure GEMM and EIG times */
            std::cout << "\t Pure EIG and GEMM times" << std::endl;

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

            // cuSOLVER FP32 eig
            duration = std::chrono::duration<double>(0.0);
            error = 0.0;
            for (int i = 0; i < restarts; ++i) {
                CHECK_CUDA(cudaMemset(A_eig, 0, nn * sizeof(double)));
                CHECK_CUDA(cudaMemset(W,     0,  n * sizeof(double)));
                duration += cusolver_FP32_psd(solverH, cublasH, A, W, A_eig, n);
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
            
            // TF16
            // TF32
            // FP32
            // FP64

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

            // composite TF16
            // composite TF32
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

            // composite FP64
            duration = std::chrono::duration<double>(0.0);
            error = 0.0;
            for (int i = 0; i < restarts; ++i) {
                CHECK_CUDA(cudaMemset(A_psd, 0, nn * sizeof(double)));
                duration += composite_FP64_psd(solverH, cublasH, A, A_psd, n);
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
            std::cout << "\t\tcomposite FP64 -- Time: " << std::scientific << duration.count() << " s" << std::endl;
            std::cout << "\t\t        Relative error: " << std::scientific << error << std::endl;

            /* Clean up */
            CHECK_CUDA(cudaFree(A));
            CHECK_CUDA(cudaFree(A_psd));
            CHECK_CUDA(cudaFree(A_psd_ref));
            CHECK_CUDA(cudaFree(A_diff));
            CHECK_CUDA(cudaFree(A_eig));
            CHECK_CUDA(cudaFree(A_eig_ref));
            CHECK_CUDA(cudaFree(W));
            CHECK_CUDA(cudaFree(W_ref));
            std::cout << std::endl;
        }
    }

    /* Clean up */
    CHECK_CUSOLVER(cusolverDnDestroy(solverH));

    return 0;
}