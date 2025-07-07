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
#include <cuda.h>
#include <random>

#define RUN_PURE_TESTS false
#define RUN_BASELINES true
#define K_DEFLATE 30 // must be greater than 0, otherwise use non-deflate versions

#define D2H cudaMemcpyDeviceToHost
#define H2D cudaMemcpyHostToDevice
#define D2D cudaMemcpyDeviceToDevice

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


/* ------------------BEGIN: helper functions -------------------------- */
unsigned long make_seed()
{
    std::random_device rd;

    std::seed_seq seq{
        rd(), rd(), rd(), rd(),
        static_cast<unsigned>(std::chrono::high_resolution_clock::now()
                              .time_since_epoch().count())   // mixes in time
    };

    std::mt19937_64 mixer(seq);   // 64-bit Mersenne Twister
    return mixer();               // one well-mixed 64-bit value
}

void load_matrix(const std::string& filename, std::vector<double>& data, const int64_t instance_size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file\n";
        throw std::runtime_error("Cannot open file");
    }

    data.resize(instance_size * instance_size);
    file.read(reinterpret_cast<char*>(data.data()), instance_size * instance_size * sizeof(double));
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

void symmetrizeFloat(
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

__global__ void convert_double_to_float_kernel(const double* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<float>(in[idx]);
    }
}

__global__ void convert_float_to_double_kernel(const float* in, double* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<double>(in[idx]);
    }
}

void convert_double_to_float(const double* d_in, float* d_out, int n, const int threadsPerBlock = 1024) {
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    convert_double_to_float_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
}

void convert_float_to_double(const float* d_in, double* d_out, int n, const int threadsPerBlock = 1024) {
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    convert_float_to_double_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
}

__global__ void build_identity_kernel(float* mat, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n)
        mat[idx] = (idx / n == idx % n) ? 1.0f : 0.0f;
}

void build_identity(
    cublasHandle_t cublasH,
    float* mat,
    int n,
    const int threadsPerBlock = 1024
) {
    const int blocksPerGrid = (n * n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel to build identity matrix
    build_identity_kernel<<<blocksPerGrid, threadsPerBlock>>>(mat, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void add_identity_kernel(float* mat, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx / n;
        int col = idx % n;
        if (row == col) {
            mat[idx] += 1.0f; // Add 1 to the diagonal elements
        }
    }
}

void add_identity(
    cublasHandle_t cublasH,
    float* mat,
    int n
) {
    const int threadsPerBlock = 1024;
    const int blocksPerGrid = (n * n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel to add identity matrix
    add_identity_kernel<<<blocksPerGrid, threadsPerBlock>>>(mat, n);
    CHECK_CUDA(cudaGetLastError());
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

void convert_float_to_half4(const float* dA, __half* dB, size_t N) {
    size_t N4 = (N + 3)/4;  // how many float4’s
    auto A4 = reinterpret_cast<const float4*>(dA);
    auto B2 = reinterpret_cast<__half2*>(dB);

    const int blk = 1024;
    int grid = (N4 + blk - 1)/blk;
    float4_to_half_kernel<<<grid,blk>>>(A4, B2, N4);
}

// Kernel to replace A by I - A
__global__ void identity_minus_kernel(const float* A_in, float* A_out, const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx / n;
        int col = idx % n;
        if (row == col) {
            A_out[idx] = 1.0f - A_in[idx]; // diagonal elements
        } else {
            A_out[idx] = -A_in[idx]; // off-diagonal elements
        }
    }
}

void identity_minus(
    const float* A_in,
    float* A_out,
    const int n
) {
    const int nn = n * n;
    const int threads = 1024;
    const int blocks = (nn + threads - 1) / threads;

    // Launch kernel to compute I - A
    identity_minus_kernel<<<blocks, threads>>>(A_in, A_out, n);
    CHECK_CUDA(cudaGetLastError());
}

// Kernel to replace A by I + A
__global__ void identity_plus_kernel(const float* A_in, float* A_out, const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx / n;
        int col = idx % n;
        if (row == col)
            A_out[idx] = 1.0f + A_in[idx]; // diagonal elements
        else
            A_out[idx] = A_in[idx]; // off-diagonal elements
    }
}

void identity_plus(
    const float* A_in, // device pointer to matrix A
    float* A_out, // device pointer to matrix A
    const int n
) {
    const int nn = n * n;
    const int threads = 1024;
    const int blocks = (nn + threads - 1) / threads;

    // Launch kernel to compute I + A
    identity_plus_kernel<<<blocks, threads>>>(A_in, A_out, n);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void fill_random_kernel(double* vec, int n, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        vec[idx] = curand_uniform_double(&state); // random double in (0,1]
    }
}

void fill_random(double* vec, int n, unsigned long seed, const int threadsPerBlock = 1024) {
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    fill_random_kernel<<<blocks, threadsPerBlock>>>(vec, n, seed);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in fill_random: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
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
    CHECK_CUDA(cudaDeviceSynchronize());
}

void approximate_two_norm(
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH,
    const double* A, size_t n,
    double* lo, double* up,
    size_t max_iter = 20, double tol = 1e-10
) {
    /* Allocations */
    // constants
    const double zero = 0.0;
    const double one = 1.0;
    
    // storage
    double *V, *V_old, *q, *w, *w1;
    max_iter = min(max_iter, n);
    CHECK_CUDA(cudaMalloc(&V,     n * max_iter * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&V_old,            n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&q,                n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&w,                n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&w1,               n * sizeof(double)));

    std::vector<double> alpha(max_iter, 0.0);
    std::vector<double> beta(max_iter, 0.0);

    double minus_alpha, minus_beta_old = 0.0;

    /* Initial vector */
    // q = randn(n, 1)
    fill_random(q, n, make_seed());

    // read from a txt file for debug
    // std::ifstream q_file("q.txt");
    // if (q_file.is_open()) {
    //     std::vector<double> q_host(n);
    //     for (size_t i = 0; i < n; ++i) {
    //         q_file >> q_host[i];
    //     }
    //     q_file.close();
    //     CHECK_CUDA(cudaMemcpy(q, q_host.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    // } else {
    //     std::cerr << "Unable to open file q.txt for reading." << std::endl;
    //     return;
    // }

    // // save q to a txt file with 17 decimal places
    // std::vector<double> q_host(n);
    // CHECK_CUDA(cudaMemcpy(q_host.data(), q, n * sizeof(double), cudaMemcpyDeviceToHost));

    // if (n == 10000) {
    //     std::ofstream q_file("q.txt");
    //     if (q_file.is_open()) {
    //         for (size_t i = 0; i < n; ++i) {
    //             q_file << std::setprecision(17) << q_host[i] << "\n";
    //         }
    //         q_file.close();
    //     } else {
    //         std::cerr << "Unable to open file q.txt for writing." << std::endl;
    //     }
    // }

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

        // w1 = At * w
        CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_T, n, n,
                                 &one, A, n, w, 1,
                                 &zero, w1, 1));
        // w = w1
        CHECK_CUBLAS(cublasDcopy(cublasH, n, w1, 1, w, 1));
        // hence w = A^T * A * q

        // alpha(k) = q^T * w
        CHECK_CUBLAS(cublasDdot(cublasH, n, q, 1, w, 1, &alpha[k]));

        // minus_alpha = -alpha[k]
        minus_alpha = -alpha[k];
        
        // w = w - alpha(k) * q - beta_old * V_old
        CHECK_CUBLAS(cublasDaxpy(cublasH, n, &minus_alpha, q, 1, w, 1));
        CHECK_CUBLAS(cublasDaxpy(cublasH, n, &minus_beta_old, V_old, 1, w, 1));
        
        // beta(k) = norm(w)
        CHECK_CUBLAS(cublasDnrm2(cublasH, n, w, 1, &beta[k]));

        // printf("alpha[k]: %5.4e, beta[k]: %5.4e \n", alpha[k], beta[k]);

        if (beta[k] <= tol * alpha[k] && k > 1)
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

    if (nb_iter == 0) {
        // in this case, the matrix is an all-zero matrix
        *lo = 0.0;
        *up = 1.0;

        CHECK_CUDA(cudaFree(V));
        CHECK_CUDA(cudaFree(V_old));
        CHECK_CUDA(cudaFree(q));
        CHECK_CUDA(cudaFree(w));
        CHECK_CUDA(cudaFree(w1));
        CHECK_CUDA(cudaDeviceSynchronize());

        return;
    }

    /* Tridiagonal T */
    // T = diag(alpha) + diag(beta(2:end),1) + diag(beta(2:end),-1);
    double *T, *d_alpha, *d_beta;
    CHECK_CUDA(cudaMalloc(&T,       nb_iter * nb_iter * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_beta,            nb_iter * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_alpha,           nb_iter * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_alpha, alpha.data(), nb_iter * sizeof(double), H2D));
    CHECK_CUDA(cudaMemcpy(d_beta,  beta.data(),  nb_iter * sizeof(double), H2D));
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

    CHECK_CUDA(cudaDeviceSynchronize());

    // print eigenvalues for debugging
    // std::vector<double> eigenvalues(nb_iter);
    // CHECK_CUDA(cudaMemcpy(eigenvalues.data(), d_eigenvalues, nb_iter * sizeof(double), cudaMemcpyDeviceToHost));
    // std::cout << "Eigenvalues: ";
    // for (const auto& ev : eigenvalues) {
    //     std::cout << ev << "\n";
    // }
    // std::cout << std::endl;

    // retrieve the max eigenvalue and corresponding eigenvector
    // int idx_max;
    // CHECK_CUBLAS(cublasIdamax(cublasH, nb_iter, d_eigenvalues, 1, &idx_max));
    // idx_max--; // convert to 0-based index
    int idx_max = nb_iter - 1;

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
    // ry = A * y
    CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_N, n, n,
                                &one, A, n, y, 1,
                                &zero, ry, 1));
    // w1 = At * ry
    CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_T, n, n,
                                &one, A, n, ry, 1,
                                &zero, w1, 1));
    // ry = w1
    CHECK_CUBLAS(cublasDcopy(cublasH, n, ry, 1, w1, 1));
    // hence ry = A^T * A * y

    // ry = ry - theta * y
    double minus_theta = -theta;
    CHECK_CUBLAS(cublasDaxpy(cublasH, n, &minus_theta, y, 1, ry, 1));

    /* Output */
    // lo = sqrt(theta)
    *lo = std::sqrt(theta);

    // up = sqrt(theta + norm(ry))
    double norm_ry;
    CHECK_CUBLAS(cublasDnrm2(cublasH, n, ry, 1, &norm_ry));
    printf("theta: %5.4e, norm_ry: %5.4e \n", theta, norm_ry);
    *up = std::sqrt(theta + norm_ry);
    printf("up: %5.4e \n", *up);

    /* Free memory */
    CHECK_CUDA(cudaFree(V));
    CHECK_CUDA(cudaFree(V_old));
    CHECK_CUDA(cudaFree(d_alpha));
    CHECK_CUDA(cudaFree(d_beta));
    CHECK_CUDA(cudaFree(q));
    CHECK_CUDA(cudaFree(w));
    CHECK_CUDA(cudaFree(w1));
    CHECK_CUDA(cudaFree(T));
    CHECK_CUDA(cudaFree(d_eigenvalues));
    CHECK_CUDA(cudaFree(d_work_eig));
    CHECK_CUDA(cudaFree(devInfo));
    CHECK_CUDA(cudaFree(uk));
    CHECK_CUDA(cudaFree(y));
    CHECK_CUDA(cudaFree(ry));
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "Upper bound: " << *up << std::endl;

    return;
}
/* ------------------ END: helper functions -------------------------- */

/* -------------------BEGIN: GEMM ----------------------- */
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
/* -------------------END: GEMM ----------------------- */

/* ---------- BEGIN: PSD cone projection based on eig ---------------------- */
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

    std::cout << "Max and min eigenvalues: "
              << *std::max_element(W_h.begin(), W_h.end()) << ", "
              << *std::min_element(W_h.begin(), W_h.end()) << std::endl;

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
/* ---------- END: PSD cone projection based on eig ---------------------- */

/* ---------- BEGIN: PSD cone projection based on minimax ---------------------- */
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12090)
void composite_FP32_emulated(
    cublasHandle_t cublasH,
    double* mat,
    const int n,
    const bool verbose = false
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
    convert_double_to_float(mat, A, nn);

    /* Coefficients */
    // const std::vector<std::vector<float>> coeff = { 
    //     { 8.3885353390, -23.7796270883, 16.8664591580 }, 
    //     { 4.1636476423, -2.9650849331, 0.5297319805 }, 
    //     { 4.0042650581, -2.8606348801, 0.5185227850 }, 
    //     { 3.4731017481, -2.5082466382, 0.4821470022 }, 
    //     { 2.4827239537, -1.7941788274, 0.4146530436 }, 
    // }; const size_t smoothing_steps = 3;

    // std::vector<std::vector<float>> coeff = { 
    //     { 8.5018632351, -24.6330845767, 17.8466614026 },
    //     { 4.2394319792, -3.0803745982, 0.5596805290 },
    //     { 4.2371780379, -3.0779047407, 0.5594995022 },
    //     { 4.1553447421, -3.0255808203, 0.5534594007 },
    //     { 3.8719053120, -2.8289969308, 0.5331377564 },
    //     { 3.0503282930, -2.2392300982, 0.4703818765 },
    //     { 2.1450160790, -1.4976204044, 0.3936105784 }
    // }; const size_t smoothing_steps = 4;

    // const std::vector<std::vector<float>> coeff = {
    //     { 8.513860379623477, -25.280005082715576,  18.766059564488327 },
    //     { 4.256391883949461,  -3.159693659036471,   0.586422854272915 },
    //     { 4.253921206146349,  -3.157984594990142,   0.586227906363050 },
    //     { 4.243442810924305,  -3.150733464368069,   0.585400800614286 },
    //     { 4.199542905437564,  -3.120304290489466,   0.581930053860478 },
    //     { 4.024728598959554,  -2.998292294405710,   0.568017312836389 },
    //     { 3.452591186626141,  -2.587800781708363,   0.521308704062118 },
    //     { 2.430558807049796,  -1.783214052500875,   0.431237717248089 },
    //     { 1.907795624713394,  -1.285976165390910,   0.378615420076569 },
    //     { 1.875011744441277,  -1.250013049314579,   0.375001304931425 }
    // }; const size_t smoothing_steps = 0; // pre-lunch

    // const std::vector<std::vector<float>> coeff = {
    //     { 8.509885302586273, -25.264304190830892,  18.753567899739625 },
    //     { 4.249573478922877,   -3.154976488114228,   0.585884782491327 },
    //     { 4.225122190777846,   -3.138044435084575,   0.583953455129916 },
    //     { 4.124838686994395,   -3.068332452805990,   0.576002953645695 },
    //     { 3.758010335802897,   -2.809273892403287,   0.546484206587685 },
    //     { 2.856177541291611,   -2.134056233175483,   0.470110769180275 },
    //     { 2.020600415776305,   -1.403721150466785,   0.390673896852026 },
    //     { 1.875875100481076,   -1.250971990481385,   0.375097212342072 },
    //     { 1.875,   -1.25,   0.375},
    //     { 1.875,   -1.25,   0.375},
    // }; const size_t smoothing_steps = 0; // best: 10 minimax original

    // const std::vector<std::vector<float>> coeff = {
    //     { 8.5117053694,  -25.2637545356,   18.7518511505 },
    //     { 4.2514746568,   -3.1551482052,    0.5855654848 },
    //     { 4.2314443096,   -3.1432483391,    0.5844187862 },
    //     { 4.1462871213,   -3.0853187659,    0.5781140029 },
    //     { 3.8679345846,   -2.8863505270,    0.5573798771 },
    //     { 3.0735744409,   -2.2984793859,    0.4942218088 },
    //     { 2.1692233704,   -1.5420827375,    0.4146319529 },
    //     { 2.0078578610,   -1.3793846146,    0.3989298303 },
    //     { 2.0029525899,   -1.3743625171,    0.3982429919 },
    //     { 1.8780193554,   -1.2544181003,    0.3764365891 },
    // }; const size_t smoothing_steps = 0; // 10 minimax refined

    const std::vector<std::vector<float>> coeff = {
        { 8.3119043343,  -23.0739115930,  16.4664144722 },
        { 4.1439360087,   -2.9176674704,   0.5246212487 },
        { 4.0257813209,   -2.9025002398,   0.5334261214 },
        { 3.5118574347,   -2.5740236523,   0.5050097282 },
        { 2.4398158400,   -1.7586675341,   0.4191290613 },
        { 1.9779835097,   -1.3337358510,   0.3772169049 },
        { 1.9559726949,   -1.3091355170,   0.3746734515 },
        { 1.9282822454,   -1.2823649693,   0.3704626545 },
        { 1.9220135179,   -1.2812524618,   0.3707011753 },
        { 1.8942192942,   -1.2613293407,   0.3676616051 }
    }; const size_t smoothing_steps = 0; // 10 polar express refined

    float scale_factor = 1.001f;

    /* Approximation of the step function */
    for (int i = 0; i < coeff.size(); i++) {
        float a = coeff[i][0];
        float b = coeff[i][1];
        float c = coeff[i][2];

        if (i < 8) {
            a /= scale_factor;
            b /= scale_factor * scale_factor * scale_factor;
            c /= scale_factor * scale_factor * scale_factor * scale_factor 
                * scale_factor;
        }

        /* Compute the powers of A*/
        // A2 = A * A
        CHECK_CUBLAS( cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            A, CUDA_R_32F, n,
            A, CUDA_R_32F, n,
            &zero,
            A2, CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ) );

        // A3 = A2 * A
        CHECK_CUBLAS( cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            A, CUDA_R_32F, n,
            A2, CUDA_R_32F, n,
            &zero,
            A3, CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ) );

        // A = a * A
        CHECK_CUBLAS( cublasSscal(cublasH, nn, &a, A, 1) );
        // A = b * A3 + A
        CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &b, A3, 1, A, 1) );
        // at this point, A = a * A + b * A3

        // A = c * A3 * A2 + A
        CHECK_CUBLAS( cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &c,
            A3, CUDA_R_32F, n,
            A2, CUDA_R_32F, n,
            &one,
            A, CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ) );

        /* Symmetrize A */
        symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace
    }

    /* Smoothing function */
    for (int i = 0; i < smoothing_steps; i++) {
        // A2 = A * A
        CHECK_CUBLAS( cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            A, CUDA_R_32F, n,
            A, CUDA_R_32F, n,
            &zero,
            A2, CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ) );

        // A3 = A2 * A
        CHECK_CUBLAS( cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            A, CUDA_R_32F, n,
            A2, CUDA_R_32F, n,
            &zero,
            A3, CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ) );

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

    // A = I + A
    add_identity(cublasH, A, n);
    // A = 0.5 * A
    CHECK_CUBLAS( cublasSscal(cublasH, nn, &half, A, 1) );

    /* Multiply the original matrix by A */
    // W = A_origin * A
    convert_double_to_float(mat, A2, nn);
    CHECK_CUBLAS( cublasGemmEx(
        cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &one,
        A2, CUDA_R_32F, n,
        A, CUDA_R_32F, n,
        &zero,
        A3, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ) );

    /* Symmetrize W */
    symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace

    /* Copy the result back to mat */
    convert_float_to_double(A3, mat, nn);

    /* Free device memory */
    CHECK_CUDA( cudaFree(A) );
    CHECK_CUDA( cudaFree(A2) );
    CHECK_CUDA( cudaFree(A3) );
}
#endif

void composite_FP32(
    cublasHandle_t cublasH,
    double* mat,
    const int n,
    const bool verbose
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
    convert_double_to_float(mat, A, nn);

    /* Coefficients */
    // const std::vector<std::vector<float>> coeff = { 
    //     { 8.3885353390, -23.7796270883, 16.8664591580 }, 
    //     { 4.1636476423, -2.9650849331, 0.5297319805 }, 
    //     { 4.0042650581, -2.8606348801, 0.5185227850 }, 
    //     { 3.4731017481, -2.5082466382, 0.4821470022 }, 
    //     { 2.4827239537, -1.7941788274, 0.4146530436 }, 
    // }; const size_t smoothing_steps = 3;

    // std::vector<std::vector<float>> coeff = { 
    //     { 8.5018632351, -24.6330845767, 17.8466614026 },
    //     { 4.2394319792, -3.0803745982, 0.5596805290 },
    //     { 4.2371780379, -3.0779047407, 0.5594995022 },
    //     { 4.1553447421, -3.0255808203, 0.5534594007 },
    //     { 3.8719053120, -2.8289969308, 0.5331377564 },
    //     { 3.0503282930, -2.2392300982, 0.4703818765 },
    //     { 2.1450160790, -1.4976204044, 0.3936105784 }
    // }; const size_t smoothing_steps = 4;

    // const std::vector<std::vector<float>> coeff = {
    //     { 8.513860379623477, -25.280005082715576,  18.766059564488327 },
    //     { 4.256391883949461,  -3.159693659036471,   0.586422854272915 },
    //     { 4.253921206146349,  -3.157984594990142,   0.586227906363050 },
    //     { 4.243442810924305,  -3.150733464368069,   0.585400800614286 },
    //     { 4.199542905437564,  -3.120304290489466,   0.581930053860478 },
    //     { 4.024728598959554,  -2.998292294405710,   0.568017312836389 },
    //     { 3.452591186626141,  -2.587800781708363,   0.521308704062118 },
    //     { 2.430558807049796,  -1.783214052500875,   0.431237717248089 },
    //     { 1.907795624713394,  -1.285976165390910,   0.378615420076569 },
    //     { 1.875011744441277,  -1.250013049314579,   0.375001304931425 }
    // }; const size_t smoothing_steps = 0; // pre-lunch

    // std::vector<std::vector<float>> coeff = {
    //     { 8.509885302586273, -25.264304190830892,  18.753567899739625 },
    //     { 4.249573478922877,   -3.154976488114228,   0.585884782491327 },
    //     { 4.225122190777846,   -3.138044435084575,   0.583953455129916 },
    //     { 4.124838686994395,   -3.068332452805990,   0.576002953645695 },
    //     { 3.758010335802897,   -2.809273892403287,   0.546484206587685 },
    //     { 2.856177541291611,   -2.134056233175483,   0.470110769180275 },
    //     { 2.020600415776305,   -1.403721150466785,   0.390673896852026 },
    //     { 1.875875100481076,   -1.250971990481385,   0.375097212342072 },
    //     { 1.875,   -1.25,   0.375},
    //     { 1.875,   -1.25,   0.375},
    // }; const size_t smoothing_steps = 0; // best: // best: 10 minimax original

    // const std::vector<std::vector<float>> coeff = {
    //     { 8.5117053694,  -25.2637545356,   18.7518511505 },
    //     { 4.2514746568,   -3.1551482052,    0.5855654848 },
    //     { 4.2314443096,   -3.1432483391,    0.5844187862 },
    //     { 4.1462871213,   -3.0853187659,    0.5781140029 },
    //     { 3.8679345846,   -2.8863505270,    0.5573798771 },
    //     { 3.0735744409,   -2.2984793859,    0.4942218088 },
    //     { 2.1692233704,   -1.5420827375,    0.4146319529 },
    //     { 2.0078578610,   -1.3793846146,    0.3989298303 },
    //     { 2.0029525899,   -1.3743625171,    0.3982429919 },
    //     { 1.8780193554,   -1.2544181003,    0.3764365891 },
    // }; const size_t smoothing_steps = 0; // 10 minimax refined

    const std::vector<std::vector<float>> coeff = {
        { 8.3119043343,  -23.0739115930,  16.4664144722 },
        { 4.1439360087,   -2.9176674704,   0.5246212487 },
        { 4.0257813209,   -2.9025002398,   0.5334261214 },
        { 3.5118574347,   -2.5740236523,   0.5050097282 },
        { 2.4398158400,   -1.7586675341,   0.4191290613 },
        { 1.9779835097,   -1.3337358510,   0.3772169049 },
        { 1.9559726949,   -1.3091355170,   0.3746734515 },
        { 1.9282822454,   -1.2823649693,   0.3704626545 },
        { 1.9220135179,   -1.2812524618,   0.3707011753 },
        { 1.8942192942,   -1.2613293407,   0.3676616051 }
    }; const size_t smoothing_steps = 0; // 10 polar express refined

    float scale_factor = 1.001f;

    /* Approximation of the step function */
    for (int i = 0; i < coeff.size(); i++) {
        float a = coeff[i][0];
        float b = coeff[i][1];
        float c = coeff[i][2];

        if (i < 8) {
            a /= scale_factor;
            b /= scale_factor * scale_factor * scale_factor;
            c /= scale_factor * scale_factor * scale_factor * scale_factor 
                * scale_factor;
        }

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

    // A = I + A
    add_identity(cublasH, A, n);
    // A = 0.5 * A
    CHECK_CUBLAS( cublasSscal(cublasH, nn, &half, A, 1) );

    /* Multiply the original matrix by A */
    // W = A_origin * A
    convert_double_to_float(mat, A2, nn);
    CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A2, n, A, n, &zero, A3, n) );

    /* Symmetrize W */
    symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace

    /* Copy the result back to mat */
    convert_float_to_double(A3, mat, nn);

    /* Free device memory */
    CHECK_CUDA( cudaFree(A) );
    CHECK_CUDA( cudaFree(A2) );
    CHECK_CUDA( cudaFree(A3) );
}


void composite_TF16(
    cublasHandle_t cublasH,
    double* mat,
    const int n
) {
    const int nn = n * n;

    /* Allocations */
    // device memory
    float *A, *A2, *A3;
    CHECK_CUDA( cudaMalloc(&A,  nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&A2, nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&A3, nn * sizeof(float)) );

    __half *hA, *hA2, *hA3;
    CHECK_CUDA(cudaMalloc(&hA,  nn * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&hA2, nn * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&hA3, nn * sizeof(__half)));

    // useful constants
    const float half       =  0.5f;
    const float one        =  1.0f;
    const float zero       =  0.0f;

    convert_double_to_float(mat, A, nn);

    /* Coefficients */
    // std::vector<std::vector<float>> coeff = {
    //     {8.4724206924, -24.5001735687, 17.7268180847},
    //     {4.2052841187, -3.0549299717, 0.5567536354},
    //     {4.0443077087, -2.9473149776, 0.5449726582},
    //     {3.5078327656, -2.5842490196, 0.5067413449},
    //     {2.5075511932, -1.8485442400, 0.4358045161}
    // };
    // std::vector<std::vector<float>> coeff = { 
    //     { 8.3885353390, -23.7796270883, 16.8664591580 }, 
    //     { 4.1636476423, -2.9650849331, 0.5297319805 }, 
    //     { 4.0042650581, -2.8606348801, 0.5185227850 }, 
    //     { 3.4731017481, -2.5082466382, 0.4821470022 }, 
    //     { 2.4827239537, -1.7941788274, 0.4146530436 }, 
    // };

    // const std::vector<std::vector<float>> coeff = { 
    //     { 8.3937001154, -23.7945582332, 16.8758390904 }, 
    //     { 4.1803895500, -2.9788012917, 0.5318143742 }, 
    //     { 4.0578478573, -2.9013956514, 0.5233571836 }, 
    //     { 3.6289664769, -2.6254124593, 0.4963343458 }, 
    //     { 2.7619020904, -1.9865006927, 0.4388497859 }, 
    //     { 2.0674922563, -1.4317903208, 0.3876934521 }, 
    //     { 1.8438914749, -1.1872290786, 0.3433825749 }, 
    // }; // current best

    // std::vector<std::vector<float>> coeff = {
    //     { 8.3864641622, -24.8594799076, 18.4448273259 },
    //     { 4.1414199835, -3.0779218910, 0.5748581003 },
    //     { 3.9226309693, -2.9248155905, 0.5574020970 },
    //     { 3.2540457594, -2.4403169649, 0.5023343504 },
    //     { 2.2512376183, -1.6283766019, 0.4120702252 },
    //     { 1.8700160370, -1.2526309162, 0.3727910154 },
    //     { 1.8564365206, -1.2376247369, 0.3712872262 },
    // };

    const std::vector<std::vector<float>> coeff = {
        { 8.2885332412,  -22.5927099246, 15.8201383114 },
        { 4.1666196466,   -2.9679004036,  0.5307623217 },
        { 4.0611848147,   -2.9698947955,  0.5492133813 },
        { 3.6678301399,   -2.7561018955,  0.5421513305 },
        { 2.7632556383,   -2.0607754898,  0.4695405857 },
        { 2.0527445797,   -1.4345145882,  0.4070669182 },
        { 1.8804816691,   -1.2583997294,  0.3779501813 }
    }; // polar express with refinement

    float scale_factor = 1.01f;

    /* Approximation of the step function */
    for (int i = 0; i < coeff.size(); i++) {
        float a = coeff[i][0];
        float b = coeff[i][1];
        float c = coeff[i][2];

        a /= scale_factor;
        b /= scale_factor * scale_factor * scale_factor;
        c /= scale_factor * scale_factor * scale_factor * scale_factor * scale_factor;

        /* Compute the powers of A*/
        // A2 = A * A
        convert_float_to_half4(A, hA, nn);
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
        convert_float_to_half4(A2, hA2, nn);
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

        // A = a * A
        CHECK_CUBLAS( cublasSscal(cublasH, nn, &a, A, 1) );
        // A = b * A3 + A
        CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &b, A3, 1, A, 1) );
        // at this point, A = a * A + b * A3

        /* Symmetrize A3, A5 */
        // symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace
        // symmetrizeFloat(cublasH, A5, n, A2); // we use A2 as a workspace

        // A = c * A3 * A2 + A
        convert_float_to_half4(A3, hA3, nn);
        CHECK_CUBLAS(cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &c,
            hA2, CUDA_R_16F, n,
            hA3, CUDA_R_16F, n,
            &one,
            A,    CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        /* Symmetrize A */
        symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace
    }

    // /* Smoothing function */
    // for (int i = 0; i < 3; i++) {
    //     // A2 = A * A
    //     convert_float_to_half4(A, hA, nn);
    //     CHECK_CUBLAS(cublasGemmEx(
    //         cublasH,
    //         CUBLAS_OP_N, CUBLAS_OP_N,
    //         n, n, n,
    //         &one,
    //         hA, CUDA_R_16F, n,
    //         hA, CUDA_R_16F, n,
    //         &zero,
    //         A2,    CUDA_R_32F, n,
    //         CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    //     // A3 = A2 * A
    //     convert_float_to_half4(A2, hA2, nn);
    //     CHECK_CUBLAS(cublasGemmEx(
    //         cublasH,
    //         CUBLAS_OP_N, CUBLAS_OP_N,
    //         n, n, n,
    //         &one,
    //         hA, CUDA_R_16F, n,
    //         hA2, CUDA_R_16F, n,
    //         &zero,
    //         A3,    CUDA_R_32F, n,
    //         CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    //     /* Symmetrize A3 */
    //     symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace

    //     /* Compute A = 1.5 * A - 0.5 * A3 */
    //     // A = 1.5 * A
    //     CHECK_CUBLAS( cublasSscal(cublasH, nn, &one_n_half, A, 1) );
    //     // A = -0.5 * A3 + A
    //     CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &minus_half, A3, 1, A, 1) );

    //     /* Symmetrize A */
    //     symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace
    // }

    add_identity(cublasH, A, n);
    // A = 0.5 * A
    CHECK_CUBLAS( cublasSscal(cublasH, nn, &half, A, 1) );

    /* Multiply the original matrix by A */
    convert_double_to_float(mat, A2, nn);
    convert_float_to_half4(A, hA, nn);
    convert_float_to_half4(A2, hA2, nn);
    CHECK_CUBLAS(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &one,
        hA2, CUDA_R_16F, n,
        hA, CUDA_R_16F, n,
        &zero,
        A3,    CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    /* Symmetrize W */
    symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace

    /* Copy the result back to mat */
    convert_float_to_double(A3, mat, nn);

    /* Free device memory */
    CHECK_CUDA( cudaFree(A) );
    CHECK_CUDA( cudaFree(A2) );
    CHECK_CUDA( cudaFree(A3) );
    CHECK_CUDA( cudaFree(hA) );
    CHECK_CUDA( cudaFree(hA2) );
    CHECK_CUDA( cudaFree(hA3) );
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

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12090)
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
#endif
/* ---------- END: PSD cone projection based on minimax ---------------------- */

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

void haoyu_TF16(
    cublasHandle_t cublasH,
    float* mat,
    const int n
) {
    const int nn = n * n;

    // Allocate device buffers
    float *dA_our, *dTmp, *dT1, *dT2, *dF;
    CHECK_CUDA(cudaMalloc(&dA_our,  nn*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dTmp,    nn*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dT1,     nn*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dT2,     nn*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dF,      nn*sizeof(float)));

    // Copy host to device
    CHECK_CUDA(cudaMemcpy(dA_our, mat, nn*sizeof(float), D2D));

    // half buffers
    __half *dT3_half, *dT4_half; 
    CHECK_CUDA(cudaMalloc(&dT3_half, nn*sizeof(__half))); 
    CHECK_CUDA(cudaMalloc(&dT4_half, nn*sizeof(__half)));

    const float one = 1.0f, zero = 0.0f;
    float half = 0.5f;

    // Iterative algorithm in float, printing after each iter
    for (int iter = 1; iter <= 4; iter++) {
        // T1 = A_our * A_our
        convert_float_to_half4(dA_our, dT3_half, nn);
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

        // T2 = I - T1
        identity_minus(dT1, dT2, n);

        // T1 = T2 * T2
        convert_float_to_half4(dT2, dT3_half, nn);
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

        // F = I + log(iter+10)*T1
        float logv = std::log(iter + 10.0f);
        CHECK_CUBLAS(cublasSscal(cublasH, nn, &logv, dT1, 1));
        identity_plus(dT1, dF, n);

        // A_our = A_our * F (to dTmp, then copy back)
        convert_float_to_half4(dA_our, dT3_half, nn);
        convert_float_to_half4(dF, dT4_half, nn);
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
        
        CHECK_CUDA(cudaMemcpy(dA_our, dTmp, nn*sizeof(float), cudaMemcpyDeviceToDevice));

        // T1 = A_our^2, T2 = I - T1
        convert_float_to_half4(dA_our, dT3_half, nn);
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

        identity_minus(dT1, dT2, n);

        // F = I + (1/sqrt(iter))*T2
        float invs = 1.0f / std::sqrt((float)iter);
        CHECK_CUBLAS(cublasSscal(cublasH, nn, &invs, dT2, 1));
        identity_plus(dT2, dF, n);
            
        // A_our = A_our * F (to dTmp)
        convert_float_to_half4(dA_our, dT3_half, nn);
        convert_float_to_half4(dF, dT4_half, nn);
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
    }
    
    // Final combine: mat = mat * (A_our + I) / 2
    identity_plus(dA_our, dF, n);

    convert_float_to_half4(mat, dT3_half, nn);
    convert_float_to_half4(dF, dT4_half, nn);
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

    /* Free memory */
    CHECK_CUDA(cudaFree(dA_our));
    CHECK_CUDA(cudaFree(dTmp));
    CHECK_CUDA(cudaFree(dT1));
    CHECK_CUDA(cudaFree(dT2));
    CHECK_CUDA(cudaFree(dF));
    CHECK_CUDA(cudaFree(dT3_half));
    CHECK_CUDA(cudaFree(dT4_half));

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
    convert_double_to_float(dA_psd, dA_psd_float, nn);

    haoyu_TF16(cublasH, dA_psd_float, n);

    convert_float_to_double(dA_psd_float, dA_psd, nn);

    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(dA_psd_float));

    return std::chrono::high_resolution_clock::now() - start;
}

/* ------------------- baselines: -------------------------- */
/* polar express FP32 */
void polarexpress_FP32(
    cublasHandle_t cublasH,
    double* mat,
    const int n,
    const bool verbose
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
    const float one        =  1.0f;
    const float zero       =  0.0f;

    /* Convert the initial matrix*/
    convert_double_to_float(mat, A, nn);

    const std::vector<std::vector<float>> coeff = {
        {8.205160414005569, -22.901934987056034, 16.460724910180303},
        {4.0669156198795875, -2.861284534588475, 0.5183804464778605},
        {3.9134926112054607, -2.824251876723087, 0.5248485625148532},
        {3.3060139701337725, -2.430227567449687, 0.4869515205509481},
        {2.304016813944474, -1.6427206546268969, 0.4009100949022211},
        {1.8771914635816975, -1.2356588245606832, 0.3590046107458665},
        {1.8564430512944288, -1.2132457535214525, 0.3568004238203446},
        {1.8564364186731164, -1.2132392045600955, 0.3568003776956684},
        {1.8564311666088567, -1.2132289074471427, 0.35679533060704244},
        {1.8564311659068637, -1.2132289060708275, 0.3567953299324476},
    }; 

    // std::vector<std::vector<float>> coeff = {
    //     { 8.509885302586273, -25.264304190830892,  18.753567899739625 },
    //     { 4.249573478922877,   -3.154976488114228,   0.585884782491327 },
    //     { 4.225122190777846,   -3.138044435084575,   0.583953455129916 },
    //     { 4.124838686994395,   -3.068332452805990,   0.576002953645695 },
    //     { 3.758010335802897,   -2.809273892403287,   0.546484206587685 },
    //     { 2.856177541291611,   -2.134056233175483,   0.470110769180275 },
    //     { 2.020600415776305,   -1.403721150466785,   0.390673896852026 },
    //     { 1.875875100481076,   -1.250971990481385,   0.375097212342072 },
    //     { 1.875,   -1.25,   0.375},
    //     { 1.875,   -1.25,   0.375},
    // }; // post-lunch: 
    // float scale_factor = 1.0001f;
    // // change coefficients first
    // for (int i = 0; i < coeff.size(); i++) {
    //     float a = coeff[i][0];
    //     float b = coeff[i][1];
    //     float c = coeff[i][2];

    //     if (i < 3) {
    //         a /= scale_factor;
    //         b /= scale_factor * scale_factor * scale_factor;
    //         c /= scale_factor * scale_factor * scale_factor * scale_factor 
    //             * scale_factor;
    //     }

    //     coeff[i][0] = a;
    //     coeff[i][1] = b;
    //     coeff[i][2] = c;
    // }

    /* DEBUG: timing */ 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    /* Approximation of the step function */
    for (int i = 0; i < coeff.size(); i++) {
        float a = coeff[i][0];
        float b = coeff[i][1];
        float c = coeff[i][2];

        // printf("i: %d, a: %5.4e, b: %5.4e, c: %5.4e \n", i, a, b, c);

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

        // A = c * A3 * A2 + A
        CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &c, A3, n, A2, n, &one, A, n) );

        /* Symmetrize A */
        symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace
    }

    /* DEBUG: timing */ 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms\n";

    // A = I + A
    add_identity(cublasH, A, n);
    // A = 0.5 * A
    CHECK_CUBLAS( cublasSscal(cublasH, nn, &half, A, 1) );

    /* Multiply the original matrix by A */
    // W = A_origin * A
    convert_double_to_float(mat, A2, nn);
    CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A2, n, A, n, &zero, A3, n) );

    /* Symmetrize W */
    symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace

    /* Copy the result back to mat */
    convert_float_to_double(A3, mat, nn);

    /* Free device memory */
    CHECK_CUDA( cudaFree(A) );
    CHECK_CUDA( cudaFree(A2) );
    CHECK_CUDA( cudaFree(A3) );
}

std::chrono::duration<double> polarexpress_FP32_psd(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dA_psd, size_t n) {
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

    polarexpress_FP32(
        cublasH, dA_psd, n, 0
    );

    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

    CHECK_CUDA(cudaDeviceSynchronize());

    return std::chrono::high_resolution_clock::now() - start;
}

/* polar express TF16 */
void polarexpress_TF16(
    cublasHandle_t cublasH,
    double* mat,
    const int n
) {
    const int nn = n * n;

    /* Allocations */
    // device memory
    float *A, *A2, *A3;
    CHECK_CUDA( cudaMalloc(&A,  nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&A2, nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&A3, nn * sizeof(float)) );

    __half *hA, *hA2, *hA3;
    CHECK_CUDA(cudaMalloc(&hA,  nn * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&hA2, nn * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&hA3, nn * sizeof(__half)));

    // useful constants
    const float half       =  0.5f;
    const float one        =  1.0f;
    const float zero       =  0.0f;

    convert_double_to_float(mat, A, nn);

    /* Coefficients */
    std::vector<std::vector<float>> coeff = { 
        {8.205160414005569, -22.901934987056034, 16.460724910180303},
        {4.0669156198795875, -2.861284534588475, 0.5183804464778605},
        {3.9134926112054607, -2.824251876723087, 0.5248485625148532},
        {3.3060139701337725, -2.430227567449687, 0.4869515205509481},
        {2.304016813944474, -1.6427206546268969, 0.4009100949022211},
        {1.8771914635816975, -1.2356588245606832, 0.3590046107458665},
        {1.8564430512944288, -1.2132457535214525, 0.3568004238203446} 
    };

    /* Approximation of the step function */
    for (int i = 0; i < coeff.size(); i++) {
        const float a = coeff[i][0];
        const float b = coeff[i][1];
        const float c = coeff[i][2];

        /* Compute the powers of A*/
        // A2 = A * A
        convert_float_to_half4(A, hA, nn);
        CHECK_CUBLAS(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, hA, CUDA_R_16F, n, hA, CUDA_R_16F, n, &zero, A2, CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // A3 = A2 * A
        convert_float_to_half4(A2, hA2, nn);
        CHECK_CUBLAS(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, hA, CUDA_R_16F, n, hA2, CUDA_R_16F, n, &zero, A3, CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // A = a * A
        CHECK_CUBLAS( cublasSscal(cublasH, nn, &a, A, 1) );
        // A = b * A3 + A
        CHECK_CUBLAS( cublasSaxpy(cublasH, nn, &b, A3, 1, A, 1) );
        // at this point, A = a * A + b * A3

        // A = c * A3 * A2 + A
        convert_float_to_half4(A3, hA3, nn);
        CHECK_CUBLAS(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &c, hA2, CUDA_R_16F, n, hA3, CUDA_R_16F, n, &one, A, CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        /* Symmetrize A */
        symmetrizeFloat(cublasH, A, n, A2); // we use A2 as a workspace
    }

    add_identity(cublasH, A, n);
    // A = 0.5 * A
    CHECK_CUBLAS( cublasSscal(cublasH, nn, &half, A, 1) );

    /* Multiply the original matrix by A */
    convert_double_to_float(mat, A2, nn);
    convert_float_to_half4(A, hA, nn);
    convert_float_to_half4(A2, hA2, nn);
    CHECK_CUBLAS(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, hA2, CUDA_R_16F, n, hA, CUDA_R_16F, n, &zero, A3, CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    /* Symmetrize W */
    symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace

    /* Copy the result back to mat */
    convert_float_to_double(A3, mat, nn);

    /* Free device memory */
    CHECK_CUDA( cudaFree(A) );
    CHECK_CUDA( cudaFree(A2) );
    CHECK_CUDA( cudaFree(A3) );
    CHECK_CUDA( cudaFree(hA) );
    CHECK_CUDA( cudaFree(hA2) );
    CHECK_CUDA( cudaFree(hA3) );
}

std::chrono::duration<double> polarexpress_TF16_psd(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dA_psd, size_t n) {
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

    polarexpress_TF16(cublasH, dA_psd, n);

    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

    CHECK_CUDA(cudaDeviceSynchronize());

    return std::chrono::high_resolution_clock::now() - start;
}

/* Newton Schulz FP32 */
void newton_FP32(
    cublasHandle_t cublasH,
    double* mat,
    const int n,
    const bool verbose
) {
    const int nn = n * n;

    const size_t smoothing_steps = 15;

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
    convert_double_to_float(mat, A, nn);

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

    // A = I + A
    add_identity(cublasH, A, n);
    // A = 0.5 * A
    CHECK_CUBLAS( cublasSscal(cublasH, nn, &half, A, 1) );

    /* Multiply the original matrix by A */
    // W = A_origin * A
    convert_double_to_float(mat, A2, nn);
    CHECK_CUBLAS( cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A2, n, A, n, &zero, A3, n) );

    /* Symmetrize W */
    symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace

    /* Copy the result back to mat */
    convert_float_to_double(A3, mat, nn);

    /* Free device memory */
    CHECK_CUDA( cudaFree(A) );
    CHECK_CUDA( cudaFree(A2) );
    CHECK_CUDA( cudaFree(A3) );
}

std::chrono::duration<double> newton_FP32_psd(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dA_psd, size_t n) {
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

    newton_FP32(
        cublasH, dA_psd, n, 0
    );

    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

    CHECK_CUDA(cudaDeviceSynchronize());

    return std::chrono::high_resolution_clock::now() - start;
}

/* newton TF16 */
void newton_TF16(
    cublasHandle_t cublasH,
    double* mat,
    const int n
) {
    const int nn = n * n;

    /* Allocations */
    // device memory
    float *A, *A2, *A3;
    CHECK_CUDA( cudaMalloc(&A,  nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&A2, nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&A3, nn * sizeof(float)) );

    __half *hA, *hA2, *hA3;
    CHECK_CUDA(cudaMalloc(&hA,  nn * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&hA2, nn * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&hA3, nn * sizeof(__half)));

    // useful constants
    const float half       =  0.5f;
    const float one        =  1.0f;
    const float minus_half = -0.5f;
    const float one_n_half =  1.5f;
    const float zero       =  0.0f;

    convert_double_to_float(mat, A, nn);

    /* Smoothing function */
    for (int i = 0; i < 10; i++) {
        // A2 = A * A
        convert_float_to_half4(A, hA, nn);
        CHECK_CUBLAS(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, hA, CUDA_R_16F, n, hA, CUDA_R_16F, n, &zero, A2, CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // A3 = A2 * A
        convert_float_to_half4(A2, hA2, nn);
        CHECK_CUBLAS(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, hA, CUDA_R_16F, n, hA2, CUDA_R_16F, n, &zero, A3, CUDA_R_32F, n,CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

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

    add_identity(cublasH, A, n);
    // A = 0.5 * A
    CHECK_CUBLAS( cublasSscal(cublasH, nn, &half, A, 1) );

    /* Multiply the original matrix by A */
    convert_double_to_float(mat, A2, nn);
    convert_float_to_half4(A, hA, nn);
    convert_float_to_half4(A2, hA2, nn);
    CHECK_CUBLAS(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, hA2, CUDA_R_16F, n, hA, CUDA_R_16F, n, &zero, A3, CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    /* Symmetrize W */
    symmetrizeFloat(cublasH, A3, n, A2); // we use A2 as a workspace

    /* Copy the result back to mat */
    convert_float_to_double(A3, mat, nn);

    /* Free device memory */
    CHECK_CUDA( cudaFree(A) );
    CHECK_CUDA( cudaFree(A2) );
    CHECK_CUDA( cudaFree(A3) );
    CHECK_CUDA( cudaFree(hA) );
    CHECK_CUDA( cudaFree(hA2) );
    CHECK_CUDA( cudaFree(hA3) );
}

std::chrono::duration<double> newton_TF16_psd(cusolverDnHandle_t solverH, cublasHandle_t cublasH, const double* dA_orig, double* dA_psd, size_t n) {
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

    newton_TF16(
        cublasH, dA_psd, n
    );

    CHECK_CUBLAS( cublasDscal(cublasH, nn, &scale, dA_psd, 1) );

    CHECK_CUDA(cudaDeviceSynchronize());

    return std::chrono::high_resolution_clock::now() - start;
}
/* -------------------------------------------------------- */


int main(int argc, char* argv[]) {
    std::vector<std::string> datasets;
    std::vector<size_t> instance_sizes;
    int restarts = 1;
    int gemm_restarts = 1;
    std::string gemm_output_file = "results/gemm_results.csv";
    std::string psd_output_file;

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
        } else if (arg == "--csv_name") {
            if (i + 1 < argc) {
                psd_output_file = argv[++i];
            } else {
                std::cerr << "Missing value for --csv_name" << std::endl;
                return 1;
            }
        }  else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    std::ofstream psd_file(psd_output_file, std::ios_base::app);
    if (psd_file.tellp() == 0) {
        psd_file << "dataset,n,method,time,relative_error\n";
    } else {
        std::cerr << "WARNING: " << psd_output_file << " already exists and is not empty." << std::endl;
        psd_file << "\ndataset,n,method,time,relative_error\n";
        // return 1;
    }
    psd_file.close();

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
            cublasH_emulated, CUBLAS_OP_N, CUBLAS_OP_N,
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

            /* ----------------- baselines ------------------- */
            // polar express FP32
            if (RUN_BASELINES) {
                duration = std::chrono::duration<double>(0.0);
                error = 0.0;
                for (int i = 0; i < restarts; ++i) {
                    CHECK_CUDA(cudaMemset(A_psd, 0, nn * sizeof(double)));
                    duration += polarexpress_FP32_psd(solverH, cublasH, A, A_psd, n);
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
                std::cout << "\t\tpolarexpress FP32 -- Time: " << std::scientific << duration.count() << " s" << std::endl;
                std::cout << "\t\t        Relative error: " << std::scientific << error << std::endl;
                append_csv(psd_output_file, "polarexpress FP32", dataset, n, duration, error);

                // polar express TF16
                duration = std::chrono::duration<double>(0.0);
                error = 0.0;
                for (int i = 0; i < restarts; ++i) {
                    CHECK_CUDA(cudaMemset(A_psd, 0, nn * sizeof(double)));
                    duration += polarexpress_TF16_psd(solverH, cublasH_TF16, A, A_psd, n);
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
                std::cout << "\t\tpolarexpress TF16 -- Time: " << std::scientific << duration.count() << " s" << std::endl;
                std::cout << "\t\t        Relative error: " << std::scientific << error << std::endl;
                append_csv(psd_output_file, "polarexpress TF16", dataset, n, duration, error);

                // newton FP32
                duration = std::chrono::duration<double>(0.0);
                error = 0.0;
                for (int i = 0; i < restarts; ++i) {
                    CHECK_CUDA(cudaMemset(A_psd, 0, nn * sizeof(double)));
                    duration += newton_FP32_psd(solverH, cublasH, A, A_psd, n);
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
                std::cout << "\t\tnewton FP32 -- Time: " << std::scientific << duration.count() << " s" << std::endl;
                std::cout << "\t\t        Relative error: " << std::scientific << error << std::endl;
                append_csv(psd_output_file, "newton FP32", dataset, n, duration, error);

                // newton TF16
                duration = std::chrono::duration<double>(0.0);
                error = 0.0;
                for (int i = 0; i < restarts; ++i) {
                    CHECK_CUDA(cudaMemset(A_psd, 0, nn * sizeof(double)));
                    duration += newton_TF16_psd(solverH, cublasH_TF16, A, A_psd, n);
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
                std::cout << "\t\tnewton TF16 -- Time: " << std::scientific << duration.count() << " s" << std::endl;
                std::cout << "\t\t        Relative error: " << std::scientific << error << std::endl;
                append_csv(psd_output_file, "newton TF16", dataset, n, duration, error);
            }
            /* ----------------------------------------------- */

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

            #if defined(CUDA_VERSION) && (CUDA_VERSION >= 12090)
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
            #endif

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