// The kernel body from cuda-gemm-from-scratch/05_2d_tiling.cu, copied without
// edits. It exists so gemm_selftest.cu can assert that the templated version in
// gemm.cuh produces bit-identical output, which keeps "the vendored kernel is
// unchanged" an assertion rather than a claim.
//
// Only the benchmark main() and the cuBLAS comparison were dropped. Compiled
// into gemm_selftest, not into the engine.

#include "gemm_reference.cuh"

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void sgemm_2d_tiling(int M, int N, int K, float alpha, const float *A,
                                const float *B, float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

    const uint threadCol = threadIdx.x % (BN / TN);
    const uint threadRow = threadIdx.x / (BN / TN);

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint strideA = numThreadsBlocktile / BK;
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    const uint strideB = numThreadsBlocktile / BN;

    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] =
                A[(innerRowA + loadOffset) * K + innerColA];
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
                B[(innerRowB + loadOffset) * N + innerColB];
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }

        __syncthreads();
    }

    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
                alpha * threadResults[resIdxM * TN + resIdxN] +
                beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
        }
    }
}

void sgemm_reference_launch(int M, int N, int K, float alpha, const float* dA,
                            const float* dB, float beta, float* dC) {
    const dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    const dim3 blockDim((BM * BN) / (TM * TN));
    sgemm_2d_tiling<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
}
