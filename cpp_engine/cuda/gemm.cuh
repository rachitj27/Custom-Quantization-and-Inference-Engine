#ifndef GEMM_CUH
#define GEMM_CUH

// 2D register-tiled SGEMM from github.com/rachitj27/cuda-gemm-from-scratch
// (05_2d_tiling.cu). Tile sizes templated, beta specialized, benchmark main
// dropped; arithmetic unchanged. Row-major, A is MxK, B is KxN, C is MxN.
// No bounds checks: callers pad to the tile and zero the padding.

#include <cuda_runtime.h>

template <int BM, int BN, int BK, int TM, int TN, bool BETA_ZERO>
__global__ void sgemm_tiled(int M, int N, int K, float alpha,
                            const float* __restrict__ A,
                            const float* __restrict__ B,
                            float beta, float* __restrict__ C) {
    constexpr int kThreads = (BM * BN) / (TM * TN);
    constexpr int kStrideA = kThreads / BK;
    constexpr int kStrideB = kThreads / BN;

    static_assert(BM % TM == 0, "BM must be a whole number of thread rows");
    static_assert(BN % TN == 0, "BN must be a whole number of thread columns");
    static_assert(kThreads % BK == 0, "A tile load must cover whole rows");
    static_assert(kThreads % BN == 0, "B tile load must cover whole rows");
    static_assert(BM % kStrideA == 0, "A tile load must tile BM evenly");
    static_assert(BK % kStrideB == 0, "B tile load must tile BK evenly");

    const unsigned cRow = blockIdx.y;
    const unsigned cCol = blockIdx.x;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const unsigned threadCol = threadIdx.x % (BN / TN);
    const unsigned threadRow = threadIdx.x / (BN / TN);

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const unsigned innerRowA = threadIdx.x / BK;
    const unsigned innerColA = threadIdx.x % BK;
    const unsigned innerRowB = threadIdx.x / BN;
    const unsigned innerColB = threadIdx.x % BN;

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // load tiles
        for (int loadOffset = 0; loadOffset < BM; loadOffset += kStrideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] =
                A[(innerRowA + loadOffset) * K + innerColA];
        }
        for (int loadOffset = 0; loadOffset < BK; loadOffset += kStrideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
                B[(innerRowB + loadOffset) * N + innerColB];
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        // outer product into registers
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (int i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }

        __syncthreads();
    }

    // store. if constexpr, not a ternary: the ternary kept C's address live
    // and cost 2 registers, which halves resident blocks at 256 threads.
    for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
            const int idx = (threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN;
            if constexpr (BETA_ZERO) {
                C[idx] = alpha * threadResults[resIdxM * TN + resIdxN];
            } else {
                C[idx] = alpha * threadResults[resIdxM * TN + resIdxN] + beta * C[idx];
            }
        }
    }
}

// tile configs, picked by N. 57.5% of the model's MACs sit at N=64.
// sm_75 occupancy: N128 128 reg 2 blocks, N64 106 reg 2, N32 72 reg 7.
enum class GemmTile { N32, N64, N128 };

struct GemmTileShape {
    int bm, bn, bk, tm, tn, threads;
};

inline GemmTileShape gemm_tile_shape(GemmTile t) {
    switch (t) {
        case GemmTile::N32:  return {64, 32, 16, 4, 4, 128};
        case GemmTile::N64:  return {128, 64, 16, 8, 4, 256};
        case GemmTile::N128: return {128, 128, 8, 8, 8, 256};
    }
    return {128, 128, 8, 8, 8, 256};
}

inline GemmTile gemm_choose_tile(int n) {
    if (n <= 32) return GemmTile::N32;
    if (n <= 64) return GemmTile::N64;
    return GemmTile::N128;
}

// M, N, K must already be rounded up to the tile.
void gemm_launch(GemmTile tile, int M, int N, int K, float alpha,
                 const float* dA, const float* dB, float beta, float* dC,
                 cudaStream_t stream);

#endif  // GEMM_CUH
