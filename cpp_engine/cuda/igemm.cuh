#ifndef IGEMM_CUH
#define IGEMM_CUH

// INT8 GEMM, int32 accumulation. Same tiling as gemm.cuh with the multiply-add
// replaced by __dp4a (four packed int8 pairs -> int32, the GPU counterpart of
// VPDPBUSD). K counted in groups of four: A is [M][K4], B is [K4][N], both
// int32, element k4 packing K values 4*k4..4*k4+3. K is always a multiple of
// 16 here, so K4 always divides BK4 = 4 and the K axis needs no padding.

#include <cuda_runtime.h>

// MINBLOCKS is per config, set to the block count that config already reaches.
// Unconstrained, 128x128 lands on 130 registers and loses half its occupancy;
// asking for fewer blocks than a tile achieves costs it one.
template <int BM, int BN, int BK4, int TM, int TN, int MINBLOCKS>
__global__ __launch_bounds__((BM * BN) / (TM * TN), MINBLOCKS)
void igemm_dp4a_tiled(int M, int N, int K4,
                      const int* __restrict__ A,
                      const int* __restrict__ B,
                      int* __restrict__ C) {
    constexpr int kThreads = (BM * BN) / (TM * TN);
    constexpr int kStrideA = kThreads / BK4;
    constexpr int kStrideB = kThreads / BN;

    static_assert(BM % TM == 0, "BM must be a whole number of thread rows");
    static_assert(BN % TN == 0, "BN must be a whole number of thread columns");
    static_assert(kThreads % BK4 == 0, "A tile load must cover whole rows");
    static_assert(kThreads % BN == 0, "B tile load must cover whole rows");
    static_assert(BM % kStrideA == 0, "A tile load must tile BM evenly");
    static_assert(BK4 % kStrideB == 0, "B tile load must tile BK4 evenly");

    const unsigned cRow = blockIdx.y;
    const unsigned cCol = blockIdx.x;

    __shared__ int As[BM * BK4];
    __shared__ int Bs[BK4 * BN];

    const unsigned threadCol = threadIdx.x % (BN / TN);
    const unsigned threadRow = threadIdx.x / (BN / TN);

    A += cRow * BM * K4;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const unsigned innerRowA = threadIdx.x / BK4;
    const unsigned innerColA = threadIdx.x % BK4;
    const unsigned innerRowB = threadIdx.x / BN;
    const unsigned innerColB = threadIdx.x % BN;

    int threadResults[TM * TN] = {0};
    int regM[TM] = {0};
    int regN[TN] = {0};

    for (int bk = 0; bk < K4; bk += BK4) {
        // load tiles
        for (int loadOffset = 0; loadOffset < BM; loadOffset += kStrideA) {
            As[(innerRowA + loadOffset) * BK4 + innerColA] =
                A[(innerRowA + loadOffset) * K4 + innerColA];
        }
        for (int loadOffset = 0; loadOffset < BK4; loadOffset += kStrideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
                B[(innerRowB + loadOffset) * N + innerColB];
        }

        __syncthreads();

        A += BK4;
        B += BK4 * N;

        // outer product into registers
        for (int dotIdx = 0; dotIdx < BK4; ++dotIdx) {
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK4 + dotIdx];
            }
            for (int i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] =
                        __dp4a(regM[resIdxM], regN[resIdxN],
                               threadResults[resIdxM * TN + resIdxN]);
                }
            }
        }

        __syncthreads();
    }

    // store. no alpha or beta: the scales are per channel, so they live in the
    // host epilogue.
    for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
                threadResults[resIdxM * TN + resIdxN];
        }
    }
}

// tile configs. BK4 stays 4 so K never needs rounding.
// sm_75: N128 124 reg 2 blocks, N64 80 reg 3, N32 62 reg 8. No spills.
enum class IgemmTile { N32, N64, N128 };

struct IgemmTileShape {
    int bm, bn, bk4, tm, tn, threads;
};

inline IgemmTileShape igemm_tile_shape(IgemmTile t) {
    switch (t) {
        case IgemmTile::N32:  return {64, 32, 4, 4, 4, 128};
        case IgemmTile::N64:  return {128, 64, 4, 8, 4, 256};
        case IgemmTile::N128: return {128, 128, 4, 8, 8, 256};
    }
    return {128, 128, 4, 8, 8, 256};
}

inline IgemmTile igemm_choose_tile(int n) {
    if (n <= 32) return IgemmTile::N32;
    if (n <= 64) return IgemmTile::N64;
    return IgemmTile::N128;
}

// M and N must already be rounded up to the tile.
void igemm_launch(IgemmTile tile, int M, int N, int K4, const int* dA,
                  const int* dB, int* dC, cudaStream_t stream);

#endif  // IGEMM_CUH
