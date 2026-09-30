#ifndef GEMM_CUH
#define GEMM_CUH

// 2D register-tiled SGEMM, from github.com/rachitj27/cuda-gemm-from-scratch
// (05_2d_tiling.cu), where it reached 3220 GFLOPS at M=N=K=4096 on a T4, or
// 76.3% of cuBLAS. Each thread computes a TM x TN block of C by outer product
// in registers, so one pass over the shared tiles feeds TM*TN multiply-adds.
//
// Changed from the source: the tile sizes are template parameters instead of
// #defines, C's beta term is specialized away, and the benchmark main() is
// gone. The arithmetic and the loop structure are untouched --
// gemm_selftest.cu asserts this version is bit-identical to the verbatim copy
// in gemm_reference.cu.
//
// No bounds checks, exactly as in the source. Callers pass dimensions already
// rounded up to the tile sizes and keep the padding region zeroed; see
// conv_cuda.cu. Row-major throughout: A is MxK, B is KxN, C is MxN.

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

    // The source always read C to form beta*C, which at beta == 0 still
    // touches uninitialized memory -- and 0.0f * NaN is NaN. if constexpr
    // rather than a ternary: the ternary kept the C address live and pushed
    // this to 130 registers, and at 256 threads anything over 128 costs half
    // the resident blocks on a T4.
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

// Tile configurations. N is the output-channel count, which in this network is
// always a power of two between 2 and 256, and 57.5% of the model's MACs sit at
// N=64 -- so the narrow configs are the ones that matter, not the 128x128 the
// kernel was originally tuned for.
//
// The narrow tiles also help occupancy. Per ptxas for sm_75, against the T4's
// 65536 registers and 64 KB of shared memory per SM:
//
//   N128  128 reg,  8 KB smem, 256 thr -> 2 blocks/SM, 16 warps, 50%
//   N64   106 reg, 12 KB smem, 256 thr -> 2 blocks/SM, 16 warps, 50%
//   N32    72 reg,  6 KB smem, 128 thr -> 7 blocks/SM, 28 warps, 87.5%
//
// 128 registers at 256 threads is exactly half the register file, so N128 sits
// right on the edge of fitting two blocks. Two registers more and it drops to
// one block and loses about 10% -- measured, not hypothetical, which is why
// the epilogue below uses if constexpr.
//
// 2D register tiling trades occupancy for instruction-level parallelism, so
// 50% is where this kernel wants to be, not a shortfall.
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

// Launch the instantiation matching `tile`. M, N and K must already be rounded
// up to the tile's bm/bn/bk.
void gemm_launch(GemmTile tile, int M, int N, int K, float alpha,
                 const float* dA, const float* dB, float beta, float* dC,
                 cudaStream_t stream);

#endif  // GEMM_CUH
