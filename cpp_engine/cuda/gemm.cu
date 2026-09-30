#include "gemm.cuh"

// Dispatch to the instantiation for the chosen tile. Written out rather than
// generated so the three configurations that actually get compiled are visible
// in one place.
void gemm_launch(GemmTile tile, int M, int N, int K, float alpha,
                 const float* dA, const float* dB, float beta, float* dC,
                 cudaStream_t stream) {
    const GemmTileShape s = gemm_tile_shape(tile);
    const dim3 grid((N + s.bn - 1) / s.bn, (M + s.bm - 1) / s.bm);
    const dim3 block(s.threads);

    const bool beta_zero = (beta == 0.0f);

    switch (tile) {
        case GemmTile::N32:
            if (beta_zero)
                sgemm_tiled<64, 32, 16, 4, 4, true><<<grid, block, 0, stream>>>(
                    M, N, K, alpha, dA, dB, beta, dC);
            else
                sgemm_tiled<64, 32, 16, 4, 4, false><<<grid, block, 0, stream>>>(
                    M, N, K, alpha, dA, dB, beta, dC);
            break;
        case GemmTile::N64:
            if (beta_zero)
                sgemm_tiled<128, 64, 16, 8, 4, true><<<grid, block, 0, stream>>>(
                    M, N, K, alpha, dA, dB, beta, dC);
            else
                sgemm_tiled<128, 64, 16, 8, 4, false><<<grid, block, 0, stream>>>(
                    M, N, K, alpha, dA, dB, beta, dC);
            break;
        case GemmTile::N128:
            if (beta_zero)
                sgemm_tiled<128, 128, 8, 8, 8, true><<<grid, block, 0, stream>>>(
                    M, N, K, alpha, dA, dB, beta, dC);
            else
                sgemm_tiled<128, 128, 8, 8, 8, false><<<grid, block, 0, stream>>>(
                    M, N, K, alpha, dA, dB, beta, dC);
            break;
    }
}
