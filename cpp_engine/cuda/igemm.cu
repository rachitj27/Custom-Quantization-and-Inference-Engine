#include "igemm.cuh"

void igemm_launch(IgemmTile tile, int M, int N, int K4, const int* dA,
                  const int* dB, int* dC, cudaStream_t stream) {
    const IgemmTileShape s = igemm_tile_shape(tile);
    const dim3 grid((N + s.bn - 1) / s.bn, (M + s.bm - 1) / s.bm);
    const dim3 block(s.threads);

    switch (tile) {
        case IgemmTile::N32:
            igemm_dp4a_tiled<64, 32, 4, 4, 4, 8><<<grid, block, 0, stream>>>(
                M, N, K4, dA, dB, dC);
            break;
        case IgemmTile::N64:
            igemm_dp4a_tiled<128, 64, 4, 8, 4, 3><<<grid, block, 0, stream>>>(
                M, N, K4, dA, dB, dC);
            break;
        case IgemmTile::N128:
            igemm_dp4a_tiled<128, 128, 4, 8, 8, 2><<<grid, block, 0, stream>>>(
                M, N, K4, dA, dB, dC);
            break;
    }
}
