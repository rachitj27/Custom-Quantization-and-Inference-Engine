// Standalone harness for the GEMM kernels, independent of the engine.
//
// Two things get checked. First, that the templated kernel in gemm.cuh is
// bit-identical to the verbatim vendored copy at M=N=K=4096, which is the shape
// it was originally tuned and measured at. Second, that all three tile
// configurations are correct on the 32 distinct (M, N, K) triples this network
// actually produces -- and how fast they are there, which is a very different
// question from the 4096 number.
//
//   ./gemm_selftest            all checks
//   ./gemm_selftest --shapes   skip the 4096 equivalence check

#include "gemm.cuh"
#include "gemm_reference.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

#define CUDA_OK(call)                                                            \
    do {                                                                         \
        const cudaError_t err_ = (call);                                          \
        if (err_ != cudaSuccess) {                                                \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_),      \
                        __FILE__, __LINE__);                                      \
            std::exit(2);                                                         \
        }                                                                        \
    } while (0)

// The GEMM shape of every convolution in the model, deduped. M is out_h*out_w,
// N is out_ch, K is kh*kw*ic_padded.
struct Shape {
    int m, n, k;
};

const Shape kShapes[] = {
    {400, 2, 64},       {400, 64, 64},      {400, 64, 576},     {400, 64, 2304},
    {400, 128, 256},    {400, 128, 1152},   {400, 256, 256},    {400, 256, 384},
    {400, 256, 512},    {400, 256, 1152},   {1600, 2, 64},      {1600, 64, 64},
    {1600, 64, 576},    {1600, 64, 1152},   {1600, 128, 128},   {1600, 128, 192},
    {1600, 128, 256},   {1600, 128, 384},   {1600, 128, 576},   {6400, 2, 64},
    {6400, 32, 288},    {6400, 64, 64},     {6400, 64, 96},     {6400, 64, 128},
    {6400, 64, 192},    {6400, 64, 288},    {6400, 64, 576},    {25600, 16, 144},
    {25600, 32, 32},    {25600, 32, 48},    {25600, 32, 144},   {102400, 16, 144},
};
constexpr int kNumShapes = static_cast<int>(sizeof(kShapes) / sizeof(kShapes[0]));

int round_up(int v, int m) { return ((v + m - 1) / m) * m; }

// Deterministic, and spread across the exponent range enough that a wrong
// index shows up instead of averaging out.
float value_at(unsigned i) {
    unsigned h = i * 2654435761u;
    h ^= h >> 15;
    return static_cast<float>(static_cast<int>(h % 2001) - 1000) / 1000.0f;
}

bool check_4096() {
    const int n = 4096;
    const size_t elems = static_cast<size_t>(n) * n;
    const size_t bytes = elems * sizeof(float);

    std::vector<float> hA(elems), hB(elems);
    for (size_t i = 0; i < elems; i++) hA[i] = value_at(static_cast<unsigned>(i));
    for (size_t i = 0; i < elems; i++) hB[i] = value_at(static_cast<unsigned>(i + 7771));

    float *dA, *dB, *dRef, *dNew;
    CUDA_OK(cudaMalloc(&dA, bytes));
    CUDA_OK(cudaMalloc(&dB, bytes));
    CUDA_OK(cudaMalloc(&dRef, bytes));
    CUDA_OK(cudaMalloc(&dNew, bytes));
    CUDA_OK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    // beta = 0, so the reference must not be handed uninitialized memory --
    // that is the bug the templated version specializes away.
    CUDA_OK(cudaMemset(dRef, 0, bytes));
    CUDA_OK(cudaMemset(dNew, 0, bytes));

    sgemm_reference_launch(n, n, n, 1.0f, dA, dB, 0.0f, dRef);
    CUDA_OK(cudaDeviceSynchronize());
    gemm_launch(GemmTile::N128, n, n, n, 1.0f, dA, dB, 0.0f, dNew, nullptr);
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<float> hRef(elems), hNew(elems);
    CUDA_OK(cudaMemcpy(hRef.data(), dRef, bytes, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(hNew.data(), dNew, bytes, cudaMemcpyDeviceToHost));

    size_t differing = 0;
    for (size_t i = 0; i < elems; i++) {
        if (std::memcmp(&hRef[i], &hNew[i], sizeof(float)) != 0) differing++;
    }

    // Time both while the buffers are still around.
    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));
    float ms_ref = 0.0f, ms_new = 0.0f;
    const int runs = 5;

    CUDA_OK(cudaEventRecord(start));
    for (int r = 0; r < runs; r++) sgemm_reference_launch(n, n, n, 1.0f, dA, dB, 0.0f, dRef);
    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    CUDA_OK(cudaEventElapsedTime(&ms_ref, start, stop));

    CUDA_OK(cudaEventRecord(start));
    for (int r = 0; r < runs; r++)
        gemm_launch(GemmTile::N128, n, n, n, 1.0f, dA, dB, 0.0f, dNew, nullptr);
    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    CUDA_OK(cudaEventElapsedTime(&ms_new, start, stop));

    const double flops = 2.0 * n * n * n;
    std::printf("  verbatim  %8.2f ms  %8.1f GFLOPS\n", ms_ref / runs,
                (flops / 1e9) / (ms_ref / runs / 1000.0));
    std::printf("  templated %8.2f ms  %8.1f GFLOPS\n", ms_new / runs,
                (flops / 1e9) / (ms_new / runs / 1000.0));
    std::printf("  differing elements: %zu / %zu\n", differing, elems);

    CUDA_OK(cudaEventDestroy(start));
    CUDA_OK(cudaEventDestroy(stop));
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dRef);
    cudaFree(dNew);
    return differing == 0;
}

// Reference dot products for a sample of output positions. A full host GEMM
// over every shape would take minutes and prove nothing extra; a spread of
// sampled positions catches an indexing error just as well.
bool check_shape(const Shape& s, double* out_gflops) {
    const GemmTile tile = gemm_choose_tile(s.n);
    const GemmTileShape ts = gemm_tile_shape(tile);

    const int mp = round_up(s.m, ts.bm);
    const int np = round_up(s.n, ts.bn);
    const int kp = round_up(s.k, ts.bk);

    const size_t bytes_a = static_cast<size_t>(mp) * kp * sizeof(float);
    const size_t bytes_b = static_cast<size_t>(kp) * np * sizeof(float);
    const size_t bytes_c = static_cast<size_t>(mp) * np * sizeof(float);

    std::vector<float> hA(static_cast<size_t>(mp) * kp, 0.0f);
    std::vector<float> hB(static_cast<size_t>(kp) * np, 0.0f);

    // Only the real region carries data; the padding stays zero, which is what
    // makes the padded GEMM exact rather than approximate.
    for (int m = 0; m < s.m; m++) {
        for (int k = 0; k < s.k; k++) {
            hA[static_cast<size_t>(m) * kp + k] = value_at(static_cast<unsigned>(m * 131 + k));
        }
    }
    for (int k = 0; k < s.k; k++) {
        for (int n = 0; n < s.n; n++) {
            hB[static_cast<size_t>(k) * np + n] = value_at(static_cast<unsigned>(k * 977 + n + 55));
        }
    }

    float *dA, *dB, *dC;
    CUDA_OK(cudaMalloc(&dA, bytes_a));
    CUDA_OK(cudaMalloc(&dB, bytes_b));
    CUDA_OK(cudaMalloc(&dC, bytes_c));
    CUDA_OK(cudaMemcpy(dA, hA.data(), bytes_a, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, hB.data(), bytes_b, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(dC, 0, bytes_c));

    gemm_launch(tile, mp, np, kp, 1.0f, dA, dB, 0.0f, dC, nullptr);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaGetLastError());

    std::vector<float> hC(static_cast<size_t>(mp) * np);
    CUDA_OK(cudaMemcpy(hC.data(), dC, bytes_c, cudaMemcpyDeviceToHost));

    double worst_rel = 0.0;
    const int samples = 2048;
    for (int t = 0; t < samples; t++) {
        const unsigned h = static_cast<unsigned>(t) * 2246822519u;
        const int m = static_cast<int>((h >> 8) % static_cast<unsigned>(s.m));
        const int n = static_cast<int>((h >> 3) % static_cast<unsigned>(s.n));

        double ref = 0.0;
        for (int k = 0; k < s.k; k++) {
            ref += static_cast<double>(hA[static_cast<size_t>(m) * kp + k]) *
                   static_cast<double>(hB[static_cast<size_t>(k) * np + n]);
        }
        const double got = hC[static_cast<size_t>(m) * np + n];
        const double denom = std::fabs(ref) > 1e-6 ? std::fabs(ref) : 1.0;
        const double rel = std::fabs(got - ref) / denom;
        if (rel > worst_rel) worst_rel = rel;
    }

    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));
    const int runs = 20;
    gemm_launch(tile, mp, np, kp, 1.0f, dA, dB, 0.0f, dC, nullptr);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaEventRecord(start));
    for (int r = 0; r < runs; r++)
        gemm_launch(tile, mp, np, kp, 1.0f, dA, dB, 0.0f, dC, nullptr);
    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
    ms /= runs;

    // Report against the useful work, not the padded work -- padding is
    // overhead, so counting it would flatter the number.
    const double useful = 2.0 * s.m * s.n * s.k;
    *out_gflops = (useful / 1e9) / (ms / 1000.0);

    const int blocks = ((np + ts.bn - 1) / ts.bn) * ((mp + ts.bm - 1) / ts.bm);
    const double pad_mult = (static_cast<double>(mp) * np * kp) /
                            (static_cast<double>(s.m) * s.n * s.k);

    const bool ok = worst_rel < 1e-4;
    std::printf("  %7d %5d %6d  %3dx%-3d %6d  %8.3f ms %8.1f GF  %5.2fx %7.1e %s\n",
                s.m, s.n, s.k, ts.bm, ts.bn, blocks, ms, *out_gflops, pad_mult,
                worst_rel, ok ? "" : "FAIL");

    CUDA_OK(cudaEventDestroy(start));
    CUDA_OK(cudaEventDestroy(stop));
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    bool skip_4096 = false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--shapes") == 0) skip_4096 = true;
    }

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("no CUDA device\n");
        return 2;
    }
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    std::printf("%s, sm_%d%d, %d SMs\n\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);

    bool ok = true;

    if (!skip_4096) {
        std::printf("=== templated vs verbatim, M=N=K=4096 ===\n");
        const bool same = check_4096();
        std::printf("  %s\n\n", same ? "identical" : "MISMATCH");
        ok = ok && same;
    }

    std::printf("=== the network's 32 distinct shapes ===\n");
    std::printf("  %7s %5s %6s  %7s %6s  %11s %11s  %5s %7s\n",
                "M", "N", "K", "tile", "blocks", "time", "useful", "pad", "rel");
    double sum_gflops = 0.0;
    for (int i = 0; i < kNumShapes; i++) {
        double gflops = 0.0;
        if (!check_shape(kShapes[i], &gflops)) ok = false;
        sum_gflops += gflops;
    }
    std::printf("\n  mean %.1f GFLOPS across %d shapes\n",
                sum_gflops / kNumShapes, kNumShapes);

    std::printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
