// Exercises the GEMM kernels on their own. Checks the templated kernel is
// bit-identical to the verbatim copy at 4096, then runs all 32 shapes the
// network produces through both the FP32 and INT8 paths.
//
//   ./gemm_selftest            all checks
//   ./gemm_selftest --shapes   skip the 4096 check

#include "gemm.cuh"
#include "gemm_reference.cuh"
#include "igemm.cuh"

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

// Every convolution's GEMM shape, deduped. M = out_h*out_w, N = out_ch,
// K = kh*kw*ic_padded, layers = how many of the 63 convs share it.
struct Shape {
    int m, n, k, layers;
};

const Shape kShapes[] = {
    {400, 2, 64, 1},      {400, 64, 64, 1},     {400, 64, 576, 2},
    {400, 64, 2304, 2},   {400, 128, 256, 1},   {400, 128, 1152, 5},
    {400, 256, 256, 1},   {400, 256, 384, 3},   {400, 256, 512, 1},
    {400, 256, 1152, 1},  {1600, 2, 64, 1},     {1600, 64, 64, 1},
    {1600, 64, 576, 11},  {1600, 64, 1152, 2},  {1600, 128, 128, 1},
    {1600, 128, 192, 3},  {1600, 128, 256, 1},  {1600, 128, 384, 1},
    {1600, 128, 576, 1},  {6400, 2, 64, 1},     {6400, 32, 288, 6},
    {6400, 64, 64, 2},    {6400, 64, 96, 1},    {6400, 64, 128, 1},
    {6400, 64, 192, 1},   {6400, 64, 288, 1},   {6400, 64, 576, 4},
    {25600, 16, 144, 2},  {25600, 32, 32, 1},   {25600, 32, 48, 1},
    {25600, 32, 144, 1},  {102400, 16, 144, 1},
};
constexpr int kNumShapes = static_cast<int>(sizeof(kShapes) / sizeof(kShapes[0]));

int round_up(int v, int m) { return ((v + m - 1) / m) * m; }

// deterministic, spread enough that a wrong index shows up
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

    // beta = 0, so the reference must not read uninitialized memory
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

    // time both
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

// sampled reference dot products; a full host GEMM proves nothing extra
bool check_shape(const Shape& s, double* out_gflops, double* out_ms) {
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

    // padding stays zero, which is what makes the padded GEMM exact
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

    double worst_abs = 0.0;
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
        const double abs_err = std::fabs(got - ref);
        if (abs_err > worst_abs) worst_abs = abs_err;
        if (std::fabs(ref) > 1e-3) {
            const double rel = abs_err / std::fabs(ref);
            if (rel > worst_rel) worst_rel = rel;
        }
    }

    // Gate on absolute error: inputs are in [-1, 1] so a K-term sum rounds by
    // at most about K * FLT_EPSILON. Relative error is the wrong gate, since a
    // dot product of random signs often lands near zero.
    const double atol = 8.0 * s.k * 1.1920929e-7;

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

    // against useful work, not padded
    const double useful = 2.0 * s.m * s.n * s.k;
    *out_gflops = (useful / 1e9) / (ms / 1000.0);
    *out_ms = ms;

    const int blocks = ((np + ts.bn - 1) / ts.bn) * ((mp + ts.bm - 1) / ts.bm);
    const double pad_mult = (static_cast<double>(mp) * np * kp) /
                            (static_cast<double>(s.m) * s.n * s.k);

    const bool ok = worst_abs <= atol;
    std::printf("  %7d %5d %6d %2d  %3dx%-3d %5d  %7.3f %7.1f  %5.2fx %8.1e %8.1e %s\n",
                s.m, s.n, s.k, s.layers, ts.bm, ts.bn, blocks, ms, *out_gflops,
                pad_mult, worst_abs, worst_rel, ok ? "" : "FAIL");

    CUDA_OK(cudaEventDestroy(start));
    CUDA_OK(cudaEventDestroy(stop));
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return ok;
}

// no tolerance: int32 accumulation is exact, so any difference is a bug
bool check_shape_int8(const Shape& s, double* out_ms) {
    const IgemmTile tile = igemm_choose_tile(s.n);
    const IgemmTileShape ts = igemm_tile_shape(tile);

    const int mp = round_up(s.m, ts.bm);
    const int np = round_up(s.n, ts.bn);
    const int kp = s.k;  // a multiple of 16, so kp/4 always divides bk4
    const int k4 = kp / 4;

    std::vector<signed char> hA((size_t)mp * kp, 0);
    std::vector<signed char> hB((size_t)k4 * np * 4, 0);

    for (int m = 0; m < s.m; m++) {
        for (int k = 0; k < kp; k++) {
            hA[(size_t)m * kp + k] =
                (signed char)(int)(value_at((unsigned)(m * 31 + k * 7)) * 127.0f);
        }
    }
    for (int k = 0; k < kp; k++) {
        for (int n = 0; n < s.n; n++) {
            hB[((size_t)(k / 4) * np + n) * 4 + (k % 4)] =
                (signed char)(int)(value_at((unsigned)(k * 17 + n * 3 + 11)) * 127.0f);
        }
    }

    void *dA, *dB, *dC;
    CUDA_OK(cudaMalloc(&dA, hA.size()));
    CUDA_OK(cudaMalloc(&dB, hB.size()));
    CUDA_OK(cudaMalloc(&dC, (size_t)mp * np * sizeof(int)));
    CUDA_OK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(dC, 0, (size_t)mp * np * sizeof(int)));

    igemm_launch(tile, mp, np, k4, (const int*)dA, (const int*)dB, (int*)dC, nullptr);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaGetLastError());

    std::vector<int> hC((size_t)mp * np);
    CUDA_OK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(int), cudaMemcpyDeviceToHost));

    int mismatches = 0;
    long long worst_acc = 0;
    const int samples = 2048;
    for (int t = 0; t < samples; t++) {
        const unsigned h = (unsigned)t * 2246822519u;
        const int m = (int)((h >> 8) % (unsigned)s.m);
        const int n = (int)((h >> 3) % (unsigned)s.n);

        int ref = 0;
        for (int k = 0; k < kp; k++) {
            ref += (int)hA[(size_t)m * kp + k] *
                   (int)hB[((size_t)(k / 4) * np + n) * 4 + (k % 4)];
        }
        if (std::abs(ref) > worst_acc) worst_acc = std::abs(ref);
        if (hC[(size_t)m * np + n] != ref) mismatches++;
    }

    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));
    const int runs = 20;
    igemm_launch(tile, mp, np, k4, (const int*)dA, (const int*)dB, (int*)dC, nullptr);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaEventRecord(start));
    for (int r = 0; r < runs; r++)
        igemm_launch(tile, mp, np, k4, (const int*)dA, (const int*)dB, (int*)dC, nullptr);
    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
    ms /= runs;
    *out_ms = ms;

    const double useful = 2.0 * s.m * s.n * s.k;
    const int blocks = (np / ts.bn) * (mp / ts.bm);
    std::printf("  %7d %5d %6d %2d  %3dx%-3d %5d  %7.3f %7.1f  %10lld %s\n",
                s.m, s.n, s.k, s.layers, ts.bm, ts.bn, blocks, ms,
                (useful / 1e9) / (ms / 1000.0), worst_acc,
                mismatches ? "MISMATCH" : "");

    CUDA_OK(cudaEventDestroy(start));
    CUDA_OK(cudaEventDestroy(stop));
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return mismatches == 0;
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
    std::printf("  %7s %5s %6s %2s  %7s %5s  %7s %7s  %5s %8s %8s\n",
                "M", "N", "K", "x", "tile", "blocks", "ms", "GFLOPS", "pad",
                "abs", "rel");

    double image_ms = 0.0;
    double image_flop = 0.0;
    int layers = 0;
    for (int i = 0; i < kNumShapes; i++) {
        double gflops = 0.0, ms = 0.0;
        if (!check_shape(kShapes[i], &gflops, &ms)) ok = false;
        image_ms += ms * kShapes[i].layers;
        image_flop += 2.0 * kShapes[i].m * kShapes[i].n * kShapes[i].k * kShapes[i].layers;
        layers += kShapes[i].layers;
    }

    // Weighted by layer count, so this is one forward pass. K includes the
    // channel rounding; the model's real arithmetic is 4.041 GMAC, and the
    // rate is reported against that.
    constexpr double kModelGmac = 4.041;
    const double padded_gmac = image_flop / 2e9;
    std::printf("\n  GEMM time for one image: %.2f ms over %d convolutions\n",
                image_ms, layers);
    std::printf("  %.3f GMAC of real convolution at %.0f GMAC/s\n", kModelGmac,
                kModelGmac / (image_ms / 1000.0));
    std::printf("  (%.3f GMAC issued; the %.1f%% excess is ic_padded rounding)\n",
                padded_gmac, 100.0 * (padded_gmac / kModelGmac - 1.0));
    std::printf("  for reference: AVX-VNNI on one laptop core is 233.5 ms,\n");
    std::printf("  PyTorch FP32 on a T4 is 8.0 ms, TensorRT INT8 on a T4 is 4.9 ms\n");

    std::printf("\n=== the same shapes through the INT8 dp4a kernel ===\n");
    std::printf("  %7s %5s %6s %2s  %7s %5s  %7s %7s  %10s\n", "M", "N", "K", "x",
                "tile", "blocks", "ms", "GOPS", "max|acc|");
    double int8_ms = 0.0;
    for (int i = 0; i < kNumShapes; i++) {
        double ms = 0.0;
        if (!check_shape_int8(kShapes[i], &ms)) ok = false;
        int8_ms += ms * kShapes[i].layers;
    }
    std::printf("\n  INT8 GEMM time for one image: %.2f ms\n", int8_ms);
    std::printf("  %.3f GMAC at %.0f GMAC/s, versus %.2f ms for the FP32 path\n",
                kModelGmac, kModelGmac / (int8_ms / 1000.0), image_ms);

    std::printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
