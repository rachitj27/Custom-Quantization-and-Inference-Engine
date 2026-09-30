#include "conv_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemm.cuh"

namespace {

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA ") + what + ": " +
                                 cudaGetErrorString(err));
    }
}

#define CU(call) cuda_check((call), #call)

int round_up(int v, int m) { return ((v + m - 1) / m) * m; }

// Grows to a high-water mark and never shrinks. Because the graph is the same
// on every image, every allocation happens during the first forward pass and
// none after -- which is the whole point of not allocating per layer.
struct DeviceBuffer {
    void* p = nullptr;
    size_t bytes = 0;

    void ensure(size_t need) {
        if (need <= bytes) return;
        if (p) CU(cudaFree(p));
        CU(cudaMalloc(&p, need));
        bytes = need;
    }

    void release() {
        if (p) cudaFree(p);
        p = nullptr;
        bytes = 0;
    }
};

struct PinnedBuffer {
    void* p = nullptr;
    size_t bytes = 0;

    void ensure(size_t need) {
        if (need <= bytes) return;
        if (p) CU(cudaFreeHost(p));
        CU(cudaHostAlloc(&p, need, cudaHostAllocDefault));
        bytes = need;
    }

    void release() {
        if (p) cudaFreeHost(p);
        p = nullptr;
        bytes = 0;
    }
};

// Per-layer device state. M is absent because it depends on the input's
// spatial size, which the caller supplies at inference time.
struct LayerPlan {
    float* d_b = nullptr;  // weight panel, [kp][np], padding zeroed
    int n = 0;
    int np = 0;
    int kp = 0;
    int icp = 0;
    GemmTile tile = GemmTile::N128;
};

std::vector<LayerPlan> g_plans;
DeviceBuffer g_act;  // INT8 CHW activation
DeviceBuffer g_a;    // im2col matrix, [mp][kp]
DeviceBuffer g_c;    // GEMM output, [mp][np]
PinnedBuffer g_stage_in;
PinnedBuffer g_stage_out;

bool g_profile = false;
CudaPhaseTimes g_times;
cudaEvent_t g_ev[6];
bool g_events_ready = false;

void ensure_events() {
    if (g_events_ready) return;
    for (int i = 0; i < 6; i++) CU(cudaEventCreate(&g_ev[i]));
    g_events_ready = true;
}

float elapsed(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    CU(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

// Builds A so that row m is output pixel m and column k is tap (r, c) of input
// channel i, with k = (r * kw + c) * icp + i. That is the same order the VNNI
// path packs its weights in, so the INT8 stage can reuse weights_hwc directly.
//
// Out-of-bounds taps and the channel padding are written as a real zero, which
// is exactly what the scalar FP32 kernel contributes for them.
__global__ void im2col_fp32(float* __restrict__ A, long long total, int kp,
                            const signed char* __restrict__ in, int in_ch,
                            int in_h, int in_w, int out_w, int kh, int kw,
                            int stride, int pad, int icp, float scale,
                            int zero_point, int m_real, int k_real) {
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int m = (int)(idx / kp);
        const int k = (int)(idx - (long long)m * kp);

        float v = 0.0f;
        if (m < m_real && k < k_real) {
            const int i = k % icp;
            const int tap = k / icp;
            const int c = tap % kw;
            const int r = tap / kw;
            if (i < in_ch && r < kh) {
                const int oh = m / out_w;
                const int ow = m - oh * out_w;
                const int ih = oh * stride + r - pad;
                const int iw = ow * stride + c - pad;
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    const int q = in[((long long)i * in_h + ih) * in_w + iw];
                    v = scale * (float)(q - zero_point);
                }
            }
        }
        A[idx] = v;
    }
}

float silu_host(float x) { return x / (1.0f + std::exp(-x)); }

}  // namespace

void cuda_profile_enable(bool on) { g_profile = on; }
bool cuda_profile_enabled() { return g_profile; }
CudaPhaseTimes cuda_phase_times() { return g_times; }
void cuda_reset_phase_times() { g_times = CudaPhaseTimes(); }

void cuda_prepare_layers(Model& model, Kernel kernel) {
    if (kernel != Kernel::CudaFp32) return;

    cuda_release();
    g_plans.clear();
    g_plans.reserve(model.conv_layers.size());

    for (size_t li = 0; li < model.conv_layers.size(); li++) {
        Layer& layer = model.conv_layers[li];
        if (layer.groups != 1) {
            throw std::runtime_error("Grouped convolution is not supported: " + layer.path);
        }
        if (layer.weights_fp32.empty()) {
            layer.cuda_slot = -1;
            continue;
        }

        const int oc = layer.weight_shape[0];
        const int ic = layer.weight_shape[1];
        const int kh = layer.weight_shape[2];
        const int kw = layer.weight_shape[3];

        LayerPlan plan;
        plan.icp = round_up(ic, 16);
        plan.n = oc;
        plan.tile = gemm_choose_tile(oc);

        const GemmTileShape ts = gemm_tile_shape(plan.tile);
        plan.np = round_up(oc, ts.bn);
        plan.kp = round_up(kh * kw * plan.icp, ts.bk);

        // Transpose the OIHW panel into [kp][np] once, here, so the kernel's B
        // loads stay coalesced along N and the hot path does no reordering.
        std::vector<float> panel((size_t)plan.kp * plan.np, 0.0f);
        for (int o = 0; o < oc; o++) {
            for (int i = 0; i < ic; i++) {
                for (int r = 0; r < kh; r++) {
                    for (int c = 0; c < kw; c++) {
                        const size_t src = (((size_t)o * ic + i) * kh + r) * kw + c;
                        const int k = (r * kw + c) * plan.icp + i;
                        panel[(size_t)k * plan.np + o] = layer.weights_fp32[src];
                    }
                }
            }
        }

        CU(cudaMalloc(&plan.d_b, panel.size() * sizeof(float)));
        CU(cudaMemcpy(plan.d_b, panel.data(), panel.size() * sizeof(float),
                      cudaMemcpyHostToDevice));

        layer.cuda_slot = (int)g_plans.size();
        g_plans.push_back(plan);
    }
}

void cuda_release() {
    for (LayerPlan& p : g_plans) {
        if (p.d_b) cudaFree(p.d_b);
        p.d_b = nullptr;
    }
    g_plans.clear();
    g_act.release();
    g_a.release();
    g_c.release();
    g_stage_in.release();
    g_stage_out.release();
    if (g_events_ready) {
        for (int i = 0; i < 6; i++) cudaEventDestroy(g_ev[i]);
        g_events_ready = false;
    }
}

FloatTensor conv_cuda_fp32(const Tensor& input, const Layer& layer, bool apply_silu) {
    if (layer.cuda_slot < 0 || layer.cuda_slot >= (int)g_plans.size()) {
        throw std::runtime_error("No CUDA plan for layer " + layer.path);
    }
    const LayerPlan& plan = g_plans[(size_t)layer.cuda_slot];

    const int in_ch = input.shape[0];
    const int in_h = input.shape[1];
    const int in_w = input.shape[2];
    if (in_ch != layer.weight_shape[1]) {
        throw std::runtime_error("Channel mismatch in " + layer.path);
    }

    const int out_ch = layer.weight_shape[0];
    const int kh = layer.weight_shape[2];
    const int kw = layer.weight_shape[3];
    const int stride = layer.stride;
    const int pad = layer.padding;
    const int out_h = (in_h + 2 * pad - kh) / stride + 1;
    const int out_w = (in_w + 2 * pad - kw) / stride + 1;

    const int m = out_h * out_w;
    const int k_real = kh * kw * plan.icp;
    const GemmTileShape ts = gemm_tile_shape(plan.tile);
    const int mp = round_up(m, ts.bm);

    FloatTensor out({out_ch, out_h, out_w});

    const size_t a_elems = (size_t)mp * plan.kp;
    const size_t c_elems = (size_t)mp * plan.np;
    g_act.ensure(input.num_elements);
    g_a.ensure(a_elems * sizeof(float));
    g_c.ensure(c_elems * sizeof(float));
    g_stage_in.ensure(input.num_elements);
    g_stage_out.ensure((size_t)m * plan.n * sizeof(float));

    const bool prof = g_profile;
    if (prof) ensure_events();

    // H2D
    std::memcpy(g_stage_in.p, input.data, input.num_elements);
    if (prof) CU(cudaEventRecord(g_ev[0]));
    CU(cudaMemcpy(g_act.p, g_stage_in.p, input.num_elements, cudaMemcpyHostToDevice));
    if (prof) CU(cudaEventRecord(g_ev[1]));

    // im2col
    const long long total = (long long)mp * plan.kp;
    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    im2col_fp32<<<blocks, threads>>>(
        (float*)g_a.p, total, plan.kp, (const signed char*)g_act.p, in_ch, in_h,
        in_w, out_w, kh, kw, stride, pad, plan.icp, input.scale,
        input.zero_point, m, k_real);
    CU(cudaGetLastError());
    if (prof) CU(cudaEventRecord(g_ev[2]));

    // GEMM
    gemm_launch(plan.tile, mp, plan.np, plan.kp, 1.0f, (const float*)g_a.p,
                plan.d_b, 0.0f, (float*)g_c.p, nullptr);
    CU(cudaGetLastError());
    if (prof) CU(cudaEventRecord(g_ev[3]));

    // D2H, useful sub-block only: C is [mp][np], we want [m][n].
    CU(cudaMemcpy2D(g_stage_out.p, (size_t)plan.n * sizeof(float), g_c.p,
                    (size_t)plan.np * sizeof(float), (size_t)plan.n * sizeof(float),
                    (size_t)m, cudaMemcpyDeviceToHost));
    if (prof) CU(cudaEventRecord(g_ev[4]));

    // Epilogue stays on the host. CUDA's expf is not glibc's, and a 1 ULP
    // difference in SiLU can flip lround at a requantization tie.
    const float* c = (const float*)g_stage_out.p;
    for (int oc = 0; oc < out_ch; oc++) {
        const float gain = layer.bn_gain[oc];
        const float bias = layer.bn_bias[oc];
        float* dst = out.data.data() + (size_t)oc * out_h * out_w;
        for (int pix = 0; pix < m; pix++) {
            float value = gain * c[(size_t)pix * plan.n + oc] + bias;
            if (apply_silu) value = silu_host(value);
            dst[pix] = value;
        }
    }

    if (prof) {
        CU(cudaEventRecord(g_ev[5]));
        CU(cudaEventSynchronize(g_ev[5]));
        g_times.h2d_ms += elapsed(g_ev[0], g_ev[1]);
        g_times.im2col_ms += elapsed(g_ev[1], g_ev[2]);
        g_times.gemm_ms += elapsed(g_ev[2], g_ev[3]);
        g_times.d2h_ms += elapsed(g_ev[3], g_ev[4]);
        g_times.epilogue_ms += elapsed(g_ev[4], g_ev[5]);
        g_times.launches += 2;
    }

    return out;
}
