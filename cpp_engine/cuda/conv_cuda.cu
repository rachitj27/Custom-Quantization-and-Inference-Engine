#include "conv_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemm.cuh"
#include "igemm.cuh"

namespace {

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA ") + what + ": " +
                                 cudaGetErrorString(err));
    }
}

#define CU(call) cuda_check((call), #call)

int round_up(int v, int m) { return ((v + m - 1) / m) * m; }

// high-water mark, never per layer
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

// per-layer device state. M comes from the input at call time.
struct LayerPlan {
    float* d_b_fp32 = nullptr;  // [kp][np], padding zeroed
    int* d_b_int8 = nullptr;    // [kp/4][np] int32 words, padding zeroed
    int n = 0;
    int np = 0;
    int kp = 0;
    int icp = 0;
    GemmTile tile_fp32 = GemmTile::N128;
    IgemmTile tile_int8 = IgemmTile::N128;
};

std::vector<LayerPlan> g_plans;
DeviceBuffer g_act;  // INT8 CHW activation
DeviceBuffer g_a;    // im2col matrix
DeviceBuffer g_c;    // GEMM output
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

// k = (r * kw + c) * icp + i, matching the VNNI weight packing
__device__ inline void decode_column(int k, int icp, int kw, int* i, int* r, int* c) {
    *i = k % icp;
    const int tap = k / icp;
    *c = tap % kw;
    *r = tap / kw;
}

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
            int i, r, c;
            decode_column(k, icp, kw, &i, &r, &c);
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

// holes fill with zero_point, not 0: the correction subtracts
// zero_point * sum(w) over every K position.
__global__ void im2col_int8(signed char* __restrict__ A, long long total, int kp,
                            const signed char* __restrict__ in, int in_ch,
                            int in_h, int in_w, int out_w, int kh, int kw,
                            int stride, int pad, int icp, int zero_point,
                            int m_real) {
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int m = (int)(idx / kp);
        const int k = (int)(idx - (long long)m * kp);

        signed char v = (signed char)zero_point;
        if (m < m_real) {
            int i, r, c;
            decode_column(k, icp, kw, &i, &r, &c);
            if (i < in_ch && r < kh) {
                const int oh = m / out_w;
                const int ow = m - oh * out_w;
                const int ih = oh * stride + r - pad;
                const int iw = ow * stride + c - pad;
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    v = in[((long long)i * in_h + ih) * in_w + iw];
                }
            }
        }
        A[idx] = v;
    }
}

float silu_host(float x) { return x / (1.0f + std::exp(-x)); }

struct ConvDims {
    int in_ch, in_h, in_w, out_ch, kh, kw, stride, pad, out_h, out_w, m;
};

ConvDims conv_dims(const Tensor& input, const Layer& layer) {
    ConvDims d;
    d.in_ch = input.shape[0];
    d.in_h = input.shape[1];
    d.in_w = input.shape[2];
    d.out_ch = layer.weight_shape[0];
    d.kh = layer.weight_shape[2];
    d.kw = layer.weight_shape[3];
    d.stride = layer.stride;
    d.pad = layer.padding;
    d.out_h = (d.in_h + 2 * d.pad - d.kh) / d.stride + 1;
    d.out_w = (d.in_w + 2 * d.pad - d.kw) / d.stride + 1;
    d.m = d.out_h * d.out_w;
    if (d.in_ch != layer.weight_shape[1]) {
        throw std::runtime_error("Channel mismatch in " + layer.path);
    }
    return d;
}

int im2col_blocks(long long total, int threads) {
    long long b = (total + threads - 1) / threads;
    if (b > 65535) b = 65535;
    return (int)b;
}

}  // namespace

void cuda_profile_enable(bool on) { g_profile = on; }
bool cuda_profile_enabled() { return g_profile; }
CudaPhaseTimes cuda_phase_times() { return g_times; }
void cuda_reset_phase_times() { g_times = CudaPhaseTimes(); }

void cuda_prepare_layers(Model& model, Kernel kernel) {
    const bool fp32 = (kernel == Kernel::CudaFp32);
    const bool int8 = (kernel == Kernel::CudaInt8);
    if (!fp32 && !int8) return;

    cuda_release();
    g_plans.reserve(model.conv_layers.size());

    for (Layer& layer : model.conv_layers) {
        if (layer.groups != 1) {
            throw std::runtime_error("Grouped convolution is not supported: " + layer.path);
        }
        if (fp32 && layer.weights_fp32.empty()) {
            layer.cuda_slot = -1;
            continue;
        }
        if (int8 && layer.weights_hwc.empty()) {
            layer.cuda_slot = -1;
            continue;
        }

        const int oc = layer.weight_shape[0];
        const int ic = layer.weight_shape[1];
        const int kh = layer.weight_shape[2];
        const int kw = layer.weight_shape[3];

        LayerPlan plan;
        plan.n = oc;
        plan.icp = int8 ? layer.ic_padded : round_up(ic, 16);

        if (fp32) {
            plan.tile_fp32 = gemm_choose_tile(oc);
            const GemmTileShape ts = gemm_tile_shape(plan.tile_fp32);
            plan.np = round_up(oc, ts.bn);
            plan.kp = round_up(kh * kw * plan.icp, ts.bk);

            // OIHW -> [kp][np], once
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
            CU(cudaMalloc(&plan.d_b_fp32, panel.size() * sizeof(float)));
            CU(cudaMemcpy(plan.d_b_fp32, panel.data(), panel.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
        } else {
            plan.tile_int8 = igemm_choose_tile(oc);
            const IgemmTileShape ts = igemm_tile_shape(plan.tile_int8);
            plan.np = round_up(oc, ts.bn);
            // no K rounding: kh*kw*icp is a multiple of 16
            plan.kp = kh * kw * plan.icp;

            // weights_hwc -> K-major, grouped in fours for dp4a
            const int k4 = plan.kp / 4;
            std::vector<signed char> bytes((size_t)k4 * plan.np * 4, 0);
            for (int o = 0; o < oc; o++) {
                for (int k = 0; k < plan.kp; k++) {
                    int i, r, c;
                    i = k % plan.icp;
                    const int tap = k / plan.icp;
                    c = tap % kw;
                    r = tap / kw;
                    if (i >= ic) continue;  // padded channel, weight stays 0
                    const size_t src =
                        (((size_t)o * kh + r) * kw + c) * plan.icp + i;
                    bytes[((size_t)(k / 4) * plan.np + o) * 4 + (k % 4)] =
                        layer.weights_hwc[src];
                }
            }
            CU(cudaMalloc(&plan.d_b_int8, bytes.size()));
            CU(cudaMemcpy(plan.d_b_int8, bytes.data(), bytes.size(),
                          cudaMemcpyHostToDevice));
        }

        layer.cuda_slot = (int)g_plans.size();
        g_plans.push_back(plan);
    }
}

void cuda_release() {
    for (LayerPlan& p : g_plans) {
        if (p.d_b_fp32) cudaFree(p.d_b_fp32);
        if (p.d_b_int8) cudaFree(p.d_b_int8);
        p.d_b_fp32 = nullptr;
        p.d_b_int8 = nullptr;
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
    const ConvDims d = conv_dims(input, layer);

    const int k_real = d.kh * d.kw * plan.icp;
    const GemmTileShape ts = gemm_tile_shape(plan.tile_fp32);
    const int mp = round_up(d.m, ts.bm);

    FloatTensor out({d.out_ch, d.out_h, d.out_w});

    g_act.ensure(input.num_elements);
    g_a.ensure((size_t)mp * plan.kp * sizeof(float));
    g_c.ensure((size_t)mp * plan.np * sizeof(float));
    g_stage_in.ensure(input.num_elements);
    g_stage_out.ensure((size_t)d.m * plan.n * sizeof(float));

    const bool prof = g_profile;
    if (prof) ensure_events();

    std::memcpy(g_stage_in.p, input.data, input.num_elements);
    if (prof) CU(cudaEventRecord(g_ev[0]));
    CU(cudaMemcpy(g_act.p, g_stage_in.p, input.num_elements, cudaMemcpyHostToDevice));
    if (prof) CU(cudaEventRecord(g_ev[1]));

    const long long total = (long long)mp * plan.kp;
    im2col_fp32<<<im2col_blocks(total, 256), 256>>>(
        (float*)g_a.p, total, plan.kp, (const signed char*)g_act.p, d.in_ch,
        d.in_h, d.in_w, d.out_w, d.kh, d.kw, d.stride, d.pad, plan.icp,
        input.scale, input.zero_point, d.m, k_real);
    CU(cudaGetLastError());
    if (prof) CU(cudaEventRecord(g_ev[2]));

    gemm_launch(plan.tile_fp32, mp, plan.np, plan.kp, 1.0f, (const float*)g_a.p,
                plan.d_b_fp32, 0.0f, (float*)g_c.p, nullptr);
    CU(cudaGetLastError());
    if (prof) CU(cudaEventRecord(g_ev[3]));

    // D2H the useful sub-block only
    CU(cudaMemcpy2D(g_stage_out.p, (size_t)plan.n * sizeof(float), g_c.p,
                    (size_t)plan.np * sizeof(float), (size_t)plan.n * sizeof(float),
                    (size_t)d.m, cudaMemcpyDeviceToHost));
    if (prof) CU(cudaEventRecord(g_ev[4]));

    // epilogue on the host: CUDA's expf is not glibc's, and 1 ULP in SiLU
    // can flip lround at a requantization tie.
    const float* c = (const float*)g_stage_out.p;
    for (int oc = 0; oc < d.out_ch; oc++) {
        const float gain = layer.bn_gain[oc];
        const float bias = layer.bn_bias[oc];
        float* dst = out.data.data() + (size_t)oc * d.m;
        for (int pix = 0; pix < d.m; pix++) {
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

FloatTensor conv_cuda_int8(const Tensor& input, const Layer& layer, bool apply_silu) {
    if (layer.cuda_slot < 0 || layer.cuda_slot >= (int)g_plans.size()) {
        throw std::runtime_error("No CUDA plan for layer " + layer.path);
    }
    const LayerPlan& plan = g_plans[(size_t)layer.cuda_slot];
    const ConvDims d = conv_dims(input, layer);

    const IgemmTileShape ts = igemm_tile_shape(plan.tile_int8);
    const int mp = round_up(d.m, ts.bm);
    const int k4 = plan.kp / 4;

    FloatTensor out({d.out_ch, d.out_h, d.out_w});

    g_act.ensure(input.num_elements);
    g_a.ensure((size_t)mp * plan.kp);
    g_c.ensure((size_t)mp * plan.np * sizeof(int32_t));
    g_stage_in.ensure(input.num_elements);
    g_stage_out.ensure((size_t)d.m * plan.n * sizeof(int32_t));

    const bool prof = g_profile;
    if (prof) ensure_events();

    std::memcpy(g_stage_in.p, input.data, input.num_elements);
    if (prof) CU(cudaEventRecord(g_ev[0]));
    CU(cudaMemcpy(g_act.p, g_stage_in.p, input.num_elements, cudaMemcpyHostToDevice));
    if (prof) CU(cudaEventRecord(g_ev[1]));

    const long long total = (long long)mp * plan.kp;
    im2col_int8<<<im2col_blocks(total, 256), 256>>>(
        (signed char*)g_a.p, total, plan.kp, (const signed char*)g_act.p, d.in_ch,
        d.in_h, d.in_w, d.out_w, d.kh, d.kw, d.stride, d.pad, plan.icp,
        input.zero_point, d.m);
    CU(cudaGetLastError());
    if (prof) CU(cudaEventRecord(g_ev[2]));

    igemm_launch(plan.tile_int8, mp, plan.np, k4, (const int*)g_a.p,
                 plan.d_b_int8, (int*)g_c.p, nullptr);
    CU(cudaGetLastError());
    if (prof) CU(cudaEventRecord(g_ev[3]));

    CU(cudaMemcpy2D(g_stage_out.p, (size_t)plan.n * sizeof(int32_t), g_c.p,
                    (size_t)plan.np * sizeof(int32_t),
                    (size_t)plan.n * sizeof(int32_t), (size_t)d.m,
                    cudaMemcpyDeviceToHost));
    if (prof) CU(cudaEventRecord(g_ev[4]));

    // same epilogue as conv_vnni_int8.
    // sum((q - z) * w) == sum(q * w) - z * sum(w)
    const int32_t* c = (const int32_t*)g_stage_out.p;
    for (int oc = 0; oc < d.out_ch; oc++) {
        const float m_scale =
            input.scale * layer.weight_scales[oc] * layer.bn_gain[oc];
        const float bias = layer.bn_bias[oc];
        const int32_t correction = input.zero_point * layer.weight_sums[oc];
        float* dst = out.data.data() + (size_t)oc * d.m;
        for (int pix = 0; pix < d.m; pix++) {
            const int32_t acc = c[(size_t)pix * plan.n + oc] - correction;
            float value = m_scale * (float)acc + bias;
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
