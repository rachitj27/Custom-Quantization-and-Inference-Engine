#ifndef CONV_CUDA_H
#define CONV_CUDA_H

// Interface to the CUDA convolution backend. Deliberately free of CUDA types
// so ops.cpp and main.cpp can include it without cuda_runtime.h reaching the
// translation unit that carries the AVX-VNNI target pragma.

#include <string>

#include "model.h"

// Device capability
bool cuda_available();
std::string cuda_device_summary();

// Uploads each layer's weight panel and picks its tile. Safe to call twice;
// the second call replaces the first.
void cuda_prepare_layers(Model& model, Kernel kernel);
void cuda_release();

// Convolution. Returns the same FP32 tensor the CPU kernels return, with the
// BatchNorm affine and optional SiLU already applied, so conv2d_quant needs no
// changes.
FloatTensor conv_cuda_fp32(const Tensor& input, const Layer& layer, bool apply_silu);

// Phase timing. Off by default: it synchronizes per phase per layer, which
// serializes the pipeline and inflates the end-to-end number.
struct CudaPhaseTimes {
    double h2d_ms = 0.0;
    double im2col_ms = 0.0;
    double gemm_ms = 0.0;
    double d2h_ms = 0.0;
    double epilogue_ms = 0.0;
    long long launches = 0;
};

void cuda_profile_enable(bool on);
bool cuda_profile_enabled();
CudaPhaseTimes cuda_phase_times();
void cuda_reset_phase_times();

#endif  // CONV_CUDA_H
