#ifndef CONV_CUDA_H
#define CONV_CUDA_H

// CUDA convolution backend. Free of CUDA types so ops.cpp can include it
// without cuda_runtime.h reaching the AVX-VNNI target pragma.

#include <string>

#include "model.h"

// Device capability
bool cuda_available();
std::string cuda_device_summary();

// uploads weight panels, picks tiles. safe to call twice.
void cuda_prepare_layers(Model& model, Kernel kernel);
void cuda_release();

// returns the same FP32 tensor the CPU kernels do, bn affine and SiLU applied
FloatTensor conv_cuda_fp32(const Tensor& input, const Layer& layer, bool apply_silu);
FloatTensor conv_cuda_int8(const Tensor& input, const Layer& layer, bool apply_silu);

// fused: epilogue and requantize on the device, returns INT8 directly.
// not byte-exact against the host path, see epilogue_int8.
std::unique_ptr<Tensor> conv_cuda_int8_fused(const Tensor& input, const Layer& layer,
                                             float out_scale, int out_zp,
                                             bool apply_silu, const Tensor* residual);

// phase timing. off by default: syncing per phase serializes the pipeline.
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
