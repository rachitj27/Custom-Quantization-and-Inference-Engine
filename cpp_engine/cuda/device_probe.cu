#include "conv_cuda.h"

#include <cuda_runtime.h>

#include <sstream>

namespace {

// dp4a needs sm_61
constexpr int kMinComputeCapability = 61;

bool pick_device(cudaDeviceProp& prop) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return false;

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return false;

    return prop.major * 10 + prop.minor >= kMinComputeCapability;
}

}  // namespace

bool cuda_available() {
    cudaDeviceProp prop{};
    return pick_device(prop);
}

std::string cuda_device_summary() {
    cudaDeviceProp prop{};
    if (!pick_device(prop)) return "no usable CUDA device";

    std::ostringstream out;
    out << prop.name
        << ", sm_" << prop.major << prop.minor
        << ", " << prop.multiProcessorCount << " SMs"
        << ", " << (prop.totalGlobalMem >> 20) << " MB";
    return out.str();
}
