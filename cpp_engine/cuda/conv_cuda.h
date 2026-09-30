#ifndef CONV_CUDA_H
#define CONV_CUDA_H

// Interface to the CUDA convolution backend. Deliberately free of CUDA types
// so ops.cpp can include it without cuda_runtime.h reaching the translation
// unit that carries the AVX-VNNI target pragma.

#include <string>

// Device capability
bool cuda_available();
std::string cuda_device_summary();

#endif  // CONV_CUDA_H
