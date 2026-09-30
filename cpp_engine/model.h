#ifndef MODEL_H
#define MODEL_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tensor.h"

// One Conv2d: INT8 weights plus everything needed to turn an INT32 accumulator
// back into real numbers.
struct Layer {
    int layer_id = -1;
    std::string path;         // e.g. "model.2.cv1.conv"
    std::string module_path;  // enclosing Conv module, e.g. "model.2.cv1"

    std::vector<int> weight_shape;  // (out_ch, in_ch, kh, kw)
    size_t byte_offset = 0;
    size_t byte_length = 0;

    // One entry per output channel. A per-tensor export is expanded on load, so
    // the convolution reads weight_scales[oc] either way and costs nothing extra.
    std::vector<float> weight_scales;
    int weight_zero_point = 0;  // always 0: weights are symmetric

    int stride = 1;
    int padding = 0;
    int groups = 1;

    // BatchNorm folded into a per-output-channel affine transform, applied
    // during requantization. For bare Conv2d layers (the Detect head's 1x1
    // predictors) gain is 1 and bias is the layer's own bias.
    std::vector<float> bn_gain;
    std::vector<float> bn_bias;

    bool silu = false;  // whether a SiLU follows this conv

    // Calibrated output quantization. Absent for layers whose output the engine
    // keeps in FP32 (the Detect head predictors).
    bool has_out_scale = false;
    float out_scale = 1.0f;
    int out_zero_point = 0;

    std::unique_ptr<Tensor> weights;

  

  
    std::vector<float> weights_fp32;

  
    std::vector<int8_t> weights_hwc;

    
    std::vector<int32_t> weight_sums;

    int ic_padded = 0;

    int cuda_slot = -1;

    int out_channels() const { return weight_shape[0]; }
};

struct BottleneckSpec {
    int cv1 = -1;
    int cv2 = -1;
    bool shortcut = false;
};

// One of the 23 top-level modules in the YOLOv8n graph.
struct ArchLayer {
    int layer_id = -1;
    std::string type;

    bool input_is_raw = false;  // true only for layer 0, which reads the image
    std::vector<int> inputs;    // producing layer ids, in order

    // Conv
    int conv = -1;

    // C2f / SPPF
    int cv1 = -1;
    int cv2 = -1;
    std::vector<BottleneckSpec> bottlenecks;
    float concat_scale = 1.0f;  // C2f: common scale for the pre-cv2 concat
    int concat_zero_point = 0;
    int k = 5;  // SPPF pooling kernel

    // Concat
    float out_scale = 1.0f;
    int out_zero_point = 0;

    // Detect
    int nc = 0;
    int reg_max = 0;
    int no = 0;
    std::vector<float> strides;
    std::vector<std::vector<int>> det_cv2;  // [level][3] box branch
    std::vector<std::vector<int>> det_cv3;  // [level][3] class branch
    int dfl = -1;
};

struct Model {
    int num_bits = 8;
    float input_scale = 1.0f / 255.0f;
    int input_zero_point = -128;

    std::vector<Layer> conv_layers;
    std::vector<ArchLayer> architecture;
};

// A decoded detection, in pixel coordinates of the 640x640 network input.
struct Detection {
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float score = 0;
    int cls = 0;
};



//   ScalarInt8  the original loop, one INT8 multiply-accumulate at a time
//   ScalarFp32  the same loop in FP32, as an arithmetic-width baseline
//   VnniInt8    INT8 through AVX-VNNI, 32 multiply-accumulates per instruction
//   CudaFp32    im2col plus an FP32 GEMM on the GPU
//   CudaInt8    im2col plus an INT8 GEMM on the GPU, dp4a for the dot product
enum class Kernel { ScalarInt8, ScalarFp32, VnniInt8, CudaFp32, CudaInt8 };

const char* kernel_name(Kernel k);
bool parse_kernel(const std::string& name, Kernel& out);

// Whether this CPU actually has the AVX-VNNI instructions.
bool vnni_supported();

// Whether a usable CUDA device is present.
bool cuda_supported();

// Build whichever weight representation the kernel needs. Safe to call twice.
void prepare_kernel(Model& model, Kernel k);

Model load_model_metadata(const std::string& json_path);
void load_model_weights(Model& model, const std::string& bin_path);

#endif
