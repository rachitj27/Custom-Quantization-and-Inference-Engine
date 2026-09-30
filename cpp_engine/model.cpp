#include "model.h"

#if ENGINE_HAS_CUDA
#include "cuda/conv_cuda.h"
#endif

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

Model load_model_metadata(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open " + json_path);
    }

    json metadata;
    file >> metadata;

    Model model;
    model.num_bits = metadata["num_bits"];

    if (metadata.contains("input")) {
        model.input_scale = metadata["input"]["scale"];
        model.input_zero_point = metadata["input"]["zero_point"];
    }

    for (const auto& j : metadata["conv_layers"]) {
        Layer layer;
        layer.layer_id = j["layer_id"];
        layer.path = j["path"];
        layer.module_path = j.value("module_path", layer.path);
        layer.weight_shape = j["weight_shape"].get<std::vector<int>>();
        layer.byte_offset = j["byte_offset"];
        layer.byte_length = j["byte_length"];
        const int out_ch = layer.weight_shape[0];
        if (j["weight_scale"].is_array()) {
            layer.weight_scales = j["weight_scale"].get<std::vector<float>>();
            if (static_cast<int>(layer.weight_scales.size()) != out_ch) {
                throw std::runtime_error("Per-channel weight scale count mismatch for " +
                                         layer.path);
            }
        } else {
            layer.weight_scales.assign(out_ch, j["weight_scale"].get<float>());
        }
        layer.weight_zero_point = j["weight_zero_point"];
        layer.stride = j.value("stride", 1);
        layer.padding = j.value("padding", 0);
        layer.groups = j.value("groups", 1);
        layer.bn_gain = j["bn_gain"].get<std::vector<float>>();
        layer.bn_bias = j["bn_bias"].get<std::vector<float>>();
        layer.silu = (j.value("act", std::string("none")) == "silu");

        if (j.contains("out_scale") && !j["out_scale"].is_null()) {
            layer.has_out_scale = true;
            layer.out_scale = j["out_scale"];
            layer.out_zero_point = j["out_zero_point"];
        }

        if (static_cast<int>(layer.bn_gain.size()) != layer.out_channels() ||
            static_cast<int>(layer.bn_bias.size()) != layer.out_channels()) {
            throw std::runtime_error("BN parameter count does not match out_channels for " +
                                     layer.path);
        }

        model.conv_layers.push_back(std::move(layer));
    }

    for (const auto& j : metadata["architecture"]) {
        ArchLayer a;
        a.layer_id = j["layer_id"];
        a.type = j["type"];

        const auto& from = j["input_from"];
        if (from.is_string()) {
            a.input_is_raw = true;
        } else if (from.is_array()) {
            a.inputs = from.get<std::vector<int>>();
        } else {
            a.inputs.push_back(from.get<int>());
        }

        if (a.type == "Conv") {
            a.conv = j["conv"];
        } else if (a.type == "C2f") {
            a.cv1 = j["cv1"];
            a.cv2 = j["cv2"];
            a.concat_scale = j["concat_scale"];
            a.concat_zero_point = j["concat_zero_point"];
            for (const auto& b : j["bottlenecks"]) {
                BottleneckSpec spec;
                spec.cv1 = b["cv1"];
                spec.cv2 = b["cv2"];
                spec.shortcut = b["shortcut"];
                a.bottlenecks.push_back(spec);
            }
        } else if (a.type == "SPPF") {
            a.cv1 = j["cv1"];
            a.cv2 = j["cv2"];
            a.k = j["k"];
        } else if (a.type == "Concat") {
            a.out_scale = j["out_scale"];
            a.out_zero_point = j["out_zero_point"];
        } else if (a.type == "Detect") {
            a.nc = j["nc"];
            a.reg_max = j["reg_max"];
            a.no = j["no"];
            a.strides = j["strides"].get<std::vector<float>>();
            a.det_cv2 = j["cv2"].get<std::vector<std::vector<int>>>();
            a.det_cv3 = j["cv3"].get<std::vector<std::vector<int>>>();
            a.dfl = j["dfl"];
        }

        model.architecture.push_back(std::move(a));
    }

    return model;
}

void load_model_weights(Model& model, const std::string& bin_path) {
    std::ifstream file(bin_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open " + bin_path);
    }

    for (auto& layer : model.conv_layers) {
        layer.weights = std::make_unique<Tensor>(layer.weight_shape, layer.weight_scales[0], 0);
        file.seekg(static_cast<std::streamoff>(layer.byte_offset));
        layer.weights->load_from_stream(file, layer.byte_length);
    }

    std::cout << "Loaded weights for " << model.conv_layers.size() << " conv layers"
              << std::endl;
}

// ---------------------------------------------------------------------------
// Kernel selection and the weight layouts each kernel needs
// ---------------------------------------------------------------------------

namespace {

// VPDPBUSD reads 32 bytes per operand in its 256-bit form and 16 in its
// 128-bit form, so input channels are padded up to a multiple of 16 and the
// dot product uses whichever width fits. Every channel count in YOLOv8n is
// already a multiple of 16 except layer 0, which has 3, so in practice this
// padding costs nothing.
constexpr int kChannelAlign = 16;

// Below this many input channels the padding would cost more than the
// vectorisation saves, so those layers stay on the scalar kernel. Only layer 0
// falls in this bucket.
constexpr int kMinVnniChannels = 16;

inline int round_up(int v, int m) { return ((v + m - 1) / m) * m; }

}  // namespace

const char* kernel_name(Kernel k) {
    switch (k) {
        case Kernel::ScalarInt8: return "scalar-int8";
        case Kernel::ScalarFp32: return "scalar-fp32";
        case Kernel::VnniInt8:   return "vnni-int8";
        case Kernel::CudaFp32:   return "cuda-fp32";
    }
    return "unknown";
}

bool parse_kernel(const std::string& name, Kernel& out) {
    if (name == "scalar-int8" || name == "int8") { out = Kernel::ScalarInt8; return true; }
    if (name == "scalar-fp32" || name == "fp32") { out = Kernel::ScalarFp32; return true; }
    if (name == "vnni-int8"   || name == "vnni") { out = Kernel::VnniInt8;   return true; }
    if (name == "cuda-fp32"   || name == "cuda32") { out = Kernel::CudaFp32; return true; }
    return false;
}

bool cuda_supported() {
#if ENGINE_HAS_CUDA
    return cuda_available();
#else
    return false;
#endif
}

bool vnni_supported() {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avxvnni") != 0;
#else
    return false;
#endif
}

void prepare_kernel(Model& model, Kernel k) {
    if (k == Kernel::ScalarInt8) return;  // runs straight off the loaded blob

    size_t vnni_layers = 0;

    for (Layer& layer : model.conv_layers) {
        const int oc = layer.weight_shape[0];
        const int ic = layer.weight_shape[1];
        const int kh = layer.weight_shape[2];
        const int kw = layer.weight_shape[3];
        const int8_t* w = layer.weights->data;
        const size_t per_oc = static_cast<size_t>(ic) * kh * kw;

        if (k == Kernel::ScalarFp32 || k == Kernel::CudaFp32) {
            if (!layer.weights_fp32.empty()) continue;
            layer.weights_fp32.resize(static_cast<size_t>(oc) * per_oc);
            // Dequantize with the same per-channel scale the INT8 path folds
            // into its requant multiplier, so both kernels compute the same
            // numbers and any timing difference is arithmetic width alone.
            for (int o = 0; o < oc; o++) {
                const float s = layer.weight_scales[o];
                const size_t base = static_cast<size_t>(o) * per_oc;
                for (size_t i = 0; i < per_oc; i++) {
                    layer.weights_fp32[base + i] = s * static_cast<float>(w[base + i]);
                }
            }
            continue;
        }

        // VnniInt8: repack [oc][ic][kh][kw] -> [oc][kh][kw][ic_padded].
        if (!layer.weights_hwc.empty()) continue;
        if (ic < kMinVnniChannels) continue;  // left on the scalar kernel

        const int icp = round_up(ic, kChannelAlign);
        layer.ic_padded = icp;
        layer.weights_hwc.assign(static_cast<size_t>(oc) * kh * kw * icp, 0);
        layer.weight_sums.assign(static_cast<size_t>(oc), 0);

        for (int o = 0; o < oc; o++) {
            int32_t sum = 0;
            for (int r = 0; r < kh; r++) {
                for (int c = 0; c < kw; c++) {
                    const size_t dst = ((static_cast<size_t>(o) * kh + r) * kw + c) * icp;
                    for (int i = 0; i < ic; i++) {
                        const int8_t v =
                            w[((static_cast<size_t>(o) * ic + i) * kh + r) * kw + c];
                        layer.weights_hwc[dst + i] = v;
                        sum += v;
                    }
                }
            }
            // Padded channels hold weight 0 and so add nothing here, which is
            // what makes the padding invisible to the zero-point correction.
            layer.weight_sums[o] = sum;
        }
        vnni_layers++;
    }

#if ENGINE_HAS_CUDA
    if (k == Kernel::CudaFp32) {
        std::cout << "CUDA device: " << cuda_device_summary() << std::endl;
        cuda_prepare_layers(model, k);
    }
#endif

    if (k == Kernel::VnniInt8) {
        std::cout << "Repacked " << vnni_layers << " of " << model.conv_layers.size()
                  << " conv layers for AVX-VNNI"
                  << " (the rest have too few channels and stay scalar)" << std::endl;
    }
}
