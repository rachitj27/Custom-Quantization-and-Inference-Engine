#include "ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>


#if ENGINE_HAS_CUDA
#include "cuda/conv_cuda.h"
#endif

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define ENGINE_HAS_VNNI 1
#else
#define ENGINE_HAS_VNNI 0
#endif

namespace {

inline int8_t quantize_one(float real, float scale, int zero_point) {
    int32_t q = static_cast<int32_t>(std::lround(real / scale)) + zero_point;
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    return static_cast<int8_t>(q);
}

inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

// Which kernel conv2d_real dispatches to. Set once at startup.
Kernel g_kernel = Kernel::ScalarInt8;


FloatTensor conv_scalar_int8(const Tensor& input, const Layer& layer, bool apply_silu) {
    const int in_ch = input.shape[0];
    const int in_h = input.shape[1];
    const int in_w = input.shape[2];

    const int out_ch = layer.weight_shape[0];
    const int w_in_ch = layer.weight_shape[1];
    const int kh = layer.weight_shape[2];
    const int kw = layer.weight_shape[3];

    if (layer.groups != 1) {
        throw std::runtime_error("Grouped convolution is not supported: " + layer.path);
    }
    if (in_ch != w_in_ch) {
        throw std::runtime_error("Channel mismatch at " + layer.path + ": input has " +
                                 std::to_string(in_ch) + ", weights expect " +
                                 std::to_string(w_in_ch));
    }

    const int stride = layer.stride;
    const int pad = layer.padding;
    const int out_h = (in_h + 2 * pad - kh) / stride + 1;
    const int out_w = (in_w + 2 * pad - kw) / stride + 1;

    FloatTensor out({out_ch, out_h, out_w});

    const int z_in = input.zero_point;
    const int8_t* w = layer.weights->data;

    for (int oc = 0; oc < out_ch; oc++) {
        // Dequantization multiplier folded together with the BatchNorm gain, so
        // per-channel weight scales cost nothing beyond this one multiply.
        const float m = input.scale * layer.weight_scales[oc] * layer.bn_gain[oc];
        const float bias = layer.bn_bias[oc];
        const int8_t* w_oc = w + static_cast<size_t>(oc) * in_ch * kh * kw;

        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int32_t acc = 0;

                for (int ic = 0; ic < in_ch; ic++) {
                    const int8_t* w_ic = w_oc + static_cast<size_t>(ic) * kh * kw;
                    const int8_t* in_c = input.data + static_cast<size_t>(ic) * in_h * in_w;

                    for (int r = 0; r < kh; r++) {
                        const int ih = oh * stride + r - pad;
                        if (ih < 0 || ih >= in_h) continue;  // padded: contributes 0

                        for (int c = 0; c < kw; c++) {
                            const int iw = ow * stride + c - pad;
                            if (iw < 0 || iw >= in_w) continue;

                            const int q_in = static_cast<int>(in_c[ih * in_w + iw]);
                            const int q_w = static_cast<int>(w_ic[r * kw + c]);
                            acc += static_cast<int32_t>(q_in - z_in) * static_cast<int32_t>(q_w);
                        }
                    }
                }

                float value = m * static_cast<float>(acc) + bias;
                if (apply_silu) value = silu(value);
                out.data[(static_cast<size_t>(oc) * out_h + oh) * out_w + ow] = value;
            }
        }
    }

    return out;
}


FloatTensor conv_scalar_fp32(const Tensor& input, const Layer& layer, bool apply_silu) {
    const int in_ch = input.shape[0];
    const int in_h = input.shape[1];
    const int in_w = input.shape[2];

    const int out_ch = layer.weight_shape[0];
    const int kh = layer.weight_shape[2];
    const int kw = layer.weight_shape[3];

    const int stride = layer.stride;
    const int pad = layer.padding;
    const int out_h = (in_h + 2 * pad - kh) / stride + 1;
    const int out_w = (in_w + 2 * pad - kw) / stride + 1;

    FloatTensor out({out_ch, out_h, out_w});

    // Dequantize the activations once per convolution. This is O(elements),
    // negligible beside the O(elements * out_ch * kh * kw) convolution itself.
    static std::vector<float> in_f;
    if (in_f.size() < input.num_elements) in_f.resize(input.num_elements);
    for (size_t i = 0; i < input.num_elements; i++) in_f[i] = input.real_at(i);

    const float* w_all = layer.weights_fp32.data();

    for (int oc = 0; oc < out_ch; oc++) {
        // The weights already carry their scale, so only the BatchNorm gain is
        // left to apply.
        const float gain = layer.bn_gain[oc];
        const float bias = layer.bn_bias[oc];
        const float* w_oc = w_all + static_cast<size_t>(oc) * in_ch * kh * kw;

        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float acc = 0.0f;

                for (int ic = 0; ic < in_ch; ic++) {
                    const float* w_ic = w_oc + static_cast<size_t>(ic) * kh * kw;
                    const float* in_c = in_f.data() + static_cast<size_t>(ic) * in_h * in_w;

                    for (int r = 0; r < kh; r++) {
                        const int ih = oh * stride + r - pad;
                        if (ih < 0 || ih >= in_h) continue;

                        for (int c = 0; c < kw; c++) {
                            const int iw = ow * stride + c - pad;
                            if (iw < 0 || iw >= in_w) continue;

                            acc += in_c[ih * in_w + iw] * w_ic[r * kw + c];
                        }
                    }
                }

                float value = gain * acc + bias;
                if (apply_silu) value = silu(value);
                out.data[(static_cast<size_t>(oc) * out_h + oh) * out_w + ow] = value;
            }
        }
    }

    return out;
}

#if ENGINE_HAS_VNNI


#pragma GCC push_options
#pragma GCC target("avx2,avxvnni")

inline int32_t hsum(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

inline int32_t hsum(__m128i s) {
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}


FloatTensor conv_vnni_int8(const Tensor& input, const Layer& layer, bool apply_silu) {
    const int in_ch = input.shape[0];
    const int in_h = input.shape[1];
    const int in_w = input.shape[2];

    const int out_ch = layer.weight_shape[0];
    const int kh = layer.weight_shape[2];
    const int kw = layer.weight_shape[3];
    const int icp = layer.ic_padded;

    const int stride = layer.stride;
    const int pad = layer.padding;
    const int out_h = (in_h + 2 * pad - kh) / stride + 1;
    const int out_w = (in_w + 2 * pad - kw) / stride + 1;

    // Scratch dimensions include the spatial padding, so the inner loop never
    // has to test whether a tap is in bounds.
    const int sh = in_h + 2 * pad;
    const int sw = in_w + 2 * pad;

    // A stored q represents scale * (q - z). Shifting to a = q + 128 gives an
    // unsigned byte standing for the same number against a zero point of z + 128.
    const int32_t z_shifted = input.zero_point + 128;
    const uint8_t z_fill = static_cast<uint8_t>(z_shifted);

    static std::vector<uint8_t> scratch;
    const size_t need = static_cast<size_t>(sh) * sw * icp;
    if (scratch.size() < need) scratch.resize(need);
    std::fill(scratch.begin(), scratch.begin() + need, z_fill);

    // CHW -> padded HWC. Writes run contiguously down the channel axis, which
    // is the side that matters, since that is what the dot product streams.
    const size_t plane = static_cast<size_t>(in_h) * in_w;
    for (int y = 0; y < in_h; y++) {
        for (int x = 0; x < in_w; x++) {
            uint8_t* dst = scratch.data() +
                           (static_cast<size_t>(y + pad) * sw + (x + pad)) * icp;
            const int8_t* src = input.data + static_cast<size_t>(y) * in_w + x;
            for (int c = 0; c < in_ch; c++) {
                dst[c] = static_cast<uint8_t>(static_cast<int>(src[c * plane]) + 128);
            }
        }
    }

    FloatTensor out({out_ch, out_h, out_w});

    // One kernel row is kw taps of icp channels, and consecutive taps are
    // adjacent in the scratch buffer, so a whole row is one contiguous run.
    const int row_bytes = kw * icp;

    for (int oc = 0; oc < out_ch; oc++) {
        const float m = input.scale * layer.weight_scales[oc] * layer.bn_gain[oc];
        const float bias = layer.bn_bias[oc];
        // sum((a - z) * w) == sum(a * w) - z * sum(w), and sum(w) is a constant
        // computed once at load time.
        const int32_t correction = z_shifted * layer.weight_sums[oc];
        const int8_t* w_oc = layer.weights_hwc.data() +
                             static_cast<size_t>(oc) * kh * row_bytes;

        for (int oh = 0; oh < out_h; oh++) {
            const int ih0 = oh * stride;

            for (int ow = 0; ow < out_w; ow++) {
                const int iw0 = ow * stride;

                __m256i acc8 = _mm256_setzero_si256();
                __m128i acc4 = _mm_setzero_si128();

                for (int r = 0; r < kh; r++) {
                    const uint8_t* a = scratch.data() +
                                       (static_cast<size_t>(ih0 + r) * sw + iw0) * icp;
                    const int8_t* w = w_oc + static_cast<size_t>(r) * row_bytes;

                    int i = 0;
                    for (; i + 32 <= row_bytes; i += 32) {
                        acc8 = _mm256_dpbusd_avx_epi32(
                            acc8,
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i)),
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i)));
                    }
                    // Channel counts are multiples of 16, so at most one
                    // 16-byte step is ever left over. The 128-bit form takes it.
                    for (; i + 16 <= row_bytes; i += 16) {
                        acc4 = _mm_dpbusd_avx_epi32(
                            acc4,
                            _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)),
                            _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i)));
                    }
                }

                const int32_t acc = hsum(acc8) + hsum(acc4) - correction;

                float value = m * static_cast<float>(acc) + bias;
                if (apply_silu) value = silu(value);
                out.data[(static_cast<size_t>(oc) * out_h + oh) * out_w + ow] = value;
            }
        }
    }

    return out;
}

#pragma GCC pop_options

#endif  // ENGINE_HAS_VNNI

// Dispatch to the selected kernel. Layers the vector path cannot serve, which
// is only layer 0 with its 3 input channels, fall back to the scalar one.
FloatTensor conv2d_real(const Tensor& input, const Layer& layer, bool apply_silu) {
#if ENGINE_HAS_CUDA
    if (g_kernel == Kernel::CudaFp32 && layer.cuda_slot >= 0) {
        return conv_cuda_fp32(input, layer, apply_silu);
    }
#endif
    if ((g_kernel == Kernel::ScalarFp32 || g_kernel == Kernel::CudaFp32) &&
        !layer.weights_fp32.empty()) {
        return conv_scalar_fp32(input, layer, apply_silu);
    }
#if ENGINE_HAS_VNNI
    if (g_kernel == Kernel::VnniInt8 && !layer.weights_hwc.empty()) {
        return conv_vnni_int8(input, layer, apply_silu);
    }
#endif
    return conv_scalar_int8(input, layer, apply_silu);
}

float iou(const Detection& a, const Detection& b) {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);

    const float iw = std::max(0.0f, ix2 - ix1);
    const float ih = std::max(0.0f, iy2 - iy1);
    const float inter = iw * ih;
    if (inter <= 0.0f) return 0.0f;

    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

}  // namespace

std::unique_ptr<Tensor> conv2d_quant(const Tensor& input,
                                     const Layer& layer,
                                     float out_scale,
                                     int out_zp,
                                     bool apply_silu,
                                     const Tensor* residual) {
    FloatTensor real = conv2d_real(input, layer, apply_silu);

    auto out = std::make_unique<Tensor>(real.shape, out_scale, out_zp);

    if (residual != nullptr) {
        if (residual->num_elements != out->num_elements) {
            throw std::runtime_error("Residual shape mismatch at " + layer.path);
        }
        // Add in the real domain so the two tensors' scales need not agree,
        // then quantize once.
        for (size_t i = 0; i < out->num_elements; i++) {
            out->data[i] = quantize_one(real.data[i] + residual->real_at(i), out_scale, out_zp);
        }
    } else {
        for (size_t i = 0; i < out->num_elements; i++) {
            out->data[i] = quantize_one(real.data[i], out_scale, out_zp);
        }
    }

    return out;
}

FloatTensor conv2d_float(const Tensor& input, const Layer& layer, bool apply_silu) {
    return conv2d_real(input, layer, apply_silu);
}

std::unique_ptr<Tensor> requantize(const Tensor& input, float scale, int zero_point) {
    auto out = std::make_unique<Tensor>(input.shape, scale, zero_point);

    if (input.scale == scale && input.zero_point == zero_point) {
        std::memcpy(out->data, input.data, input.num_elements);
        return out;
    }

    for (size_t i = 0; i < input.num_elements; i++) {
        out->data[i] = quantize_one(input.real_at(i), scale, zero_point);
    }
    return out;
}

std::unique_ptr<Tensor> concat(const std::vector<const Tensor*>& tensors,
                               float scale,
                               int zero_point) {
    if (tensors.empty()) {
        throw std::runtime_error("Concat needs at least one tensor");
    }

    const int height = tensors[0]->shape[1];
    const int width = tensors[0]->shape[2];
    int total_channels = 0;
    for (const Tensor* t : tensors) {
        if (t->shape[1] != height || t->shape[2] != width) {
            throw std::runtime_error("Concat tensors must have matching spatial dims");
        }
        total_channels += t->shape[0];
    }

    auto out = std::make_unique<Tensor>(std::vector<int>{total_channels, height, width},
                                        scale, zero_point);

    const size_t plane = static_cast<size_t>(height) * width;
    size_t offset = 0;
    for (const Tensor* t : tensors) {
        const size_t n = static_cast<size_t>(t->shape[0]) * plane;
        if (t->scale == scale && t->zero_point == zero_point) {
            std::memcpy(out->data + offset, t->data, n);
        } else {
            for (size_t i = 0; i < n; i++) {
                out->data[offset + i] = quantize_one(t->real_at(i), scale, zero_point);
            }
        }
        offset += n;
    }

    return out;
}

std::unique_ptr<Tensor> upsample2x(const Tensor& input) {
    const int channels = input.shape[0];
    const int in_h = input.shape[1];
    const int in_w = input.shape[2];
    const int out_h = in_h * 2;
    const int out_w = in_w * 2;

    // Nearest-neighbour copies values verbatim, so the quantization parameters
    // carry through unchanged.
    auto out = std::make_unique<Tensor>(std::vector<int>{channels, out_h, out_w},
                                        input.scale, input.zero_point);

    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            const int ih = oh / 2;
            for (int ow = 0; ow < out_w; ow++) {
                const int iw = ow / 2;
                out->data[(static_cast<size_t>(c) * out_h + oh) * out_w + ow] =
                    input.data[(static_cast<size_t>(c) * in_h + ih) * in_w + iw];
            }
        }
    }

    return out;
}

std::unique_ptr<Tensor> maxpool2d(const Tensor& input, int kernel_size) {
    const int channels = input.shape[0];
    const int height = input.shape[1];
    const int width = input.shape[2];
    const int pad = kernel_size / 2;

    // Max is monotonic, so the output lives on the input's scale.
    auto out = std::make_unique<Tensor>(std::vector<int>{channels, height, width},
                                        input.scale, input.zero_point);

    for (int c = 0; c < channels; c++) {
        const int8_t* in_c = input.data + static_cast<size_t>(c) * height * width;
        int8_t* out_c = out->data + static_cast<size_t>(c) * height * width;

        for (int oh = 0; oh < height; oh++) {
            for (int ow = 0; ow < width; ow++) {
                // PyTorch pads MaxPool with -inf, so out-of-bounds positions are
                // skipped rather than treated as zero.
                int8_t best = -128;
                bool seen = false;

                for (int r = 0; r < kernel_size; r++) {
                    const int ih = oh + r - pad;
                    if (ih < 0 || ih >= height) continue;
                    for (int c2 = 0; c2 < kernel_size; c2++) {
                        const int iw = ow + c2 - pad;
                        if (iw < 0 || iw >= width) continue;
                        const int8_t v = in_c[ih * width + iw];
                        if (!seen || v > best) {
                            best = v;
                            seen = true;
                        }
                    }
                }
                out_c[oh * width + ow] = best;
            }
        }
    }

    return out;
}

std::unique_ptr<Tensor> c2f_block(const Tensor& input, const Model& model,
                                  const ArchLayer& arch) {
    const Layer& cv1 = model.conv_layers[arch.cv1];
    const Layer& cv2 = model.conv_layers[arch.cv2];

    // cv1 produces 2c channels which are split in half.
    auto y = conv2d_quant(input, cv1, cv1.out_scale, cv1.out_zero_point, cv1.silu);

    const int channels = y->shape[0];
    const int half = channels / 2;
    const int height = y->shape[1];
    const int width = y->shape[2];
    const size_t half_bytes = static_cast<size_t>(half) * height * width;

    // Every branch is put on one common scale -- the calibrated range of the
    // tensor that actually feeds cv2 -- before being concatenated.
    const float cs = arch.concat_scale;
    const int czp = arch.concat_zero_point;

    Tensor a({half, height, width}, y->scale, y->zero_point);
    Tensor b({half, height, width}, y->scale, y->zero_point);
    std::memcpy(a.data, y->data, half_bytes);
    std::memcpy(b.data, y->data + half_bytes, half_bytes);

    auto a_rq = requantize(a, cs, czp);
    auto b_rq = requantize(b, cs, czp);

    std::vector<std::unique_ptr<Tensor>> owned;
    std::vector<const Tensor*> parts{a_rq.get(), b_rq.get()};

    const Tensor* current = b_rq.get();
    for (const BottleneckSpec& spec : arch.bottlenecks) {
        const Layer& bcv1 = model.conv_layers[spec.cv1];
        const Layer& bcv2 = model.conv_layers[spec.cv2];

        auto hidden = conv2d_quant(*current, bcv1, bcv1.out_scale, bcv1.out_zero_point,
                                   bcv1.silu);
        // YOLOv8's Bottleneck is x + cv2(cv1(x)), so the skip is added after
        // cv2's activation. Quantize straight onto the concat scale: the
        // calibration for that tensor already covers these outputs.
        auto next = conv2d_quant(*hidden, bcv2, cs, czp, bcv2.silu,
                                 spec.shortcut ? current : nullptr);

        current = next.get();
        parts.push_back(current);
        owned.push_back(std::move(next));
    }

    auto merged = concat(parts, cs, czp);
    return conv2d_quant(*merged, cv2, cv2.out_scale, cv2.out_zero_point, cv2.silu);
}

std::unique_ptr<Tensor> sppf_block(const Tensor& input, const Model& model,
                                   const ArchLayer& arch) {
    const Layer& cv1 = model.conv_layers[arch.cv1];
    const Layer& cv2 = model.conv_layers[arch.cv2];

    auto x = conv2d_quant(input, cv1, cv1.out_scale, cv1.out_zero_point, cv1.silu);

    auto y1 = maxpool2d(*x, arch.k);
    auto y2 = maxpool2d(*y1, arch.k);
    auto y3 = maxpool2d(*y2, arch.k);

    // MaxPool preserves the scale, so all four branches already agree and the
    // concat is exact.
    std::vector<const Tensor*> parts{x.get(), y1.get(), y2.get(), y3.get()};
    auto merged = concat(parts, x->scale, x->zero_point);

    return conv2d_quant(*merged, cv2, cv2.out_scale, cv2.out_zero_point, cv2.silu);
}

std::vector<std::unique_ptr<Tensor>> run_backbone(const Model& model, const Tensor& input) {
    std::vector<std::unique_ptr<Tensor>> outputs(model.architecture.size());

    for (size_t i = 0; i < model.architecture.size(); i++) {
        const ArchLayer& arch = model.architecture[i];

        if (arch.type == "Detect") continue;  // handled by detect_head()

        const Tensor* in_tensor = nullptr;
        if (arch.input_is_raw) {
            in_tensor = &input;
        } else if (!arch.inputs.empty()) {
            in_tensor = outputs[arch.inputs[0]].get();
        }

        std::unique_ptr<Tensor> result;

        if (arch.type == "Conv") {
            const Layer& conv = model.conv_layers[arch.conv];
            result = conv2d_quant(*in_tensor, conv, conv.out_scale, conv.out_zero_point,
                                  conv.silu);
        } else if (arch.type == "C2f") {
            result = c2f_block(*in_tensor, model, arch);
        } else if (arch.type == "SPPF") {
            result = sppf_block(*in_tensor, model, arch);
        } else if (arch.type == "Upsample") {
            result = upsample2x(*in_tensor);
        } else if (arch.type == "Concat") {
            std::vector<const Tensor*> parts;
            for (int idx : arch.inputs) {
                parts.push_back(outputs[idx].get());
            }
            result = concat(parts, arch.out_scale, arch.out_zero_point);
        } else {
            throw std::runtime_error("Unhandled layer type: " + arch.type);
        }

        outputs[i] = std::move(result);
    }

    return outputs;
}

std::vector<Detection> detect_head(const Model& model,
                                   const std::vector<std::unique_ptr<Tensor>>& features,
                                   float conf_threshold,
                                   float iou_threshold) {
    const ArchLayer* det = nullptr;
    for (const ArchLayer& a : model.architecture) {
        if (a.type == "Detect") det = &a;
    }
    if (det == nullptr) throw std::runtime_error("No Detect layer in the model");

    const int reg_max = det->reg_max;
    const int nc = det->nc;

    std::vector<Detection> candidates;

    for (size_t level = 0; level < det->inputs.size(); level++) {
        const Tensor& feat = *features[det->inputs[level]];
        const float stride = det->strides[level];

        // Box branch: two 3x3 Convs (INT8) then a 1x1 predictor kept in FP32.
        const Layer& b0 = model.conv_layers[det->det_cv2[level][0]];
        const Layer& b1 = model.conv_layers[det->det_cv2[level][1]];
        const Layer& b2 = model.conv_layers[det->det_cv2[level][2]];
        auto bh0 = conv2d_quant(feat, b0, b0.out_scale, b0.out_zero_point, b0.silu);
        auto bh1 = conv2d_quant(*bh0, b1, b1.out_scale, b1.out_zero_point, b1.silu);
        FloatTensor box = conv2d_float(*bh1, b2, false);

        // Class branch.
        const Layer& c0 = model.conv_layers[det->det_cv3[level][0]];
        const Layer& c1 = model.conv_layers[det->det_cv3[level][1]];
        const Layer& c2 = model.conv_layers[det->det_cv3[level][2]];
        auto ch0 = conv2d_quant(feat, c0, c0.out_scale, c0.out_zero_point, c0.silu);
        auto ch1 = conv2d_quant(*ch0, c1, c1.out_scale, c1.out_zero_point, c1.silu);
        FloatTensor cls = conv2d_float(*ch1, c2, false);

        const int h = box.shape[1];
        const int w = box.shape[2];
        const size_t plane = static_cast<size_t>(h) * w;

        std::vector<float> prob(reg_max);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                const size_t pos = static_cast<size_t>(y) * w + x;

                // Best class at this anchor.
                int best_cls = 0;
                float best_logit = -1e30f;
                for (int k = 0; k < nc; k++) {
                    const float v = cls.data[static_cast<size_t>(k) * plane + pos];
                    if (v > best_logit) {
                        best_logit = v;
                        best_cls = k;
                    }
                }
                const float score = 1.0f / (1.0f + std::exp(-best_logit));
                if (score < conf_threshold) continue;

                // DFL: each of the 4 box sides is a distribution over reg_max
                // bins; its expected value is the distance to that edge, in
                // stride units.
                float dist[4];
                for (int side = 0; side < 4; side++) {
                    float max_logit = -1e30f;
                    for (int j = 0; j < reg_max; j++) {
                        const float v =
                            box.data[static_cast<size_t>(side * reg_max + j) * plane + pos];
                        prob[j] = v;
                        if (v > max_logit) max_logit = v;
                    }
                    float sum = 0.0f;
                    for (int j = 0; j < reg_max; j++) {
                        prob[j] = std::exp(prob[j] - max_logit);
                        sum += prob[j];
                    }
                    float expectation = 0.0f;
                    for (int j = 0; j < reg_max; j++) {
                        expectation += static_cast<float>(j) * prob[j];
                    }
                    dist[side] = expectation / sum;
                }

                // Anchor centre sits at the middle of the cell.
                const float ax = static_cast<float>(x) + 0.5f;
                const float ay = static_cast<float>(y) + 0.5f;

                Detection d;
                d.x1 = (ax - dist[0]) * stride;
                d.y1 = (ay - dist[1]) * stride;
                d.x2 = (ax + dist[2]) * stride;
                d.y2 = (ay + dist[3]) * stride;
                d.score = score;
                d.cls = best_cls;
                candidates.push_back(d);
            }
        }
    }

    // Greedy per-class NMS.
    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });

    std::vector<Detection> kept;
    std::vector<bool> suppressed(candidates.size(), false);
    for (size_t i = 0; i < candidates.size(); i++) {
        if (suppressed[i]) continue;
        kept.push_back(candidates[i]);
        for (size_t j = i + 1; j < candidates.size(); j++) {
            if (suppressed[j]) continue;
            if (candidates[j].cls != candidates[i].cls) continue;
            if (iou(candidates[i], candidates[j]) > iou_threshold) suppressed[j] = true;
        }
    }

    return kept;
}

std::vector<Detection> run_inference(const Model& model, const Tensor& input,
                                     float conf_threshold, float iou_threshold) {
    auto features = run_backbone(model, input);
    return detect_head(model, features, conf_threshold, iou_threshold);
}

void set_kernel(Kernel k) { g_kernel = k; }
