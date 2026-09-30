// Custom INT8 inference engine for YOLOv8n fire/smoke detection.
//
//   ./custom_engine image.jpg                  detect and write image_pred.jpg
//   ./custom_engine image.jpg -o out.jpg       choose the output path
//   ./custom_engine --input-bin test_input.bin --dump-dir dumps
//                                              layer-parity validation mode
//   ./custom_engine image.jpg --bench 5        time the forward pass
//   ./custom_engine image.jpg --kernel vnni     pick the convolution kernel
//
// Model files are found automatically in ../../quantization/, preferring the
// more accurate per-channel model, and can be overridden with --model-json and
// --model-bin.

#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "image_io.h"
#include "model.h"
#include "ops.h"

namespace {

constexpr int kNetSize = 640;

struct Options {
    std::string image;
    std::string input_bin;
    std::string output = "";
    std::string model_json;
    std::string model_bin;
    std::string dump_dir;
    std::string csv;
    bool csv_append = false;
    float conf = 0.25f;
    float iou = 0.45f;
    int bench = 0;
    Kernel kernel = Kernel::VnniInt8;
};

bool file_exists(const std::string& p) {
    std::ifstream f(p);
    return f.good();
}

// Locate a model file without depending on the working directory. Looks next to
// the executable first, then relative to wherever we were invoked from.
std::string find_in_usual_places(const std::string& argv0, const std::string& filename) {
    std::string exe_dir = ".";
    const size_t slash = argv0.find_last_of("/\\");
    if (slash != std::string::npos) exe_dir = argv0.substr(0, slash);

    const std::string candidates[] = {
        exe_dir + "/../../quantization/" + filename,
        exe_dir + "/../quantization/" + filename,
        exe_dir + "/" + filename,
        "../../quantization/" + filename,
        "quantization/" + filename,
        filename,
    };
    for (const std::string& c : candidates) {
        if (file_exists(c)) return c;
    }
    return std::string();
}

// Prefer the per-channel model, which scores 0.8826 against 0.7680 for the
// per-tensor one. Running the less accurate model by default just because its
// filename is shorter is a trap, so the accurate one wins and per-tensor is
// still reachable with --model-json / --model-bin.
std::string find_model_file(const std::string& argv0, const std::string& suffix) {
    const std::string preferred = "model_int8_pc" + suffix;
    const std::string fallback = "model_int8" + suffix;

    std::string path = find_in_usual_places(argv0, preferred);
    if (!path.empty()) return path;

    path = find_in_usual_places(argv0, fallback);
    if (!path.empty()) return path;

    return "../../quantization/" + preferred;  // name the expected location
}

void print_usage() {
    std::cout <<
        "Usage: custom_engine <image> [options]\n"
        "       custom_engine --input-bin <file> [options]\n\n"
        "Options:\n"
        "  -o, --output <path>     annotated image output (default <image>_pred.jpg)\n"
        "      --conf <float>      confidence threshold (default 0.25)\n"
        "      --iou <float>       NMS IoU threshold (default 0.45)\n"
        "      --input-bin <path>  use a pre-quantized 3x640x640 INT8 input\n"
        "      --dump-dir <dir>    write each layer's INT8 output for validation\n"
        "      --bench <n>         run the forward pass n extra times and report timing\n"
        "      --kernel <name>     convolution kernel (default vnni-int8):\n"
        "                            scalar-int8  one INT8 multiply at a time\n"
        "                            scalar-fp32  the same loop in FP32\n"
        "                            vnni-int8    INT8 via AVX-VNNI, 32 per instruction\n"
        "                            cuda-fp32    im2col plus an FP32 GEMM on the GPU\n"
        "      --model-json <path> / --model-bin <path>\n";
}

bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value after ") + what);
            return argv[++i];
        };

        if (a == "-h" || a == "--help") return false;
        else if (a == "-o" || a == "--output") o.output = next("--output");
        else if (a == "--conf") o.conf = std::stof(next("--conf"));
        else if (a == "--iou") o.iou = std::stof(next("--iou"));
        else if (a == "--input-bin") o.input_bin = next("--input-bin");
        else if (a == "--dump-dir") o.dump_dir = next("--dump-dir");
        else if (a == "--csv") o.csv = next("--csv");
        else if (a == "--csv-append") { o.csv = next("--csv-append"); o.csv_append = true; }
        else if (a == "--bench") o.bench = std::stoi(next("--bench"));
        else if (a == "--kernel") {
            const std::string name = next("--kernel");
            if (!parse_kernel(name, o.kernel)) {
                throw std::runtime_error("Unknown kernel: " + name +
                                         " (want scalar-int8, scalar-fp32 or vnni-int8)");
            }
        }
        else if (a == "--model-json") o.model_json = next("--model-json");
        else if (a == "--model-bin") o.model_bin = next("--model-bin");
        else if (!a.empty() && a[0] == '-') throw std::runtime_error("Unknown option: " + a);
        else o.image = a;
    }
    return !(o.image.empty() && o.input_bin.empty());
}

std::string default_output_path(const std::string& image) {
    const size_t dot = image.find_last_of('.');
    const std::string stem = (dot == std::string::npos) ? image : image.substr(0, dot);
    return stem + "_pred.jpg";
}

std::vector<std::string> read_class_names(const std::string& json_path) {
    std::ifstream f(json_path);
    nlohmann::json j;
    f >> j;
    if (j.contains("class_names")) {
        return j["class_names"].get<std::vector<std::string>>();
    }
    return {};
}

void dump_layers(const std::vector<std::unique_ptr<Tensor>>& outputs, const std::string& dir) {
    int written = 0;
    for (size_t i = 0; i < outputs.size(); i++) {
        if (!outputs[i]) continue;
        char name[512];
        std::snprintf(name, sizeof(name), "%s/L%02zu_cpp.bin", dir.c_str(), i);
        std::ofstream f(name, std::ios::binary);
        if (!f) {
            std::cerr << "Warning: could not write " << name << std::endl;
            continue;
        }
        f.write(reinterpret_cast<const char*>(outputs[i]->data),
                static_cast<std::streamsize>(outputs[i]->num_elements));
        written++;
    }
    std::cout << "Dumped " << written << " layer outputs to " << dir << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    try {
        if (!parse_args(argc, argv, opt)) {
            print_usage();
            return argc > 1 ? 0 : 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        print_usage();
        return 1;
    }

    if (opt.model_json.empty()) opt.model_json = find_model_file(argv[0], ".json");
    if (opt.model_bin.empty()) opt.model_bin = find_model_file(argv[0], ".bin");

    try {
        Model model = load_model_metadata(opt.model_json);
        load_model_weights(model, opt.model_bin);
        const std::vector<std::string> class_names = read_class_names(opt.model_json);

        // AVX-VNNI is not on every x86 CPU, so fall back rather than crash with
        // an illegal instruction on a machine that lacks it.
        if (opt.kernel == Kernel::VnniInt8 && !vnni_supported()) {
            std::cerr << "This CPU has no AVX-VNNI; falling back to scalar-int8"
                      << std::endl;
            opt.kernel = Kernel::ScalarInt8;
        }
        if (opt.kernel == Kernel::CudaFp32 && !cuda_supported()) {
            std::cerr << "No usable CUDA device; falling back to scalar-fp32"
                      << std::endl;
            opt.kernel = Kernel::ScalarFp32;
        }
        prepare_kernel(model, opt.kernel);
        set_kernel(opt.kernel);
        std::cout << "Kernel: " << kernel_name(opt.kernel) << std::endl;

        // Build the INT8 input, either from an image or a pre-quantized blob.
        std::unique_ptr<Tensor> input;
        Image image;
        if (!opt.input_bin.empty()) {
            input = std::make_unique<Tensor>(std::vector<int>{3, kNetSize, kNetSize},
                                             model.input_scale, model.input_zero_point);
            std::ifstream f(opt.input_bin, std::ios::binary);
            if (!f) throw std::runtime_error("Could not open " + opt.input_bin);
            input->load_from_stream(f, input->num_elements);
            std::cout << "Loaded pre-quantized input from " << opt.input_bin << std::endl;
        } else {
            image = load_image(opt.image);
            std::cout << "Loaded " << opt.image << " (" << image.width << "x" << image.height
                      << ")" << std::endl;
            input = preprocess(image, kNetSize, model);
        }

        const auto t0 = std::chrono::high_resolution_clock::now();
        auto features = run_backbone(model, *input);
        auto detections = detect_head(model, features, opt.conf, opt.iou);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (!opt.dump_dir.empty()) {
            dump_layers(features, opt.dump_dir);
        }

        std::cout << "\nInference: " << std::fixed << std::setprecision(1) << ms << " ms\n";
        std::cout << "Detections: " << detections.size() << "\n";
        for (const Detection& d : detections) {
            const std::string name = (d.cls < static_cast<int>(class_names.size()))
                                         ? class_names[d.cls]
                                         : ("class" + std::to_string(d.cls));
            std::cout << "  " << std::setw(6) << std::left << name << " "
                      << std::setprecision(3) << d.score << "  box=["
                      << std::setprecision(1) << d.x1 << ", " << d.y1 << ", " << d.x2 << ", "
                      << d.y2 << "]\n";
        }

        // Machine-readable output: image,class_id,class_name,conf,x1,y1,x2,y2
        // in 640x640 network pixels. Consumed by quantization/eval_map.py.
        if (!opt.csv.empty()) {
            std::ofstream f(opt.csv, opt.csv_append ? std::ios::app : std::ios::trunc);
            if (!f) throw std::runtime_error("Could not write " + opt.csv);
            const size_t slash = opt.image.find_last_of("/\\");
            const std::string base =
                (slash == std::string::npos) ? opt.image : opt.image.substr(slash + 1);
            for (const Detection& d : detections) {
                const std::string name = (d.cls < static_cast<int>(class_names.size()))
                                             ? class_names[d.cls]
                                             : ("class" + std::to_string(d.cls));
                f << base << "," << d.cls << "," << name << ","
                  << std::setprecision(6) << d.score << "," << d.x1 << "," << d.y1 << ","
                  << d.x2 << "," << d.y2 << "\n";
            }
        }

        if (!opt.image.empty()) {
            const std::string out_path =
                opt.output.empty() ? default_output_path(opt.image) : opt.output;
            draw_detections(image, detections, kNetSize, class_names);
            save_jpeg(out_path, image);
            std::cout << "\nWrote " << out_path << std::endl;
        }

        if (opt.bench > 0) {
            std::vector<double> times;
            for (int i = 0; i < opt.bench; i++) {
                const auto a = std::chrono::high_resolution_clock::now();
                auto f = run_backbone(model, *input);
                auto d = detect_head(model, f, opt.conf, opt.iou);
                (void)d;
                const auto b = std::chrono::high_resolution_clock::now();
                times.push_back(std::chrono::duration<double, std::milli>(b - a).count());
            }
            double sum = 0, lo = times[0], hi = times[0];
            for (double t : times) {
                sum += t;
                lo = std::min(lo, t);
                hi = std::max(hi, t);
            }
            std::cout << "\n=== Benchmark (" << opt.bench << " runs) ===\n"
                      << "Avg: " << std::setprecision(1) << sum / times.size() << " ms\n"
                      << "Min: " << lo << " ms\nMax: " << hi << " ms" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
