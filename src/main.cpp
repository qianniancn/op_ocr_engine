#include "ocr_engine.h"
#include "ocr_server.h"

#ifdef OCR_HAS_NCNN
#include "ncnn_ocr_engine.h"
#endif

#include <filesystem>
#include <iostream>
#include <string>
#include <csignal>
#include <memory>
#include <vector>
#include <algorithm>
#include <utility>

namespace fs = std::filesystem;

static std::unique_ptr<ocr::OcrServer> g_server;

static void signal_handler(int sig) {
    std::cout << "\n[main] Received signal " << sig << ", shutting down..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
}

static void print_usage(const char* prog, bool advanced = false) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --model-type <type> PP-OCRv6 model type: tiny, small, medium (default: small)\n"
              << "  --device <device>   Inference device: cpu or gpu (default: cpu)\n"
              << "  --model-dir <path>  Directory containing LiteOCR ncnn models (default: ..\\LiteOCR\\models)\n"
              << "  --port <port>       HTTP listen port (default: 8082)\n"
              << "  --host <host>       HTTP listen host (default: 127.0.0.1)\n"
              << "  --help              Show this help message\n"
              << "  --help-advanced     Show developer and tuning options\n";
    if (!advanced) {
        return;
    }

    std::cout << "\nAdvanced options:\n"
              << "  --backend <name>    OCR backend: tesseract or ncnn (default: auto)\n"
              << "  --datapath <path>   Path to tessdata directory for Tesseract\n"
              << "  --lang <language>   Tesseract OCR language, e.g. eng, chi_sim (default: eng)\n"
              << "  --model-version <v> ncnn model version: v5 or v6 (default: v6)\n"
              << "  --quality <level>   Compatibility alias: fast, balanced, accurate\n"
              << "  --use-vulkan        Enable ncnn Vulkan compute (same as --device gpu)\n"
              << "  --gpu-device <id>   ncnn Vulkan GPU device id (default: 0 when GPU is enabled)\n"
              << "  --threads <n>       ncnn CPU inference thread count (default: 4)\n"
              << "  --fp16              Enable ncnn FP16 arithmetic/storage/packing\n"
              << "  --fast-mode         Skip direction classifier and 180-degree fallback recognition\n"
              << "  --bf16              Enable ncnn BF16 storage/packing\n";
}

#ifdef OCR_HAS_NCNN
static bool contains(const std::vector<std::string>& values, const std::string& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

static fs::path first_existing_path(const fs::path& root, const std::vector<std::string>& names) {
    for (const auto& name : names) {
        fs::path path = root / name;
        if (fs::exists(path)) {
            return path;
        }
    }
    return root / names.front();
}

static std::vector<std::string> ocr_model_file_candidates(const std::vector<std::string>& prefixes,
                                                          const std::string& role,
                                                          const std::string& extension) {
    std::vector<std::string> names;
    for (const auto& prefix : prefixes) {
        names.push_back(prefix + "_" + role + "." + extension);
    }
    return names;
}

static void set_first_existing_model_pair(const fs::path& root,
                                          const std::vector<std::pair<std::string, std::string>>& candidates,
                                          std::string& param_path,
                                          std::string& bin_path) {
    for (const auto& [param_name, bin_name] : candidates) {
        const fs::path param = root / param_name;
        const fs::path bin = root / bin_name;
        if (fs::exists(param) && fs::exists(bin)) {
            param_path = param.string();
            bin_path = bin.string();
            return;
        }
    }

    param_path = (root / candidates.front().first).string();
    bin_path = (root / candidates.front().second).string();
}

static bool apply_quality_alias(const std::string& quality, std::string& model_version, std::string& model_type) {
    if (quality == "fast") {
        model_version = "v6";
        model_type = "tiny";
        return true;
    }
    if (quality == "balanced") {
        model_version = "v6";
        model_type = "small";
        return true;
    }
    if (quality == "accurate") {
        model_version = "v6";
        model_type = "medium";
        return true;
    }

    std::cerr << "Error: --quality must be fast, balanced, or accurate" << std::endl;
    return false;
}

static bool resolve_ncnn_model_paths(const fs::path& root, const std::string& model_version,
                                     const std::string& requested_model_type,
                                     ocr::NcnnOcrModelPaths& model_paths,
                                     std::string& resolved_model_type) {
    std::vector<std::string> prefixes;
    std::vector<std::string> dict_candidates;

    if (model_version == "v5") {
        resolved_model_type = requested_model_type == "auto" ? "mobile" : requested_model_type;
        if (!contains({"mobile", "server"}, resolved_model_type)) {
            std::cerr << "Error: v5 --model-type must be auto, mobile, or server" << std::endl;
            return false;
        }

        prefixes = {
            "PP-OCRv5_" + resolved_model_type,
        };
        dict_candidates = {
            "PP-OCRv5_vocab.txt",
        };
    } else if (model_version == "v6") {
        resolved_model_type = requested_model_type == "auto" ? "small" : requested_model_type;
        if (!contains({"tiny", "small", "medium"}, resolved_model_type)) {
            std::cerr << "Error: v6 --model-type must be auto, tiny, small, or medium" << std::endl;
            return false;
        }

        prefixes = {
            "PP-OCRv6_" + resolved_model_type,
        };
        dict_candidates = resolved_model_type == "tiny"
            ? std::vector<std::string>{"PP-OCRv6_vocab_tiny.txt", "PP-OCRv6_vocab.txt"}
            : std::vector<std::string>{"PP-OCRv6_vocab.txt"};
    } else {
        std::cerr << "Error: --model-version must be v5 or v6" << std::endl;
        return false;
    }

    model_paths.det_param = first_existing_path(root, ocr_model_file_candidates(prefixes, "det", "param")).string();
    model_paths.det_bin = first_existing_path(root, ocr_model_file_candidates(prefixes, "det", "bin")).string();
    model_paths.rec_param = first_existing_path(root, ocr_model_file_candidates(prefixes, "rec", "param")).string();
    model_paths.rec_bin = first_existing_path(root, ocr_model_file_candidates(prefixes, "rec", "bin")).string();
    model_paths.dict_path = first_existing_path(root, dict_candidates).string();
    set_first_existing_model_pair(root,
        std::vector<std::pair<std::string, std::string>>{
            {"PP-LCNet_x1_0_textline_ori.param", "PP-LCNet_x1_0_textline_ori.bin"},
        },
        model_paths.cls_param, model_paths.cls_bin);
    return true;
}
#endif

int main(int argc, char* argv[]) {
    std::string backend = "auto";
    std::string datapath;
    std::string lang = "eng";
    std::string model_dir = "../LiteOCR/models";
    std::string model_version = "v6";
    std::string model_type = "small";
    std::string resolved_model_type;
    std::string quality_alias;
    bool model_version_specified = false;
    bool model_type_specified = false;
    std::string host = "127.0.0.1";
    int port = 8082;
#ifdef OCR_HAS_NCNN
    ocr::NcnnOcrOptions ncnn_options;
#endif

    // 这里保持轻量解析，避免为了少量启动参数引入额外依赖。
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--backend" && i + 1 < argc) {
            backend = argv[++i];
        } else if (arg == "--datapath" && i + 1 < argc) {
            datapath = argv[++i];
        } else if (arg == "--lang" && i + 1 < argc) {
            lang = argv[++i];
        } else if (arg == "--model-dir" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (arg == "--quality" && i + 1 < argc) {
            quality_alias = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            const std::string device = argv[++i];
            if (device == "cpu") {
#ifdef OCR_HAS_NCNN
                ncnn_options.use_vulkan = false;
                ncnn_options.gpu_device_id = -1;
#endif
            } else if (device == "gpu") {
#ifdef OCR_HAS_NCNN
                ncnn_options.use_vulkan = true;
                if (ncnn_options.gpu_device_id < 0) {
                    ncnn_options.gpu_device_id = 0;
                }
#endif
            } else {
                std::cerr << "Error: --device must be cpu or gpu" << std::endl;
                return 1;
            }
        } else if (arg == "--model-version" && i + 1 < argc) {
            model_version = argv[++i];
            model_version_specified = true;
        } else if (arg == "--model-type" && i + 1 < argc) {
            model_type = argv[++i];
            model_type_specified = true;
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--use-vulkan") {
#ifdef OCR_HAS_NCNN
            ncnn_options.use_vulkan = true;
            if (ncnn_options.gpu_device_id < 0) {
                ncnn_options.gpu_device_id = 0;
            }
#endif
        } else if (arg == "--gpu-device" && i + 1 < argc) {
#ifdef OCR_HAS_NCNN
            ncnn_options.gpu_device_id = std::stoi(argv[++i]);
            ncnn_options.use_vulkan = ncnn_options.gpu_device_id >= 0;
#else
            ++i;
#endif
        } else if (arg == "--threads" && i + 1 < argc) {
#ifdef OCR_HAS_NCNN
            ncnn_options.num_threads = std::max(1, std::stoi(argv[++i]));
#else
            ++i;
#endif
        } else if (arg == "--fp16") {
#ifdef OCR_HAS_NCNN
            ncnn_options.use_fp16 = true;
#endif
        } else if (arg == "--fast-mode") {
#ifdef OCR_HAS_NCNN
            ncnn_options.fast_mode = true;
#endif
        } else if (arg == "--bf16") {
#ifdef OCR_HAS_NCNN
            ncnn_options.use_bf16 = true;
#endif
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--help-advanced") {
            print_usage(argv[0], true);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    if (backend == "auto") {
#ifdef OCR_HAS_NCNN
        backend = "ncnn";
#elif defined(OCR_HAS_TESSERACT)
        backend = "tesseract";
#else
        std::cerr << "Error: no OCR backend is enabled in this build" << std::endl;
        return 1;
#endif
    }

    // 服务常驻运行，收到终止信号时先让 HTTP server 退出监听。
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

#ifdef OCR_HAS_TESSERACT
    std::unique_ptr<ocr::OcrEngine> tess_engine;
#endif
#ifdef OCR_HAS_NCNN
    std::unique_ptr<ocr::NcnnOcrEngine> ncnn_engine;
#endif
    const char* version = "unknown";

    if (backend == "tesseract") {
#ifndef OCR_HAS_TESSERACT
        std::cerr << "Error: this build does not include Tesseract support" << std::endl;
        return 1;
#else
        if (datapath.empty()) {
            std::cerr << "Error: --datapath is required for Tesseract\n" << std::endl;
            print_usage(argv[0]);
            return 1;
        }

        tess_engine = std::make_unique<ocr::OcrEngine>();
        if (!tess_engine->init(datapath, lang)) {
            std::cerr << "Failed to initialize Tesseract OCR engine" << std::endl;
            return 1;
        }
        version = ocr::OcrEngine::version();

        g_server = std::make_unique<ocr::OcrServer>(
            [&tess_engine](const uint8_t* image, int width, int height, int bpp) {
                ocr::OcrResponse response;
                response.results = tess_engine->recognize(image, width, height, bpp);
                return response;
            },
            [version]() { return version; });
#endif
    } else if (backend == "ncnn") {
#ifndef OCR_HAS_NCNN
        std::cerr << "Error: this build does not include ncnn support" << std::endl;
        return 1;
#else
        const fs::path root = fs::path(model_dir);
        ocr::NcnnOcrModelPaths model_paths;
        if (!quality_alias.empty() && (model_version_specified || model_type_specified)) {
            std::cerr << "Error: --quality cannot be used with --model-version or --model-type" << std::endl;
            return 1;
        }
        if (!quality_alias.empty() && !apply_quality_alias(quality_alias, model_version, model_type)) {
            return 1;
        }
        if (!model_version_specified && model_type_specified && contains({"mobile", "server"}, model_type)) {
            model_version = "v5";
        }
        if (model_version_specified && !model_type_specified && model_version == "v5") {
            model_type = "auto";
        }
        if (!resolve_ncnn_model_paths(root, model_version, model_type, model_paths, resolved_model_type)) {
            return 1;
        }

        ncnn_engine = std::make_unique<ocr::NcnnOcrEngine>();
        if (!ncnn_engine->init(model_paths, ncnn_options)) {
            std::cerr << "Failed to initialize ncnn OCR engine" << std::endl;
            return 1;
        }
        version = ocr::NcnnOcrEngine::version();

        g_server = std::make_unique<ocr::OcrServer>(
            [&ncnn_engine](const uint8_t* image, int width, int height, int bpp) {
                return ncnn_engine->recognize_with_profile(image, width, height, bpp);
            },
            [version]() { return version; });
#endif
    } else {
        std::cerr << "Unknown backend: " << backend << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "=== OCR HTTP Service ===" << std::endl;
    std::cout << "Backend: " << backend << std::endl;
    if (backend == "ncnn") {
        if (!quality_alias.empty()) {
            std::cout << "Quality alias: " << quality_alias << std::endl;
        }
        std::cout << "Model version: " << model_version << std::endl;
        std::cout << "Model type: " << resolved_model_type << std::endl;
#ifdef OCR_HAS_NCNN
        std::cout << "Threads: " << ncnn_options.num_threads << std::endl;
        std::cout << "GPU device: " << ncnn_options.gpu_device_id << std::endl;
        std::cout << "FP16: " << (ncnn_options.use_fp16 ? "on" : "off") << std::endl;
        std::cout << "BF16: " << (ncnn_options.use_bf16 ? "on" : "off") << std::endl;
        std::cout << "Fast mode: " << (ncnn_options.fast_mode ? "on" : "off") << std::endl;
#endif
    }
    std::cout << "Version: " << version << std::endl;
    std::cout << "Endpoints:" << std::endl;
    std::cout << "  GET  /health          - Health check" << std::endl;
    std::cout << "  GET  /api/v1/version  - Engine version" << std::endl;
    std::cout << "  POST /api/v1/ocr      - Perform OCR" << std::endl;
    std::cout << "========================" << std::endl;

    if (!g_server->listen(host, port)) {
        std::cerr << "Failed to start server on " << host << ":" << port << std::endl;
        return 1;
    }

    return 0;
}
