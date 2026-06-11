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

namespace fs = std::filesystem;

static std::unique_ptr<ocr::OcrServer> g_server;

static void signal_handler(int sig) {
    std::cout << "\n[main] Received signal " << sig << ", shutting down..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
}

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --backend <name>    OCR backend: tesseract or ncnn (default: auto)\n"
              << "  --datapath <path>   Path to tessdata directory for Tesseract\n"
              << "  --lang <language>   Tesseract OCR language, e.g. eng, chi_sim (default: eng)\n"
              << "  --model-dir <path>  Directory containing ncnn PP-OCRv5 models (default: models)\n"
              << "  --model-type <type> ncnn model type: mobile or server (default: mobile)\n"
              << "  --use-vulkan        Enable ncnn Vulkan compute\n"
              << "  --port <port>       HTTP listen port (default: 8080)\n"
              << "  --host <host>       HTTP listen host (default: 0.0.0.0)\n"
              << "  --help              Show this help message\n";
}

int main(int argc, char* argv[]) {
    std::string backend = "auto";
    std::string datapath;
    std::string lang = "eng";
    std::string model_dir = "models";
    std::string model_type = "mobile";
    std::string host = "0.0.0.0";
    int port = 8080;
    bool use_vulkan = false;

    // Simple argument parsing
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
        } else if (arg == "--model-type" && i + 1 < argc) {
            model_type = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--use-vulkan") {
            use_vulkan = true;
        } else if (arg == "--help") {
            print_usage(argv[0]);
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

    // Set up signal handlers for graceful shutdown
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
        if (model_type != "mobile" && model_type != "server") {
            std::cerr << "Error: --model-type must be mobile or server" << std::endl;
            return 1;
        }

        const std::string model_prefix = model_type == "server" ? "PP_OCRv5_server" : "PP_OCRv5_mobile";
        ocr::NcnnOcrModelPaths model_paths;
        model_paths.det_param = (root / (model_prefix + "_det.param")).string();
        model_paths.det_bin = (root / (model_prefix + "_det.bin")).string();
        model_paths.rec_param = (root / (model_prefix + "_rec.param")).string();
        model_paths.rec_bin = (root / (model_prefix + "_rec.bin")).string();
        model_paths.cls_param = (root / "PP_LCNet_x0_25_textline_ori.param").string();
        model_paths.cls_bin = (root / "PP_LCNet_x0_25_textline_ori.bin").string();

        ncnn_engine = std::make_unique<ocr::NcnnOcrEngine>();
        if (!ncnn_engine->init(model_paths, use_vulkan)) {
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
        std::cout << "Model type: " << model_type << std::endl;
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
