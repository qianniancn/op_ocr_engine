#pragma once

#include "ocr_engine.h"

#include <memory>
#include <string>

namespace ocr {

struct NcnnOcrModelPaths {
    std::string det_param;
    std::string det_bin;
    std::string rec_param;
    std::string rec_bin;
    std::string dict_path;
    std::string cls_param;
    std::string cls_bin;
};

struct NcnnOcrOptions {
    bool use_vulkan = false;
    int gpu_device_id = -1;
    int num_threads = 4;
    bool use_fp16 = false;
    bool use_int8 = false;
    bool use_bf16 = false;
};

class NcnnOcrEngine {
public:
    NcnnOcrEngine();
    ~NcnnOcrEngine();

    NcnnOcrEngine(const NcnnOcrEngine&) = delete;
    NcnnOcrEngine& operator=(const NcnnOcrEngine&) = delete;

    /// 使用已转换的 PP-OCRv5/PP-OCRv6 ncnn 模型初始化 OCR 引擎。
    bool init(const NcnnOcrModelPaths& model_paths, const NcnnOcrOptions& options = {});
    bool init(const NcnnOcrModelPaths& model_paths, bool use_vulkan);

    /// 从原始图像字节中识别文字，返回文字内容、外接矩形和置信度。
    std::vector<OcrResult> recognize(const uint8_t* image, int width, int height, int bpp);
    OcrResponse recognize_with_profile(const uint8_t* image, int width, int height, int bpp);

    static const char* version();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ocr
