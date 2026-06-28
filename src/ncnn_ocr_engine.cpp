#include "ncnn_ocr_engine.h"

#ifdef OCR_HAS_NCNN
#include <net.h>
#if NCNN_VULKAN
#include <gpu.h>
#endif
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ocr {

#ifdef OCR_HAS_NCNN
namespace {

using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct CharacterToken {
    int id = -1;
    float prob = 0.f;
};

struct PointF {
    float x = 0.f;
    float y = 0.f;
};

struct PointI {
    int x = 0;
    int y = 0;
};

struct SizeF {
    float width = 0.f;
    float height = 0.f;
};

struct RotatedRect {
    PointF center;
    SizeF size;
    float angle = 0.f;
};

struct RotatedTextBox {
    std::array<PointF, 4> corners;
    float score = 0.f;
};

static std::vector<std::string> load_character_dict(const fs::path& dict_path) {
    std::vector<std::string> dict;
    std::ifstream ifs(dict_path, std::ios::binary);
    if (!ifs) {
        std::cerr << "[NcnnOcrEngine] Failed to open dict file: " << dict_path.string() << std::endl;
        return dict;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        dict.push_back(line);
    }
    return dict;
}

static int source_pixel_type(int bpp) {
    if (bpp == 1)
        return ncnn::Mat::PIXEL_GRAY2BGR;
    if (bpp == 3)
        return ncnn::Mat::PIXEL_RGB2BGR;
    return ncnn::Mat::PIXEL_RGBA2BGR;
}

static std::vector<uint8_t> to_bgr_pixels(const uint8_t* image, int width, int height, int bpp) {
    std::vector<uint8_t> bgr(static_cast<size_t>(width) * height * 3);
    ncnn::Mat in = ncnn::Mat::from_pixels(image, source_pixel_type(bpp), width, height);
    in.to_pixels(bgr.data(), ncnn::Mat::PIXEL_BGR);
    return bgr;
}

static std::vector<CharacterToken> decode_ctc_tokens(const ncnn::Mat& out) {
    std::vector<CharacterToken> tokens;
    int last_token = 0;

    for (int i = 0; i < out.h; i++) {
        const float* p = out.row(i);

        int index = 0;
        float max_score = -9999.f;
        for (int j = 0; j < out.w; j++) {
            const float score = *p++;
            if (score > max_score) {
                max_score = score;
                index = j;
            }
        }

        if (last_token == index)
            continue;

        last_token = index;

        if (index <= 0)
            continue;

        CharacterToken token;
        token.id = index - 1;
        token.prob = max_score;
        tokens.push_back(token);
    }

    return tokens;
}

static std::string decode_text(const std::vector<std::string>& dict, const std::vector<CharacterToken>& tokens,
                               float& avg_confidence) {
    std::string text;
    float conf_sum = 0.f;
    int conf_count = 0;

    for (const auto& token : tokens) {
        if (token.id < 0)
            continue;

        if (token.id < static_cast<int>(dict.size())) {
            text += dict[token.id];
        } else if (token.id == static_cast<int>(dict.size()) && !text.empty() && text.back() != ' ') {
            text += ' ';
        } else {
            continue;
        }
        conf_sum += token.prob;
        conf_count++;
    }

    avg_confidence = conf_count > 0 ? (conf_sum / conf_count) : 0.f;
    return text;
}

static int run_net(ncnn::Net& net, const ncnn::Mat& in, ncnn::Mat& out) {
    ncnn::Extractor ex = net.create_extractor();
    if (ex.input("in0", in) != 0 && ex.input("input", in) != 0) {
        return -1;
    }

    if (ex.extract("out0", out) != 0 && ex.extract("output", out) != 0) {
        return -1;
    }

    return 0;
}

static float distance_between(const PointF& a, const PointF& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

static float polygon_area(const std::array<PointF, 4>& points) {
    float area = 0.f;
    for (size_t i = 0; i < points.size(); ++i) {
        const auto& a = points[i];
        const auto& b = points[(i + 1) % points.size()];
        area += a.x * b.y - a.y * b.x;
    }
    return std::abs(area) * 0.5f;
}

static float polygon_area(const std::vector<PointF>& points) {
    if (points.size() < 3)
        return 0.f;

    float area = 0.f;
    for (size_t i = 0; i < points.size(); ++i) {
        const auto& a = points[i];
        const auto& b = points[(i + 1) % points.size()];
        area += a.x * b.y - a.y * b.x;
    }
    return std::abs(area) * 0.5f;
}

static std::vector<PointI> convex_hull(std::vector<PointI> points) {
    if (points.size() <= 1)
        return points;

    std::sort(points.begin(), points.end(), [](const PointI& a, const PointI& b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });

    const auto cross = [](const PointI& o, const PointI& a, const PointI& b) -> long long {
        return static_cast<long long>(a.x - o.x) * (b.y - o.y) -
               static_cast<long long>(a.y - o.y) * (b.x - o.x);
    };

    std::vector<PointI> hull;
    hull.reserve(points.size() * 2);
    for (const auto& p : points) {
        while (hull.size() >= 2 && cross(hull[hull.size() - 2], hull.back(), p) <= 0)
            hull.pop_back();
        hull.push_back(p);
    }

    const size_t lower_size = hull.size();
    for (int i = static_cast<int>(points.size()) - 2; i >= 0; --i) {
        const auto& p = points[static_cast<size_t>(i)];
        while (hull.size() > lower_size && cross(hull[hull.size() - 2], hull.back(), p) <= 0)
            hull.pop_back();
        hull.push_back(p);
    }

    if (!hull.empty())
        hull.pop_back();
    return hull;
}

static RotatedRect min_area_rect(const std::vector<PointI>& contour) {
    if (contour.empty())
        return {};
    if (contour.size() == 1)
        return {{static_cast<float>(contour[0].x), static_cast<float>(contour[0].y)}, {}, 0.f};

    const auto hull = convex_hull(contour);
    if (hull.size() == 2) {
        const float cx = (hull[0].x + hull[1].x) * 0.5f;
        const float cy = (hull[0].y + hull[1].y) * 0.5f;
        const float dx = static_cast<float>(hull[1].x - hull[0].x);
        const float dy = static_cast<float>(hull[1].y - hull[0].y);
        const float len = std::sqrt(dx * dx + dy * dy);
        const float angle = std::atan2(dy, dx) * 180.f / static_cast<float>(M_PI);
        return {{cx, cy}, {len, 0.f}, angle};
    }

    float best_area = std::numeric_limits<float>::max();
    RotatedRect best;
    for (size_t i = 0; i < hull.size(); ++i) {
        const auto& a = hull[i];
        const auto& b = hull[(i + 1) % hull.size()];
        const float ex = static_cast<float>(b.x - a.x);
        const float ey = static_cast<float>(b.y - a.y);
        const float len = std::sqrt(ex * ex + ey * ey);
        if (len < 1e-6f)
            continue;

        const float ux = ex / len;
        const float uy = ey / len;
        const float vx = -uy;
        const float vy = ux;
        float min_u = std::numeric_limits<float>::max();
        float max_u = std::numeric_limits<float>::lowest();
        float min_v = std::numeric_limits<float>::max();
        float max_v = std::numeric_limits<float>::lowest();

        for (const auto& p : hull) {
            const float px = static_cast<float>(p.x - a.x);
            const float py = static_cast<float>(p.y - a.y);
            const float pu = px * ux + py * uy;
            const float pv = px * vx + py * vy;
            min_u = std::min(min_u, pu);
            max_u = std::max(max_u, pu);
            min_v = std::min(min_v, pv);
            max_v = std::max(max_v, pv);
        }

        const float rect_w = max_u - min_u;
        const float rect_h = max_v - min_v;
        const float area = rect_w * rect_h;
        if (area >= best_area)
            continue;

        const float cu = (min_u + max_u) * 0.5f;
        const float cv = (min_v + max_v) * 0.5f;
        best_area = area;
        best.center = {
            static_cast<float>(a.x) + cu * ux + cv * vx,
            static_cast<float>(a.y) + cu * uy + cv * vy,
        };
        best.size = {rect_w, rect_h};
        best.angle = std::atan2(uy, ux) * 180.f / static_cast<float>(M_PI);
    }

    if (best.angle >= 90.f)
        best.angle -= 180.f;
    if (best.angle < -90.f)
        best.angle += 180.f;
    if (best.angle >= 0.f) {
        std::swap(best.size.width, best.size.height);
        best.angle -= 90.f;
    }
    return best;
}

static std::array<PointF, 4> rotated_rect_points(const RotatedRect& rect) {
    const float half_w = rect.size.width * 0.5f;
    const float half_h = rect.size.height * 0.5f;
    const float angle = rect.angle * static_cast<float>(M_PI) / 180.f;
    const float c = std::cos(angle);
    const float s = std::sin(angle);

    return {
        PointF{rect.center.x - half_w * c + half_h * s, rect.center.y - half_w * s - half_h * c},
        PointF{rect.center.x + half_w * c + half_h * s, rect.center.y + half_w * s - half_h * c},
        PointF{rect.center.x + half_w * c - half_h * s, rect.center.y + half_w * s + half_h * c},
        PointF{rect.center.x - half_w * c - half_h * s, rect.center.y - half_w * s + half_h * c},
    };
}

static std::array<PointF, 4> order_box_points(std::array<PointF, 4> points) {
    std::sort(points.begin(), points.end(), [](const PointF& a, const PointF& b) {
        return a.x < b.x;
    });

    const PointF tl = points[0].y < points[1].y ? points[0] : points[1];
    const PointF bl = points[0].y < points[1].y ? points[1] : points[0];
    const PointF tr = points[2].y < points[3].y ? points[2] : points[3];
    const PointF br = points[2].y < points[3].y ? points[3] : points[2];
    return {tl, tr, br, bl};
}

static bool point_in_polygon(float x, float y, const std::vector<PointF>& polygon) {
    bool inside = false;
    for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
        const auto& a = polygon[i];
        const auto& b = polygon[j];
        const bool crosses = ((a.y > y) != (b.y > y)) &&
            (x < (b.x - a.x) * (y - a.y) / ((b.y - a.y) + 1e-6f) + a.x);
        if (crosses)
            inside = !inside;
    }
    return inside;
}

static float box_score(const ncnn::Mat& prob, const std::vector<PointF>& box) {
    if (box.size() < 4)
        return 0.f;

    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();
    for (const auto& p : box) {
        min_x = std::min(min_x, p.x);
        min_y = std::min(min_y, p.y);
        max_x = std::max(max_x, p.x);
        max_y = std::max(max_y, p.y);
    }

    const int x1 = std::clamp(static_cast<int>(std::floor(min_x)), 0, prob.w - 1);
    const int y1 = std::clamp(static_cast<int>(std::floor(min_y)), 0, prob.h - 1);
    const int x2 = std::clamp(static_cast<int>(std::ceil(max_x)), x1, prob.w - 1);
    const int y2 = std::clamp(static_cast<int>(std::ceil(max_y)), y1, prob.h - 1);

    double sum = 0.0;
    int count = 0;
    for (int y = y1; y <= y2; ++y) {
        const float* row = prob.row(y);
        for (int x = x1; x <= x2; ++x) {
            if (point_in_polygon(x + 0.5f, y + 0.5f, box)) {
                sum += row[x];
                count++;
            }
        }
    }
    return count > 0 ? static_cast<float>(sum / count) : 0.f;
}

static std::vector<PointF> unclip_box(const std::vector<PointF>& box, float ratio) {
    if (box.size() < 4)
        return box;

    float cx = 0.f;
    float cy = 0.f;
    for (const auto& p : box) {
        cx += p.x;
        cy += p.y;
    }
    cx /= static_cast<float>(box.size());
    cy /= static_cast<float>(box.size());

    float perimeter = 0.f;
    for (size_t i = 0; i < box.size(); ++i) {
        perimeter += distance_between(box[i], box[(i + 1) % box.size()]);
    }
    const float area = polygon_area(box);
    const float distance = perimeter > 1e-6f ? area * ratio / perimeter : 0.f;

    std::vector<PointF> expanded;
    expanded.reserve(box.size());
    for (const auto& p : box) {
        const float dx = p.x - cx;
        const float dy = p.y - cy;
        const float len = std::sqrt(dx * dx + dy * dy);
        const float scale = len > 1e-6f ? (len + distance) / len : 1.f;
        expanded.push_back({cx + dx * scale, cy + dy * scale});
    }
    return expanded;
}

static void axis_aligned_bounds(const RotatedTextBox& box, int image_width, int image_height, int& x1, int& y1,
                                int& x2, int& y2) {
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();

    for (const auto& p : box.corners) {
        min_x = std::min(min_x, p.x);
        min_y = std::min(min_y, p.y);
        max_x = std::max(max_x, p.x);
        max_y = std::max(max_y, p.y);
    }

    x1 = std::clamp(static_cast<int>(std::floor(min_x)), 0, image_width - 1);
    y1 = std::clamp(static_cast<int>(std::floor(min_y)), 0, image_height - 1);
    x2 = std::clamp(static_cast<int>(std::ceil(max_x)), x1 + 1, image_width);
    y2 = std::clamp(static_cast<int>(std::ceil(max_y)), y1 + 1, image_height);
}

static uint8_t sample_bgr(const std::vector<uint8_t>& bgr, int image_width, int image_height, float x, float y,
                          int channel) {
    x = std::clamp(x, 0.f, static_cast<float>(image_width - 1));
    y = std::clamp(y, 0.f, static_cast<float>(image_height - 1));

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, image_width - 1);
    const int y1 = std::min(y0 + 1, image_height - 1);
    const float wx = x - x0;
    const float wy = y - y0;

    const auto at = [&](int px, int py) -> float {
        return bgr[(static_cast<size_t>(py) * image_width + px) * 3 + channel];
    };

    const float top = at(x0, y0) * (1.f - wx) + at(x1, y0) * wx;
    const float bottom = at(x0, y1) * (1.f - wx) + at(x1, y1) * wx;
    const float value = top * (1.f - wy) + bottom * wy;
    return static_cast<uint8_t>(std::clamp(std::round(value), 0.f, 255.f));
}

static std::vector<uint8_t> warp_text_crop(const std::vector<uint8_t>& bgr, int image_width, int image_height,
                                           const RotatedTextBox& box, int& crop_width, int& crop_height) {
    // 按检测到的四点区域做透视近似裁剪，保证识别模型看到横向文本行。
    const float src_width = std::max(1.f, distance_between(box.corners[0], box.corners[1]));
    const float src_height = std::max(1.f, distance_between(box.corners[0], box.corners[3]));

    crop_height = 48;
    crop_width = std::max(16, static_cast<int>(std::round(src_width * crop_height / src_height)));
    crop_width = std::min(crop_width, 4096);
    crop_width = (crop_width + 3) / 4 * 4;

    std::vector<uint8_t> crop(static_cast<size_t>(crop_width) * crop_height * 3);
    for (int y = 0; y < crop_height; ++y) {
        const float v = (y + 0.5f) / crop_height;
        for (int x = 0; x < crop_width; ++x) {
            const float u = (x + 0.5f) / crop_width;
            const PointF src{
                box.corners[0].x + (box.corners[1].x - box.corners[0].x) * u +
                    (box.corners[3].x - box.corners[0].x) * v,
                box.corners[0].y + (box.corners[1].y - box.corners[0].y) * u +
                    (box.corners[3].y - box.corners[0].y) * v,
            };

            uint8_t* dst = crop.data() + (static_cast<size_t>(y) * crop_width + x) * 3;
            dst[0] = sample_bgr(bgr, image_width, image_height, src.x, src.y, 0);
            dst[1] = sample_bgr(bgr, image_width, image_height, src.x, src.y, 1);
            dst[2] = sample_bgr(bgr, image_width, image_height, src.x, src.y, 2);
        }
    }

    return crop;
}

static std::vector<uint8_t> rotate_90_counterclockwise_bgr(const uint8_t* bgr, int width, int height,
                                                           int& rotated_width, int& rotated_height) {
    rotated_width = height;
    rotated_height = width;
    std::vector<uint8_t> rotated(static_cast<size_t>(rotated_width) * rotated_height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const uint8_t* src = bgr + (static_cast<size_t>(y) * width + x) * 3;
            uint8_t* dst = rotated.data() + (static_cast<size_t>(width - 1 - x) * rotated_width + y) * 3;
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
        }
    }
    return rotated;
}

static std::vector<uint8_t> rotate_180_bgr(const uint8_t* bgr, int width, int height) {
    std::vector<uint8_t> rotated(static_cast<size_t>(width) * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const uint8_t* src = bgr + (static_cast<size_t>(y) * width + x) * 3;
            uint8_t* dst = rotated.data() + (static_cast<size_t>(height - 1 - y) * width + (width - 1 - x)) * 3;
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
        }
    }
    return rotated;
}

static std::vector<uint8_t> make_cls_input_bgr(const uint8_t* bgr, int width, int height, int& out_width,
                                               int& out_height) {
    constexpr int target_width = 160;
    constexpr int target_height = 80;
    constexpr float max_downscale = 3.f;

    const float ratio = static_cast<float>(target_height) / height;
    int resized_width = static_cast<int>(width * ratio);
    out_width = target_width;
    out_height = target_height;

    if (resized_width < target_width) {
        std::vector<uint8_t> output(static_cast<size_t>(target_width) * target_height * 3, 114);
        std::vector<uint8_t> resized(static_cast<size_t>(std::max(1, resized_width)) * target_height * 3);
        ncnn::Mat::from_pixels_resize(bgr, ncnn::Mat::PIXEL_BGR, width, height, std::max(1, resized_width),
                                      target_height)
            .to_pixels(resized.data(), ncnn::Mat::PIXEL_BGR);
        for (int y = 0; y < target_height; ++y) {
            std::copy(resized.data() + static_cast<size_t>(y) * resized_width * 3,
                      resized.data() + (static_cast<size_t>(y) * resized_width + resized_width) * 3,
                      output.data() + static_cast<size_t>(y) * target_width * 3);
        }
        return output;
    }

    int crop_width = width;
    if (resized_width >= static_cast<int>(target_width * max_downscale)) {
        crop_width = std::max(1, static_cast<int>(max_downscale * target_width / ratio));
    }

    std::vector<uint8_t> output(static_cast<size_t>(target_width) * target_height * 3);
    ncnn::Mat::from_pixels_resize(bgr, ncnn::Mat::PIXEL_BGR, crop_width, height, target_width, target_height)
        .to_pixels(output.data(), ncnn::Mat::PIXEL_BGR);
    return output;
}

static std::vector<RotatedTextBox> detect_text_boxes(ncnn::Net& det_net, const std::vector<uint8_t>& bgr, int width,
                                                     int height) {
    const int target_size = 768;
    const int target_stride = 32;

    int resized_width = width;
    int resized_height = height;
    float scale = 1.f;
    if (std::max(resized_width, resized_height) > target_size) {
        if (resized_width > resized_height) {
            scale = static_cast<float>(target_size) / resized_width;
            resized_width = target_size;
            resized_height = static_cast<int>(resized_height * scale);
        } else {
            scale = static_cast<float>(target_size) / resized_height;
            resized_height = target_size;
            resized_width = static_cast<int>(resized_width * scale);
        }
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data(), ncnn::Mat::PIXEL_BGR, width, height, resized_width,
                                                 resized_height);

    const int wpad = (resized_width + target_stride - 1) / target_stride * target_stride - resized_width;
    const int hpad = (resized_height + target_stride - 1) / target_stride * target_stride - resized_height;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2,
                           ncnn::BORDER_CONSTANT, 114.f);

    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const float norm_vals[3] = {1.f / 0.229f / 255.f, 1.f / 0.224f / 255.f, 1.f / 0.225f / 255.f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Mat out;
    if (run_net(det_net, in_pad, out) != 0) {
        std::cerr << "[NcnnOcrEngine] Failed to extract detection output" << std::endl;
        return {};
    }

    // DB 检测输出是文字概率图。这里按 LiteOCR/PaddleOCR 的常规后处理：
    // 二值连通区域 -> 最小外接旋转矩形 -> mask 平均分 -> unclip。
    const int map_width = out.w;
    const int map_height = out.h;
    std::vector<uint8_t> visited(static_cast<size_t>(map_width) * map_height, 0);
    std::vector<RotatedTextBox> boxes;

    const float threshold = 0.3f;
    const int min_area = 6;
    const float box_thresh = 0.6f;
    const float unclip_ratio = 1.5f;
    const float min_size = 3.f;

    for (int y = 0; y < map_height; ++y) {
        const float* row = out.row(y);
        for (int x = 0; x < map_width; ++x) {
            const size_t index = static_cast<size_t>(y) * map_width + x;
            if (visited[index] || row[x] < threshold)
                continue;

            int count = 0;
            std::vector<PointI> component_points;
            std::queue<std::pair<int, int>> pending;
            pending.push({x, y});
            visited[index] = 1;

            while (!pending.empty()) {
                const auto [cx, cy] = pending.front();
                pending.pop();

                count++;
                component_points.push_back({cx, cy});

                constexpr int dx[4] = {1, -1, 0, 0};
                constexpr int dy[4] = {0, 0, 1, -1};
                for (int k = 0; k < 4; ++k) {
                    const int nx = cx + dx[k];
                    const int ny = cy + dy[k];
                    if (nx < 0 || ny < 0 || nx >= map_width || ny >= map_height)
                        continue;

                    const size_t ni = static_cast<size_t>(ny) * map_width + nx;
                    if (visited[ni] || out.row(ny)[nx] < threshold)
                        continue;

                    visited[ni] = 1;
                    pending.push({nx, ny});
                }
            }

            if (count < min_area)
                continue;

            const RotatedRect rect = min_area_rect(component_points);
            if (std::min(rect.size.width, rect.size.height) < min_size)
                continue;

            const auto raw_array = order_box_points(rotated_rect_points(rect));
            const std::vector<PointF> raw_box(raw_array.begin(), raw_array.end());
            const float component_score = box_score(out, raw_box);
            if (component_score < box_thresh)
                continue;

            const auto expanded = unclip_box(raw_box, unclip_ratio);
            std::vector<PointI> expanded_contour;
            expanded_contour.reserve(expanded.size());
            for (const auto& p : expanded) {
                expanded_contour.push_back({static_cast<int>(std::round(p.x)), static_cast<int>(std::round(p.y))});
            }
            const RotatedRect expanded_rect = min_area_rect(expanded_contour);
            if (std::min(expanded_rect.size.width, expanded_rect.size.height) < min_size + 2.f)
                continue;

            const auto mapped_box = order_box_points(rotated_rect_points(expanded_rect));

            const float sx = static_cast<float>(in_pad.w) / map_width;
            const float sy = static_cast<float>(in_pad.h) / map_height;
            const auto to_original = [&](const PointF& p) -> PointF {
                return {
                    std::clamp((p.x * sx - wpad / 2.f) / scale, 0.f, static_cast<float>(width - 1)),
                    std::clamp((p.y * sy - hpad / 2.f) / scale, 0.f, static_cast<float>(height - 1)),
                };
            };

            RotatedTextBox box;
            for (size_t i = 0; i < box.corners.size(); ++i) {
                box.corners[i] = to_original(mapped_box[i]);
            }
            box.score = component_score;
            boxes.push_back(box);
        }
    }

    std::sort(boxes.begin(), boxes.end(), [](const RotatedTextBox& a, const RotatedTextBox& b) {
        const auto a_points = order_box_points(a.corners);
        const auto b_points = order_box_points(b.corners);
        if (std::abs(a_points[0].y - b_points[0].y) > 10.f)
            return a_points[0].y < b_points[0].y;
        return a_points[0].x < b_points[0].x;
    });

    return boxes;
}

static bool recognize_crop(ncnn::Net& rec_net, const std::vector<std::string>& dict, const uint8_t* bgr, int width,
                           int height, std::string& text, float& confidence) {
    const int target_height = 48;
    int target_width = std::max(16, static_cast<int>(std::round(width * (target_height / static_cast<float>(height)))));
    target_width = std::min(target_width, 4096);
    target_width = (target_width + 3) / 4 * 4;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr, ncnn::Mat::PIXEL_BGR, width, height, target_width,
                                                 target_height);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Mat out;
    if (run_net(rec_net, in, out) != 0) {
        std::cerr << "[NcnnOcrEngine] Failed to extract recognition output" << std::endl;
        return false;
    }

    const auto tokens = decode_ctc_tokens(out);
    text = decode_text(dict, tokens, confidence);
    return !text.empty();
}

static bool classify_needs_180(ncnn::Net& cls_net, const uint8_t* bgr, int width, int height, bool& needs_180,
                               float& score) {
    // 方向分类只判断 0/180 度，避免倒置文本被识别成乱码。
    int cls_width = 0;
    int cls_height = 0;
    const auto cls_input = make_cls_input_bgr(bgr, width, height, cls_width, cls_height);
    ncnn::Mat in = ncnn::Mat::from_pixels(cls_input.data(), ncnn::Mat::PIXEL_BGR, cls_width, cls_height);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Mat out;
    if (run_net(cls_net, in, out) != 0) {
        std::cerr << "[NcnnOcrEngine] Failed to extract classifier output" << std::endl;
        return false;
    }

    const float* values = reinterpret_cast<const float*>(out.data);
    const int count = out.w * out.h * out.c;
    if (count < 2)
        return false;

    int best_index = 0;
    float best_score = values[0];
    for (int i = 1; i < count; ++i) {
        if (values[i] > best_score) {
            best_score = values[i];
            best_index = i;
        }
    }

    needs_180 = best_index == 1;
    score = best_score;
    return true;
}

static bool recognize_crop_with_classifier(ncnn::Net& rec_net, ncnn::Net* cls_net, const std::vector<std::string>& dict,
                                           const uint8_t* bgr, int width, int height, bool force_180,
                                           bool fast_mode, std::string& text, float& confidence) {
    // 竖排长框先转成横向，再交给 PP-OCR 识别模型处理。
    if (height >= static_cast<int>(width * 1.5f)) {
        int rotated_width = 0;
        int rotated_height = 0;
        const auto rotated = rotate_90_counterclockwise_bgr(bgr, width, height, rotated_width, rotated_height);
        return recognize_crop_with_classifier(rec_net, cls_net, dict, rotated.data(), rotated_width, rotated_height,
                                              force_180, fast_mode, text, confidence);
    }

    if (!fast_mode && force_180) {
        const auto rotated = rotate_180_bgr(bgr, width, height);
        return recognize_crop(rec_net, dict, rotated.data(), width, height, text, confidence);
    }

    bool prefer_180 = false;
    if (!fast_mode && cls_net != nullptr) {
        float cls_score = 0.f;
        classify_needs_180(*cls_net, bgr, width, height, prefer_180, cls_score);
    }

    std::string first_text;
    float first_confidence = 0.f;
    std::string fallback_text;
    float fallback_confidence = 0.f;

    if (prefer_180) {
        const auto rotated = rotate_180_bgr(bgr, width, height);
        const bool first_ok = recognize_crop(rec_net, dict, rotated.data(), width, height, first_text, first_confidence);
        if (first_ok && first_confidence >= 0.80f) {
            text = std::move(first_text);
            confidence = first_confidence;
            return true;
        }

        const bool fallback_ok = recognize_crop(rec_net, dict, bgr, width, height, fallback_text, fallback_confidence);
        if (fallback_ok && (!first_ok || fallback_confidence > first_confidence)) {
            text = std::move(fallback_text);
            confidence = fallback_confidence;
            return true;
        }

        if (first_ok) {
            text = std::move(first_text);
            confidence = first_confidence;
        }
        return first_ok;
    }

    const bool first_ok = recognize_crop(rec_net, dict, bgr, width, height, first_text, first_confidence);
    if (fast_mode) {
        text = std::move(first_text);
        confidence = first_confidence;
        return first_ok;
    }
    if (first_ok && first_confidence >= 0.80f) {
        text = std::move(first_text);
        confidence = first_confidence;
        return true;
    }

    const auto rotated = rotate_180_bgr(bgr, width, height);
    const bool fallback_ok = recognize_crop(rec_net, dict, rotated.data(), width, height, fallback_text, fallback_confidence);
    if (fallback_ok && (!first_ok || fallback_confidence > first_confidence)) {
        text = std::move(fallback_text);
        confidence = fallback_confidence;
        return true;
    }

    if (first_ok) {
        text = std::move(first_text);
        confidence = first_confidence;
    }
    return first_ok;
}

} // namespace
#endif

struct NcnnOcrEngine::Impl {
    NcnnOcrModelPaths model_paths;
    NcnnOcrOptions options;
    std::vector<std::string> character_dict;
    bool initialized = false;
    bool use_vulkan = false;
    bool cls_initialized = false;

#ifdef OCR_HAS_NCNN
    ncnn::Net det_net;
    ncnn::Net rec_net;
    ncnn::Net cls_net;
#endif
};

NcnnOcrEngine::NcnnOcrEngine() : impl_(std::make_unique<Impl>()) {}

NcnnOcrEngine::~NcnnOcrEngine() = default;

const char* NcnnOcrEngine::version() {
#ifdef OCR_HAS_NCNN
    return "ncnn-ocr-service 1.0";
#else
    return "ncnn-ocr-service unavailable (ncnn not linked)";
#endif
}

bool NcnnOcrEngine::init(const NcnnOcrModelPaths& model_paths, bool use_vulkan) {
    NcnnOcrOptions options;
    options.use_vulkan = use_vulkan;
    options.gpu_device_id = use_vulkan ? 0 : -1;
    return init(model_paths, options);
}

bool NcnnOcrEngine::init(const NcnnOcrModelPaths& model_paths, const NcnnOcrOptions& options) {
#ifndef OCR_HAS_NCNN
    (void)model_paths;
    (void)options;
    std::cerr << "[NcnnOcrEngine] ncnn support is not enabled in this build" << std::endl;
    return false;
#else
    const auto require_file = [](const std::string& path, const char* label) -> bool {
        if (path.empty()) {
            std::cerr << "[NcnnOcrEngine] Missing path for " << label << std::endl;
            return false;
        }
        if (!fs::exists(path)) {
            std::cerr << "[NcnnOcrEngine] File not found for " << label << ": " << path << std::endl;
            return false;
        }
        return true;
    };

    if (!require_file(model_paths.det_param, "det_param") ||
        !require_file(model_paths.det_bin, "det_bin") ||
        !require_file(model_paths.rec_param, "rec_param") ||
        !require_file(model_paths.rec_bin, "rec_bin")) {
        return false;
    }

    const fs::path dict_path = model_paths.dict_path.empty()
        ? fs::path(model_paths.rec_param).parent_path() / "PP-OCRv5_vocab.txt"
        : fs::path(model_paths.dict_path);
    if (!require_file(dict_path.string(), "dict_path")) {
        return false;
    }
    impl_->character_dict = load_character_dict(dict_path);
    if (impl_->character_dict.empty()) {
        std::cerr << "[NcnnOcrEngine] Character dict is empty: " << dict_path.string() << std::endl;
        return false;
    }

    impl_->det_net.clear();
    impl_->rec_net.clear();
    impl_->cls_net.clear();

    // det/rec/cls 共用同一套推理选项，Vulkan 设备只探测一次，避免重复告警。
    bool enable_vulkan = false;
    int vulkan_device_id = -1;
#if NCNN_VULKAN
    if (options.use_vulkan || options.gpu_device_id >= 0) {
        const int gpu_count = ncnn::get_gpu_count();
        if (gpu_count > 0) {
            const int requested_device = options.gpu_device_id >= 0 ? options.gpu_device_id : 0;
            if (requested_device < gpu_count) {
                enable_vulkan = true;
                vulkan_device_id = requested_device;
            } else {
                std::cerr << "[NcnnOcrEngine] GPU device " << requested_device
                          << " is unavailable; falling back to CPU" << std::endl;
            }
        } else {
            std::cerr << "[NcnnOcrEngine] Vulkan requested but no GPU device is available; falling back to CPU"
                      << std::endl;
        }
    }
#else
    if (options.use_vulkan || options.gpu_device_id >= 0) {
        std::cerr << "[NcnnOcrEngine] Vulkan requested but this ncnn build has no Vulkan support; falling back to CPU"
                  << std::endl;
    }
#endif

    auto apply_options = [&options, enable_vulkan, vulkan_device_id](ncnn::Net& net) {
        net.opt.num_threads = std::max(1, options.num_threads);
        net.opt.use_fp16_arithmetic = options.use_fp16;
        net.opt.use_fp16_storage = options.use_fp16;
        net.opt.use_fp16_packed = options.use_fp16;
        if (options.use_bf16) {
            net.opt.use_fp16_arithmetic = false;
            net.opt.use_fp16_storage = false;
            net.opt.use_fp16_packed = false;
            net.opt.use_bf16_storage = true;
            net.opt.use_bf16_packed = true;
        }
#if NCNN_VULKAN
        if (enable_vulkan) {
            net.set_vulkan_device(vulkan_device_id);
            net.opt.use_vulkan_compute = true;
        }
#endif
    };
    apply_options(impl_->det_net);
    apply_options(impl_->rec_net);
    apply_options(impl_->cls_net);

    if (impl_->det_net.load_param(model_paths.det_param.c_str()) != 0 ||
        impl_->det_net.load_model(model_paths.det_bin.c_str()) != 0) {
        std::cerr << "[NcnnOcrEngine] Failed to load detection model" << std::endl;
        return false;
    }

    if (impl_->rec_net.load_param(model_paths.rec_param.c_str()) != 0 ||
        impl_->rec_net.load_model(model_paths.rec_bin.c_str()) != 0) {
        std::cerr << "[NcnnOcrEngine] Failed to load recognition model" << std::endl;
        return false;
    }

    impl_->cls_initialized = false;
    if (options.fast_mode) {
        impl_->cls_net.clear();
    } else if (!model_paths.cls_param.empty() && !model_paths.cls_bin.empty() &&
        fs::exists(model_paths.cls_param) && fs::exists(model_paths.cls_bin)) {
        if (impl_->cls_net.load_param(model_paths.cls_param.c_str()) != 0 ||
            impl_->cls_net.load_model(model_paths.cls_bin.c_str()) != 0) {
            std::cerr << "[NcnnOcrEngine] Failed to load direction classifier model; continuing without cls"
                      << std::endl;
            impl_->cls_net.clear();
        } else {
            impl_->cls_initialized = true;
        }
    } else {
        std::cerr << "[NcnnOcrEngine] Direction classifier model not found; continuing without cls" << std::endl;
    }

    impl_->model_paths = model_paths;
    impl_->options = options;
    impl_->use_vulkan = impl_->det_net.opt.use_vulkan_compute;
    impl_->initialized = true;

    std::cout << "[NcnnOcrEngine] Initialized"
              << " det=" << model_paths.det_param
              << " rec=" << model_paths.rec_param
              << " dict=" << dict_path.string()
              << " cls=" << (impl_->cls_initialized ? model_paths.cls_param : "disabled")
              << " threads=" << std::max(1, options.num_threads)
              << " gpu_device=" << options.gpu_device_id
              << " fp16=" << (options.use_fp16 ? "true" : "false")
              << " bf16=" << (options.use_bf16 ? "true" : "false")
              << " fast_mode=" << (options.fast_mode ? "true" : "false")
              << " use_vulkan=" << (impl_->use_vulkan ? "true" : "false")
              << std::endl;
    return true;
#endif
}

std::vector<OcrResult> NcnnOcrEngine::recognize(const uint8_t* image, int width, int height, int bpp) {
    return recognize_with_profile(image, width, height, bpp).results;
}

OcrResponse NcnnOcrEngine::recognize_with_profile(const uint8_t* image, int width, int height, int bpp) {
    OcrResponse response;
    const auto total_start = Clock::now();

    if (!impl_->initialized) {
        std::cerr << "[NcnnOcrEngine] Engine not initialized" << std::endl;
        return response;
    }

    if (image == nullptr || width <= 0 || height <= 0 || (bpp != 1 && bpp != 3 && bpp != 4)) {
        std::cerr << "[NcnnOcrEngine] Invalid image buffer" << std::endl;
        return response;
    }

#ifndef OCR_HAS_NCNN
    return response;
#else
    const auto bgr = to_bgr_pixels(image, width, height, bpp);
    const auto det_start = Clock::now();
    const auto boxes = detect_text_boxes(impl_->det_net, bgr, width, height);
    response.profile.det_ms = elapsed_ms(det_start, Clock::now());
    const bool fast_mode = impl_->options.fast_mode;
    ncnn::Net* cls_net = (!fast_mode && impl_->cls_initialized) ? &impl_->cls_net : nullptr;

    // 先用所有文本行投票判断整张图是否整体倒置，再逐行识别。
    bool force_180 = false;
    if (cls_net != nullptr && !boxes.empty()) {
        const auto cls_start = Clock::now();
        float rot_weight = 0.f;
        float normal_weight = 0.f;
        for (const auto& box : boxes) {
            int crop_width = 0;
            int crop_height = 0;
            const auto crop = warp_text_crop(bgr, width, height, box, crop_width, crop_height);
            bool needs_180 = false;
            float cls_score = 0.f;
            if (classify_needs_180(*cls_net, crop.data(), crop_width, crop_height, needs_180, cls_score)) {
                if (needs_180)
                    rot_weight += cls_score;
                else
                    normal_weight += cls_score;
            }
        }
        force_180 = rot_weight > normal_weight;
        response.profile.cls_ms += elapsed_ms(cls_start, Clock::now());
    }

    for (const auto& box : boxes) {
        int crop_width = 0;
        int crop_height = 0;
        const auto crop = warp_text_crop(bgr, width, height, box, crop_width, crop_height);

        // 每个文本框单独裁剪识别，结果 bbox 保持原图坐标，方便上层点击。
        std::string text;
        float confidence = 0.f;
        const auto rec_start = Clock::now();
        if (!recognize_crop_with_classifier(impl_->rec_net, cls_net, impl_->character_dict, crop.data(), crop_width,
                                            crop_height, force_180, fast_mode, text, confidence)) {
            response.profile.rec_ms += elapsed_ms(rec_start, Clock::now());
            continue;
        }
        response.profile.rec_ms += elapsed_ms(rec_start, Clock::now());

        int x1 = 0;
        int y1 = 0;
        int x2 = 0;
        int y2 = 0;
        axis_aligned_bounds(box, width, height, x1, y1, x2, y2);

        OcrResult result;
        result.x1 = x1;
        result.y1 = y1;
        result.x2 = x2;
        result.y2 = y2;
        result.text = text;
        result.confidence = confidence;
        response.results.push_back(std::move(result));
    }

    response.profile.total_ms = elapsed_ms(total_start, Clock::now());
    return response;
#endif
}

} // namespace ocr
