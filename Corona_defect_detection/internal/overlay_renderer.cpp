#include "overlay_renderer.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <sstream>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "HalconCpp.h"

#ifdef _WIN32
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif

namespace InspectionOverlay {
namespace {

// 辅助函数：将文本适配到指定宽度
std::string FitTextToWidth(const std::string& text, int max_width, double font_scale, int thickness) {
    if (max_width <= 0) return std::string();
    int baseline = 0;
    if (cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline).width <= max_width) {
        return text;
    }

    const std::string ellipsis = "...";
    const int ellipsis_w = cv::getTextSize(ellipsis, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline).width;
    if (ellipsis_w > max_width) return std::string();

    int lo = 0;
    int hi = static_cast<int>(text.size());
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        const std::string cand = text.substr(0, static_cast<size_t>(mid)) + ellipsis;
        const int w = cv::getTextSize(cand, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline).width;
        if (w <= max_width) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    if (lo <= 0) return ellipsis;
    return text.substr(0, static_cast<size_t>(lo)) + ellipsis;
}

// 辅助函数：绘制标签块
void DrawLabelBlock(
    cv::Mat& vis,
    const cv::Rect& r,
    const std::vector<std::string>& lines,
    const cv::Scalar& bg_color,
    int img_w,
    int img_h
) {
    if (vis.empty() || lines.empty() || img_w <= 0 || img_h <= 0) return;

    const double font_scale = 0.55;
    const int thickness = 1;
    const int padding = 3;
    const int line_gap = 2;

    int max_line_w = 0;
    std::vector<cv::Size> sizes;
    std::vector<int> baselines;
    sizes.reserve(lines.size());
    baselines.reserve(lines.size());

    for (const auto& line : lines) {
        int baseline = 0;
        const cv::Size ts = cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
        sizes.push_back(ts);
        baselines.push_back(baseline);
        max_line_w = std::max(max_line_w, ts.width);
    }

    int block_w = max_line_w + padding * 2;
    int block_h = padding * 2;
    for (size_t i = 0; i < lines.size(); ++i) {
        block_h += sizes[i].height + baselines[i];
        if (i + 1 < lines.size()) block_h += line_gap;
    }

    if (block_w > img_w) block_w = img_w;
    if (block_h > img_h) block_h = img_h;

    int x = r.x + r.width + 3;
    int y = r.y;

    if (x + block_w > img_w) x = r.x - block_w - 3;
    if (x < 0) x = std::max(0, std::min(r.x, img_w - block_w));
    if (y + block_h > img_h) y = std::max(0, img_h - block_h);
    if (y < 0) y = 0;

    const int max_text_w = std::max(0, block_w - padding * 2);
    int cursor_y = y + padding;
    for (size_t i = 0; i < lines.size(); ++i) {
        const std::string t = FitTextToWidth(lines[i], max_text_w, font_scale, thickness);
        int baseline = 0;
        const cv::Size ts = cv::getTextSize(t, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
        const int org_x = x + padding;
        const int org_y = cursor_y + ts.height;
        if (org_x >= 0 && org_x < img_w && org_y >= 0 && org_y < img_h) {
            cv::putText(vis, t, cv::Point(org_x, org_y), cv::FONT_HERSHEY_SIMPLEX, font_scale, bg_color, thickness);
        }
        cursor_y += ts.height + baseline + line_gap;
        if (cursor_y >= y + block_h) break;
    }
}

cv::Mat BuildFullImageMask(const cv::Mat& source, const cv::Rect& detection, const cv::Size& image_size) {
    if (source.empty() || source.channels() != 1 || detection.empty()) return {};

    cv::Mat binary;
    source.convertTo(binary, CV_8U);
    cv::threshold(binary, binary, 0, 255, cv::THRESH_BINARY);

    cv::Mat full_mask(image_size, CV_8U, cv::Scalar(0));
    if (binary.size() == image_size) {
        binary(detection).copyTo(full_mask(detection));
    } else {
        cv::Mat local;
        if (binary.size() == detection.size()) {
            local = binary;
        } else {
            cv::resize(binary, local, detection.size(), 0.0, 0.0, cv::INTER_NEAREST);
        }
        local.copyTo(full_mask(detection));
    }
    return full_mask;
}

void DrawDefectMask(cv::Mat& vis,
                    const cv::Mat& source_mask,
                    const cv::Rect& detection,
                    const InspectionDLL::MaskOverlayOptions& options) {
    const cv::Mat mask = BuildFullImageMask(source_mask, detection, vis.size());
    if (mask.empty() || cv::countNonZero(mask) == 0) return;

    cv::Mat color_layer(vis.size(), vis.type(),
                        cv::Scalar(options.color_b, options.color_g, options.color_r));
    cv::Mat blended;
    cv::addWeighted(vis, 1.0 - options.alpha, color_layer, options.alpha, 0.0, blended);
    blended.copyTo(vis, mask);
}

}  // namespace

// 渲染推理结果到Halcon图像句柄
bool RenderResultOverlayToHalconHandle(
    const float* image_array,
    int32_t width,
    int32_t height,
    const InspectionDLL::InferenceResult& result,
    bool draw_defect_box,
    bool draw_box_details,
    const InspectionDLL::MaskOverlayOptions& mask_options,
    int32_t o_imageHandle[1],
    std::string& err) {
    if (!image_array) {
        err = "overlay input image_array is nullptr";
        return false;
    }
    if (width <= 0 || height <= 0) {
        err = "overlay input width or height is invalid";
        return false;
    }

    try {
        cv::Mat img32(height, width, CV_32FC1, const_cast<float*>(image_array));

        double min_val = 0.0;
        double max_val = 0.0;
        cv::minMaxLoc(img32, &min_val, &max_val);

        cv::Mat img_u8;
        if (max_val > min_val + 1e-12) {
            cv::Mat norm = (img32 - min_val) / (max_val - min_val);
            norm.convertTo(img_u8, CV_8U, 255.0);
        } else {
            img_u8 = cv::Mat::zeros(img32.size(), CV_8U);
        }

        cv::Mat vis;
        cv::cvtColor(img_u8, vis, cv::COLOR_GRAY2BGR);

        const cv::Scalar color_ng(0, 51, 255);
        const cv::Scalar color_ok(0, 255, 0);
        const bool is_ok = (result.result == "OK");
        const cv::Scalar color = is_ok ? color_ok : color_ng;

        {
            const std::string res_text = is_ok ? "OK" : "NG";
            int baseline = 0;
            const double font_scale = 0.9;
            const int thickness = 2;
            const cv::Size ts = cv::getTextSize(res_text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);

            const int padding = 6;

            cv::putText(
                vis,
                res_text,
                cv::Point(padding, padding + ts.height),
                cv::FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness
            );
        }

        const size_t det_n = result.details.size();
        if (mask_options.enabled && mask_options.alpha > 0.0f) {
            for (size_t i = 0; i < det_n; ++i) {
                const auto& d = result.details[i];
                cv::Rect r(
                    static_cast<int>(std::round(d.x)),
                    static_cast<int>(std::round(d.y)),
                    static_cast<int>(std::round(d.w)),
                    static_cast<int>(std::round(d.h)));
                r &= cv::Rect(0, 0, width, height);
                if (!r.empty()) DrawDefectMask(vis, d.mask, r, mask_options);
            }
        }

        for (size_t i = 0; i < det_n; ++i) {
            const auto& d = result.details[i];
            const int x = static_cast<int>(std::round(d.x));
            const int y = static_cast<int>(std::round(d.y));
            const int w = static_cast<int>(std::round(d.w));
            const int h = static_cast<int>(std::round(d.h));

            if (w <= 0 || h <= 0) continue;

            cv::Rect r(x, y, w, h);
            r &= cv::Rect(0, 0, width, height);
            if (r.width <= 0 || r.height <= 0) continue;

            if (draw_defect_box) {
                cv::rectangle(vis, r, color, 2);
            }

            if (draw_box_details) {
                std::string name = d.name;
                if (name.empty()) name = "defect";

                std::vector<std::string> lines;
                lines.reserve(3);
                {
                    std::ostringstream oss;
                    oss << name << " ";
                    if (name == "Abnormal") {
                        oss << "Score:" << std::fixed << std::setprecision(3) << d.score;
                    } else {
                        oss << "Conf:" << std::fixed << std::setprecision(3) << d.score;
                    }
                    lines.push_back(oss.str());
                }
                {
                    std::ostringstream oss;
                    oss << "Contrast:" << std::fixed << std::setprecision(2) << d.contrast << " Area:" << d.area;
                    lines.push_back(oss.str());
                }
                {
                    std::ostringstream oss;
                    oss << "W:" << w << " H:" << h;
                    lines.push_back(oss.str());
                }

                DrawLabelBlock(vis, r, lines, color, width, height);
            }
        }

        if (!o_imageHandle) {
            return true;
        }

        if (vis.empty()) {
            err = "overlay image is empty";
            return false;
        }
        if (vis.type() != CV_8UC3) {
            err = "overlay image must be CV_8UC3";
            return false;
        }

        cv::Mat continuous_image;
        const cv::Mat* image_to_serialize = &vis;
        if (!vis.isContinuous()) {
            continuous_image = vis.clone();
            image_to_serialize = &continuous_image;
        }

        HalconCpp::HObject ho_image;
        HalconCpp::HTuple hv_serialized;
        HalconCpp::GenImageInterleaved(
            &ho_image,
            reinterpret_cast<Hlong>(image_to_serialize->data),
            "bgr",
            image_to_serialize->cols,
            image_to_serialize->rows,
            0,
            "byte",
            image_to_serialize->cols,
            image_to_serialize->rows,
            0,
            0,
            -1,
            0
        );
        HalconCpp::SerializeObject(ho_image, &hv_serialized);
        o_imageHandle[0] = static_cast<int32_t>(hv_serialized[0].L());
        hv_serialized.Clear();
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    } catch (...) {
        err = "overlay rendering failed with unknown exception";
        return false;
    }
}

// 测试用辅助函数：将文本适配到指定宽度
std::string FitTextToWidthForTest(const std::string& text, int max_width, double font_scale, int thickness) {
    return FitTextToWidth(text, max_width, font_scale, thickness);
}

}  // namespace InspectionOverlay
