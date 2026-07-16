#include "inference_c_api.h"
#include "inference_dll.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <iomanip>

#include <opencv2/imgcodecs.hpp>
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



static bool EqualsIgnoreCase(const char* a, const char* b) {
    if (!a || !b) return false;
    while (*a && *b) {
        const unsigned char ca = static_cast<unsigned char>(*a);
        const unsigned char cb = static_cast<unsigned char>(*b);
        if (std::tolower(ca) != std::tolower(cb)) return false;
        ++a;
        ++b;
    }
    return *a == '\0' && *b == '\0';
}

static int32_t DefectNameToIndex(const char* name) {
    if (!name || name[0] == '\0') return -1;
    if (EqualsIgnoreCase(name, "Abnormal")) return static_cast<int32_t>(Defect_name::Abnormal);
    if (EqualsIgnoreCase(name, "Stain")) return static_cast<int32_t>(Defect_name::Stain);
    if (EqualsIgnoreCase(name, "BrightStripes")) return static_cast<int32_t>(Defect_name::BrightStripes);
    if (EqualsIgnoreCase(name, "LineArtifacts")) return static_cast<int32_t>(Defect_name::LineArtifacts);
    if (EqualsIgnoreCase(name, "DarkClusters")) return static_cast<int32_t>(Defect_name::DarkClusters);
    return -1;
}

void CopyExceptionMessage(const char* msg, char o_message[100])
{
    if (o_message == NULL)
        return;

    if (msg == NULL)
    {
        strcpy(o_message, "Unknown error");
        return;
    }

#ifdef _MSC_VER
    strncpy_s(o_message, 100, msg, _TRUNCATE);
#else
    strncpy(o_message, msg, 99);
    o_message[99] = '\0';
#endif
}

static InspectionDLL::InspectionEngine* ToEngine(InspectionHandle handle) {
    return reinterpret_cast<InspectionDLL::InspectionEngine*>(handle);
}

static void ZeroResult(InspectionResultC* out_result) {
    if (!out_result) return;
    std::memset(out_result, 0, sizeof(InspectionResultC));
}

static void CopyName(char* dst, size_t dst_size, const std::string& src) {
    if (!dst || dst_size == 0) return;
#ifdef _WIN32
    strncpy_s(dst, dst_size, src.c_str(), _TRUNCATE);
#else
    std::strncpy(dst, src.c_str(), dst_size - 1);
    dst[dst_size - 1] = '\0';
#endif
}

static void FillResult(const InspectionDLL::InferenceResult& src, InspectionResultC* dst) {
    if (!dst) return;

    dst->result = (src.result == "OK") ? INSPECTION_RESULT_OK : INSPECTION_RESULT_NG;
    dst->yolo_score = src.yolo_score;
    dst->patchcore_score = src.patchcore_score;
    dst->patchcore_area_ratio = src.patchcore_area_ratio;

    const int32_t n = static_cast<int32_t>(std::min<size_t>(src.details.size(), INSPECTION_MAX_DETECTIONS));
    dst->details_count = n;
    for (int32_t i = 0; i < n; ++i) {
        const auto& s = src.details[static_cast<size_t>(i)];
        auto& d = dst->details[i];
        std::memset(d.name, 0, sizeof(d.name));
        CopyName(d.name, sizeof(d.name), s.name);
        d.area = static_cast<int32_t>(s.area);
        d.x = s.x;
        d.y = s.y;
        d.w = s.w;
        d.h = s.h;
        d.contrast = s.contrast;
    }
}

static std::string FitTextToWidth(const std::string& text, int max_width, double font_scale, int thickness) {
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

static void DrawLabelBlock(
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

    const cv::Rect bg(x, y, block_w, block_h);
    cv::rectangle(vis, bg, bg_color, cv::FILLED);

    const int max_text_w = std::max(0, block_w - padding * 2);
    int cursor_y = y + padding;
    for (size_t i = 0; i < lines.size(); ++i) {
        const std::string t = FitTextToWidth(lines[i], max_text_w, font_scale, thickness);
        int baseline = 0;
        const cv::Size ts = cv::getTextSize(t, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
        const int org_x = x + padding;
        const int org_y = cursor_y + ts.height;
        if (org_x >= 0 && org_x < img_w && org_y >= 0 && org_y < img_h) {
            cv::putText(vis, t, cv::Point(org_x, org_y), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 0, 0), thickness);
        }
        cursor_y += ts.height + baseline + line_gap;
        if (cursor_y >= y + block_h) break;
    }
}

static bool SaveResultOverlay(
    const float* image_array,
    int32_t width,
    int32_t height,
    const InspectionDLL::InferenceResult& result,
    bool draw_box_details,
    int32_t o_imageHandle[1],
    char o_message[100]
) {

    if (!image_array || width <= 0 || height <= 0) return false;

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
            const int tx = 0;
            const int ty = 0;
            const int bg_w = std::min(ts.width + padding * 2, width);
            const int bg_h = std::min(ts.height + baseline + padding * 2, height);
            const cv::Rect bg(tx, ty, bg_w, bg_h);

            cv::rectangle(vis, bg, color, cv::FILLED);
            cv::putText(
                vis,
                res_text,
                cv::Point(tx + padding, ty + padding + ts.height),
                cv::FONT_HERSHEY_SIMPLEX,
                font_scale,
                cv::Scalar(0, 0, 0),
                thickness
            );
        }

        const size_t det_n = std::min<size_t>(result.details.size(), INSPECTION_MAX_DETECTIONS);
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

            cv::rectangle(vis, r, color, 2);

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
        
        if (o_imageHandle) {
            try {
                if (vis.empty()) {
                    CopyExceptionMessage("input image is empty", o_message);
                    return false;
                }

                if (vis.type() != CV_8UC3) {
                    CopyExceptionMessage("input image must be CV_8UC3", o_message);
                    return false;
                }

                // GenImageInterleaved 要求连续内存
                cv::Mat continuousImage;
                const cv::Mat* pImage = &vis;
                if (!vis.isContinuous()) {
                    continuousImage = vis.clone();
                    pImage = &continuousImage;
                }

                HalconCpp::HObject ho_image;
                HalconCpp::HTuple hv_serialized;

                HalconCpp::GenImageInterleaved(
                    &ho_image,
                    reinterpret_cast<Hlong>(pImage->data),
                    "bgr",
                    pImage->cols,
                    pImage->rows,
                    0,
                    "byte",
                    pImage->cols,
                    pImage->rows,
                    0,
                    0,
                    -1,
                    0
                );
                HalconCpp::SerializeObject(ho_image, &hv_serialized);
                o_imageHandle[0] = static_cast<int32_t>(hv_serialized[0].L());
                hv_serialized.Clear();
                return true;
            }
            catch (...) {
                CopyExceptionMessage("halcon serialize cv::Mat wrong", o_message);
                return false;
            }
        }

        return true;
    } catch (...) {
        return false;
    }
}

static bool SaveInputTiff32F(
    const float* image_array,
    int32_t array_length,
    int32_t width,
    int32_t height,
    const char* input_tif_path
) {
    if (!input_tif_path || input_tif_path[0] == '\0') return false;
    if (!image_array || width <= 0 || height <= 0) return false;

    const int64_t expected = static_cast<int64_t>(width) * static_cast<int64_t>(height);
    if (expected <= 0 || expected > INT32_MAX) return false;
    if (array_length != static_cast<int32_t>(expected)) return false;

    try {
        cv::Mat img32(height, width, CV_32FC1);
        std::memcpy(img32.ptr<float>(0), image_array, static_cast<size_t>(array_length) * sizeof(float));
        return cv::imwrite(input_tif_path, img32);
        
    } catch (...) {
        return false;
    }
}

INSPECTION_C_EXPORT InspectionHandle INSPECTION_CALL Inspection_Create() {
    try {
        return reinterpret_cast<InspectionHandle>(new InspectionDLL::InspectionEngine());
    } catch (...) {
        return nullptr;
    }
}

INSPECTION_C_EXPORT void INSPECTION_CALL Inspection_Destroy(InspectionHandle handle) {
    auto* eng = ToEngine(handle);
    delete eng;
}

INSPECTION_C_EXPORT int32_t INSPECTION_CALL Inspection_Initialize(
    InspectionHandle handle,
    const char* i_config_path,
    char* o_message
) {
    auto* eng = ToEngine(handle);
    if (!eng) {
        CopyExceptionMessage("to engine fail", o_message);
        return 1;
    }
    if (!i_config_path) {
        CopyExceptionMessage("config path is empty", o_message);
        return 1;
    }

    const bool ok = eng->Initialize(std::string(i_config_path));

    if (!ok) {
        CopyExceptionMessage("model initialize fail", o_message);
        return 1;
    }

    CopyExceptionMessage("model initilize success", o_message);
    return 0;  
 
}

INSPECTION_C_EXPORT int32_t INSPECTION_CALL Inspection_ProcessImagePath(
    InspectionHandle handle,
    const char* image_path,
    InspectionResultC* out_result
) {
    ZeroResult(out_result);
    auto* eng = ToEngine(handle);
    if (!eng || !image_path || !out_result) return 1;

    InspectionDLL::InferenceResult tmp;
    const bool ok = eng->ProcessImagePath(std::string(image_path), tmp);
    if (!ok) return 1;
    FillResult(tmp, out_result);
    return 0;
}

INSPECTION_C_EXPORT int32_t INSPECTION_CALL Inspection_ProcessFloatArray(
    InspectionHandle handle,
    const float* image_array,
    int32_t array_length,
    int32_t width,
    int32_t height,
    InspectionResultC* out_result,
    int32_t* o_imageHandle,
    char* o_message
) {
    ZeroResult(out_result);
    auto* eng = ToEngine(handle);
    if (!eng) {
        CopyExceptionMessage("Invalid handle or engine is not initialized.", o_message);
        return 1;
    }

    if (!image_array) {
        CopyExceptionMessage("image_array is nullptr.", o_message);
        return 1;
    }

    if (!out_result) {
        CopyExceptionMessage("out_result is nullptr.", o_message);
        return 1;
    }



    const int64_t expected = static_cast<int64_t>(width) * static_cast<int64_t>(height);
    if (expected <= 0 || expected > INT32_MAX) {
        CopyExceptionMessage("width or height out of bounds.", o_message);
        return 1;
    }
    if (array_length != static_cast<int32_t>(expected)) {
        CopyExceptionMessage("The size of numpy is not equal width plus height", o_message);
        return 1;
    }


    InspectionDLL::InferenceResult tmp;
    const bool ok = eng->ProcessFloatArry(image_array, tmp, static_cast<int>(width), static_cast<int>(height));
    if (!ok) {
        CopyExceptionMessage("inference fail", o_message);
        return 1;
    }
    FillResult(tmp, out_result);

    SaveResultOverlay(image_array, width, height, tmp, eng->ShouldDrawBoxDetails(), o_imageHandle, o_message);
    CopyExceptionMessage("inference success", o_message);
    
    return 0;
}


INSPECTION_C_EXPORT int32_t INSPECTION_CALL Inspection_ProcessFloatArray_npy(
    InspectionHandle handle,
    const float* i_image_array,
    int32_t i_width,
    int32_t i_height,
    int32_t o_detect_num[INSPECTION_DETECT_CATEGORIES],
    int32_t (*o_detect_area)[INSPECTION_MAX_DETECTIONS],
    float (*o_detect_contrast)[INSPECTION_MAX_DETECTIONS],
    int32_t* o_imageHandle,
    char* o_message
) {

    auto* eng = ToEngine(handle);
    if (!eng) {
        CopyExceptionMessage("Invalid handle or engine is not initialized.", o_message);
        return 1;
    }

    if (!i_image_array) {
        CopyExceptionMessage("image_array is nullptr.", o_message);
        return 1;
    }

    const int64_t expected = static_cast<int64_t>(i_width) * static_cast<int64_t>(i_height);
    if (expected <= 0 || expected > INT32_MAX) {
        CopyExceptionMessage("width or height out of bounds.", o_message);
        return 1;
    }
    
    InspectionDLL::InferenceResult tmp;
    const bool ok = eng->ProcessFloatArry(i_image_array, tmp, static_cast<int>(i_width), static_cast<int>(i_height));
    if (!ok) {
        CopyExceptionMessage("inference fail", o_message);
        return 1;
    }

    InspectionResultC result_c;
    ZeroResult(&result_c);
    FillResult(tmp, &result_c);


    if (o_detect_num) {
        std::memset(o_detect_num, 0, sizeof(int32_t) * INSPECTION_DETECT_CATEGORIES);
    }
    if (o_detect_area) {
        std::memset(o_detect_area, 0, sizeof(int32_t) * INSPECTION_DETECT_CATEGORIES * INSPECTION_MAX_DETECTIONS);
    }
    if (o_detect_contrast) {
        std::memset(o_detect_contrast, 0, sizeof(float) * INSPECTION_DETECT_CATEGORIES * INSPECTION_MAX_DETECTIONS);
    }

    int32_t counts[INSPECTION_DETECT_CATEGORIES] = { 0 };
    int32_t write_pos[INSPECTION_DETECT_CATEGORIES] = { 0 };

    for (int32_t i = 0; i < result_c.details_count; ++i) {
        const auto& d = result_c.details[i];
        const int32_t cat = DefectNameToIndex(d.name);
        if (cat < 0 || cat >= INSPECTION_DETECT_CATEGORIES) continue;

        counts[cat] += 1;

        const int32_t pos = write_pos[cat];
        if (pos >= INSPECTION_MAX_DETECTIONS) continue;

        if (o_detect_area) {
            o_detect_area[cat][pos] = d.area;
        }
        if (o_detect_contrast) {
            o_detect_contrast[cat][pos] = d.contrast;
        }
        write_pos[cat] += 1;
    }

    if (o_detect_num) {
        for (int32_t c = 0; c < INSPECTION_DETECT_CATEGORIES; ++c) {
            o_detect_num[c] = std::min<int32_t>(counts[c], INSPECTION_MAX_DETECTIONS);
        }
    }

    SaveResultOverlay(i_image_array, i_width, i_height, tmp, eng->ShouldDrawBoxDetails(), o_imageHandle, o_message);
    CopyExceptionMessage("inference success", o_message);
    
    return 0;
}

INSPECTION_C_EXPORT void INSPECTION_CALL Inspection_SetThresholds(
    InspectionHandle handle,
    float score_thresh,
    float area_thresh,
    float mask_area_thresh
) {
    auto* eng = ToEngine(handle);
    if (!eng) return;
    eng->SetThresholds(score_thresh, area_thresh, mask_area_thresh);
}

INSPECTION_C_EXPORT void INSPECTION_CALL Inspection_SetDarkClustersThreshold(
    InspectionHandle handle,
    float dark_clusters_thresh
) {
    auto* eng = ToEngine(handle);
    if (!eng) return;
    eng->SetDarkClustersThreshold(dark_clusters_thresh);
}

INSPECTION_C_EXPORT void INSPECTION_CALL Inspection_SetYoloNmsMode(
    InspectionHandle handle,
    int32_t class_aware
) {
    auto* eng = ToEngine(handle);
    if (!eng) return;
    eng->SetYoloNmsMode(class_aware != 0);
}

INSPECTION_C_EXPORT void INSPECTION_CALL Inspection_Release(InspectionHandle handle) {
    auto* eng = ToEngine(handle);
    if (!eng) return;
    eng->Release();
}

