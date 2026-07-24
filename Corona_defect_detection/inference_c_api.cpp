#include "inference_c_api.h"
#include "inference_dll.h"
#include "internal/overlay_renderer.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>
#include <stdio.h>
#include <string.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#ifdef _WIN32
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif

// 内部辅助函数
namespace {

// 对比字符串是否相等，不区分大小写
bool EqualsIgnoreCase(const char* a, const char* b) {
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

// 将缺陷名称转换为缺陷索引
int32_t DefectNameToIndex(const char* name) {
    if (!name || name[0] == '\0') return -1;
    if (EqualsIgnoreCase(name, "Abnormal")) return static_cast<int32_t>(Defect_name::Abnormal);
    if (EqualsIgnoreCase(name, "Stain")) return static_cast<int32_t>(Defect_name::Stain);
    if (EqualsIgnoreCase(name, "BrightStripes")) return static_cast<int32_t>(Defect_name::BrightStripes);
    if (EqualsIgnoreCase(name, "LineArtifacts")) return static_cast<int32_t>(Defect_name::LineArtifacts);
    if (EqualsIgnoreCase(name, "DarkClusters")) return static_cast<int32_t>(Defect_name::DarkClusters);
    return -1;
}

// 将异常信息复制到输出缓冲区
void CopyExceptionMessage(const char* msg, char o_message[O_MESSAGE_LEN]) {
    if (!o_message) {
        return;
    }

    if (!msg) {
        snprintf(o_message, O_MESSAGE_LEN, "Unknown error");
        return;
    }

#ifdef _MSC_VER
    strncpy_s(o_message, O_MESSAGE_LEN, msg, _TRUNCATE);
#else
    snprintf(o_message, O_MESSAGE_LEN, "%s", msg);
#endif
}

// 将句柄转换为引擎指针
InspectionDLL::InspectionEngine* ToEngine(InspectionHandle handle) {
    return reinterpret_cast<InspectionDLL::InspectionEngine*>(handle);
}

// 将结果结构体置零
void ZeroResult(InspectionResultC* out_result) {
    if (!out_result) return;
    std::memset(out_result, 0, sizeof(InspectionResultC));
}

// 将字符串复制到输出缓冲区，确保 null 终止
void CopyName(char* dst, size_t dst_size, const std::string& src) {
    if (!dst || dst_size == 0) return;
#ifdef _WIN32
    strncpy_s(dst, dst_size, src.c_str(), _TRUNCATE);
#else
    std::strncpy(dst, src.c_str(), dst_size - 1);
    dst[dst_size - 1] = '\0';
#endif
}

// 将推理结果填充到输出结构体
void FillResult(const InspectionDLL::InferenceResult& src, InspectionResultC* dst) {
    if (!dst) return;

    dst->result = (src.result == "OK") ? INSPECTION_RESULT_OK : INSPECTION_RESULT_NG;
    dst->yolo_score = src.yolo_score;
    dst->patchcore_score = src.patchcore_score;
    dst->patchcore_area_ratio = src.patchcore_area_ratio;

    // 填充details
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

// 将输入图像数组保存为 32 位浮点数 TIFF 图像
bool SaveInputTiff32F(
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

}  // namespace

// 创建 inspection 引擎实例
INSPECTION_C_EXPORT InspectionHandle INSPECTION_CALL Inspection_Create() {
    try {
        return reinterpret_cast<InspectionHandle>(new InspectionDLL::InspectionEngine());
    } catch (...) {
        return nullptr;
    }
}

// 销毁 inspection 引擎实例
INSPECTION_C_EXPORT void INSPECTION_CALL Inspection_Destroy(InspectionHandle handle) {
    auto* eng = ToEngine(handle);
    delete eng;
}

// 初始化 inspection 引擎
INSPECTION_C_EXPORT int32_t INSPECTION_CALL Inspection_Initialize(
    InspectionHandle handle,
    const char* i_config_path,
    char* o_message
) {
    auto* eng = ToEngine(handle);
    if (!eng) {
        CopyExceptionMessage("inspection handle is nullptr.", o_message);
        return INSPECTION_STATUS_INVALID_ARGUMENT;
    }
    if (!i_config_path) {
        CopyExceptionMessage("config path is empty.", o_message);
        return INSPECTION_STATUS_INVALID_ARGUMENT;
    }

    if (!eng->Initialize(std::string(i_config_path))) {
        // 初始化失败，复制错误信息
        CopyExceptionMessage(eng->GetLastError().c_str(), o_message);
        return INSPECTION_STATUS_INITIALIZE_FAILED;
    }

    CopyExceptionMessage("model initialize success", o_message);
    return INSPECTION_STATUS_OK;
}

// 处理单张图像路径
INSPECTION_C_EXPORT int32_t INSPECTION_CALL Inspection_ProcessImagePath(
    InspectionHandle handle,
    const char* image_path,
    InspectionResultC* out_result
) {
    ZeroResult(out_result);
    auto* eng = ToEngine(handle);
    if (!eng || !image_path || !out_result)  return INSPECTION_STATUS_INVALID_ARGUMENT;

    InspectionDLL::InferenceResult tmp;
    if (!eng->ProcessImagePath(std::string(image_path), tmp)) {
        return INSPECTION_STATUS_INFERENCE_FAILED;
    }

    FillResult(tmp, out_result);
    return INSPECTION_STATUS_OK;
}

// 处理单张图像数组
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
        CopyExceptionMessage("invalid inspection handle.", o_message);
        return INSPECTION_STATUS_INVALID_ARGUMENT;
    }
    if (!image_array) {
        CopyExceptionMessage("image_array is nullptr.", o_message);
        return INSPECTION_STATUS_INVALID_ARGUMENT;
    }
    if (!out_result) {
        CopyExceptionMessage("out_result is nullptr.", o_message);
        return INSPECTION_STATUS_INVALID_ARGUMENT;
    }

    const int64_t expected = static_cast<int64_t>(width) * static_cast<int64_t>(height);
    if (expected <= 0 || expected > INT32_MAX) {
        CopyExceptionMessage("width or height out of bounds.", o_message);
        return INSPECTION_STATUS_INVALID_ARGUMENT;
    }
    if (array_length != static_cast<int32_t>(expected)) {
        CopyExceptionMessage("array_length does not match width times height.", o_message);
        return INSPECTION_STATUS_INVALID_ARGUMENT;
    }

    InspectionDLL::InferenceResult tmp;
    if (!eng->ProcessFloatArry(image_array, tmp, static_cast<int>(width), static_cast<int>(height))) {
        CopyExceptionMessage(eng->GetLastError().c_str(), o_message);
        return INSPECTION_STATUS_INFERENCE_FAILED;
    }

    FillResult(tmp, out_result);

    if (o_imageHandle) {
        std::string overlay_err;
        if (!InspectionOverlay::RenderResultOverlayToHalconHandle(
                image_array,
                width,
                height,
                tmp,
                eng->ShouldDrawDefectBox(),
                eng->ShouldDrawBoxDetails(),
                eng->GetMaskOverlayOptions(),
                o_imageHandle,
                overlay_err)) {
            CopyExceptionMessage(overlay_err.c_str(), o_message);
            return INSPECTION_STATUS_OVERLAY_FAILED;
        }
    }

    CopyExceptionMessage("inference success", o_message);
    return INSPECTION_STATUS_OK;
}

// 处理单张图像数组（返回检测结果，结果格式满足sunny之前风格）
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
        CopyExceptionMessage("invalid inspection handle", o_message);
        return INSPECTION_STATUS_INVALID_ARGUMENT;
    }
    if (!i_image_array) {
        CopyExceptionMessage("image_array is nullptr.", o_message);
        return INSPECTION_STATUS_INVALID_ARGUMENT;
    }

    const int64_t expected = static_cast<int64_t>(i_width) * static_cast<int64_t>(i_height);
    if (expected <= 0 || expected > INT32_MAX) {
        CopyExceptionMessage("width or height out of bounds.", o_message);
        return INSPECTION_STATUS_INVALID_ARGUMENT;
    }

    InspectionDLL::InferenceResult tmp;
    if (!eng->ProcessFloatArry(i_image_array, tmp, static_cast<int>(i_width), static_cast<int>(i_height))) {
        CopyExceptionMessage(eng->GetLastError().c_str(), o_message);
        return INSPECTION_STATUS_INFERENCE_FAILED;
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

    int32_t counts[INSPECTION_DETECT_CATEGORIES] = {0};
    int32_t write_pos[INSPECTION_DETECT_CATEGORIES] = {0};
    // 遍历检测结果，填充结果数组
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

    // 填充检测数量, 每个类别最多INSPECTION_MAX_DETECTIONS个检测结果
    if (o_detect_num) {
        for (int32_t c = 0; c < INSPECTION_DETECT_CATEGORIES; ++c) {
            o_detect_num[c] = std::min<int32_t>(counts[c], INSPECTION_MAX_DETECTIONS);
        }
    }

    // 渲染推理结果到Halcon图像句柄
    if (o_imageHandle) {
        std::string overlay_err;
        if (!InspectionOverlay::RenderResultOverlayToHalconHandle(
                i_image_array,
                i_width,
                i_height,
                tmp,
                eng->ShouldDrawDefectBox(),
                eng->ShouldDrawBoxDetails(),
                eng->GetMaskOverlayOptions(),
                o_imageHandle,
                overlay_err)) {
            CopyExceptionMessage(overlay_err.c_str(), o_message);
            return INSPECTION_STATUS_OVERLAY_FAILED;
        }
    }

    CopyExceptionMessage("inference success", o_message);
    return INSPECTION_STATUS_OK;
}

// 设置推理阈值
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

// 设置暗区聚类阈值
INSPECTION_C_EXPORT void INSPECTION_CALL Inspection_SetDarkClustersThreshold(
    InspectionHandle handle,
    float dark_clusters_thresh
) {
    auto* eng = ToEngine(handle);
    if (!eng) return;
    eng->SetDarkClustersThreshold(dark_clusters_thresh);
}

// 设置YOLO NMS模式（是否类别感知）
INSPECTION_C_EXPORT void INSPECTION_CALL Inspection_SetYoloNmsMode(
    InspectionHandle handle,
    int32_t class_aware
) {
    auto* eng = ToEngine(handle);
    if (!eng) return;
    eng->SetYoloNmsMode(class_aware != 0);
}

// 释放 inspection 引擎资源
INSPECTION_C_EXPORT void INSPECTION_CALL Inspection_Release(InspectionHandle handle) {
    auto* eng = ToEngine(handle);
    if (!eng) return;
    eng->Release();
}
