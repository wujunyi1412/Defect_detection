#pragma once

#include <stdint.h>

#ifdef _WIN32
#ifdef INSPECTION_CAPI_EXPORTS
#define INSPECTION_C_EXPORT extern "C" __declspec(dllexport)
#else
#define INSPECTION_C_EXPORT extern "C" __declspec(dllimport)
#endif
#define INSPECTION_CALL __cdecl
#else
#define INSPECTION_C_EXPORT extern "C" __attribute__((visibility("default")))
#define INSPECTION_CALL
#endif

#define INSPECTION_MAX_DETECTIONS 128
#define INSPECTION_MAX_NAME_LEN 32
#define INSPECTION_DETECT_CATEGORIES 9


#define INSPECTION_RESULT_OK 0
#define INSPECTION_RESULT_NG 1

#define INSPECTION_STATUS_OK 0
#define INSPECTION_STATUS_INVALID_ARGUMENT 2  // 无效参数
#define INSPECTION_STATUS_INITIALIZE_FAILED 3 // 初始化失败
#define INSPECTION_STATUS_INFERENCE_FAILED 4 // 推理失败
#define INSPECTION_STATUS_OVERLAY_FAILED 5   // 图像变成句柄失败

#define O_MESSAGE_LEN 100
#define O_IMAGE_HANDLE 1


enum class Defect_name {
    Abnormal = 0,
    Glue_overflow = 1,
    Decolorization = 2,
    Stain = 3,
    Stripes = 4,
    BrightStripes = 5,
    Bright_clusters = 6,
    Line_artifacts = 7,
    LineArtifacts = 8,
};

typedef void* InspectionHandle;

typedef struct InspectionDetectionC {
    char name[INSPECTION_MAX_NAME_LEN];
    int32_t area;
    float x;
    float y;
    float w;
    float h;
    float contrast;
} InspectionDetectionC;

typedef struct InspectionResultC {
    int32_t result;
    int32_t details_count;
    InspectionDetectionC details[INSPECTION_MAX_DETECTIONS];
    float yolo_score;
    float patchcore_score;
    float patchcore_area_ratio;
} InspectionResultC;

INSPECTION_C_EXPORT InspectionHandle INSPECTION_CALL Inspection_Create();
INSPECTION_C_EXPORT void INSPECTION_CALL Inspection_Destroy(InspectionHandle handle);

INSPECTION_C_EXPORT int32_t INSPECTION_CALL Inspection_Initialize(
    InspectionHandle handle,
    const char* i_config_path,
    char o_message[O_MESSAGE_LEN]
);

INSPECTION_C_EXPORT int32_t INSPECTION_CALL Inspection_ProcessImagePath(
    InspectionHandle handle,
    const char* image_path,
    InspectionResultC* out_result
);

// Processes one image and returns native inference time (image decoding excluded).
INSPECTION_C_EXPORT int32_t INSPECTION_CALL Inspection_ProcessImagePathTimed(
    InspectionHandle handle,
    const char* image_path,
    InspectionResultC* out_result,
    double o_inference_ms[1]
);

INSPECTION_C_EXPORT int32_t INSPECTION_CALL Inspection_ProcessImagePathToOverlayFile(
    InspectionHandle handle,
    const char* image_path,
    const char* output_path,
    InspectionResultC* out_result,
    char o_message[O_MESSAGE_LEN],
    double o_inference_ms[1],
    double o_save_ms[1]
);

INSPECTION_C_EXPORT int32_t INSPECTION_CALL Inspection_ProcessFloatArray(
    InspectionHandle handle,
    const float* image_array,
    int32_t array_length,
    int32_t width,
    int32_t height,
    InspectionResultC* out_result,
    int32_t o_imageHandle[O_IMAGE_HANDLE],
    char o_message[O_MESSAGE_LEN]
);

INSPECTION_C_EXPORT int32_t INSPECTION_CALL Inspection_ProcessFloatArray_npy(
    InspectionHandle handle,
    const float* i_image_array,
    int32_t i_width,
    int32_t i_height,
    int32_t o_detect_num[INSPECTION_DETECT_CATEGORIES],
    int32_t o_detect_area[INSPECTION_DETECT_CATEGORIES][INSPECTION_MAX_DETECTIONS],
    float o_detect_contrast[INSPECTION_DETECT_CATEGORIES][INSPECTION_MAX_DETECTIONS],
    int32_t o_imageHandle[O_IMAGE_HANDLE],
    char o_message[O_MESSAGE_LEN]
);

INSPECTION_C_EXPORT void INSPECTION_CALL Inspection_SetThresholds(
    InspectionHandle handle,
    float score_thresh,
    float area_thresh,
    float mask_area_thresh
);

INSPECTION_C_EXPORT void INSPECTION_CALL Inspection_SetYoloNmsMode(
    InspectionHandle handle,
    int32_t class_aware
);

INSPECTION_C_EXPORT void INSPECTION_CALL Inspection_Release(InspectionHandle handle);

