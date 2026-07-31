#pragma once

#include <cstdint>
#include <string>

#include "inference_dll.h"

namespace InspectionOverlay {

bool RenderResultOverlayToHalconHandle(
    const float* image_array,
    int32_t width,
    int32_t height,
    const InspectionDLL::InferenceResult& result,
    bool draw_defect_box,
    bool draw_box_details,
    const InspectionDLL::MaskOverlayOptions& mask_options,
    bool concat_original_image,
    int32_t o_imageHandle[1],
    std::string& err);

bool RenderResultOverlayToFile(
    const float* image_array,
    int32_t width,
    int32_t height,
    const InspectionDLL::InferenceResult& result,
    bool draw_defect_box,
    bool draw_box_details,
    const InspectionDLL::MaskOverlayOptions& mask_options,
    bool concat_original_image,
    const char* output_path,
    std::string& err);

std::string FitTextToWidthForTest(const std::string& text, int max_width, double font_scale, int thickness);

}  // namespace InspectionOverlay
