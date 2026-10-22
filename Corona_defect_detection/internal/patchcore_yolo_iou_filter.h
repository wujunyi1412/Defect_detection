#pragma once

#include <vector>

#include "../inference_dll.h"

namespace InspectionDLL::Internal {

void ApplyPatchCoreYoloIouFilter(
    std::vector<DetectionResult>& patchcore_details,
    const std::vector<DetectionResult>& yolo_details,
    float iou_threshold,
    std::vector<DetectionResult>* removed_details = nullptr);

}  // namespace InspectionDLL::Internal
