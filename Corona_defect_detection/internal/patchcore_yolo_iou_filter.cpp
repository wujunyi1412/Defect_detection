#include "patchcore_yolo_iou_filter.h"

#include <algorithm>
#include <cmath>

namespace InspectionDLL::Internal {
namespace {

float CalculateBoxIou(const DetectionResult& lhs, const DetectionResult& rhs) {
    if (!std::isfinite(lhs.x) || !std::isfinite(lhs.y) ||
        !std::isfinite(lhs.w) || !std::isfinite(lhs.h) ||
        !std::isfinite(rhs.x) || !std::isfinite(rhs.y) ||
        !std::isfinite(rhs.w) || !std::isfinite(rhs.h) ||
        lhs.w <= 0.0f || lhs.h <= 0.0f || rhs.w <= 0.0f || rhs.h <= 0.0f) {
        return 0.0f;
    }

    const float intersection_left = std::max(lhs.x, rhs.x);
    const float intersection_top = std::max(lhs.y, rhs.y);
    const float intersection_right = std::min(lhs.x + lhs.w, rhs.x + rhs.w);
    const float intersection_bottom = std::min(lhs.y + lhs.h, rhs.y + rhs.h);
    const float intersection_width = std::max(0.0f, intersection_right - intersection_left);
    const float intersection_height = std::max(0.0f, intersection_bottom - intersection_top);
    const float intersection_area = intersection_width * intersection_height;
    const float union_area = lhs.w * lhs.h + rhs.w * rhs.h - intersection_area;
    if (!std::isfinite(union_area) || union_area <= 0.0f) return 0.0f;
    return std::clamp(intersection_area / union_area, 0.0f, 1.0f);
}

}  // namespace

void ApplyPatchCoreYoloIouFilter(
    std::vector<DetectionResult>& patchcore_details,
    const std::vector<DetectionResult>& yolo_details,
    float minimum_iou_threshold) {
    if (patchcore_details.empty() || yolo_details.empty()) return;

    patchcore_details.erase(
        std::remove_if(
            patchcore_details.begin(),
            patchcore_details.end(),
            [&](const DetectionResult& patchcore_detail) {
                float minimum_iou = 1.0f;
                for (const auto& yolo_detail : yolo_details) {
                    minimum_iou = std::min(
                        minimum_iou,
                        CalculateBoxIou(patchcore_detail, yolo_detail));
                }
                return minimum_iou > minimum_iou_threshold;
            }),
        patchcore_details.end());
}

}  // namespace InspectionDLL::Internal
