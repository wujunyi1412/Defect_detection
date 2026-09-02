#include "yolo_postprocess.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "image_utils.h"
#include "yolo_categories.h"

namespace InspectionDLL::Internal {

YoloDerived AnalyzeYolo(const std::vector<YOLO::Detection>& yolo_detections,
                        const cv::Mat& gray_yolo) {
    YoloDerived result;
    result.has_detections = false;
    result.score = 0.0f;

    result.details.clear();
    result.details.reserve(yolo_detections.size());
    for (const auto& det : yolo_detections) {
        DetectionResult detail;
        const YoloCategoryDefinition* category = FindYoloCategory(det.class_id);
        if (!category) continue;
        detail.name = category->name;
        detail.score = det.confidence;
        result.score = std::max(result.score, detail.score);
        result.has_detections = true;

        detail.x = det.x1;
        detail.y = det.y1;
        detail.w = det.x2 - det.x1;
        detail.h = det.y2 - det.y1;

        const cv::Mat& mask_uint8 = det.mask;
        detail.mask = mask_uint8;

        const int W = gray_yolo.cols;
        const int H = gray_yolo.rows;
        int x1i = static_cast<int>(std::round(static_cast<double>(det.x1)));
        int y1i = static_cast<int>(std::round(static_cast<double>(det.y1)));
        int x2i = static_cast<int>(std::round(static_cast<double>(det.x2)));
        int y2i = static_cast<int>(std::round(static_cast<double>(det.y2)));
        x1i = std::max(0, std::min(x1i, std::max(0, W - 1)));
        y1i = std::max(0, std::min(y1i, std::max(0, H - 1)));
        x2i = std::max(x1i + 1, std::min(x2i, W));
        y2i = std::max(y1i + 1, std::min(y2i, H));
        const int bbox_w = std::max(1, x2i - x1i);
        const int bbox_h = std::max(1, y2i - y1i);

        int mask_area_in_bbox = 0;
        if (mask_uint8.rows == H && mask_uint8.cols == W) {
            mask_area_in_bbox = cv::countNonZero(mask_uint8(cv::Rect(x1i, y1i, bbox_w, bbox_h)));
        } else if (mask_uint8.rows == bbox_h && mask_uint8.cols == bbox_w) {
            mask_area_in_bbox = cv::countNonZero(mask_uint8);
        } else {
            cv::Mat resized_mask;
            cv::resize(mask_uint8, resized_mask, cv::Size(bbox_w, bbox_h), 0, 0, cv::INTER_NEAREST);
            mask_area_in_bbox = cv::countNonZero(resized_mask);
        }

        const int bbox_area = std::max(1, bbox_w * bbox_h);
        detail.area = std::min(mask_area_in_bbox, bbox_area);
        if (detail.area <= 0 && det.confidence > 0.0f) {
            detail.area = 1;
        }

        detail.contrast = CalculateContrastRatioAdaptive(gray_yolo, mask_uint8, x1i, y1i, bbox_w, bbox_h,
                                                 category->contrast_polarity);

        result.details.push_back(detail);
    }

    return result;
}

}  // namespace InspectionDLL::Internal
