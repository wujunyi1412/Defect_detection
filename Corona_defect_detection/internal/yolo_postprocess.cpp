#include "yolo_postprocess.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "image_utils.h"
#include "yolo_categories.h"

namespace InspectionDLL::Internal {

YoloDerived AnalyzeYolo(const std::vector<YOLO::Detection>& yolo_detections,
                        const cv::Mat& gray_yolo,
                        ContrastCalculationMode contrast_mode,
                        const InspectionConfig::CategoryFilterMap* category_filters,
                        const InspectionConfig::YoloLargeBoxShrinkConfig* large_box_shrink) {
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

        float box_x1 = det.x1;
        float box_y1 = det.y1;
        float box_x2 = det.x2;
        float box_y2 = det.y2;
        const float original_width = box_x2 - box_x1;
        const float original_height = box_y2 - box_y1;
        if (large_box_shrink && large_box_shrink->enable &&
            original_width > large_box_shrink->min_width &&
            original_height > large_box_shrink->min_height) {
            const float center_x = 0.5f * (box_x1 + box_x2);
            const float center_y = 0.5f * (box_y1 + box_y2);
            const float half_width = 0.5f * original_width * (1.0f - large_box_shrink->ratio);
            const float half_height = 0.5f * original_height * (1.0f - large_box_shrink->ratio);
            box_x1 = std::clamp(center_x - half_width, 0.0f, static_cast<float>(gray_yolo.cols));
            box_y1 = std::clamp(center_y - half_height, 0.0f, static_cast<float>(gray_yolo.rows));
            box_x2 = std::clamp(center_x + half_width, 0.0f, static_cast<float>(gray_yolo.cols));
            box_y2 = std::clamp(center_y + half_height, 0.0f, static_cast<float>(gray_yolo.rows));
        }
        detail.x = box_x1;
        detail.y = box_y1;
        detail.w = box_x2 - box_x1;
        detail.h = box_y2 - box_y1;

        const cv::Mat& mask_uint8 = det.mask;
        detail.mask = mask_uint8;

        const int W = gray_yolo.cols;
        const int H = gray_yolo.rows;
        int x1i = static_cast<int>(std::round(static_cast<double>(box_x1)));
        int y1i = static_cast<int>(std::round(static_cast<double>(box_y1)));
        int x2i = static_cast<int>(std::round(static_cast<double>(box_x2)));
        int y2i = static_cast<int>(std::round(static_cast<double>(box_y2)));
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

        ContrastPolarity contrast_polarity = category->contrast_polarity;
        if (category_filters) {
            const auto filter_it = category_filters->find(category->name);
            if (filter_it != category_filters->end()) {
                switch (filter_it->second.defect_polarity) {
                    case InspectionConfig::DefectPolarity::Dark:
                        contrast_polarity = ContrastPolarity::Dark;
                        break;
                    case InspectionConfig::DefectPolarity::Bright:
                        contrast_polarity = ContrastPolarity::Bright;
                        break;
                    case InspectionConfig::DefectPolarity::Auto:
                    case InspectionConfig::DefectPolarity::Invalid:
                        break;
                }
            }
        }
        detail.contrast = CalculateContrastRatioByMode(
            gray_yolo, mask_uint8, x1i, y1i, bbox_w, bbox_h,
            contrast_polarity, contrast_mode);

        result.details.push_back(detail);
    }

    return result;
}

}  // namespace InspectionDLL::Internal
