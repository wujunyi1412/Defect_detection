#include "yolo_postprocess.h"

#include <algorithm>
#include <cmath>

#include "image_utils.h"

namespace InspectionDLL::Internal {

YoloDerived AnalyzeYolo(const std::vector<YOLO::Detection>& yolo_detections,
                        const cv::Mat& gray_yolo,
                        float dark_clusters_threshold) {
    YoloDerived result;
    result.has_detections = false;
    result.score = 0.0f;

    result.details.clear();
    result.details.reserve(yolo_detections.size());
    for (const auto& det : yolo_detections) {
        DetectionResult detail;
        switch (det.class_id) {
            case 0: detail.name = "Stain"; break;
            case 1: detail.name = "BrightStripes"; break;
            case 2: detail.name = "LineArtifacts"; break;
            default: continue;
        }
        detail.score = det.confidence;
        result.score = std::max(result.score, detail.score);
        result.has_detections = true;

        detail.x = det.x1;
        detail.y = det.y1;
        detail.w = det.x2 - det.x1;
        detail.h = det.y2 - det.y1;

        cv::Mat mask_uint8;
        det.mask.convertTo(mask_uint8, CV_8U);
        cv::threshold(mask_uint8, mask_uint8, 0, 255, cv::THRESH_BINARY);
        detail.area = static_cast<int>(cv::countNonZero(mask_uint8));

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
            cv::threshold(resized_mask, resized_mask, 0, 255, cv::THRESH_BINARY);
            mask_area_in_bbox = cv::countNonZero(resized_mask);
        }

        const int bbox_area = std::max(1, bbox_w * bbox_h);
        detail.area = std::min(mask_area_in_bbox, bbox_area);
        if (detail.area <= 0 && det.confidence > 0.0f) {
            detail.area = 1;
        }

        detail.contrast = CalculateContrastRatio(gray_yolo, mask_uint8, x1i, y1i, bbox_w, bbox_h);
        if (detail.name == "Stain" && detail.contrast > 1.0f) {
            detail.contrast = 1.0f / std::max(detail.contrast, 1e-6f);
        }

        if (detail.name == "Stain" && (detail.contrast <= dark_clusters_threshold || detail.contrast == 0.0f)) {
            detail.name = "DarkClusters";
        }

        result.details.push_back(detail);
    }

    return result;
}

}  // namespace InspectionDLL::Internal
