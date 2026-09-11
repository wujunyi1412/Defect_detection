#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include "../inference_dll.h"
#include "yolo_inference.h"
#include "image_utils.h"

namespace InspectionDLL::Internal {

struct YoloDerived {
    float score = 0.0f;
    bool has_detections = false;
    std::vector<DetectionResult> details;
};

YoloDerived AnalyzeYolo(const std::vector<YOLO::Detection>& yolo_detections,
                        const cv::Mat& gray_yolo,
                        ContrastCalculationMode contrast_mode =
                            ContrastCalculationMode::AdaptiveLocalPlane);

}  // namespace InspectionDLL::Internal
