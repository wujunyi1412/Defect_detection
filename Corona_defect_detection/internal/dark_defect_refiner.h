#pragma once

#include <opencv2/opencv.hpp>

#include "../inference_dll.h"
#include "image_utils.h"

namespace InspectionDLL::Internal {

bool RefineDarkDefectGeometryOnly(const cv::Mat& gray_yolo, DetectionResult& detail);
bool RefineDarkDefectGeometryAdaptive(
    const cv::Mat& gray_yolo,
    DetectionResult& detail,
    ContrastCalculationMode contrast_mode = ContrastCalculationMode::AdaptiveLocalPlane);

}  // namespace InspectionDLL::Internal
