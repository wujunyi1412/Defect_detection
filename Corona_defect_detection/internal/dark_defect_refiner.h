#pragma once

#include <opencv2/opencv.hpp>

#include "../inference_dll.h"

namespace InspectionDLL::Internal {

bool RefineDarkDefectGeometryOnly(const cv::Mat& gray_yolo, DetectionResult& detail);
bool RefineDarkDefectGeometryAdaptive(const cv::Mat& gray_yolo, DetectionResult& detail);

}  // namespace InspectionDLL::Internal
