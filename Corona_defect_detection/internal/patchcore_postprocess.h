#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include "patchcore_inference.h"

namespace InspectionDLL::Internal {

struct Component {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
    int area = 0;
    float cx = 0.0f;
    float cy = 0.0f;
};

struct PatchCoreDerived {
    float score = 0.0f;
    float area_ratio = 0.0f;
    int area = 0;
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
    float contrast = 0.0f;
    bool has_defect = false;
    float scale_x = 0.0f;
    float scale_y = 0.0f;
    cv::Mat area_cropped;
    cv::Mat area_orig;
    cv::Mat score_cropped;
    cv::Mat score_orig;
    std::vector<Component> crosshair_components;
    bool has_crosshair_grid = false;
};

PatchCoreDerived AnalyzePatchCore(const PatchCore::PatchCoreResult& patchcore_result,
                                  const cv::Size& img_shape,
                                  const cv::Mat& gray_patchcore,
                                  float score_threshold,
                                  float area_threshold,
                                  float mask_area_threshold);

}  // namespace InspectionDLL::Internal
