#pragma once

#include "config.h"
#include "patchcore_postprocess.h"
#include "yolo_postprocess.h"

namespace InspectionDLL::Internal {

struct ResultComposeContext {
    float score_threshold = 1.4f;
    float patchcore_yolo_iou_threshold = 0.0f;
    InspectionConfig::AaFilterConfig aa_filter;
    InspectionConfig::AbnormalFilterConfig abnormal_filter;
    InspectionConfig::CategoryFilterConfig stain_filter;
    InspectionConfig::CategoryFilterConfig brightstripes_filter;
    InspectionConfig::CategoryFilterConfig lineartifacts_filter;
};

void ComposeOutput(const PatchCoreDerived& pc,
                   const YoloDerived& yolo,
                   const cv::Size& img_shape,
                   const cv::Mat& gray_patchcore,
                   InferenceResult& output,
                   const ResultComposeContext& context);

void ComposeOutputWithDefectFilter(const PatchCoreDerived& pc,
                                   const YoloDerived& yolo,
                                   const cv::Size& img_shape,
                                   const cv::Mat& gray_patchcore,
                                   const cv::Mat& gray_yolo,
                                   InferenceResult& output,
                                   const ResultComposeContext& context);

}  // namespace InspectionDLL::Internal
