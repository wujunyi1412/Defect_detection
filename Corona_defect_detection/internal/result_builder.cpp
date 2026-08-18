#include "result_builder.h"

#include <algorithm>
#include <cmath>

#include "aa_yolo_filter.h"
#include "dark_defect_refiner.h"
#include "image_utils.h"

namespace InspectionDLL::Internal {
namespace {

bool IsDarkDefectCategory(const std::string& name) {
    return name == "Stain" || name == "DarkClusters";
}

bool PassContrastThreshold(float contrast, float threshold, bool keep_below_threshold) {
    if (threshold <= 0.0f) return true;
    if (!IsContrastRatioValid(contrast)) return false;
    return keep_below_threshold ? contrast < threshold : contrast >= threshold;
}

}  // namespace

void ComposeOutput(const PatchCoreDerived& pc,
                   const YoloDerived& yolo,
                   const cv::Size& img_shape,
                   const cv::Mat& gray_patchcore,
                   InferenceResult& output,
                   const ResultComposeContext& context) {
    std::vector<DetectionResult> yolo_details_filtered = yolo.details;
    ApplyAaYoloFilter(yolo_details_filtered, context.aa_filter);

    output.yolo_score = 0.0f;
    for (const auto& detail : yolo_details_filtered) {
        output.yolo_score = std::max(output.yolo_score, detail.score);
    }
    output.patchcore_score = pc.score;
    output.patchcore_area_ratio = pc.area_ratio;

    const bool has_yolo_detections = !yolo_details_filtered.empty();
    if (pc.score >= context.score_threshold || has_yolo_detections) {
        output.result = "NG";
    } else {
        output.result = "OK";
    }

    output.details.clear();
    const bool include_patchcore_details =
        !(context.suppress_patchcore_when_yolo_detected && has_yolo_detections);

    if (include_patchcore_details && pc.has_crosshair_grid) {
        std::vector<Component> comps = pc.crosshair_components;
        std::sort(comps.begin(), comps.end(),
                  [](const Component& a, const Component& b) {
                      if (a.cy != b.cy) return a.cy < b.cy;
                      return a.cx < b.cx;
                  });

        for (const auto& c : comps) {
            int x1 = static_cast<int>(std::round(static_cast<float>(c.x) * pc.scale_x));
            int y1 = static_cast<int>(std::round(static_cast<float>(c.y) * pc.scale_y));
            int w1 = static_cast<int>(std::round(static_cast<float>(c.w) * pc.scale_x));
            int h1 = static_cast<int>(std::round(static_cast<float>(c.h) * pc.scale_y));

            x1 = std::max(0, std::min(x1, img_shape.width - 1));
            y1 = std::max(0, std::min(y1, img_shape.height - 1));
            w1 = std::max(1, std::min(w1, img_shape.width - x1));
            h1 = std::max(1, std::min(h1, img_shape.height - y1));

            const int area = static_cast<int>(std::round(static_cast<float>(c.area) * pc.scale_x * pc.scale_y));
            const float contrast = CalculateContrastRatio(gray_patchcore, pc.score_orig, x1, y1, w1, h1);

            DetectionResult detail;
            detail.name = "Abnormal";
            detail.area = area;
            detail.x = static_cast<float>(x1);
            detail.y = static_cast<float>(y1);
            detail.w = static_cast<float>(w1);
            detail.h = static_cast<float>(h1);
            detail.contrast = contrast;
            detail.score = pc.score;
            detail.mask = pc.score_orig;
            output.details.push_back(detail);
        }
    }

    if (has_yolo_detections && !pc.has_defect) {
        output.details.insert(output.details.end(), yolo_details_filtered.begin(), yolo_details_filtered.end());
    } else if (!has_yolo_detections && pc.has_defect && !pc.has_crosshair_grid) {
        DetectionResult detail;
        detail.name = "Abnormal";
        detail.area = pc.area;
        detail.x = static_cast<float>(pc.x);
        detail.y = static_cast<float>(pc.y);
        detail.w = static_cast<float>(pc.w);
        detail.h = static_cast<float>(pc.h);
        detail.contrast = pc.contrast;
        detail.score = pc.score;
        detail.mask = pc.area_orig;
        output.details.push_back(detail);
    } else if (has_yolo_detections && pc.has_defect) {
        if (include_patchcore_details && !pc.has_crosshair_grid) {
            DetectionResult detail;
            detail.name = "Abnormal";
            detail.area = pc.area;
            detail.x = static_cast<float>(pc.x);
            detail.y = static_cast<float>(pc.y);
            detail.w = static_cast<float>(pc.w);
            detail.h = static_cast<float>(pc.h);
            detail.contrast = pc.contrast;
            detail.score = pc.score;
            detail.mask = pc.area_orig;
            output.details.push_back(detail);
        }
        output.details.insert(output.details.end(), yolo_details_filtered.begin(), yolo_details_filtered.end());
    }
}

void ComposeOutputWithDefectFilter(const PatchCoreDerived& pc,
                                   const YoloDerived& yolo,
                                   const cv::Size& img_shape,
                                   const cv::Mat& gray_patchcore,
                                   const cv::Mat& gray_yolo,
                                   InferenceResult& output,
                                   const ResultComposeContext& context) {
    std::vector<DetectionResult> yolo_details_filtered;
    yolo_details_filtered.reserve(yolo.details.size());
    for (const auto& d : yolo.details) {
        const InspectionConfig::CategoryFilterConfig* cf = nullptr;
        if (d.name == "Stain") cf = &context.stain_filter;
        else if (d.name == "DarkClusters") cf = &context.darkclusters_filter;
        else if (d.name == "BrightStripes") cf = &context.brightstripes_filter;
        else if (d.name == "LineArtifacts") cf = &context.lineartifacts_filter;

        if (!cf) continue;

        DetectionResult detail = d;
        if ((detail.name == "Stain" || detail.name == "DarkClusters") && cf->use_traditional_measure) {
            if (RefineDarkDefectGeometryAdaptive(gray_yolo, detail)) {
                if (IsContrastRatioValid(detail.contrast)) {
                    detail.name = detail.contrast <= context.dark_clusters_threshold
                                      ? "DarkClusters"
                                      : "Stain";
                    cf = detail.name == "Stain" ? &context.stain_filter : &context.darkclusters_filter;
                }
            }
        }
        if (!cf->enable) continue;
        if (detail.score < cf->confidence_threshold) continue;
        if (!PassContrastThreshold(
                detail.contrast,
                cf->contrast_threshold,
                IsDarkDefectCategory(detail.name))) continue;
        if (cf->min_width > 0 && detail.w < static_cast<float>(cf->min_width)) continue;
        if (cf->min_height > 0 && detail.h < static_cast<float>(cf->min_height)) continue;
        if (cf->min_area > 0 && detail.area < cf->min_area) continue;
        yolo_details_filtered.push_back(detail);
    }
    ApplyAaYoloFilter(yolo_details_filtered, context.aa_filter);
    const bool has_yolo_detections = !yolo_details_filtered.empty();
    const bool include_patchcore_details =
        !(context.suppress_patchcore_when_yolo_detected && has_yolo_detections);

    std::vector<DetectionResult> pc_details_filtered;
    const bool abnormal_enable = context.abnormal_filter.enable;
    const bool abnormal_by_score_area =
        (pc.score >= context.abnormal_filter.score_threshold_2) ||
        (pc.score >= context.abnormal_filter.score_threshold_1 && pc.area_ratio >= context.abnormal_filter.area_threshold_1);
    const bool abnormal_is_ng = abnormal_enable && abnormal_by_score_area;

    const bool abnormal_pass_base =
        abnormal_enable &&
        abnormal_by_score_area &&
        (context.abnormal_filter.min_width <= 0 || pc.w >= context.abnormal_filter.min_width) &&
        (context.abnormal_filter.min_height <= 0 || pc.h >= context.abnormal_filter.min_height) &&
        (context.abnormal_filter.min_area <= 0 || pc.area >= context.abnormal_filter.min_area) &&
        PassContrastThreshold(pc.contrast, context.abnormal_filter.contrast_threshold, false);

    if (include_patchcore_details && pc.has_crosshair_grid && abnormal_is_ng) {
        std::vector<Component> comps = pc.crosshair_components;
        std::sort(comps.begin(), comps.end(),
                  [](const Component& a, const Component& b) {
                      if (a.cy != b.cy) return a.cy < b.cy;
                      return a.cx < b.cx;
                  });

        pc_details_filtered.reserve(comps.size());
        for (const auto& c : comps) {
            int x1 = static_cast<int>(std::round(static_cast<float>(c.x) * pc.scale_x));
            int y1 = static_cast<int>(std::round(static_cast<float>(c.y) * pc.scale_y));
            int w1 = static_cast<int>(std::round(static_cast<float>(c.w) * pc.scale_x));
            int h1 = static_cast<int>(std::round(static_cast<float>(c.h) * pc.scale_y));

            x1 = std::max(0, std::min(x1, img_shape.width - 1));
            y1 = std::max(0, std::min(y1, img_shape.height - 1));
            w1 = std::max(1, std::min(w1, img_shape.width - x1));
            h1 = std::max(1, std::min(h1, img_shape.height - y1));

            const int area = static_cast<int>(std::round(static_cast<float>(c.area) * pc.scale_x * pc.scale_y));
            const float contrast = CalculateContrastRatio(gray_patchcore, pc.score_orig, x1, y1, w1, h1);
            if (context.abnormal_filter.min_width > 0 && w1 < context.abnormal_filter.min_width) continue;
            if (context.abnormal_filter.min_height > 0 && h1 < context.abnormal_filter.min_height) continue;
            if (context.abnormal_filter.min_area > 0 && area < context.abnormal_filter.min_area) continue;
            if (!PassContrastThreshold(
                    contrast,
                    context.abnormal_filter.contrast_threshold,
                    false)) continue;

            DetectionResult detail;
            detail.name = "Abnormal";
            detail.area = area;
            detail.x = static_cast<float>(x1);
            detail.y = static_cast<float>(y1);
            detail.w = static_cast<float>(w1);
            detail.h = static_cast<float>(h1);
            detail.contrast = contrast;
            detail.score = pc.score;
            detail.mask = pc.score_orig;
            pc_details_filtered.push_back(detail);
        }
    } else if (include_patchcore_details && abnormal_pass_base) {
        if (pc.area > 0 && pc.w > 0 && pc.h > 0) {
            DetectionResult detail;
            detail.name = "Abnormal";
            detail.area = pc.area;
            detail.x = static_cast<float>(pc.x);
            detail.y = static_cast<float>(pc.y);
            detail.w = static_cast<float>(pc.w);
            detail.h = static_cast<float>(pc.h);
            detail.contrast = pc.contrast;
            detail.score = pc.score;
            detail.mask = pc.area_orig;
            pc_details_filtered.push_back(detail);
        }
    }

    if (has_yolo_detections) {
        float max_score = 0.0f;
        for (const auto& d : yolo_details_filtered) max_score = std::max(max_score, d.score);
        output.yolo_score = max_score;
    } else {
        output.yolo_score = 0.0f;
    }
    output.patchcore_score = pc.score;
    output.patchcore_area_ratio = pc.area_ratio;

    if (has_yolo_detections || abnormal_is_ng) {
        output.result = "NG";
    } else {
        output.result = "OK";
    }

    output.details.clear();
    output.details.reserve(pc_details_filtered.size() + yolo_details_filtered.size());
    output.details.insert(output.details.end(), pc_details_filtered.begin(), pc_details_filtered.end());
    output.details.insert(output.details.end(), yolo_details_filtered.begin(), yolo_details_filtered.end());
}

}  // namespace InspectionDLL::Internal
