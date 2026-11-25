#include "result_builder.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "aa_yolo_filter.h"
#include "dark_defect_refiner.h"
#include "image_utils.h"
#include "iqt_yolo_filter.h"
#include "patchcore_yolo_iou_filter.h"
#include "result_category_mapper.h"
#include "yolo_categories.h"

namespace InspectionDLL::Internal {
namespace {

bool PassContrastThreshold(float contrast, float threshold, bool keep_below_threshold) {
    if (threshold <= 0.0f) return true;
    if (!IsContrastRatioValid(contrast)) return false;
    return keep_below_threshold ? contrast < threshold : contrast >= threshold;
}

bool PassMinimumGeometry(
    const DetectionResult& detail,
    float confidence_threshold,
    int min_width,
    int min_height,
    int min_area) {
    return detail.score >= confidence_threshold &&
           (min_width <= 0 || detail.w >= static_cast<float>(min_width)) &&
           (min_height <= 0 || detail.h >= static_cast<float>(min_height)) &&
           (min_area <= 0 || detail.area >= min_area);
}

bool UsesTieredContrastFilter(const InspectionConfig::CategoryFilterConfig& filter) {
    return filter.contrast_threshold_1 > 0.0f || filter.contrast_threshold_2 > 0.0f;
}

bool PassTieredContrastAndSize(
    const DetectionResult& detail,
    const InspectionConfig::CategoryFilterConfig& filter) {
    if (!UsesTieredContrastFilter(filter)) return true;
    if (!IsContrastRatioValid(detail.contrast)) return false;

    if (filter.contrast_direction == InspectionConfig::ContrastFilterDirection::LessEqual) {
        if (detail.contrast <= filter.contrast_threshold_1) {
            return PassMinimumGeometry(
                detail, filter.confidence_threshold_0,
                filter.min_width_0, filter.min_height_0, filter.min_area_0);
        }
        if (detail.contrast <= filter.contrast_threshold_2) {
            return PassMinimumGeometry(
                detail, filter.confidence_threshold_1,
                filter.min_width_1, filter.min_height_1, filter.min_area_1);
        }
    } else {
        if (detail.contrast >= filter.contrast_threshold_2) {
            return PassMinimumGeometry(
                detail, filter.confidence_threshold_0,
                filter.min_width_0, filter.min_height_0, filter.min_area_0);
        }
        if (detail.contrast >= filter.contrast_threshold_1) {
            return PassMinimumGeometry(
                detail, filter.confidence_threshold_1,
                filter.min_width_1, filter.min_height_1, filter.min_area_1);
        }
    }
    return PassMinimumGeometry(
        detail, filter.confidence_threshold_2,
        filter.min_width_2, filter.min_height_2, filter.min_area_2);
}

bool RefineYoloDetail(const cv::Mat& gray_yolo,
                      DetectionResult& detail,
                      const YoloCategoryDefinition& category,
                      const InspectionConfig::CategoryFilterConfig& filter,
                      ContrastCalculationMode contrast_mode) {
    if (filter.refinement_method == InspectionConfig::DefectRefinementMethod::None) {
        return false;
    }

    InspectionConfig::DefectPolarity polarity = filter.defect_polarity;
    if (polarity == InspectionConfig::DefectPolarity::Auto) {
        if (category.contrast_polarity == ContrastPolarity::Dark) {
            polarity = InspectionConfig::DefectPolarity::Dark;
        } else if (category.contrast_polarity == ContrastPolarity::Bright) {
            polarity = InspectionConfig::DefectPolarity::Bright;
        } else if (IsContrastRatioValid(detail.contrast)) {
            polarity = detail.contrast <= 1.0f
                           ? InspectionConfig::DefectPolarity::Dark
                           : InspectionConfig::DefectPolarity::Bright;
        } else {
            return false;
        }
    }

    const auto refine_mask_guided = [&](DetectionResult& candidate) {
        return polarity == InspectionConfig::DefectPolarity::Dark
                   ? RefineDarkDefectGeometryMaskGuided(gray_yolo, candidate, contrast_mode)
                   : RefineBrightDefectGeometryMaskGuided(gray_yolo, candidate, contrast_mode);
    };
    const auto refine_gray_adaptive = [&](DetectionResult& candidate) {
        return polarity == InspectionConfig::DefectPolarity::Dark
                   ? RefineDarkDefectGeometryAdaptive(gray_yolo, candidate, contrast_mode)
                   : RefineBrightDefectGeometryAdaptive(gray_yolo, candidate, contrast_mode);
    };
    const auto refine_gray_adaptive_v2 = [&](DetectionResult& candidate) {
        return polarity == InspectionConfig::DefectPolarity::Dark
                   ? RefineDarkDefectGeometryAdaptiveV2(gray_yolo, candidate, contrast_mode)
                   : RefineBrightDefectGeometryAdaptiveV2(gray_yolo, candidate, contrast_mode);
    };

    if (polarity != InspectionConfig::DefectPolarity::Dark &&
        polarity != InspectionConfig::DefectPolarity::Bright) {
        return false;
    }

    if (filter.refinement_method == InspectionConfig::DefectRefinementMethod::MaskGuided) {
        return refine_mask_guided(detail);
    }
    if (filter.refinement_method == InspectionConfig::DefectRefinementMethod::GrayAdaptive) {
        return refine_gray_adaptive(detail);
    }
    if (filter.refinement_method == InspectionConfig::DefectRefinementMethod::HybridCascade) {
        // Gray-adaptive V2 keeps grayscale segmentation authoritative. The YOLO
        // mask is only a reliability-gated soft prior during component ranking.
        DetectionResult enhanced_candidate = detail;
        if (refine_gray_adaptive_v2(enhanced_candidate)) {
            detail = std::move(enhanced_candidate);
            return true;
        }
        // The enhanced path uses stricter evidence. Preserve baseline recall if
        // those additional checks reject an otherwise valid grayscale defect.
        return refine_gray_adaptive(detail);
    }
    return false;
}

}  // namespace

void ComposeOutput(const PatchCoreDerived& pc,
                   const YoloDerived& yolo,
                   const cv::Size& img_shape,
                   const cv::Mat& gray_patchcore,
                   InferenceResult& output,
                   const ResultComposeContext& context) {
    output.filtered_details.clear();
    std::vector<DetectionResult> yolo_details_filtered = yolo.details;
    ApplyAaYoloFilter(yolo_details_filtered, context.aa_filter);

    output.patchcore_score = pc.score;
    output.patchcore_area_ratio = pc.area_ratio;

    std::vector<DetectionResult> pc_details;

    if (pc.has_crosshair_grid) {
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
            const float contrast = CalculateContrastRatioByMode(
                gray_patchcore, pc.score_orig, x1, y1, w1, h1,
                ContrastPolarity::Auto, context.contrast_mode);

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
            pc_details.push_back(detail);
        }
    }

    if (pc.has_defect && !pc.has_crosshair_grid) {
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
        pc_details.push_back(detail);
    }

    ApplyPatchCoreYoloIouFilter(
        pc_details,
        yolo_details_filtered,
        context.patchcore_yolo_iou_threshold);
    ApplyIqtYoloFilter(yolo_details_filtered, context.iqt_filter);

    output.yolo_score = 0.0f;
    for (const auto& detail : yolo_details_filtered) {
        output.yolo_score = std::max(output.yolo_score, detail.score);
    }
    const bool has_yolo_detections = !yolo_details_filtered.empty();
    output.result = (pc.score >= context.score_threshold || has_yolo_detections) ? "NG" : "OK";

    output.details.clear();
    output.details.reserve(pc_details.size() + yolo_details_filtered.size());
    output.details.insert(output.details.end(), pc_details.begin(), pc_details.end());
    output.details.insert(output.details.end(), yolo_details_filtered.begin(), yolo_details_filtered.end());
}

void ComposeOutputWithDefectFilter(const PatchCoreDerived& pc,
                                   const YoloDerived& yolo,
                                   const cv::Size& img_shape,
                                   const cv::Mat& gray_patchcore,
                                   const cv::Mat& gray_yolo,
                                   InferenceResult& output,
                                   const ResultComposeContext& context) {
    output.filtered_details.clear();
    auto record_filtered = [&](const DetectionResult& detail,
                               const char* source,
                               const char* reason) {
        if (!context.draw_filtered_defects) return;
        output.filtered_details.push_back({detail, source, reason});
    };

    std::vector<DetectionResult> yolo_details_filtered;
    yolo_details_filtered.reserve(yolo.details.size());
    for (const auto& d : yolo.details) {
        const auto filter_it = context.category_filters.find(d.name);
        const YoloCategoryDefinition* category = FindYoloCategory(d.name);
        if (filter_it == context.category_filters.end() || !category) {
            record_filtered(d, "YOLO", "category_config");
            continue;
        }
        const InspectionConfig::CategoryFilterConfig* cf = &filter_it->second;

        DetectionResult detail = d;
        RefineYoloDetail(gray_yolo, detail, *category, *cf, context.contrast_mode);
        if (!cf->enable) {
            record_filtered(detail, "YOLO", "category_disabled");
            continue;
        }
        if (detail.score < cf->confidence_threshold) {
            record_filtered(detail, "YOLO", "confidence");
            continue;
        }
        if (UsesTieredContrastFilter(*cf)) {
            if (!PassTieredContrastAndSize(detail, *cf)) {
                record_filtered(detail, "YOLO", "tiered_threshold");
                continue;
            }
        } else {
            if (!PassContrastThreshold(
                    detail.contrast,
                    cf->contrast_threshold,
                    category->keep_contrast_below_threshold)) {
                record_filtered(detail, "YOLO", "contrast");
                continue;
            }
            if (!PassMinimumGeometry(
                    detail, 0.0f, cf->min_width, cf->min_height, cf->min_area)) {
                record_filtered(detail, "YOLO", "geometry");
                continue;
            }
        }
        yolo_details_filtered.push_back(detail);
    }
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

    if (pc.has_crosshair_grid && (abnormal_is_ng || context.draw_filtered_defects)) {
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
            const float contrast = CalculateContrastRatioByMode(
                gray_patchcore, pc.score_orig, x1, y1, w1, h1,
                ContrastPolarity::Auto, context.contrast_mode);
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
            if (!abnormal_is_ng) {
                record_filtered(detail, "PatchCore", "base_threshold");
                continue;
            }
            if (context.abnormal_filter.min_width > 0 && w1 < context.abnormal_filter.min_width) {
                record_filtered(detail, "PatchCore", "geometry");
                continue;
            }
            if (context.abnormal_filter.min_height > 0 && h1 < context.abnormal_filter.min_height) {
                record_filtered(detail, "PatchCore", "geometry");
                continue;
            }
            if (context.abnormal_filter.min_area > 0 && area < context.abnormal_filter.min_area) {
                record_filtered(detail, "PatchCore", "geometry");
                continue;
            }
            if (!PassContrastThreshold(
                    contrast,
                    context.abnormal_filter.contrast_threshold,
                    false)) {
                record_filtered(detail, "PatchCore", "contrast");
                continue;
            }
            pc_details_filtered.push_back(detail);
        }
    } else if (pc.area > 0 && pc.w > 0 && pc.h > 0) {
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
        if (abnormal_pass_base) {
            pc_details_filtered.push_back(detail);
        } else {
            record_filtered(detail, "PatchCore", "base_threshold");
        }
    }

    std::vector<DetectionResult> removed_by_iou;
    std::vector<DetectionResult> removed_by_aa;
    std::vector<DetectionResult> removed_by_iqt;
    ApplyPatchCoreYoloIouFilter(
        pc_details_filtered,
        yolo_details_filtered,
        context.patchcore_yolo_iou_threshold,
        context.draw_filtered_defects ? &removed_by_iou : nullptr);
    ApplyAaYoloFilter(
        yolo_details_filtered,
        context.aa_filter,
        context.draw_filtered_defects ? &removed_by_aa : nullptr,
        gray_yolo);
    ApplyIqtYoloFilter(
        yolo_details_filtered,
        context.iqt_filter,
        context.draw_filtered_defects ? &removed_by_iqt : nullptr);
    for (const auto& detail : removed_by_iou) record_filtered(detail, "PatchCore", "yolo_iou");
    for (const auto& detail : removed_by_aa) record_filtered(detail, "YOLO", "AA");
    for (const auto& detail : removed_by_iqt) record_filtered(detail, "YOLO", "IQT");

    const bool has_yolo_detections = !yolo_details_filtered.empty();
    const bool has_patchcore_detections = !pc_details_filtered.empty();

    if (has_yolo_detections) {
        float max_score = 0.0f;
        for (const auto& d : yolo_details_filtered) max_score = std::max(max_score, d.score);
        output.yolo_score = max_score;
    } else {
        output.yolo_score = 0.0f;
    }
    output.patchcore_score = pc.score;
    output.patchcore_area_ratio = pc.area_ratio;

    if (has_yolo_detections || has_patchcore_detections) {
        output.result = "NG";
    } else {
        output.result = "OK";
    }

    output.details.clear();
    output.details.reserve(pc_details_filtered.size() + yolo_details_filtered.size());
    ApplyResultCategoryMapping(pc_details_filtered, context.category_name_mapping);
    ApplyResultCategoryMapping(yolo_details_filtered, context.category_name_mapping);
    for (auto& filtered : output.filtered_details) {
        filtered.detail.name = MapResultCategoryName(
            filtered.detail.name, context.category_name_mapping);
    }
    output.details.insert(output.details.end(), pc_details_filtered.begin(), pc_details_filtered.end());
    output.details.insert(output.details.end(), yolo_details_filtered.begin(), yolo_details_filtered.end());
}

}  // namespace InspectionDLL::Internal
