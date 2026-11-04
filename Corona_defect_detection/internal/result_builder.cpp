#include "result_builder.h"

#include <algorithm>
#include <cmath>
#include <limits>
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

cv::Rect DetailRect(const DetectionResult& detail, const cv::Size& image_size) {
    if (image_size.width <= 0 || image_size.height <= 0 ||
        !std::isfinite(detail.x) || !std::isfinite(detail.y) ||
        !std::isfinite(detail.w) || !std::isfinite(detail.h) ||
        detail.w <= 0.0f || detail.h <= 0.0f) {
        return {};
    }
    const int x1 = std::clamp(static_cast<int>(std::floor(detail.x)), 0, image_size.width);
    const int y1 = std::clamp(static_cast<int>(std::floor(detail.y)), 0, image_size.height);
    const int x2 = std::clamp(static_cast<int>(std::ceil(detail.x + detail.w)), 0, image_size.width);
    const int y2 = std::clamp(static_cast<int>(std::ceil(detail.y + detail.h)), 0, image_size.height);
    return x2 > x1 && y2 > y1 ? cv::Rect(x1, y1, x2 - x1, y2 - y1) : cv::Rect();
}

cv::Mat DetailMaskInImage(const DetectionResult& detail, const cv::Size& image_size) {
    const cv::Rect rect = DetailRect(detail, image_size);
    if (rect.empty() || detail.mask.empty() || detail.mask.channels() != 1) return {};

    cv::Mat local;
    if (detail.mask.size() == image_size) {
        local = detail.mask(rect);
    } else if (detail.mask.size() == rect.size()) {
        local = detail.mask;
    } else {
        cv::resize(detail.mask, local, rect.size(), 0.0, 0.0, cv::INTER_NEAREST);
    }
    cv::Mat local_u8;
    local.convertTo(local_u8, CV_8U);
    cv::threshold(local_u8, local_u8, 0, 255, cv::THRESH_BINARY);

    cv::Mat result(image_size, CV_8U, cv::Scalar(0));
    local_u8.copyTo(result(rect));
    return result;
}

double RefinementQuality(const DetectionResult& candidate,
                         const cv::Mat& candidate_mask,
                         const cv::Mat& original_mask,
                         const cv::Rect& original_rect,
                         InspectionConfig::DefectPolarity polarity) {
    const int candidate_area = cv::countNonZero(candidate_mask);
    if (candidate_area <= 0) return -std::numeric_limits<double>::infinity();

    double mask_agreement = 0.5;
    const int original_area = original_mask.empty() ? 0 : cv::countNonZero(original_mask);
    if (original_area > 0) {
        cv::Mat overlap;
        cv::bitwise_and(candidate_mask, original_mask, overlap);
        const int overlap_area = cv::countNonZero(overlap);
        mask_agreement = 2.0 * overlap_area / static_cast<double>(candidate_area + original_area);
    }

    double contrast_strength = 0.0;
    if (IsContrastRatioValid(candidate.contrast) && candidate.contrast > 0.0f) {
        const double signed_log = polarity == InspectionConfig::DefectPolarity::Dark
                                      ? -std::log(candidate.contrast)
                                      : std::log(candidate.contrast);
        contrast_strength = 1.0 - std::exp(-6.0 * std::max(0.0, signed_log));
    }

    double area_plausibility = 0.5;
    if (original_area > 0) {
        const double ratio = candidate_area / static_cast<double>(original_area);
        area_plausibility = std::exp(-0.65 * std::abs(std::log(std::max(1e-6, ratio))));
    }

    double center_score = 0.5;
    if (!original_rect.empty()) {
        const cv::Rect candidate_rect = DetailRect(candidate, candidate_mask.size());
        const cv::Point2d original_center(original_rect.x + original_rect.width * 0.5,
                                          original_rect.y + original_rect.height * 0.5);
        const cv::Point2d candidate_center(candidate_rect.x + candidate_rect.width * 0.5,
                                           candidate_rect.y + candidate_rect.height * 0.5);
        const double scale = std::max(1.0, std::hypot(original_rect.width, original_rect.height));
        center_score = std::exp(-3.0 * cv::norm(candidate_center - original_center) / scale);
    }

    // Gray evidence is primary. Mask agreement stabilizes ambiguous low-contrast
    // cases without making the ensemble blindly follow an inaccurate YOLO mask.
    return 0.55 * contrast_strength + 0.25 * mask_agreement +
           0.12 * area_plausibility + 0.08 * center_score;
}

bool FuseRefinementCandidates(const cv::Mat& gray_yolo,
                              const DetectionResult& original,
                              const DetectionResult& mask_candidate,
                              const DetectionResult& adaptive_candidate,
                              InspectionConfig::DefectPolarity polarity,
                              ContrastCalculationMode contrast_mode,
                              DetectionResult& result) {
    const cv::Mat mask_result = DetailMaskInImage(mask_candidate, gray_yolo.size());
    const cv::Mat adaptive_result = DetailMaskInImage(adaptive_candidate, gray_yolo.size());
    if (mask_result.empty() || adaptive_result.empty()) return false;

    cv::Mat intersection;
    cv::Mat union_mask;
    cv::bitwise_and(mask_result, adaptive_result, intersection);
    cv::bitwise_or(mask_result, adaptive_result, union_mask);
    const int intersection_area = cv::countNonZero(intersection);
    const int union_area = cv::countNonZero(union_mask);
    if (intersection_area <= 0 || union_area <= 0) return false;

    // Conflicting localizations must not be averaged into a geometrically invalid
    // result. They are handled by quality-based selection in the caller.
    const double iou = intersection_area / static_cast<double>(union_area);
    if (iou < 0.35) return false;

    const cv::Mat original_mask = DetailMaskInImage(original, gray_yolo.size());
    const cv::Rect original_rect = DetailRect(original, gray_yolo.size());
    const double mask_quality = RefinementQuality(
        mask_candidate, mask_result, original_mask, original_rect, polarity);
    const double adaptive_quality = RefinementQuality(
        adaptive_candidate, adaptive_result, original_mask, original_rect, polarity);
    const double quality_sum = std::max(1e-6, mask_quality + adaptive_quality);
    // Keep both algorithms influential even when one quality score is dominant.
    const double mask_weight = std::clamp(mask_quality / quality_sum, 0.25, 0.75);

    std::vector<cv::Point> union_points;
    cv::findNonZero(union_mask, union_points);
    cv::Rect roi = cv::boundingRect(union_points);
    roi.x = std::max(0, roi.x - 2);
    roi.y = std::max(0, roi.y - 2);
    roi.width = std::min(gray_yolo.cols - roi.x, roi.width + 4);
    roi.height = std::min(gray_yolo.rows - roi.y, roi.height + 4);

    auto signed_distance = [&](const cv::Mat& global_mask) -> cv::Mat {
        cv::Mat local = global_mask(roi);
        cv::Mat inside_distance;
        cv::Mat outside_distance;
        cv::distanceTransform(local, inside_distance, cv::DIST_L2, 3);
        cv::Mat inverse;
        cv::bitwise_not(local, inverse);
        cv::distanceTransform(inverse, outside_distance, cv::DIST_L2, 3);
        cv::Mat signed_result;
        cv::subtract(inside_distance, outside_distance, signed_result);
        return signed_result;
    };

    const cv::Mat mask_distance = signed_distance(mask_result);
    const cv::Mat adaptive_distance = signed_distance(adaptive_result);
    cv::Mat blended;
    cv::addWeighted(mask_distance, mask_weight,
                    adaptive_distance, 1.0 - mask_weight, 0.0, blended);
    cv::Mat fused_local;
    cv::compare(blended, 0.0, fused_local, cv::CMP_GE);

    // Retain only fused components connected to pixels supported by both methods.
    cv::Mat labels;
    const int label_count = cv::connectedComponents(fused_local, labels, 8, CV_32S);
    cv::Mat trusted_labels(label_count, 1, CV_8U, cv::Scalar(0));
    const cv::Mat intersection_local = intersection(roi);
    for (int y = 0; y < labels.rows; ++y) {
        const int* label_row = labels.ptr<int>(y);
        const uchar* intersection_row = intersection_local.ptr<uchar>(y);
        for (int x = 0; x < labels.cols; ++x) {
            if (intersection_row[x]) trusted_labels.at<uchar>(label_row[x], 0) = 255;
        }
    }
    trusted_labels.at<uchar>(0, 0) = 0;
    for (int y = 0; y < fused_local.rows; ++y) {
        uchar* fused_row = fused_local.ptr<uchar>(y);
        const int* label_row = labels.ptr<int>(y);
        for (int x = 0; x < fused_local.cols; ++x) {
            if (!trusted_labels.at<uchar>(label_row[x], 0)) fused_row[x] = 0;
        }
    }

    std::vector<cv::Point> fused_points;
    cv::findNonZero(fused_local, fused_points);
    if (fused_points.empty()) return false;
    const cv::Rect local_box = cv::boundingRect(fused_points);
    const int fused_area = cv::countNonZero(fused_local);
    const int smaller_area = std::min(cv::countNonZero(mask_result),
                                      cv::countNonZero(adaptive_result));
    const int larger_area = std::max(cv::countNonZero(mask_result),
                                     cv::countNonZero(adaptive_result));
    if (fused_area < static_cast<int>(std::floor(smaller_area * 0.60)) ||
        fused_area > static_cast<int>(std::ceil(larger_area * 1.25))) {
        return false;
    }

    result = original;
    result.x = static_cast<float>(roi.x + local_box.x);
    result.y = static_cast<float>(roi.y + local_box.y);
    result.w = static_cast<float>(local_box.width);
    result.h = static_cast<float>(local_box.height);
    result.area = fused_area;
    result.mask = fused_local(local_box).clone();
    const ContrastPolarity contrast_polarity =
        polarity == InspectionConfig::DefectPolarity::Dark
            ? ContrastPolarity::Dark
            : ContrastPolarity::Bright;
    const float contrast = CalculateContrastRatioByMode(
        gray_yolo, result.mask, static_cast<int>(result.x), static_cast<int>(result.y),
        local_box.width, local_box.height, contrast_polarity, contrast_mode);
    if (IsContrastRatioValid(contrast)) result.contrast = contrast;
    return true;
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
        // Run both refiners from the untouched YOLO result. When both succeed,
        // combine agreeing boundaries and select by evidence when they disagree.
        const DetectionResult original = detail;
        DetectionResult mask_candidate = detail;
        DetectionResult adaptive_candidate = detail;
        const bool mask_succeeded = refine_mask_guided(mask_candidate);
        const bool adaptive_succeeded = refine_gray_adaptive(adaptive_candidate);
        if (mask_succeeded && !adaptive_succeeded) {
            detail = std::move(mask_candidate);
            return true;
        }
        if (!mask_succeeded && adaptive_succeeded) {
            detail = std::move(adaptive_candidate);
            return true;
        }
        if (!mask_succeeded) return false;

        DetectionResult fused_candidate;
        if (FuseRefinementCandidates(gray_yolo, original, mask_candidate,
                                     adaptive_candidate, polarity, contrast_mode,
                                     fused_candidate)) {
            detail = std::move(fused_candidate);
            return true;
        }

        const cv::Mat original_mask = DetailMaskInImage(original, gray_yolo.size());
        const cv::Mat mask_result = DetailMaskInImage(mask_candidate, gray_yolo.size());
        const cv::Mat adaptive_result = DetailMaskInImage(adaptive_candidate, gray_yolo.size());
        const cv::Rect original_rect = DetailRect(original, gray_yolo.size());
        const double mask_quality = RefinementQuality(
            mask_candidate, mask_result, original_mask, original_rect, polarity);
        const double adaptive_quality = RefinementQuality(
            adaptive_candidate, adaptive_result, original_mask, original_rect, polarity);
        detail = adaptive_quality > mask_quality
                     ? std::move(adaptive_candidate)
                     : std::move(mask_candidate);
        return true;
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
        context.draw_filtered_defects ? &removed_by_aa : nullptr);
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
