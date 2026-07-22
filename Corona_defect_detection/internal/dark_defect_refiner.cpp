#include "dark_defect_refiner.h"

#include <algorithm>
#include <cmath>

#include "image_utils.h"

namespace InspectionDLL::Internal {

bool RefineDarkDefectGeometryOnly(const cv::Mat& gray_yolo, DetectionResult& detail) {
    if (gray_yolo.empty()) return false;

    const int W = gray_yolo.cols;
    const int H = gray_yolo.rows;
    int x1 = static_cast<int>(std::round(detail.x));
    int y1 = static_cast<int>(std::round(detail.y));
    int x2 = static_cast<int>(std::round(detail.x + detail.w));
    int y2 = static_cast<int>(std::round(detail.y + detail.h));

    x1 = std::max(0, std::min(x1, std::max(0, W - 1)));
    y1 = std::max(0, std::min(y1, std::max(0, H - 1)));
    x2 = std::max(x1 + 1, std::min(x2, W));
    y2 = std::max(y1 + 1, std::min(y2, H));
    if (x2 <= x1 || y2 <= y1) return false;

    const cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
    if (roi.width <= 0 || roi.height <= 0) return false;
    const int roi_area = std::max(1, roi.width * roi.height);
    const int original_box_area = std::max(1, static_cast<int>(std::round(detail.w * detail.h)));
    const int original_mask_area = std::max(1, detail.area);
    const bool is_small_box = roi.width <= 30 && roi.height <= 30;

    const cv::Mat roi_gray = gray_yolo(roi);
    cv::Mat roi_u8 = ConvertGrayToU8Normalized(roi_gray);
    if (roi_u8.empty()) return false;

    if (roi.width < 4 || roi.height < 4) {
        detail.area = std::min(original_mask_area, roi_area);
        detail.x = static_cast<float>(x1);
        detail.y = static_cast<float>(y1);
        detail.w = static_cast<float>(roi.width);
        detail.h = static_cast<float>(roi.height);
        return true;
    }

    if (is_small_box) {
        cv::Mat med_bg;
        cv::medianBlur(roi_u8, med_bg, 3);

        cv::Mat dark_response_small;
        cv::subtract(med_bg, roi_u8, dark_response_small);

        cv::Scalar mean_gray_small, std_gray_small;
        cv::Scalar mean_resp_small, std_resp_small;
        cv::meanStdDev(roi_u8, mean_gray_small, std_gray_small); 
        cv::meanStdDev(dark_response_small, mean_resp_small, std_resp_small);

        const double seed_gray_thr = mean_gray_small[0] - std::max(6.0, 0.30 * std_gray_small[0]);
        const double seed_resp_thr = mean_resp_small[0] + std::max(3.0, 0.65 * std_resp_small[0]);
        const double grow_gray_thr = mean_gray_small[0] - std::max(2.0, 0.08 * std_gray_small[0]);
        const double grow_resp_thr = mean_resp_small[0] + std::max(1.0, 0.12 * std_resp_small[0]);

        cv::Mat seed_gray = roi_u8 <= seed_gray_thr;
        cv::Mat seed_resp = dark_response_small >= seed_resp_thr;
        cv::Mat seed_mask;
        cv::bitwise_and(seed_gray, seed_resp, seed_mask);
        if (cv::countNonZero(seed_mask) == 0) {
            cv::bitwise_or(seed_gray, seed_resp, seed_mask);
        }
        if (cv::countNonZero(seed_mask) == 0) {
            return false;
        }

        cv::Mat grow_gray = roi_u8 <= grow_gray_thr;
        cv::Mat grow_resp = dark_response_small >= grow_resp_thr;
        cv::Mat grow_mask;
        cv::bitwise_or(grow_gray, grow_resp, grow_mask);

        const int kernel_size = (std::min(roi.width, roi.height) >= 9) ? 3 : 1;
        if (kernel_size > 1) {
            cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
            cv::morphologyEx(seed_mask, seed_mask, cv::MORPH_OPEN, k, cv::Point(-1, -1), 1);
            cv::morphologyEx(grow_mask, grow_mask, cv::MORPH_CLOSE, k, cv::Point(-1, -1), 1);
        }

        cv::Mat labels_small, stats_small, centroids_small;
        const int num_labels_small = cv::connectedComponentsWithStats(grow_mask, labels_small, stats_small, centroids_small, 8, CV_32S);
        if (num_labels_small <= 1) return false;

        double best_score = -1.0;
        cv::Rect best_roi;
        int best_label = -1;
        for (int lab = 1; lab < num_labels_small; ++lab) {
            const int area = stats_small.at<int>(lab, cv::CC_STAT_AREA);
            if (area <= 0) continue;

            const int left = stats_small.at<int>(lab, cv::CC_STAT_LEFT);
            const int top = stats_small.at<int>(lab, cv::CC_STAT_TOP);
            const int width = stats_small.at<int>(lab, cv::CC_STAT_WIDTH);
            const int height = stats_small.at<int>(lab, cv::CC_STAT_HEIGHT);
            if (width <= 0 || height <= 0) continue;

            cv::Rect comp_roi(left, top, width, height);
            const cv::Mat comp_mask = (labels_small(comp_roi) == lab);
            const int seed_overlap = cv::countNonZero(seed_mask(comp_roi) & comp_mask);
            if (seed_overlap <= 0) continue;
            const cv::Scalar mean_resp = cv::mean(dark_response_small(comp_roi), comp_mask);
            const cv::Scalar mean_inv = cv::mean(255 - roi_u8(comp_roi), comp_mask);
            const double fill_ratio = static_cast<double>(area) / static_cast<double>(std::max(1, width * height));

            double score = mean_resp[0] * 2.4 + mean_inv[0] * 1.2 + fill_ratio * 16.0;
            score += static_cast<double>(seed_overlap) * 6.0;
            score -= std::max(0, width * height - 120) * 0.3;
            if (score > best_score) {
                best_score = score;
                best_roi = comp_roi;
                best_label = lab;
            }
        }

        if (best_score < 0.0 || best_roi.width <= 0 || best_roi.height <= 0 || best_label <= 0) {
            return false;
        }

        const int pad = (std::min(roi.width, roi.height) >= 12) ? 1 : 0;
        best_roi.x = std::max(0, best_roi.x - pad);
        best_roi.y = std::max(0, best_roi.y - pad);
        best_roi.width = std::min(roi.width - best_roi.x, best_roi.width + 2 * pad);
        best_roi.height = std::min(roi.height - best_roi.y, best_roi.height + 2 * pad);

        const int best_box_area = std::max(1, best_roi.width * best_roi.height);
        const cv::Mat final_component_mask = (labels_small(best_roi) == best_label);
        int refined_area = cv::countNonZero(final_component_mask);
        refined_area = std::min(refined_area, best_box_area);
        if (refined_area <= 0) return false;

        detail.x = static_cast<float>(x1 + best_roi.x);
        detail.y = static_cast<float>(y1 + best_roi.y);
        detail.w = static_cast<float>(best_roi.width);
        detail.h = static_cast<float>(best_roi.height);
        detail.area = refined_area;
        return true;
    }

    cv::Mat border_mask = cv::Mat::zeros(roi.size(), CV_8U);
    const int border_band = std::max(1, std::min(std::min(roi.width, roi.height) / 6, 6));
    border_mask.rowRange(0, border_band).setTo(255);
    border_mask.rowRange(std::max(0, roi.height - border_band), roi.height).setTo(255);
    border_mask.colRange(0, border_band).setTo(255);
    border_mask.colRange(std::max(0, roi.width - border_band), roi.width).setTo(255);

    cv::Scalar border_mean_scalar;
    cv::Scalar border_std_scalar;
    cv::meanStdDev(roi_u8, border_mean_scalar, border_std_scalar, border_mask);
    const double border_mean = border_mean_scalar[0];
    const double border_std = border_std_scalar[0];

    cv::Mat blur_bg;
    const int blur_size = std::max(5, ((std::min(roi.width, roi.height) / 5) | 1));
    cv::GaussianBlur(roi_u8, blur_bg, cv::Size(blur_size, blur_size), 0.0);

    cv::Mat dark_response;
    cv::subtract(blur_bg, roi_u8, dark_response);

    cv::Scalar mean_resp_scalar;
    cv::Scalar std_resp_scalar;
    cv::meanStdDev(dark_response, mean_resp_scalar, std_resp_scalar);
    cv::Scalar mean_gray_scalar;
    cv::Scalar std_gray_scalar;
    cv::meanStdDev(roi_u8, mean_gray_scalar, std_gray_scalar);

    const double seed_intensity_thr = std::min(border_mean - std::max(8.0, border_std * 0.75),
                                               mean_gray_scalar[0] - 0.30 * std_gray_scalar[0]);
    const double grow_intensity_thr = std::min(border_mean - std::max(3.0, border_std * 0.25),
                                               mean_gray_scalar[0] - 0.08 * std_gray_scalar[0]);
    const double seed_resp_thr = mean_resp_scalar[0] + 0.90 * std_resp_scalar[0];
    const double grow_resp_thr = mean_resp_scalar[0] + 0.20 * std_resp_scalar[0];

    cv::Mat seed_dark = roi_u8 <= seed_intensity_thr;
    cv::Mat seed_resp = dark_response >= seed_resp_thr;
    cv::Mat seed_mask;
    cv::bitwise_and(seed_dark, seed_resp, seed_mask);
    if (cv::countNonZero(seed_mask) == 0) {
        cv::bitwise_or(seed_dark, seed_resp, seed_mask);
    }

    cv::Mat grow_dark = roi_u8 <= grow_intensity_thr;
    cv::Mat grow_resp = dark_response >= grow_resp_thr;
    cv::Mat grow_mask;
    cv::bitwise_and(grow_dark, grow_resp, grow_mask);
    if (cv::countNonZero(grow_mask) < std::max(9, roi_area / 40)) {
        cv::bitwise_or(grow_dark, grow_resp, grow_mask);
    }

    const int seed_kernel_size = std::max(3, ((std::min(roi.width, roi.height) / 24) | 1));
    const int grow_kernel_size = std::max(3, ((std::min(roi.width, roi.height) / 18) | 1));
    cv::Mat seed_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(seed_kernel_size, seed_kernel_size));
    cv::Mat grow_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(grow_kernel_size, grow_kernel_size));
    cv::morphologyEx(seed_mask, seed_mask, cv::MORPH_OPEN, seed_kernel, cv::Point(-1, -1), 1);
    cv::morphologyEx(seed_mask, seed_mask, cv::MORPH_CLOSE, seed_kernel, cv::Point(-1, -1), 1);
    cv::morphologyEx(grow_mask, grow_mask, cv::MORPH_CLOSE, grow_kernel, cv::Point(-1, -1), 1);

    cv::Rect final_roi(0, 0, roi.width, roi.height);
    int final_area = std::min(original_mask_area, roi_area);

    cv::Mat labels, stats, centroids;
    const int num_labels = cv::connectedComponentsWithStats(grow_mask, labels, stats, centroids, 8, CV_32S);
    if (num_labels > 1) {
        const int min_component_area = std::max(12, roi_area / 250);
        cv::Mat seed_dilated;
        cv::dilate(seed_mask, seed_dilated, grow_kernel, cv::Point(-1, -1), 1);

        double best_score = -1.0;
        cv::Rect best_roi;
        int best_area = 0;

        for (int lab = 1; lab < num_labels; ++lab) {
            const int area = stats.at<int>(lab, cv::CC_STAT_AREA);
            if (area < min_component_area) continue;

            const int left = stats.at<int>(lab, cv::CC_STAT_LEFT);
            const int top = stats.at<int>(lab, cv::CC_STAT_TOP);
            const int width = stats.at<int>(lab, cv::CC_STAT_WIDTH);
            const int height = stats.at<int>(lab, cv::CC_STAT_HEIGHT);
            if (width <= 0 || height <= 0) continue;

            cv::Rect comp_roi(left, top, width, height);
            const cv::Mat comp_mask = (labels(comp_roi) == lab);
            const int overlap = cv::countNonZero(seed_dilated(comp_roi) & comp_mask);
            const double overlap_ratio = static_cast<double>(overlap) / static_cast<double>(std::max(1, area));
            const double area_ratio = static_cast<double>(area) / static_cast<double>(roi_area);
            const cv::Scalar mean_resp = cv::mean(dark_response(comp_roi), comp_mask);
            const cv::Scalar mean_inv = cv::mean(255 - roi_u8(comp_roi), comp_mask);
            const double fill_ratio = static_cast<double>(area) / static_cast<double>(std::max(1, width * height));

            double score = mean_resp[0] * 2.0 + mean_inv[0] * 1.0 + area_ratio * 60.0 + fill_ratio * 15.0;
            if (overlap > 0) score += 80.0 + overlap_ratio * 80.0;
            else score -= 25.0;
            if (area_ratio < 0.08) score -= 35.0;

            if (score > best_score) {
                best_score = score;
                best_roi = comp_roi;
                best_area = area;
            }
        }

        if (best_score >= 0.0 && best_roi.width > 0 && best_roi.height > 0) {
            const double selected_ratio = static_cast<double>(best_area) / static_cast<double>(roi_area);
            if (selected_ratio < 0.15) {
                final_roi = cv::Rect(0, 0, roi.width, roi.height);
                final_area = std::min(roi_area, original_mask_area);
            } else {
                const int pad_w = std::min(std::max(1, roi.width / 10), std::max(1, static_cast<int>(std::round(best_roi.width * 0.12f))));
                const int pad_h = std::min(std::max(1, roi.height / 10), std::max(1, static_cast<int>(std::round(best_roi.height * 0.12f))));
                final_roi.x = std::max(0, best_roi.x - pad_w);
                final_roi.y = std::max(0, best_roi.y - pad_h);
                final_roi.width = std::min(roi.width - final_roi.x, best_roi.width + 2 * pad_w);
                final_roi.height = std::min(roi.height - final_roi.y, best_roi.height + 2 * pad_h);
                final_area = std::min(best_area, original_mask_area);
            }
        }
    }

    const double roi_dark_offset = border_mean - mean_gray_scalar[0];
    const double weak_dark_ratio = static_cast<double>(cv::countNonZero(grow_mask)) / static_cast<double>(roi_area);
    if (weak_dark_ratio > 0.55 && roi_dark_offset > std::max(2.0, border_std * 0.2)) {
        final_roi = cv::Rect(0, 0, roi.width, roi.height);
        final_area = std::min(original_mask_area, roi_area);
    }

    if (final_roi.width <= 0 || final_roi.height <= 0) return false;
    const int final_box_area = std::max(1, final_roi.width * final_roi.height);
    final_area = std::min(final_area, std::min(final_box_area, original_mask_area));
    if (final_area <= 0) final_area = std::min(original_mask_area, final_box_area);
    if (final_box_area < original_box_area / 6) {
        final_roi = cv::Rect(0, 0, roi.width, roi.height);
        final_area = std::min(original_mask_area, roi_area);
    }

    const double refined_box_ratio = static_cast<double>(final_box_area) / static_cast<double>(original_box_area);
    const double refined_area_ratio = static_cast<double>(final_area) / static_cast<double>(original_mask_area);
    if (refined_box_ratio > 0.98 && refined_area_ratio > 0.98) {
        return false;
    }

    detail.x = static_cast<float>(x1 + final_roi.x);
    detail.y = static_cast<float>(y1 + final_roi.y);
    detail.w = static_cast<float>(final_roi.width);
    detail.h = static_cast<float>(final_roi.height);
    detail.area = final_area;
    return true;
}

}  // namespace InspectionDLL::Internal
