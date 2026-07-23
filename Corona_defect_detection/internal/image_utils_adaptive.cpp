#include "image_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace InspectionDLL::Internal {
namespace {

float InvalidContrastRatio() {
    return std::numeric_limits<float>::quiet_NaN();
}

struct BackgroundSample {
    double x;
    double y;
    double value;
};

cv::Rect ClampRect(int x, int y, int w, int h, const cv::Size& image_size) {
    if (image_size.width <= 0 || image_size.height <= 0 || w <= 0 || h <= 0) return {};
    const long long right = static_cast<long long>(x) + w;
    const long long bottom = static_cast<long long>(y) + h;
    const int x1 = std::clamp(x, 0, image_size.width);
    const int y1 = std::clamp(y, 0, image_size.height);
    const int x2 = static_cast<int>(std::clamp<long long>(right, 0, image_size.width));
    const int y2 = static_cast<int>(std::clamp<long long>(bottom, 0, image_size.height));
    if (x2 <= x1 || y2 <= y1) return {};
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

cv::Rect ExpandRect(const cv::Rect& rect, int margin, const cv::Size& image_size) {
    const int x1 = std::max(0, rect.x - margin);
    const int y1 = std::max(0, rect.y - margin);
    const int x2 = std::min(image_size.width, rect.x + rect.width + margin);
    const int y2 = std::min(image_size.height, rect.y + rect.height + margin);
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

cv::Mat BuildDetectionMask(const cv::Mat& binary_mask,
                           const cv::Rect& detection,
                           const cv::Size& image_size) {
    if (binary_mask.empty() || binary_mask.channels() != 1) return {};

    cv::Mat local;
    if (binary_mask.size() == image_size) {
        local = binary_mask(detection).clone();
    } else if (binary_mask.size() == detection.size()) {
        local = binary_mask.clone();
    } else {
        cv::resize(binary_mask, local, detection.size(), 0.0, 0.0, cv::INTER_NEAREST);
    }

    cv::Mat local_u8;
    if (local.depth() == CV_8U) {
        local_u8 = local;
    } else {
        local.convertTo(local_u8, CV_8U);
    }
    cv::threshold(local_u8, local_u8, 0, 255, cv::THRESH_BINARY);
    return local_u8;
}

cv::Mat FindOuterDarkRegion(const cv::Mat& gray_f32, int transition_guard) {
    cv::Mat result(gray_f32.size(), CV_8U, cv::Scalar(0));
    std::vector<float> values;
    values.reserve(gray_f32.total());
    for (int y = 0; y < gray_f32.rows; ++y) {
        const float* row = gray_f32.ptr<float>(y);
        for (int x = 0; x < gray_f32.cols; ++x) {
            if (std::isfinite(row[x])) values.push_back(row[x]);
        }
    }
    if (values.empty()) return result;

    const float low_reference = Percentile(values, 10.0f);
    const float material_reference = Percentile(values, 80.0f);
    if (material_reference <= 0.0f) return result;
    const float dark_threshold = std::min(material_reference * 0.35f,
                                          low_reference + 0.20f *
                                                              (material_reference - low_reference));

    cv::Mat dark_candidates;
    cv::compare(gray_f32, dark_threshold, dark_candidates, cv::CMP_LE);
    cv::Mat labels;
    const int label_count = cv::connectedComponents(dark_candidates, labels, 8, CV_32S);
    if (label_count <= 1) return result;

    std::vector<uchar> outer_labels(static_cast<size_t>(label_count), 0);
    for (int y = 0; y < labels.rows; ++y) {
        outer_labels[labels.at<int>(y, 0)] = 1;
        outer_labels[labels.at<int>(y, labels.cols - 1)] = 1;
    }
    for (int x = 0; x < labels.cols; ++x) {
        outer_labels[labels.at<int>(0, x)] = 1;
        outer_labels[labels.at<int>(labels.rows - 1, x)] = 1;
    }
    outer_labels[0] = 0;

    for (int y = 0; y < labels.rows; ++y) {
        const int* label_row = labels.ptr<int>(y);
        uchar* result_row = result.ptr<uchar>(y);
        for (int x = 0; x < labels.cols; ++x) {
            if (outer_labels[label_row[x]]) result_row[x] = 255;
        }
    }
    if (transition_guard > 0 && cv::countNonZero(result) > 0) {
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::dilate(result, result, kernel, cv::Point(-1, -1), transition_guard);
    }
    return result;
}

bool FitPlane(const std::vector<BackgroundSample>& samples,
              const std::vector<uchar>* selected,
              cv::Vec3d& coefficients) {
    cv::Matx33d normal = cv::Matx33d::zeros();
    cv::Vec3d rhs(0.0, 0.0, 0.0);
    int count = 0;
    for (size_t i = 0; i < samples.size(); ++i) {
        if (selected && !(*selected)[i]) continue;
        const BackgroundSample& sample = samples[i];
        const cv::Vec3d basis(sample.x, sample.y, 1.0);
        normal += basis * basis.t();
        rhs += basis * sample.value;
        ++count;
    }
    return count >= 3 && cv::solve(normal, rhs, coefficients, cv::DECOMP_SVD);
}

double RobustMedian(std::vector<float> values) {
    return values.empty() ? 0.0 : static_cast<double>(Median(values));
}

bool FitRobustBackgroundPlane(const std::vector<BackgroundSample>& samples,
                              cv::Vec3d& coefficients,
                              double& background_median) {
    if (samples.size() < 12 || !FitPlane(samples, nullptr, coefficients)) return false;

    std::vector<float> background_values;
    std::vector<float> residuals;
    background_values.reserve(samples.size());
    residuals.reserve(samples.size());
    for (const BackgroundSample& sample : samples) {
        background_values.push_back(static_cast<float>(sample.value));
        const double predicted = coefficients[0] * sample.x +
                                 coefficients[1] * sample.y + coefficients[2];
        residuals.push_back(static_cast<float>(sample.value - predicted));
    }
    background_median = RobustMedian(background_values);
    const double residual_median = RobustMedian(residuals);

    std::vector<float> deviations;
    deviations.reserve(residuals.size());
    for (float residual : residuals) {
        deviations.push_back(static_cast<float>(std::abs(residual - residual_median)));
    }
    const double sigma = std::max(0.5, 1.4826 * RobustMedian(deviations));
    const double inlier_limit = 3.0 * sigma;

    std::vector<uchar> inliers(samples.size(), 0);
    int inlier_count = 0;
    for (size_t i = 0; i < residuals.size(); ++i) {
        if (std::abs(residuals[i] - residual_median) <= inlier_limit) {
            inliers[i] = 1;
            ++inlier_count;
        }
    }
    if (inlier_count >= 12) FitPlane(samples, &inliers, coefficients);
    return true;
}

float RobustForegroundRatio(std::vector<float> ratios, ContrastPolarity polarity) {
    if (ratios.empty()) return 0.0f;
    std::sort(ratios.begin(), ratios.end());

    if (polarity == ContrastPolarity::Auto) {
        polarity = ratios[ratios.size() / 2] <= 1.0f
                       ? ContrastPolarity::Dark
                       : ContrastPolarity::Bright;
    }
    if (ratios.size() < 5) return ratios[ratios.size() / 2];

    const float low_percentile = polarity == ContrastPolarity::Bright ? 60.0f : 10.0f;
    const float high_percentile = polarity == ContrastPolarity::Bright ? 90.0f : 40.0f;
    const float low = Percentile(ratios, low_percentile);
    const float high = Percentile(ratios, high_percentile);

    double sum = 0.0;
    int count = 0;
    for (float ratio : ratios) {
        if (ratio >= low && ratio <= high) {
            sum += ratio;
            ++count;
        }
    }
    return count > 0 ? static_cast<float>(sum / count) : ratios[ratios.size() / 2];
}

}  // namespace

float CalculateContrastRatioAdaptive(const cv::Mat& gray_img,
                                     const cv::Mat& binary_mask,
                                     int x,
                                     int y,
                                     int w,
                                     int h,
                                     ContrastPolarity polarity) {
    if (gray_img.empty() || binary_mask.empty() || gray_img.channels() != 1) {
        return InvalidContrastRatio();
    }

    const cv::Rect detection = ClampRect(x, y, w, h, gray_img.size());
    if (detection.empty()) return InvalidContrastRatio();
    const cv::Mat detection_mask = BuildDetectionMask(binary_mask, detection, gray_img.size());
    const int foreground_area = detection_mask.empty() ? 0 : cv::countNonZero(detection_mask);
    if (foreground_area <= 0) return InvalidContrastRatio();

    const double equivalent_radius = std::sqrt(static_cast<double>(foreground_area) / CV_PI);
    const int inner_radius = std::clamp(static_cast<int>(std::round(equivalent_radius * 0.20)), 2, 8);
    const int outer_radius = std::clamp(static_cast<int>(std::round(equivalent_radius * 0.80)),
                                        inner_radius + 4, 32);
    const cv::Rect context = ExpandRect(detection, outer_radius + 2, gray_img.size());
    const cv::Rect detection_in_context(detection.x - context.x, detection.y - context.y,
                                        detection.width, detection.height);

    cv::Mat gray_f32;
    gray_img(context).convertTo(gray_f32, CV_32F);
    cv::Mat foreground_mask(context.size(), CV_8U, cv::Scalar(0));
    detection_mask.copyTo(foreground_mask(detection_in_context));

    const cv::Mat inner_kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(2 * inner_radius + 1, 2 * inner_radius + 1));
    const cv::Mat outer_kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(2 * outer_radius + 1, 2 * outer_radius + 1));
    cv::Mat inner_dilated;
    cv::Mat outer_dilated;
    cv::dilate(foreground_mask, inner_dilated, inner_kernel);
    cv::dilate(foreground_mask, outer_dilated, outer_kernel);
    cv::Mat background_ring;
    cv::bitwise_and(outer_dilated, ~inner_dilated, background_ring);
    const int transition_guard = std::clamp(std::min(detection.width, detection.height) / 12, 1, 3);
    const cv::Mat outer_dark_region = FindOuterDarkRegion(gray_f32, transition_guard);
    background_ring.setTo(0, outer_dark_region);

    std::vector<BackgroundSample> background_samples;
    background_samples.reserve(static_cast<size_t>(cv::countNonZero(background_ring)));
    for (int yy = 0; yy < gray_f32.rows; ++yy) {
        const float* gray_row = gray_f32.ptr<float>(yy);
        const uchar* ring_row = background_ring.ptr<uchar>(yy);
        for (int xx = 0; xx < gray_f32.cols; ++xx) {
            if (ring_row[xx] && std::isfinite(gray_row[xx])) {
                background_samples.push_back({static_cast<double>(xx), static_cast<double>(yy),
                                              static_cast<double>(gray_row[xx])});
            }
        }
    }

    cv::Vec3d background_plane;
    double background_median = 0.0;
    if (!FitRobustBackgroundPlane(background_samples, background_plane, background_median)) {
        return InvalidContrastRatio();
    }

    const double denominator_floor = std::max(1.0, std::abs(background_median) * 0.10);
    std::vector<float> foreground_ratios;
    foreground_ratios.reserve(static_cast<size_t>(foreground_area));
    for (int yy = 0; yy < gray_f32.rows; ++yy) {
        const float* gray_row = gray_f32.ptr<float>(yy);
        const uchar* foreground_row = foreground_mask.ptr<uchar>(yy);
        for (int xx = 0; xx < gray_f32.cols; ++xx) {
            if (!foreground_row[xx] || !std::isfinite(gray_row[xx])) continue;
            const double predicted_background = background_plane[0] * xx +
                                                background_plane[1] * yy + background_plane[2];
            if (!std::isfinite(predicted_background) || predicted_background < denominator_floor) continue;
            const double ratio = gray_row[xx] / predicted_background;
            if (std::isfinite(ratio) && ratio >= 0.0 && ratio <= 10.0) {
                foreground_ratios.push_back(static_cast<float>(ratio));
            }
        }
    }
    if (foreground_ratios.empty()) return InvalidContrastRatio();
    const float ratio = RobustForegroundRatio(std::move(foreground_ratios), polarity);
    return IsContrastRatioValid(ratio) ? ratio : InvalidContrastRatio();
}

}  // namespace InspectionDLL::Internal
