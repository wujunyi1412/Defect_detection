#include "aa_yolo_filter.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>

namespace InspectionDLL::Internal {
namespace {

std::string ToLowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

bool ContainsCategory(const std::vector<std::string>& categories,
                      const std::string& defect_name) {
    const std::string lowered_name = ToLowerCopy(defect_name);
    return std::any_of(categories.begin(), categories.end(),
                       [&](const std::string& category) {
                           return ToLowerCopy(category) == lowered_name;
                       });
}

bool ShouldFilterByBandRule(const DetectionResult& detail,
                            const InspectionConfig::AaFilterConfig& config) {
    if (!ContainsCategory(config.categories, detail.name)) return false;
    if (!std::isfinite(detail.y) || !std::isfinite(detail.w) ||
        !std::isfinite(detail.h) || detail.w < 0.0f || detail.h <= 0.0f) return false;

    const float center_y = detail.y + detail.h * 0.5f;
    if (center_y < config.center_y_min || center_y > config.center_y_max) return false;

    const bool exceeds_ratio = detail.w / detail.h > config.min_width_height_ratio;
    const bool exceeds_width = config.max_width > 0.0f && detail.w > config.max_width;
    return exceeds_ratio || exceeds_width;
}

bool ShouldFilterByPositionRule(const DetectionResult& detail,
                                const InspectionConfig::AaFilterConfig& config) {
    if (!ContainsCategory(config.position_categories, detail.name)) return false;
    if (!std::isfinite(detail.x) || !std::isfinite(detail.y) ||
        !std::isfinite(detail.w) || !std::isfinite(detail.h)) return false;

    return detail.x >= config.position_x_min && detail.x <= config.position_x_max &&
           detail.y >= config.position_y_min && detail.y <= config.position_y_max &&
           detail.w >= config.position_width_min && detail.w <= config.position_width_max &&
           detail.h >= config.position_height_min && detail.h <= config.position_height_max;
}

bool MatchesBlackSpotGeometry(const DetectionResult& detail,
                              const InspectionConfig::AaFilterConfig& config) {
    if (!ContainsCategory(config.black_spot_categories, detail.name)) return false;
    if (!std::isfinite(detail.x) || !std::isfinite(detail.y) ||
        !std::isfinite(detail.w) || !std::isfinite(detail.h) ||
        detail.area < 0) return false;

    return detail.w >= config.black_spot_width_min &&
           detail.w <= config.black_spot_width_max &&
           detail.h >= config.black_spot_height_min &&
           detail.h <= config.black_spot_height_max &&
           detail.area >= config.black_spot_mask_area_min &&
           detail.area <= config.black_spot_mask_area_max;
}

bool HasBlackSpotAppearance(const DetectionResult& detail,
                            const cv::Mat& gray_yolo,
                            const InspectionConfig::AaFilterConfig& config) {
    if (gray_yolo.empty() || gray_yolo.channels() != 1) return false;

    const int x0 = std::clamp(static_cast<int>(std::floor(detail.x)), 0, gray_yolo.cols);
    const int y0 = std::clamp(static_cast<int>(std::floor(detail.y)), 0, gray_yolo.rows);
    const int x1 = std::clamp(static_cast<int>(std::ceil(detail.x + detail.w)), 0, gray_yolo.cols);
    const int y1 = std::clamp(static_cast<int>(std::ceil(detail.y + detail.h)), 0, gray_yolo.rows);
    if (x1 - x0 < 12 || y1 - y0 < 12) return false;

    cv::Mat roi32;
    gray_yolo(cv::Rect(x0, y0, x1 - x0, y1 - y0)).convertTo(roi32, CV_32F);
    cv::Mat smooth;
    const int min_dim = std::min(roi32.cols, roi32.rows);
    int kernel = std::max(5, static_cast<int>(std::round(min_dim * 0.07f)) | 1);
    kernel = std::min(kernel, (min_dim % 2 == 0) ? min_dim - 1 : min_dim);
    cv::GaussianBlur(roi32, smooth, cv::Size(kernel, kernel), 0.0);

    double min_value = 0.0;
    cv::Point dark_center;
    cv::minMaxLoc(smooth, &min_value, nullptr, &dark_center, nullptr);

    const float half_w = std::max(1.0f, 0.5f * static_cast<float>(smooth.cols));
    const float half_h = std::max(1.0f, 0.5f * static_cast<float>(smooth.rows));
    const float center_dx = (static_cast<float>(dark_center.x) + 0.5f - half_w) / half_w;
    const float center_dy = (static_cast<float>(dark_center.y) + 0.5f - half_h) / half_h;
    if (std::sqrt(center_dx * center_dx + center_dy * center_dy) >
        config.black_spot_max_center_offset_ratio) return false;

    constexpr int kRingCount = 8;
    float sums[kRingCount] = {};
    int counts[kRingCount] = {};
    const float radius_x = std::max(1.0f, static_cast<float>(std::min(dark_center.x + 1, smooth.cols - dark_center.x)));
    const float radius_y = std::max(1.0f, static_cast<float>(std::min(dark_center.y + 1, smooth.rows - dark_center.y)));
    for (int y = 0; y < smooth.rows; ++y) {
        const float* row = smooth.ptr<float>(y);
        for (int x = 0; x < smooth.cols; ++x) {
            const float dx = (static_cast<float>(x - dark_center.x)) / radius_x;
            const float dy = (static_cast<float>(y - dark_center.y)) / radius_y;
            const float radius = std::sqrt(dx * dx + dy * dy);
            if (radius > 1.0f) continue;
            const int ring = std::min(kRingCount - 1, static_cast<int>(radius * kRingCount));
            sums[ring] += row[x];
            ++counts[ring];
        }
    }

    float means[kRingCount] = {};
    for (int i = 0; i < kRingCount; ++i) {
        if (counts[i] == 0) return false;
        means[i] = sums[i] / static_cast<float>(counts[i]);
    }
    const float center_mean = 0.5f * (means[0] + means[1]);
    const float outer_mean = 0.5f * (means[6] + means[7]);
    if (outer_mean - center_mean < config.black_spot_min_center_darkness) return false;

    int rising_steps = 0;
    for (int i = 0; i + 1 < kRingCount; ++i) {
        if (means[i + 1] + config.black_spot_monotonic_tolerance >= means[i]) {
            ++rising_steps;
        }
    }
    const float monotonic_ratio = static_cast<float>(rising_steps) /
                                  static_cast<float>(kRingCount - 1);
    return monotonic_ratio >= config.black_spot_min_monotonic_ratio;
}

bool ShouldFilterByBlackSpotRule(const DetectionResult& detail,
                                 const cv::Mat& gray_yolo,
                                 const InspectionConfig::AaFilterConfig& config) {
    return MatchesBlackSpotGeometry(detail, config) &&
           HasBlackSpotAppearance(detail, gray_yolo, config);
}

}  // namespace

void ApplyAaYoloFilter(std::vector<DetectionResult>& details,
                       const InspectionConfig::AaFilterConfig& config,
                       std::vector<DetectionResult>* removed_details,
                       const cv::Mat& gray_yolo) {
    if (!config.enable && !config.position_filter_enable &&
        !config.black_spot_filter_enable) return;
    details.erase(
        std::remove_if(details.begin(), details.end(),
                       [&](const DetectionResult& detail) {
                           const bool removed =
                               (config.enable && ShouldFilterByBandRule(detail, config)) ||
                               (config.position_filter_enable &&
                                ShouldFilterByPositionRule(detail, config)) ||
                               (config.black_spot_filter_enable &&
                                ShouldFilterByBlackSpotRule(detail, gray_yolo, config));
                           if (removed && removed_details) removed_details->push_back(detail);
                           return removed;
                       }),
        details.end());
}

}  // namespace InspectionDLL::Internal
