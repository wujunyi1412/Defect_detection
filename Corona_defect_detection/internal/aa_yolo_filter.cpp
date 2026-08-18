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
    if (detail.h <= 0.0f) return false;

    const float center_y = detail.y + detail.h * 0.5f;
    if (center_y < config.center_y_min || center_y > config.center_y_max) return false;

    return detail.w / detail.h > config.min_width_height_ratio;
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

}  // namespace

void ApplyAaYoloFilter(std::vector<DetectionResult>& details,
                       const InspectionConfig::AaFilterConfig& config) {
    if (!config.enable && !config.position_filter_enable) return;
    details.erase(
        std::remove_if(details.begin(), details.end(),
                       [&](const DetectionResult& detail) {
                           return (config.enable && ShouldFilterByBandRule(detail, config)) ||
                                  (config.position_filter_enable &&
                                   ShouldFilterByPositionRule(detail, config));
                       }),
        details.end());
}

}  // namespace InspectionDLL::Internal
