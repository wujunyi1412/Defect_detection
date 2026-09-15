#include "result_category_mapper.h"

#include <algorithm>
#include <cctype>

namespace InspectionDLL::Internal {
namespace {

std::string ToLowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

}  // namespace

std::string MapResultCategoryName(
    const std::string& source_name,
    const InspectionConfig::CategoryNameMap& mapping) {
    const auto mapped = mapping.find(ToLowerCopy(source_name));
    return mapped == mapping.end() || mapped->second.empty()
               ? source_name
               : mapped->second;
}

void ApplyResultCategoryMapping(
    std::vector<DetectionResult>& details,
    const InspectionConfig::CategoryNameMap& mapping) {
    if (mapping.empty()) return;
    for (auto& detail : details) {
        detail.name = MapResultCategoryName(detail.name, mapping);
    }
}

}  // namespace InspectionDLL::Internal
