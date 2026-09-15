#pragma once

#include <string>
#include <vector>

#include "config.h"
#include "../inference_dll.h"

namespace InspectionDLL::Internal {

std::string MapResultCategoryName(
    const std::string& source_name,
    const InspectionConfig::CategoryNameMap& mapping);

void ApplyResultCategoryMapping(
    std::vector<DetectionResult>& details,
    const InspectionConfig::CategoryNameMap& mapping);

}  // namespace InspectionDLL::Internal
