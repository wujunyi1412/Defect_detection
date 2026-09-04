#pragma once

#include <vector>

#include "config.h"
#include "../inference_dll.h"

namespace InspectionDLL::Internal {

void ApplyIqtYoloFilter(std::vector<DetectionResult>& details,
                        const InspectionConfig::IqtFilterConfig& config);

}  // namespace InspectionDLL::Internal
