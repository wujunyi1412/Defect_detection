#pragma once

#include <vector>

#include "config.h"
#include "../inference_dll.h"

namespace InspectionDLL::Internal {

void ApplyAaYoloFilter(std::vector<DetectionResult>& details,
                       const InspectionConfig::AaFilterConfig& config);

}  // namespace InspectionDLL::Internal
