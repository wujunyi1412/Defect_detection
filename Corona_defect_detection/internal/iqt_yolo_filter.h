#pragma once

#include <vector>

#include "config.h"
#include "../inference_dll.h"

namespace InspectionDLL::Internal {

void ApplyIqtYoloFilter(std::vector<DetectionResult>& details,
                        const InspectionConfig::IqtFilterConfig& config,
                        std::vector<DetectionResult>* removed_details = nullptr);

}  // namespace InspectionDLL::Internal
