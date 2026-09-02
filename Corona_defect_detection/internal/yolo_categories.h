#pragma once

#include <array>
#include <cstddef>
#include <string>

#include "image_utils.h"

namespace InspectionDLL::Internal {

struct YoloCategoryDefinition {
    const char* name;
    ContrastPolarity contrast_polarity;
    bool keep_contrast_below_threshold;
};

inline constexpr std::array<YoloCategoryDefinition, 8> kYoloCategories = {{
    {"Glue_overflow", ContrastPolarity::Dark, true},
    {"Decolorization", ContrastPolarity::Dark, true},
    {"Stain", ContrastPolarity::Dark, true},
    {"Stripes", ContrastPolarity::Auto, false},
    {"BrightStripes", ContrastPolarity::Bright, false},
    {"Bright_clusters", ContrastPolarity::Bright, false},
    {"Line_artifacts", ContrastPolarity::Dark, true},
    {"LineArtifacts", ContrastPolarity::Bright, false},
}};

inline const YoloCategoryDefinition* FindYoloCategory(int class_id) {
    if (class_id < 0 || static_cast<std::size_t>(class_id) >= kYoloCategories.size()) {
        return nullptr;
    }
    return &kYoloCategories[static_cast<std::size_t>(class_id)];
}

inline const YoloCategoryDefinition* FindYoloCategory(const std::string& name) {
    for (const auto& category : kYoloCategories) {
        if (name == category.name) return &category;
    }
    return nullptr;
}

}  // namespace InspectionDLL::Internal
