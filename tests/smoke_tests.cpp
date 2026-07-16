#include <iostream>
#include <string>

#include "config.h"
#include "internal/overlay_renderer.h"

namespace {

int AssertTrue(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "[FAIL] " << message << std::endl;
        return 1;
    }
    return 0;
}

int TestOverlayTextFit() {
    const std::string original = "VeryLongDefectLabel123456";
    const std::string clipped = InspectionOverlay::FitTextToWidthForTest(original, 30, 0.55, 1);
    return AssertTrue(!clipped.empty(), "FitTextToWidth should produce non-empty output for narrow width");
}

int TestConfigDefaults() {
    InspectionConfig::InspectionConfigData cfg;
    if (AssertTrue(cfg.draw_box_details, "draw_box_details default should be true")) return 1;
    if (AssertTrue(cfg.log_enabled, "log_enabled default should be true")) return 1;
    return 0;
}

}  // namespace

int main() {
    if (TestOverlayTextFit()) return 1;
    if (TestConfigDefaults()) return 1;
    std::cout << "[PASS] smoke_tests" << std::endl;
    return 0;
}
