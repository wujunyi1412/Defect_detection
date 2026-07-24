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
    if (AssertTrue(cfg.draw_defect_box, "draw_defect_box default should be true")) return 1;
    if (AssertTrue(cfg.draw_box_details, "draw_box_details default should be true")) return 1;
    if (AssertTrue(!cfg.draw_defect_mask, "draw_defect_mask default should be false")) return 1;
    if (AssertTrue(cfg.defect_mask_color_r == 255 &&
                   cfg.defect_mask_color_g == 0 &&
                   cfg.defect_mask_color_b == 0,
                   "defect mask default color should be red")) return 1;
    if (AssertTrue(cfg.defect_mask_alpha == 0.35f, "defect_mask_alpha default should be 0.35")) return 1;
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
