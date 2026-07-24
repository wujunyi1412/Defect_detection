#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>

#include "config.h"
#include "internal/image_utils.h"
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

int TestCombinedInferencePreprocessingMatchesLegacyPath() {
    cv::Mat input(96, 128, CV_32FC1);
    for (int y = 0; y < input.rows; ++y) {
        float* row = input.ptr<float>(y);
        for (int x = 0; x < input.cols; ++x) {
            row[x] = static_cast<float>((x * 7 + y * 13) % 257);
        }
    }
    input.rowRange(0, 10).setTo(0.0f);

    const cv::Mat legacy_patchcore =
        InspectionDLL::Internal::ProcessTIF32ForPatchcore(input);
    const cv::Mat legacy_yolo =
        InspectionDLL::Internal::ProcessForYolo(input);
    const InspectionDLL::Internal::InferenceImages combined =
        InspectionDLL::Internal::PrepareInferenceImages(input);

    if (AssertTrue(cv::norm(legacy_patchcore, combined.patchcore_bgr, cv::NORM_INF) == 0.0,
                   "combined PatchCore BGR preprocessing should match the legacy path")) return 1;
    if (AssertTrue(cv::norm(legacy_yolo, combined.yolo_bgr, cv::NORM_INF) == 0.0,
                   "combined YOLO BGR preprocessing should match the legacy path")) return 1;

    cv::Mat legacy_patchcore_gray;
    cv::Mat legacy_yolo_gray;
    cv::cvtColor(legacy_patchcore, legacy_patchcore_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(legacy_yolo, legacy_yolo_gray, cv::COLOR_BGR2GRAY);
    if (AssertTrue(cv::norm(legacy_patchcore_gray, combined.patchcore_gray, cv::NORM_INF) == 0.0,
                   "combined PatchCore gray preprocessing should match the legacy path")) return 1;
    if (AssertTrue(cv::norm(legacy_yolo_gray, combined.yolo_gray, cv::NORM_INF) == 0.0,
                   "combined YOLO gray preprocessing should match the legacy path")) return 1;
    return 0;
}

}  // namespace

int main() {
    if (TestOverlayTextFit()) return 1;
    if (TestConfigDefaults()) return 1;
    if (TestCombinedInferencePreprocessingMatchesLegacyPath()) return 1;
    std::cout << "[PASS] smoke_tests" << std::endl;
    return 0;
}
