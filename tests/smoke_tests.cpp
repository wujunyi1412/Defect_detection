#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>

#include "config.h"
#include "internal/image_utils.h"
#include "internal/overlay_renderer.h"
#include "internal/yolo_postprocess.h"
#include "logger.h"

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
    if (AssertTrue(cfg.faiss_threads == 16, "faiss_threads default should be 16")) return 1;
    return 0;
}

int TestLoggerLevelQuery() {
    const InspectionLogging::LoggerConfig original = InspectionLogging::GetLoggerConfig();
    InspectionLogging::LoggerConfig config;
    config.enabled = true;
    config.min_level = InspectionLogging::LogLevel::Info;
    config.log_to_stderr = false;
    InspectionLogging::SetLoggerConfig(config);

    const bool info_enabled =
        InspectionLogging::IsLogEnabled(InspectionLogging::LogLevel::Info);
    const bool debug_enabled =
        InspectionLogging::IsLogEnabled(InspectionLogging::LogLevel::Debug);
    InspectionLogging::SetLoggerConfig(original);

    if (AssertTrue(info_enabled, "info logging should be enabled at info level")) return 1;
    if (AssertTrue(!debug_enabled, "debug logging should be disabled at info level")) return 1;
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

int TestYoloPostprocessReusesBinaryMask() {
    cv::Mat gray(80, 80, CV_8U, cv::Scalar(100));
    gray(cv::Rect(20, 20, 20, 20)).setTo(50);

    YOLO::Detection detection{};
    detection.class_id = 0;
    detection.confidence = 0.9f;
    detection.x1 = 20.0f;
    detection.y1 = 20.0f;
    detection.x2 = 40.0f;
    detection.y2 = 40.0f;
    detection.mask = cv::Mat::zeros(gray.size(), CV_8U);
    detection.mask(cv::Rect(20, 20, 20, 20)).setTo(255);

    const InspectionDLL::Internal::YoloDerived result =
        InspectionDLL::Internal::AnalyzeYolo({detection}, gray, 0.8f);
    if (AssertTrue(result.details.size() == 1, "YOLO postprocess should keep the detection")) return 1;
    if (AssertTrue(result.details[0].area == 400, "YOLO mask area should remain unchanged")) return 1;
    if (AssertTrue(result.details[0].mask.data == detection.mask.data,
                   "YOLO postprocess should share the existing binary mask")) return 1;
    if (AssertTrue(cv::norm(result.details[0].mask, detection.mask, cv::NORM_INF) == 0.0,
                   "shared YOLO mask contents should remain unchanged")) return 1;
    return 0;
}

}  // namespace

int main() {
    if (TestOverlayTextFit()) return 1;
    if (TestConfigDefaults()) return 1;
    if (TestLoggerLevelQuery()) return 1;
    if (TestCombinedInferencePreprocessingMatchesLegacyPath()) return 1;
    if (TestYoloPostprocessReusesBinaryMask()) return 1;
    std::cout << "[PASS] smoke_tests" << std::endl;
    return 0;
}
