#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "config.h"
#include "internal/aa_yolo_filter.h"
#include "internal/image_utils.h"
#include "internal/overlay_renderer.h"
#include "internal/result_builder.h"
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

int TestFloatOverlayFileIs8Bit() {
    cv::Mat input(48, 64, CV_32FC1);
    for (int y = 0; y < input.rows; ++y) {
        float* row = input.ptr<float>(y);
        for (int x = 0; x < input.cols; ++x) {
            row[x] = static_cast<float>(x * 0.25 + y * 0.1);
        }
    }

    InspectionDLL::InferenceResult result;
    result.result = "OK";
    result.yolo_score = 0.0f;
    result.patchcore_score = 0.0f;
    result.patchcore_area_ratio = 0.0f;
    InspectionDLL::MaskOverlayOptions mask_options;
    const std::filesystem::path output_path =
        std::filesystem::temp_directory_path() / "corona_overlay_smoke.png";

    std::string error;
    const bool saved = InspectionOverlay::RenderResultOverlayToFile(
        input.ptr<float>(0),
        input.cols,
        input.rows,
        result,
        true,
        false,
        true,
        mask_options,
        false,
        output_path.string().c_str(),
        error);
    if (AssertTrue(saved, "float overlay export should succeed: " + error)) return 1;

    const cv::Mat output = cv::imread(output_path.string(), cv::IMREAD_UNCHANGED);
    std::error_code remove_error;
    std::filesystem::remove(output_path, remove_error);
    if (AssertTrue(!output.empty(), "exported overlay should be readable")) return 1;
    if (AssertTrue(output.depth() == CV_8U, "exported overlay should be 8-bit")) return 1;
    if (AssertTrue(output.channels() == 3, "exported overlay should be three-channel")) return 1;
    return 0;
}

int TestConfigDefaults() {
    InspectionConfig::InspectionConfigData cfg;
    if (AssertTrue(cfg.yolo_enabled, "yolo_enabled default should be true")) return 1;
    if (AssertTrue(cfg.patchcore_enabled, "patchcore_enabled default should be true")) return 1;
    if (AssertTrue(!cfg.suppress_patchcore_when_yolo_detected,
                   "suppress_patchcore_when_yolo_detected default should be false")) return 1;
    if (AssertTrue(!cfg.aa_filter.enable, "AA filter default should be disabled")) return 1;
    if (AssertTrue(cfg.aa_filter.center_y_min == 665.0f &&
                   cfg.aa_filter.center_y_max == 715.0f,
                   "AA filter default center Y range should be [665, 715]")) return 1;
    if (AssertTrue(cfg.aa_filter.min_width_height_ratio == 3.0f,
                   "AA filter default width-height ratio should be 3")) return 1;
    if (AssertTrue(cfg.draw_defect_box, "draw_defect_box default should be true")) return 1;
    if (AssertTrue(!cfg.expand_defect_box, "expand_defect_box default should be false")) return 1;
    if (AssertTrue(cfg.draw_box_details, "draw_box_details default should be true")) return 1;
    if (AssertTrue(!cfg.draw_defect_mask, "draw_defect_mask default should be false")) return 1;
    if (AssertTrue(!cfg.concat_original_image, "concat_original_image default should be false")) return 1;
    if (AssertTrue(cfg.defect_mask_color_r == 255 &&
                   cfg.defect_mask_color_g == 0 &&
                   cfg.defect_mask_color_b == 0,
                   "defect mask default color should be red")) return 1;
    if (AssertTrue(cfg.defect_mask_alpha == 0.35f, "defect_mask_alpha default should be 0.35")) return 1;
    if (AssertTrue(cfg.log_enabled, "log_enabled default should be true")) return 1;
    return 0;
}

int TestDefectBoxExpansionClipsToImage() {
    const cv::Size image_size(20, 15);
    const cv::Rect centered = InspectionOverlay::ExpandDefectRectForTest(
        cv::Rect(5, 4, 5, 4), image_size);
    if (AssertTrue(centered == cv::Rect(2, 1, 11, 10),
                   "centered defect box should expand by three pixels on every side")) return 1;

    const cv::Rect at_edge = InspectionOverlay::ExpandDefectRectForTest(
        cv::Rect(0, 0, 5, 4), image_size);
    if (AssertTrue(at_edge == cv::Rect(0, 0, 8, 7),
                   "expanded defect box should be clipped at image boundaries")) return 1;
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

int TestPatchCoreDetailSuppressionWhenYoloDetected() {
    InspectionDLL::Internal::PatchCoreDerived patchcore;
    patchcore.score = 2.1f;
    patchcore.area_ratio = 0.5f;
    patchcore.has_defect = true;
    patchcore.area = 100;
    patchcore.x = 5;
    patchcore.y = 6;
    patchcore.w = 10;
    patchcore.h = 10;

    InspectionDLL::DetectionResult yolo_detail;
    yolo_detail.name = "BrightStripes";
    yolo_detail.score = 0.9f;
    yolo_detail.area = 25;
    yolo_detail.w = 5.0f;
    yolo_detail.h = 5.0f;

    InspectionDLL::Internal::YoloDerived yolo;
    yolo.has_detections = true;
    yolo.details.push_back(yolo_detail);

    InspectionDLL::Internal::ResultComposeContext context;
    InspectionDLL::InferenceResult output;
    const cv::Mat gray(32, 32, CV_8U, cv::Scalar(128));
    InspectionDLL::Internal::ComposeOutputWithDefectFilter(
        patchcore, yolo, gray.size(), gray, gray, output, context);

    if (AssertTrue(output.details.size() == 2,
                   "default composition should preserve PatchCore and YOLO details")) return 1;

    context.suppress_patchcore_when_yolo_detected = true;
    InspectionDLL::Internal::ComposeOutputWithDefectFilter(
        patchcore, yolo, gray.size(), gray, gray, output, context);

    if (AssertTrue(output.result == "NG", "suppression should not change the NG decision")) return 1;
    if (AssertTrue(output.patchcore_score == patchcore.score,
                   "suppression should preserve the PatchCore score")) return 1;
    if (AssertTrue(output.details.size() == 1 && output.details[0].name == "BrightStripes",
                   "suppression should keep only the filtered YOLO detail")) return 1;

    context.brightstripes_filter.confidence_threshold = 0.95f;
    InspectionDLL::Internal::ComposeOutputWithDefectFilter(
        patchcore, yolo, gray.size(), gray, gray, output, context);
    if (AssertTrue(output.details.size() == 1 && output.details[0].name == "Abnormal",
                   "a filtered-out YOLO detection should not suppress PatchCore details")) return 1;

    context.brightstripes_filter.confidence_threshold = 0.0f;
    context.aa_filter.enable = true;
    context.aa_filter.categories = {"BrightStripes"};
    yolo.details[0].y = 680.0f;
    yolo.details[0].w = 80.0f;
    yolo.details[0].h = 20.0f;
    InspectionDLL::Internal::ComposeOutputWithDefectFilter(
        patchcore, yolo, cv::Size(128, 800), gray, gray, output, context);
    if (AssertTrue(output.yolo_score == 0.0f &&
                   output.details.size() == 1 && output.details[0].name == "Abnormal",
                   "AA-filtered YOLO details should not suppress PatchCore details")) return 1;
    return 0;
}

int TestAaYoloFilterRules() {
    auto MakeDetail = [](const std::string& name, float x, float y, float width, float height) {
        InspectionDLL::DetectionResult detail;
        detail.name = name;
        detail.x = x;
        detail.y = y;
        detail.w = width;
        detail.h = height;
        return detail;
    };

    std::vector<InspectionDLL::DetectionResult> details = {
        MakeDetail("Stain", 0.0f, 645.0f, 121.0f, 40.0f),
        MakeDetail("DarkClusters", 0.0f, 705.0f, 80.0f, 20.0f),
        MakeDetail("Stain", 0.0f, 680.0f, 60.0f, 20.0f),
        MakeDetail("Stain", 0.0f, 700.0f, 121.0f, 40.0f),
        MakeDetail("BrightStripes", 0.0f, 680.0f, 80.0f, 20.0f)
    };

    InspectionConfig::AaFilterConfig config;
    InspectionDLL::Internal::ApplyAaYoloFilter(details, config);
    if (AssertTrue(details.size() == 5, "disabled AA filter should preserve all details")) return 1;

    config.enable = true;
    config.categories = {"stain", "DARKCLUSTERS"};
    InspectionDLL::Internal::ApplyAaYoloFilter(details, config);
    if (AssertTrue(details.size() == 3,
                   "AA filter should remove only matching category, Y range, and ratio details")) return 1;
    if (AssertTrue(details[0].name == "Stain" && details[0].w == 60.0f,
                   "ratio equal to the threshold should be preserved")) return 1;
    if (AssertTrue(details[1].name == "Stain" && details[1].y == 700.0f,
                   "center Y above the configured range should be preserved")) return 1;
    if (AssertTrue(details[2].name == "BrightStripes",
                   "categories outside the configured list should be preserved")) return 1;

    std::vector<InspectionDLL::DetectionResult> position_details = {
        MakeDetail("Stain", 454.22f, 304.30f, 170.52f, 211.39f),
        MakeDetail("DarkClusters", 430.0f, 280.0f, 145.0f, 175.0f),
        MakeDetail("Stain", 429.0f, 304.30f, 170.52f, 211.39f),
        MakeDetail("BrightStripes", 454.22f, 304.30f, 170.52f, 211.39f)
    };
    config = {};
    config.position_filter_enable = true;
    config.position_categories = {"stain", "DARKCLUSTERS"};
    InspectionDLL::Internal::ApplyAaYoloFilter(position_details, config);
    if (AssertTrue(position_details.size() == 2,
                   "AA position filter should remove only details matching every configured range")) return 1;
    if (AssertTrue(position_details[0].name == "Stain" && position_details[0].x == 429.0f,
                   "a detail outside the left-coordinate range should be preserved")) return 1;
    if (AssertTrue(position_details[1].name == "BrightStripes",
                   "AA position filter should preserve categories outside its list")) return 1;
    return 0;
}

int TestAaConfigParsing() {
    const std::filesystem::path config_path =
        std::filesystem::temp_directory_path() / "corona_aa_filter_smoke.ini";
    {
        std::ofstream stream(config_path);
        stream << "[yolo]\n"
               << "enabled=0\n"
               << "[patchcore]\n"
               << "enabled=0\n"
               << "[AA]\n"
               << "enabled=1\n"
               << "center_y_min=100.5\n"
               << "center_y_max=200.5\n"
               << "min_width_height_ratio=4.5\n"
               << "categories= stain, LineArtifacts\n"
               << "position_filter_enabled=1\n"
               << "position_x_min=10.5\n"
               << "position_x_max=20.5\n"
               << "position_y_min=30.5\n"
               << "position_y_max=40.5\n"
               << "position_width_min=50.5\n"
               << "position_width_max=60.5\n"
               << "position_height_min=70.5\n"
               << "position_height_max=80.5\n"
               << "position_categories= darkclusters, BrightStripes\n";
    }

    InspectionConfig::InspectionConfigData config;
    std::string error;
    const bool loaded = InspectionConfig::LoadInspectionConfig(config_path.string(), config, error);
    std::error_code remove_error;
    std::filesystem::remove(config_path, remove_error);

    if (AssertTrue(loaded, "AA config should load: " + error)) return 1;
    if (AssertTrue(config.aa_filter.enable, "AA config enabled flag should be parsed")) return 1;
    if (AssertTrue(config.aa_filter.center_y_min == 100.5f &&
                   config.aa_filter.center_y_max == 200.5f,
                   "AA center Y range should be parsed")) return 1;
    if (AssertTrue(config.aa_filter.min_width_height_ratio == 4.5f,
                   "AA ratio should be parsed")) return 1;
    if (AssertTrue(config.aa_filter.categories.size() == 2 &&
                   config.aa_filter.categories[0] == "stain" &&
                   config.aa_filter.categories[1] == "LineArtifacts",
                   "AA category list should be trimmed and parsed")) return 1;
    if (AssertTrue(config.aa_filter.position_filter_enable,
                   "AA position filter enabled flag should be parsed")) return 1;
    if (AssertTrue(config.aa_filter.position_x_min == 10.5f &&
                   config.aa_filter.position_x_max == 20.5f &&
                   config.aa_filter.position_y_min == 30.5f &&
                   config.aa_filter.position_y_max == 40.5f,
                   "AA position coordinate ranges should be parsed")) return 1;
    if (AssertTrue(config.aa_filter.position_width_min == 50.5f &&
                   config.aa_filter.position_width_max == 60.5f &&
                   config.aa_filter.position_height_min == 70.5f &&
                   config.aa_filter.position_height_max == 80.5f,
                   "AA position size ranges should be parsed")) return 1;
    if (AssertTrue(config.aa_filter.position_categories.size() == 2 &&
                   config.aa_filter.position_categories[0] == "darkclusters" &&
                   config.aa_filter.position_categories[1] == "BrightStripes",
                   "AA position category list should be trimmed and parsed")) return 1;
    return 0;
}

int TestContrastThresholdUsesCategoryDirection() {
    InspectionDLL::DetectionResult yolo_detail;
    yolo_detail.name = "Stain";
    yolo_detail.score = 0.9f;
    yolo_detail.contrast = 0.7f;
    yolo_detail.area = 25;
    yolo_detail.w = 5.0f;
    yolo_detail.h = 5.0f;

    InspectionDLL::Internal::YoloDerived yolo;
    yolo.has_detections = true;
    yolo.details.push_back(yolo_detail);

    InspectionDLL::Internal::PatchCoreDerived patchcore;
    InspectionDLL::Internal::ResultComposeContext context;
    context.stain_filter.contrast_threshold = 0.8f;

    InspectionDLL::InferenceResult output;
    const cv::Mat gray(32, 32, CV_8U, cv::Scalar(128));
    InspectionDLL::Internal::ComposeOutputWithDefectFilter(
        patchcore, yolo, gray.size(), gray, gray, output, context);
    if (AssertTrue(output.details.size() == 1,
                   "Stain contrast below the threshold should be preserved")) return 1;

    yolo.details[0].contrast = 0.8f;
    InspectionDLL::Internal::ComposeOutputWithDefectFilter(
        patchcore, yolo, gray.size(), gray, gray, output, context);
    if (AssertTrue(output.details.empty(),
                   "Stain contrast equal to the threshold should be filtered")) return 1;

    yolo.details[0].contrast = 0.9f;
    InspectionDLL::Internal::ComposeOutputWithDefectFilter(
        patchcore, yolo, gray.size(), gray, gray, output, context);
    if (AssertTrue(output.details.empty(),
                   "Stain contrast above the threshold should be filtered")) return 1;

    yolo.details[0].name = "DarkClusters";
    yolo.details[0].contrast = 0.7f;
    context.darkclusters_filter.contrast_threshold = 0.8f;
    InspectionDLL::Internal::ComposeOutputWithDefectFilter(
        patchcore, yolo, gray.size(), gray, gray, output, context);
    if (AssertTrue(output.details.size() == 1,
                   "DarkClusters contrast below the threshold should be preserved")) return 1;

    yolo.details[0].name = "BrightStripes";
    yolo.details[0].contrast = 0.7f;
    context.brightstripes_filter.contrast_threshold = 0.8f;
    InspectionDLL::Internal::ComposeOutputWithDefectFilter(
        patchcore, yolo, gray.size(), gray, gray, output, context);
    if (AssertTrue(output.details.empty(),
                   "BrightStripes contrast below the threshold should be filtered")) return 1;

    yolo.details[0].contrast = 0.8f;
    InspectionDLL::Internal::ComposeOutputWithDefectFilter(
        patchcore, yolo, gray.size(), gray, gray, output, context);
    if (AssertTrue(output.details.size() == 1,
                   "BrightStripes contrast equal to the threshold should be preserved")) return 1;

    yolo.details[0].name = "LineArtifacts";
    yolo.details[0].contrast = 0.9f;
    context.lineartifacts_filter.contrast_threshold = 0.8f;
    InspectionDLL::Internal::ComposeOutputWithDefectFilter(
        patchcore, yolo, gray.size(), gray, gray, output, context);
    if (AssertTrue(output.details.size() == 1,
                   "LineArtifacts contrast above the threshold should be preserved")) return 1;

    context.lineartifacts_filter.contrast_threshold = 0.0f;
    yolo.details[0].contrast = 0.1f;
    InspectionDLL::Internal::ComposeOutputWithDefectFilter(
        patchcore, yolo, gray.size(), gray, gray, output, context);
    if (AssertTrue(output.details.size() == 1,
                   "zero contrast threshold should disable contrast filtering")) return 1;

    yolo = {};
    patchcore.score = 2.1f;
    patchcore.area_ratio = 0.5f;
    patchcore.has_defect = true;
    patchcore.area = 100;
    patchcore.w = 10;
    patchcore.h = 10;
    patchcore.contrast = 0.7f;
    context.abnormal_filter.contrast_threshold = 0.8f;
    InspectionDLL::Internal::ComposeOutputWithDefectFilter(
        patchcore, yolo, gray.size(), gray, gray, output, context);
    if (AssertTrue(output.details.empty(),
                   "PatchCore contrast below the threshold should be filtered")) return 1;

    patchcore.contrast = 0.8f;
    InspectionDLL::Internal::ComposeOutputWithDefectFilter(
        patchcore, yolo, gray.size(), gray, gray, output, context);
    if (AssertTrue(output.details.size() == 1 && output.details[0].name == "Abnormal",
                   "PatchCore contrast equal to the threshold should be preserved")) return 1;
    return 0;
}

}  // namespace

int main() {
    if (TestOverlayTextFit()) return 1;
    if (TestFloatOverlayFileIs8Bit()) return 1;
    if (TestConfigDefaults()) return 1;
    if (TestAaYoloFilterRules()) return 1;
    if (TestAaConfigParsing()) return 1;
    if (TestContrastThresholdUsesCategoryDirection()) return 1;
    if (TestPatchCoreDetailSuppressionWhenYoloDetected()) return 1;
    if (TestDefectBoxExpansionClipsToImage()) return 1;
    if (TestLoggerLevelQuery()) return 1;
    if (TestCombinedInferencePreprocessingMatchesLegacyPath()) return 1;
    if (TestYoloPostprocessReusesBinaryMask()) return 1;
    std::cout << "[PASS] smoke_tests" << std::endl;
    return 0;
}
