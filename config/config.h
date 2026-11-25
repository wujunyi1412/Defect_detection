#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace InspectionConfig {

enum class TraditionalMeasureMode {
    None,
    Dark,
    Bright,
};

// Selects the geometry-refinement algorithm independently from defect polarity.
// MaskGuided uses the YOLO mask as a spatial prior and image residuals to refine
// the final foreground boundary. HybridCascade is the GrayAdaptive V2 path: its
// grayscale segmentation remains authoritative and the mask is only a soft prior.
enum class DefectRefinementMethod {
    None,
    GrayAdaptive,
    MaskGuided,
    HybridCascade,
    Invalid,
};

enum class DefectPolarity {
    Auto,
    Dark,
    Bright,
    Invalid,
};

// Type-name compatibility for code built during the transition.
using DefectRefinementPolarity = DefectPolarity;

enum class ContrastFilterDirection {
    LessEqual,
    GreaterEqual,
    Invalid,
};

struct CategoryFilterConfig {
    bool enable = true;
    float confidence_threshold = 0.0f;
    float contrast_threshold = 0.0f;
    int min_width = 0;
    int min_height = 0;
    float contrast_threshold_1 = 0.0f;
    float contrast_threshold_2 = 0.0f;
    ContrastFilterDirection contrast_direction = ContrastFilterDirection::LessEqual;
    float confidence_threshold_0 = 0.0f;
    int min_width_0 = 0;
    int min_height_0 = 0;
    int min_area_0 = 0;
    float confidence_threshold_1 = 0.0f;
    int min_width_1 = 0;
    int min_height_1 = 0;
    int min_area_1 = 0;
    float confidence_threshold_2 = 0.0f;
    int min_width_2 = 0;
    int min_height_2 = 0;
    int min_area_2 = 0;
    int min_area = 0;
    DefectRefinementMethod refinement_method = DefectRefinementMethod::None;
    DefectPolarity defect_polarity = DefectPolarity::Auto;
    // Deprecated compatibility field. New code should use refinement_method
    // and defect_polarity.
    TraditionalMeasureMode traditional_measure_mode = TraditionalMeasureMode::None;
};

struct AbnormalFilterConfig {
    bool enable = true;
    float score_threshold_1 = 1.4f;
    float score_threshold_2 = 2.0f;
    float area_threshold_1 = 0.3f;
    float contrast_threshold = 0.0f;
    int min_width = 0;
    int min_height = 0;
    int min_area = 0;
};

using CategoryFilterMap = std::unordered_map<std::string, CategoryFilterConfig>;
using CategoryNameMap = std::unordered_map<std::string, std::string>;

inline CategoryFilterMap DefaultCategoryFilters() {
    return {
        {"Glue_overflow", {}},
        {"Decolorization", {}},
        {"Stain", {}},
        {"Stripes", {}},
        {"BrightStripes", {}},
        {"Bright_clusters", {}},
        {"Line_artifacts", {}},
        {"LineArtifacts", {}},
        {"Stripe_horizontal", {}},
        {"Stripe_vertical", {}},
    };
}

struct AaFilterConfig {
    bool enable = false;
    float center_y_min = 665.0f;
    float center_y_max = 715.0f;
    float min_width_height_ratio = 3.0f;
    float max_width = 0.0f;
    std::vector<std::string> categories = {"Stain"};

    bool position_filter_enable = false;
    float position_x_min = 430.0f;
    float position_x_max = 480.0f;
    float position_y_min = 280.0f;
    float position_y_max = 340.0f;
    float position_width_min = 145.0f;
    float position_width_max = 200.0f;
    float position_height_min = 175.0f;
    float position_height_max = 245.0f;
    std::vector<std::string> position_categories = {"Stain"};

    bool black_spot_filter_enable = false;
    bool black_spot_debug_log_enable = false;
    float black_spot_width_min = 145.0f;
    float black_spot_width_max = 240.0f;
    float black_spot_height_min = 145.0f;
    float black_spot_height_max = 260.0f;
    int black_spot_mask_area_min = 3000;
    int black_spot_mask_area_max = 50000;
    float black_spot_min_center_darkness = 12.0f;
    float black_spot_min_monotonic_ratio = 0.71f;
    float black_spot_monotonic_tolerance = 2.0f;
    float black_spot_max_center_offset_ratio = 0.45f;
    std::vector<std::string> black_spot_categories = {"Stain"};
};

struct IqtFilterConfig {
    bool enable = false;
    float center_y_min = 665.0f;
    float center_y_max = 715.0f;
    float min_width_height_ratio = 3.0f;
    float max_width = 0.0f;
    std::vector<std::string> categories = {"Stain"};
};

struct YoloLargeBoxShrinkConfig {
    bool enable = true;
    float min_width = 200.0f;
    float min_height = 200.0f;
    float ratio = 0.05f;
};

struct InspectionConfigData {
    bool yolo_enabled = true;
    bool patchcore_enabled = true;
    std::string yolo_model_path;
    std::string patchcore_model_path;
    std::string faiss_index_path;
    std::string metadata_path;
    float yolo_score_threshold = 0.25f;
    float yolo_mask_threshold = 0.25f;
    float yolo_iou_threshold = 0.2f;
    bool yolo_nms_class_aware = false;
    YoloLargeBoxShrinkConfig yolo_large_box_shrink;
    int ort_intra_threads = 4;
    float patchcore_score_threshold = 1.4f;
    float patchcore_area_threshold = 1.4f;
    float patchcore_mask_area_threshold = 0.3f;
    float patchcore_yolo_iou_threshold = 0.0f;
    int contrast_mode = 1;
    bool draw_defect_box = true;
    bool expand_defect_box = false;
    bool draw_box_details = true;
    bool draw_filtered_defects = false;
    bool draw_defect_mask = false;
    bool concat_original_image = false;
    int defect_mask_color_r = 255;
    int defect_mask_color_g = 0;
    int defect_mask_color_b = 0;
    float defect_mask_alpha = 0.35f;
    bool log_enabled = true;
    std::string log_level = "info";
    bool log_to_stderr = true;
    bool log_to_file = false;
    std::string log_file_path;
    AaFilterConfig aa_filter;
    IqtFilterConfig iqt_filter;
    AbnormalFilterConfig abnormal_filter;
    CategoryFilterMap category_filters = DefaultCategoryFilters();
    CategoryNameMap category_name_mapping;
};

bool LoadInspectionConfig(const std::string& config_path, InspectionConfigData& out, std::string& err);

}  // namespace InspectionConfig
