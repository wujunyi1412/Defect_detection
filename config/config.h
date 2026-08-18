#pragma once

#include <string>
#include <vector>

namespace InspectionConfig {

struct CategoryFilterConfig {
    bool enable = true;
    float confidence_threshold = 0.0f;
    float contrast_threshold = 0.0f;
    int min_width = 0;
    int min_height = 0;
    int min_area = 0;
    bool use_traditional_measure = false;
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

struct AaFilterConfig {
    bool enable = false;
    float center_y_min = 665.0f;
    float center_y_max = 715.0f;
    float min_width_height_ratio = 3.0f;
    std::vector<std::string> categories = {"Stain", "DarkClusters"};

    bool position_filter_enable = false;
    float position_x_min = 430.0f;
    float position_x_max = 480.0f;
    float position_y_min = 280.0f;
    float position_y_max = 340.0f;
    float position_width_min = 145.0f;
    float position_width_max = 200.0f;
    float position_height_min = 175.0f;
    float position_height_max = 245.0f;
    std::vector<std::string> position_categories = {"Stain", "DarkClusters"};
};

struct InspectionConfigData {
    bool yolo_enabled = true;
    bool patchcore_enabled = true;
    std::string yolo_model_path;
    std::string patchcore_model_path;
    std::string faiss_index_path;
    std::string metadata_path;
    float yolo_score_threshold = 0.25f;
    float yolo_iou_threshold = 0.2f;
    bool yolo_nms_class_aware = false;
    int ort_intra_threads = 4;
    float patchcore_score_threshold = 1.4f;
    float patchcore_area_threshold = 1.4f;
    float patchcore_mask_area_threshold = 0.3f;
    float dark_clusters_threshold = 0.8f;
    bool suppress_patchcore_when_yolo_detected = false;
    bool draw_defect_box = true;
    bool expand_defect_box = false;
    bool draw_box_details = true;
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
    AbnormalFilterConfig abnormal_filter;
    CategoryFilterConfig stain_filter;
    CategoryFilterConfig darkclusters_filter;
    CategoryFilterConfig brightstripes_filter;
    CategoryFilterConfig lineartifacts_filter;
};

bool LoadInspectionConfig(const std::string& config_path, InspectionConfigData& out, std::string& err);

}  // namespace InspectionConfig
