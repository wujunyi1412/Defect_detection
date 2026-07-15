#pragma once

#include <string>

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

struct InspectionConfigData {
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
    AbnormalFilterConfig abnormal_filter;
    CategoryFilterConfig stain_filter;
    CategoryFilterConfig darkclusters_filter;
    CategoryFilterConfig brightstripes_filter;
    CategoryFilterConfig lineartifacts_filter;
};

bool LoadInspectionConfig(const std::string& config_path, InspectionConfigData& out, std::string& err);

}  // namespace InspectionConfig
