#include "config.h"

#include "ini_parser.h"

#include <sstream>
#include <vector>

namespace InspectionConfig {
namespace {

CategoryFilterConfig LoadCategoryFilter(const IniData& ini, const std::string& section) {
    CategoryFilterConfig filter;
    filter.enable = GetBoolOr(ini, section, "Enable", true);
    filter.confidence_threshold = GetFloatOr(ini, section, "confidence_threshold", 0.0f);
    filter.contrast_threshold = GetFloatOr(ini, section, "contrast_threshold", 0.0f);
    filter.min_width = GetIntOr(ini, section, "min_width", 0);
    filter.min_height = GetIntOr(ini, section, "min_height", 0);
    filter.min_area = GetIntOr(ini, section, "min_area", 0);
    filter.use_traditional_measure = GetBoolOr(ini, section, "use_traditional_measure", false);
    return filter;
}

AbnormalFilterConfig LoadAbnormalFilter(const IniData& ini) {
    const bool use_legacy_section = ini.sections.find("abormal_config") != ini.sections.end();
    const std::string section = use_legacy_section ? "Abormal_config" : "Abnormal_config";

    AbnormalFilterConfig filter;
    filter.enable = GetBoolOr(ini, section, "Enable", true);
    filter.score_threshold_1 = GetFloatOr(ini, section, "score_threshold_1", 1.4f);
    filter.score_threshold_2 = GetFloatOr(ini, section, "score_threshold_2", 2.0f);
    filter.area_threshold_1 = GetFloatOr(ini, section, "area_threshold_1", 0.3f);
    filter.contrast_threshold = GetFloatOr(ini, section, "contrast_threshold", 0.0f);
    filter.min_width = GetIntOr(ini, section, "min_width", 0);
    filter.min_height = GetIntOr(ini, section, "min_height", 0);
    filter.min_area = GetIntOr(ini, section, "min_area", 0);
    return filter;
}

}  // namespace

bool LoadInspectionConfig(const std::string& config_path, InspectionConfigData& out, std::string& err) {
    IniData ini;
    if (!ParseIniFile(config_path, ini, err)) {
        return false;
    }

    std::vector<std::string> missing_keys;
    if (!TryGetString(ini, "models", "yolo_model_path", out.yolo_model_path)) missing_keys.push_back("yolo_model_path");
    if (!TryGetString(ini, "models", "patchcore_model_path", out.patchcore_model_path)) missing_keys.push_back("patchcore_model_path");
    if (!TryGetString(ini, "models", "faiss_index_path", out.faiss_index_path)) missing_keys.push_back("faiss_index_path");
    if (!TryGetString(ini, "models", "metadata_path", out.metadata_path)) missing_keys.push_back("metadata_path");

    if (!missing_keys.empty()) {
        std::ostringstream oss;
        oss << "ini missing required keys in [models]: ";
        for (size_t i = 0; i < missing_keys.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << missing_keys[i];
        }
        err = oss.str();
        return false;
    }

    out.yolo_model_path = ResolvePathRelativeToIni(config_path, out.yolo_model_path);
    out.patchcore_model_path = ResolvePathRelativeToIni(config_path, out.patchcore_model_path);
    out.faiss_index_path = ResolvePathRelativeToIni(config_path, out.faiss_index_path);
    out.metadata_path = ResolvePathRelativeToIni(config_path, out.metadata_path);

    out.yolo_score_threshold = GetFloatOr(ini, "yolo", "score_threshold", out.yolo_score_threshold);
    out.yolo_iou_threshold = GetFloatOr(ini, "yolo", "iou_threshold", out.yolo_iou_threshold);
    out.yolo_nms_class_aware = GetBoolOr(ini, "yolo", "nms_class_aware", out.yolo_nms_class_aware);
    out.ort_intra_threads = GetIntOr(ini, "ort", "intra_threads", out.ort_intra_threads);

    out.patchcore_score_threshold = GetFloatOr(ini, "patchcore", "score_threshold", out.patchcore_score_threshold);
    out.patchcore_area_threshold = GetFloatOr(ini, "patchcore", "area_threshold", out.patchcore_area_threshold);
    out.patchcore_mask_area_threshold = GetFloatOr(ini, "patchcore", "mask_area_threshold", out.patchcore_mask_area_threshold);
    out.dark_clusters_threshold = GetFloatOr(ini, "post", "dark_clusters_threshold", out.dark_clusters_threshold);

    out.abnormal_filter = LoadAbnormalFilter(ini);
    out.stain_filter = LoadCategoryFilter(ini, "stain_config");
    out.darkclusters_filter = LoadCategoryFilter(ini, "darkclusters_config");
    out.brightstripes_filter = LoadCategoryFilter(ini, "brightstripes_config");
    out.lineartifacts_filter = LoadCategoryFilter(ini, "lineartifacts_config");

    return true;
}

}  // namespace InspectionConfig
