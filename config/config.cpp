#include "config.h"

#include "ini_parser.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <sstream>
#include <vector>

namespace InspectionConfig {
namespace {

namespace fs = std::filesystem;

constexpr std::array<std::pair<const char*, const char*>, 8> kYoloCategorySections = {{
    {"Glue_overflow", "glue_overflow_config"},
    {"Decolorization", "decolorization_config"},
    {"Stain", "stain_config"},
    {"Stripes", "stripes_config"},
    {"BrightStripes", "brightstripes_config"},
    {"Bright_clusters", "bright_clusters_config"},
    {"Line_artifacts", "line_artifacts_config"},
    {"LineArtifacts", "lineartifacts_config"},
}};

std::string TrimCopy(const std::string& value) {
    const auto begin = std::find_if_not(value.begin(), value.end(),
                                        [](unsigned char ch) { return std::isspace(ch); });
    if (begin == value.end()) return {};
    const auto end = std::find_if_not(value.rbegin(), value.rend(),
                                      [](unsigned char ch) { return std::isspace(ch); }).base();
    return std::string(begin, end);
}

std::string ToLowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::string CanonicalYoloCategory(const std::string& category) {
    const std::string lowered = ToLowerCopy(TrimCopy(category));
    for (const auto& entry : kYoloCategorySections) {
        if (lowered == ToLowerCopy(entry.first)) return entry.first;
    }
    return {};
}

std::vector<std::string> ParseFilterCategories(const std::string& value) {
    std::vector<std::string> categories;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const std::string trimmed = TrimCopy(item);
        if (!trimmed.empty()) categories.push_back(trimmed);
    }
    return categories;
}

AaFilterConfig LoadAaFilter(const IniData& ini) {
    AaFilterConfig filter;
    filter.enable = GetBoolOr(ini, "AA", "enabled", filter.enable);
    filter.center_y_min = GetFloatOr(ini, "AA", "center_y_min", filter.center_y_min);
    filter.center_y_max = GetFloatOr(ini, "AA", "center_y_max", filter.center_y_max);
    filter.min_width_height_ratio = GetFloatOr(
        ini, "AA", "min_width_height_ratio", filter.min_width_height_ratio);

    std::string categories;
    if (TryGetString(ini, "AA", "categories", categories)) {
        filter.categories = ParseFilterCategories(categories);
    }

    filter.position_filter_enable = GetBoolOr(
        ini, "AA", "position_filter_enabled", filter.position_filter_enable);
    filter.position_x_min = GetFloatOr(ini, "AA", "position_x_min", filter.position_x_min);
    filter.position_x_max = GetFloatOr(ini, "AA", "position_x_max", filter.position_x_max);
    filter.position_y_min = GetFloatOr(ini, "AA", "position_y_min", filter.position_y_min);
    filter.position_y_max = GetFloatOr(ini, "AA", "position_y_max", filter.position_y_max);
    filter.position_width_min = GetFloatOr(
        ini, "AA", "position_width_min", filter.position_width_min);
    filter.position_width_max = GetFloatOr(
        ini, "AA", "position_width_max", filter.position_width_max);
    filter.position_height_min = GetFloatOr(
        ini, "AA", "position_height_min", filter.position_height_min);
    filter.position_height_max = GetFloatOr(
        ini, "AA", "position_height_max", filter.position_height_max);

    std::string position_categories;
    if (TryGetString(ini, "AA", "position_categories", position_categories)) {
        filter.position_categories = ParseFilterCategories(position_categories);
    }
    return filter;
}

IqtFilterConfig LoadIqtFilter(const IniData& ini) {
    IqtFilterConfig filter;
    filter.enable = GetBoolOr(ini, "IQT", "enabled", filter.enable);
    filter.center_y_min = GetFloatOr(ini, "IQT", "center_y_min", filter.center_y_min);
    filter.center_y_max = GetFloatOr(ini, "IQT", "center_y_max", filter.center_y_max);
    filter.min_width_height_ratio = GetFloatOr(
        ini, "IQT", "min_width_height_ratio", filter.min_width_height_ratio);
    filter.max_width = GetFloatOr(ini, "IQT", "max_width", filter.max_width);

    std::string categories;
    if (TryGetString(ini, "IQT", "categories", categories)) {
        filter.categories = ParseFilterCategories(categories);
    }
    return filter;
}

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

bool IsSupportedLogLevel(const std::string& level) {
    return level == "error" || level == "warn" || level == "warning" || level == "info" || level == "debug";
}

bool ValidateNonNegative(float value, const char* name, std::string& err) {
    if (value < 0.0f) {
        err = std::string(name) + " must be >= 0";
        return false;
    }
    return true;
}

bool ValidateColorChannel(int value, const char* name, std::string& err) {
    if (value < 0 || value > 255) {
        err = std::string(name) + " must be in [0, 255]";
        return false;
    }
    return true;
}

bool ValidateConfig(const InspectionConfigData& out, std::string& err) {
    if (out.yolo_enabled && !fs::exists(out.yolo_model_path)) {
        err = "yolo_model_path does not exist: " + out.yolo_model_path;
        return false;
    }
    if (out.patchcore_enabled && !fs::exists(out.patchcore_model_path)) {
        err = "patchcore_model_path does not exist: " + out.patchcore_model_path;
        return false;
    }
    if (out.patchcore_enabled && !fs::exists(out.faiss_index_path)) {
        err = "faiss_index_path does not exist: " + out.faiss_index_path;
        return false;
    }
    if (out.patchcore_enabled && !fs::exists(out.metadata_path)) {
        err = "metadata_path does not exist: " + out.metadata_path;
        return false;
    }
    if (out.ort_intra_threads <= 0) {
        err = "ort.intra_threads must be > 0";
        return false;
    }
    if (out.yolo_iou_threshold < 0.0f || out.yolo_iou_threshold > 1.0f) {
        err = "yolo.iou_threshold must be in [0, 1]";
        return false;
    }
    if (!ValidateNonNegative(out.yolo_score_threshold, "yolo.score_threshold", err)) return false;
    if (!ValidateNonNegative(out.patchcore_score_threshold, "patchcore.score_threshold", err)) return false;
    if (!ValidateNonNegative(out.patchcore_area_threshold, "patchcore.area_threshold", err)) return false;
    if (!ValidateNonNegative(out.patchcore_mask_area_threshold, "patchcore.mask_area_threshold", err)) return false;
    if (out.patchcore_yolo_iou_threshold < 0.0f || out.patchcore_yolo_iou_threshold > 1.0f) {
        err = "post.patchcore_yolo_iou_threshold must be in [0, 1]";
        return false;
    }
    if (out.contrast_mode < 0 || out.contrast_mode > 2) {
        err = "post.contrast_mode must be one of: 0, 1, 2";
        return false;
    }
    if (!ValidateColorChannel(out.defect_mask_color_r, "post.defect_mask_color_r", err)) return false;
    if (!ValidateColorChannel(out.defect_mask_color_g, "post.defect_mask_color_g", err)) return false;
    if (!ValidateColorChannel(out.defect_mask_color_b, "post.defect_mask_color_b", err)) return false;
    if (out.defect_mask_alpha < 0.0f || out.defect_mask_alpha > 1.0f) {
        err = "post.defect_mask_alpha must be in [0, 1]";
        return false;
    }

    if (out.aa_filter.enable) {
        if (out.aa_filter.center_y_max < out.aa_filter.center_y_min) {
            err = "AA.center_y_max must be >= AA.center_y_min";
            return false;
        }
        if (out.aa_filter.min_width_height_ratio < 0.0f) {
            err = "AA.min_width_height_ratio must be >= 0";
            return false;
        }
        if (out.aa_filter.categories.empty()) {
            err = "AA.categories must contain at least one category when AA is enabled";
            return false;
        }
        for (const auto& category : out.aa_filter.categories) {
            if (CanonicalYoloCategory(category).empty()) {
                err = "AA.categories contains unsupported category: " + category;
                return false;
            }
        }
    }
    if (out.aa_filter.position_filter_enable) {
        if (out.aa_filter.position_x_max < out.aa_filter.position_x_min) {
            err = "AA.position_x_max must be >= AA.position_x_min";
            return false;
        }
        if (out.aa_filter.position_y_max < out.aa_filter.position_y_min) {
            err = "AA.position_y_max must be >= AA.position_y_min";
            return false;
        }
        if (out.aa_filter.position_width_max < out.aa_filter.position_width_min) {
            err = "AA.position_width_max must be >= AA.position_width_min";
            return false;
        }
        if (out.aa_filter.position_height_max < out.aa_filter.position_height_min) {
            err = "AA.position_height_max must be >= AA.position_height_min";
            return false;
        }
        if (out.aa_filter.position_categories.empty()) {
            err = "AA.position_categories must contain at least one category when the position filter is enabled";
            return false;
        }
        for (const auto& category : out.aa_filter.position_categories) {
            if (CanonicalYoloCategory(category).empty()) {
                err = "AA.position_categories contains unsupported category: " + category;
                return false;
            }
        }
    }

    if (out.iqt_filter.enable) {
        if (out.iqt_filter.center_y_max < out.iqt_filter.center_y_min) {
            err = "IQT.center_y_max must be >= IQT.center_y_min";
            return false;
        }
        if (out.iqt_filter.min_width_height_ratio < 0.0f) {
            err = "IQT.min_width_height_ratio must be >= 0";
            return false;
        }
        if (out.iqt_filter.max_width < 0.0f) {
            err = "IQT.max_width must be >= 0";
            return false;
        }
        if (out.iqt_filter.categories.empty()) {
            err = "IQT.categories must contain at least one category when IQT is enabled";
            return false;
        }
        for (const auto& category : out.iqt_filter.categories) {
            if (CanonicalYoloCategory(category).empty()) {
                err = "IQT.categories contains unsupported category: " + category;
                return false;
            }
        }
    }

    if (!IsSupportedLogLevel(out.log_level)) {
        err = "log.level must be one of: error, warn, info, debug";
        return false;
    }

    if (out.log_to_file) {
        if (out.log_file_path.empty()) {
            err = "log.file_path must be set when log_to_file=1";
            return false;
        }
        const fs::path parent = fs::path(out.log_file_path).parent_path();
        if (!parent.empty() && !fs::exists(parent)) {
            err = "log file parent directory does not exist: " + parent.string();
            return false;
        }
    }

    return true;
}

}  // namespace

bool LoadInspectionConfig(const std::string& config_path, InspectionConfigData& out, std::string& err) {
    IniData ini;
    if (!ParseIniFile(config_path, ini, err)) {
        return false;
    }

    out.yolo_enabled = GetBoolOr(ini, "yolo", "enabled", out.yolo_enabled);
    out.patchcore_enabled = GetBoolOr(ini, "patchcore", "enabled", out.patchcore_enabled);

    std::vector<std::string> missing_keys;
    if (!TryGetString(ini, "models", "yolo_model_path", out.yolo_model_path) && out.yolo_enabled) missing_keys.push_back("yolo_model_path");
    if (!TryGetString(ini, "models", "patchcore_model_path", out.patchcore_model_path) && out.patchcore_enabled) missing_keys.push_back("patchcore_model_path");
    if (!TryGetString(ini, "models", "faiss_index_path", out.faiss_index_path) && out.patchcore_enabled) missing_keys.push_back("faiss_index_path");
    if (!TryGetString(ini, "models", "metadata_path", out.metadata_path) && out.patchcore_enabled) missing_keys.push_back("metadata_path");

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

    if (!out.yolo_model_path.empty()) out.yolo_model_path = ResolvePathRelativeToIni(config_path, out.yolo_model_path);
    if (!out.patchcore_model_path.empty()) out.patchcore_model_path = ResolvePathRelativeToIni(config_path, out.patchcore_model_path);
    if (!out.faiss_index_path.empty()) out.faiss_index_path = ResolvePathRelativeToIni(config_path, out.faiss_index_path);
    if (!out.metadata_path.empty()) out.metadata_path = ResolvePathRelativeToIni(config_path, out.metadata_path);

    out.yolo_score_threshold = GetFloatOr(ini, "yolo", "score_threshold", out.yolo_score_threshold);
    out.yolo_iou_threshold = GetFloatOr(ini, "yolo", "iou_threshold", out.yolo_iou_threshold);
    out.yolo_nms_class_aware = GetBoolOr(ini, "yolo", "nms_class_aware", out.yolo_nms_class_aware);
    out.ort_intra_threads = GetIntOr(ini, "ort", "intra_threads", out.ort_intra_threads);

    out.patchcore_score_threshold = GetFloatOr(ini, "patchcore", "score_threshold", out.patchcore_score_threshold);
    out.patchcore_area_threshold = GetFloatOr(ini, "patchcore", "area_threshold", out.patchcore_area_threshold);
    out.patchcore_mask_area_threshold = GetFloatOr(ini, "patchcore", "mask_area_threshold", out.patchcore_mask_area_threshold);
    out.patchcore_yolo_iou_threshold = GetFloatOr(
        ini,
        "post",
        "patchcore_yolo_iou_threshold",
        out.patchcore_yolo_iou_threshold);
    out.contrast_mode = GetIntOr(ini, "post", "contrast_mode", out.contrast_mode);
    out.draw_defect_box = GetBoolOr(ini, "post", "draw_defect_box", out.draw_defect_box);
    out.expand_defect_box = GetBoolOr(ini, "post", "expand_defect_box", out.expand_defect_box);
    out.draw_box_details = GetBoolOr(ini, "post", "draw_box_details", out.draw_box_details);
    out.draw_defect_mask = GetBoolOr(ini, "post", "draw_defect_mask", out.draw_defect_mask);
    out.concat_original_image = GetBoolOr(ini, "post", "concat_original_image", out.concat_original_image);
    out.defect_mask_color_r = GetIntOr(ini, "post", "defect_mask_color_r", out.defect_mask_color_r);
    out.defect_mask_color_g = GetIntOr(ini, "post", "defect_mask_color_g", out.defect_mask_color_g);
    out.defect_mask_color_b = GetIntOr(ini, "post", "defect_mask_color_b", out.defect_mask_color_b);
    out.defect_mask_alpha = GetFloatOr(ini, "post", "defect_mask_alpha", out.defect_mask_alpha);
    out.log_enabled = GetBoolOr(ini, "log", "enabled", out.log_enabled);
    out.log_level = GetStringOr(ini, "log", "level", out.log_level);
    out.log_to_stderr = GetBoolOr(ini, "log", "log_to_stderr", out.log_to_stderr);
    out.log_to_file = GetBoolOr(ini, "log", "log_to_file", out.log_to_file);
    out.log_file_path = GetStringOr(ini, "log", "file_path", out.log_file_path);
    if (!out.log_file_path.empty()) {
        out.log_file_path = ResolvePathRelativeToIni(config_path, out.log_file_path);
    }

    out.abnormal_filter = LoadAbnormalFilter(ini);
    out.category_filters.clear();
    for (const auto& entry : kYoloCategorySections) {
        out.category_filters.emplace(entry.first, LoadCategoryFilter(ini, entry.second));
    }
    out.aa_filter = LoadAaFilter(ini);
    out.iqt_filter = LoadIqtFilter(ini);

    if (!ValidateConfig(out, err)) {
        return false;
    }

    return true;
}

}  // namespace InspectionConfig
