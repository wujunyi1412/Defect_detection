#include "patchcore_postprocess.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include "image_utils.h"

namespace InspectionDLL::Internal {

namespace {

cv::Mat BuildPatchCoreCroppedBinaryMask(const cv::Mat& patch_scores,
                                        const PatchCore::PatchCoreResult::MetaData& meta,
                                        float threshold) {
    if (patch_scores.empty()) return cv::Mat();

    cv::Mat binary = patch_scores > threshold;
    binary.convertTo(binary, CV_8U, 255.0);

    cv::Mat square;
    cv::resize(binary, square, cv::Size(meta.target_size, meta.target_size), 0, 0, cv::INTER_NEAREST);

    cv::Rect crop_roi(meta.pad_left, meta.pad_top, meta.new_w, meta.new_h);
    crop_roi &= cv::Rect(0, 0, meta.target_size, meta.target_size);
    if (crop_roi.width <= 0 || crop_roi.height <= 0) return cv::Mat();
    return square(crop_roi).clone();
}

cv::Mat ResizePatchcoreMaskFromCropped(const cv::Mat& cropped, const cv::Size& img_shape, int interpolation) {
    if (cropped.empty()) return cv::Mat();
    cv::Mat resized;
    cv::resize(cropped, resized, img_shape, 0, 0, interpolation);
    return resized;
}

float CalculatePatchCoreAreaRatio(const cv::Mat& cropped_binary,
                                  const PatchCore::PatchCoreResult::MetaData& meta) {
    if (cropped_binary.empty()) return 0.0f;
    const int total_pixels = meta.new_w * meta.new_h;
    const int anomaly_pixels = cv::countNonZero(cropped_binary);
    return total_pixels > 0 ? static_cast<float>(anomaly_pixels) / static_cast<float>(total_pixels) : 0.0f;
}

void GetPatchCoreDefectStats(const cv::Mat& cropped_binary,
                             const PatchCore::PatchCoreResult::MetaData& meta,
                             const cv::Size& img_shape,
                             int& area,
                             int& x,
                             int& y,
                             int& w,
                             int& h) {
    area = 0;
    x = 0;
    y = 0;
    w = 0;
    h = 0;

    if (cropped_binary.empty()) return;
    if (meta.new_w <= 0 || meta.new_h <= 0) return;
    const int area_224 = cv::countNonZero(cropped_binary);
    if (area_224 <= 0) return;

    std::vector<cv::Point> nz;
    cv::findNonZero(cropped_binary, nz);
    if (nz.empty()) return;
    int x_min_224 = cropped_binary.cols;
    int y_min_224 = cropped_binary.rows;
    int x_max_224 = 0;
    int y_max_224 = 0;
    for (const auto& p : nz) {
        x_min_224 = std::min(x_min_224, p.x);
        y_min_224 = std::min(y_min_224, p.y);
        x_max_224 = std::max(x_max_224, p.x);
        y_max_224 = std::max(y_max_224, p.y);
    }
    const int w_224 = x_max_224 - x_min_224 + 1;
    const int h_224 = y_max_224 - y_min_224 + 1;

    const float scale_x = static_cast<float>(img_shape.width) / meta.new_w;
    const float scale_y = static_cast<float>(img_shape.height) / meta.new_h;

    area = static_cast<int>(std::round(area_224 * scale_x * scale_y));
    x = static_cast<int>(std::round(x_min_224 * scale_x));
    y = static_cast<int>(std::round(y_min_224 * scale_y));
    w = static_cast<int>(std::round(w_224 * scale_x));
    h = static_cast<int>(std::round(h_224 * scale_y));
}

void KMeans1D(const std::vector<float>& values, int k, int iters, std::vector<int>& labels, std::vector<float>& centers) {
    labels.assign(values.size(), 0);
    centers.clear();
    if (values.empty()) return;
    std::vector<float> v_sorted = values;
    std::sort(v_sorted.begin(), v_sorted.end());
    if (static_cast<int>(values.size()) < k) {
        const float m = std::accumulate(values.begin(), values.end(), 0.0f) / static_cast<float>(values.size());
        centers = {m};
        return;
    }
    centers = {v_sorted.front(), v_sorted[v_sorted.size() / 2], v_sorted.back()};

    std::mt19937 rng(0);
    std::uniform_int_distribution<int> uni(0, static_cast<int>(v_sorted.size() - 1));

    for (int it = 0; it < iters; ++it) {
        for (size_t i = 0; i < values.size(); ++i) {
            float best = std::abs(values[i] - centers[0]);
            int best_i = 0;
            for (int c = 1; c < k; ++c) {
                const float d = std::abs(values[i] - centers[c]);
                if (d < best) {
                    best = d;
                    best_i = c;
                }
            }
            labels[i] = best_i;
        }
        for (int c = 0; c < k; ++c) {
            float sum = 0.0f;
            int cnt = 0;
            for (size_t i = 0; i < values.size(); ++i) {
                if (labels[i] == c) {
                    sum += values[i];
                    ++cnt;
                }
            }
            if (cnt > 0) {
                centers[c] = sum / static_cast<float>(cnt);
            } else {
                centers[c] = v_sorted[static_cast<size_t>(uni(rng))];
            }
        }
    }

    std::vector<int> order(k);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return centers[a] < centers[b]; });
    std::vector<int> remap(k, 0);
    for (int i = 0; i < k; ++i) remap[order[i]] = i;
    std::vector<float> centers_sorted(k);
    for (int i = 0; i < k; ++i) centers_sorted[i] = centers[order[i]];
    centers = std::move(centers_sorted);
    for (auto& l : labels) l = remap[l];
}

std::vector<Component> Detect3x3GridComponents(const cv::Mat& cropped_binary) {
    std::vector<Component> empty;
    if (cropped_binary.empty()) return empty;

    cv::Mat labels, stats, centroids;
    const int num_labels = cv::connectedComponentsWithStats((cropped_binary > 0), labels, stats, centroids, 8, CV_32S);
    std::vector<Component> comps;
    for (int lab = 1; lab < num_labels; ++lab) {
        const int area = stats.at<int>(lab, cv::CC_STAT_AREA);
        if (area <= 0) continue;
        Component c;
        c.x = stats.at<int>(lab, cv::CC_STAT_LEFT);
        c.y = stats.at<int>(lab, cv::CC_STAT_TOP);
        c.w = stats.at<int>(lab, cv::CC_STAT_WIDTH);
        c.h = stats.at<int>(lab, cv::CC_STAT_HEIGHT);
        c.area = area;
        c.cx = static_cast<float>(centroids.at<double>(lab, 0));
        c.cy = static_cast<float>(centroids.at<double>(lab, 1));
        comps.push_back(c);
    }
    if (comps.size() < 9) return empty;

    std::sort(comps.begin(), comps.end(), [](const Component& a, const Component& b) { return a.area > b.area; });
    comps.resize(9);

    std::vector<float> xs;
    std::vector<float> ys;
    xs.reserve(9);
    ys.reserve(9);
    for (const auto& c : comps) {
        xs.push_back(c.cx);
        ys.push_back(c.cy);
    }

    std::vector<int> x_labels, y_labels;
    std::vector<float> x_centers, y_centers;
    KMeans1D(xs, 3, 30, x_labels, x_centers);
    KMeans1D(ys, 3, 30, y_labels, y_centers);
    if (x_centers.size() != 3 || y_centers.size() != 3) return empty;

    for (int i = 0; i < 3; ++i) {
        int cxn = 0;
        int cyn = 0;
        for (int j = 0; j < 9; ++j) {
            if (x_labels[j] == i) ++cxn;
            if (y_labels[j] == i) ++cyn;
        }
        if (cxn != 3 || cyn != 3) return empty;
    }

    const float dx1 = x_centers[1] - x_centers[0];
    const float dx2 = x_centers[2] - x_centers[1];
    const float dy1 = y_centers[1] - y_centers[0];
    const float dy2 = y_centers[2] - y_centers[1];
    if (dx1 <= 0 || dx2 <= 0 || dy1 <= 0 || dy2 <= 0) return empty;
    if (std::abs(dx1 - dx2) / std::max(dx1, dx2) > 0.35f) return empty;
    if (std::abs(dy1 - dy2) / std::max(dy1, dy2) > 0.35f) return empty;

    bool grid[3][3] = {};
    for (int i = 0; i < 9; ++i) {
        const int r = y_labels[i];
        const int c = x_labels[i];
        if (grid[r][c]) return empty;
        grid[r][c] = true;
    }

    return comps;
}

}  // namespace

PatchCoreDerived AnalyzePatchCore(const PatchCore::PatchCoreResult& patchcore_result,
                                  const cv::Size& img_shape,
                                  const cv::Mat& gray_patchcore,
                                  float score_threshold,
                                  float area_threshold,
                                  float mask_area_threshold) {
    PatchCoreDerived result;
    result.score = patchcore_result.image_score;
    result.scale_x = patchcore_result.meta.new_w > 0 ? static_cast<float>(img_shape.width) / patchcore_result.meta.new_w : 0.0f;
    result.scale_y = patchcore_result.meta.new_h > 0 ? static_cast<float>(img_shape.height) / patchcore_result.meta.new_h : 0.0f;

    result.area_cropped = BuildPatchCoreCroppedBinaryMask(patchcore_result.patch_scores, patchcore_result.meta, area_threshold);
    result.area_ratio = CalculatePatchCoreAreaRatio(result.area_cropped, patchcore_result.meta);
    GetPatchCoreDefectStats(result.area_cropped, patchcore_result.meta, img_shape, result.area, result.x, result.y, result.w, result.h);
    result.has_defect = (result.score >= score_threshold) && (result.area_ratio >= mask_area_threshold);

    if (!result.area_cropped.empty()) {
        result.area_orig = ResizePatchcoreMaskFromCropped(result.area_cropped, img_shape, cv::INTER_NEAREST);
        result.contrast = CalculateContrastRatio(gray_patchcore, result.area_orig, result.x, result.y, result.w, result.h);
    }

    if (std::abs(score_threshold - area_threshold) < 1e-6f) {
        result.score_cropped = result.area_cropped;
    } else {
        result.score_cropped = BuildPatchCoreCroppedBinaryMask(patchcore_result.patch_scores, patchcore_result.meta, score_threshold);
    }
    if (!result.score_cropped.empty()) {
        result.score_orig = ResizePatchcoreMaskFromCropped(result.score_cropped, img_shape, cv::INTER_NEAREST);
    }

    result.crosshair_components = Detect3x3GridComponents(result.score_cropped);
    result.has_crosshair_grid = result.crosshair_components.size() == 9;
    return result;
}

}  // namespace InspectionDLL::Internal
