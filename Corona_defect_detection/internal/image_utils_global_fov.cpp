#include "image_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace InspectionDLL::Internal {
namespace {

float InvalidContrastRatio() {
    return std::numeric_limits<float>::quiet_NaN();
}

cv::Rect ClampRect(int x, int y, int w, int h, const cv::Size& size) {
    if (w <= 0 || h <= 0 || size.width <= 0 || size.height <= 0) return {};
    const int x1 = std::clamp(x, 0, size.width);
    const int y1 = std::clamp(y, 0, size.height);
    const int x2 = static_cast<int>(std::clamp<long long>(
        static_cast<long long>(x) + w, 0, size.width));
    const int y2 = static_cast<int>(std::clamp<long long>(
        static_cast<long long>(y) + h, 0, size.height));
    return x2 > x1 && y2 > y1 ? cv::Rect(x1, y1, x2 - x1, y2 - y1) : cv::Rect();
}

cv::Mat BuildDetectionMask(const cv::Mat& binary_mask,
                           const cv::Rect& detection,
                           const cv::Size& image_size) {
    if (binary_mask.empty() || binary_mask.channels() != 1) return {};
    cv::Mat local;
    if (binary_mask.size() == image_size) {
        local = binary_mask(detection).clone();
    } else if (binary_mask.size() == detection.size()) {
        local = binary_mask.clone();
    } else {
        cv::resize(binary_mask, local, detection.size(), 0.0, 0.0, cv::INTER_NEAREST);
    }
    if (local.depth() != CV_8U) local.convertTo(local, CV_8U);
    cv::threshold(local, local, 0, 255, cv::THRESH_BINARY);
    return local;
}

// Only low-valued components connected to an image edge are treated as the
// black frame. Dark defects inside the FOV therefore remain valid samples.
cv::Mat FindBlackFrame(const cv::Mat& gray_f32) {
    cv::Mat black_frame(gray_f32.size(), CV_8U, cv::Scalar(0));
    std::vector<float> values;
    values.reserve(gray_f32.total());
    for (int y = 0; y < gray_f32.rows; ++y) {
        const float* row = gray_f32.ptr<float>(y);
        for (int x = 0; x < gray_f32.cols; ++x) {
            if (std::isfinite(row[x])) values.push_back(row[x]);
        }
    }
    if (values.empty()) return black_frame;

    const float low = Percentile(values, 10.0f);
    const float material = Percentile(values, 80.0f);
    if (!std::isfinite(material) || material <= 0.0f) return black_frame;
    const float threshold = std::min(material * 0.35f,
                                     low + 0.20f * (material - low));

    cv::Mat candidates;
    cv::compare(gray_f32, threshold, candidates, cv::CMP_LE);
    cv::Mat labels;
    const int count = cv::connectedComponents(candidates, labels, 8, CV_32S);
    if (count <= 1) return black_frame;

    std::vector<uchar> edge_labels(static_cast<size_t>(count), 0);
    for (int y = 0; y < labels.rows; ++y) {
        edge_labels[labels.at<int>(y, 0)] = 1;
        edge_labels[labels.at<int>(y, labels.cols - 1)] = 1;
    }
    for (int x = 0; x < labels.cols; ++x) {
        edge_labels[labels.at<int>(0, x)] = 1;
        edge_labels[labels.at<int>(labels.rows - 1, x)] = 1;
    }
    edge_labels[0] = 0;
    for (int y = 0; y < labels.rows; ++y) {
        const int* label_row = labels.ptr<int>(y);
        uchar* frame_row = black_frame.ptr<uchar>(y);
        for (int x = 0; x < labels.cols; ++x) {
            if (edge_labels[label_row[x]]) frame_row[x] = 255;
        }
    }

    if (cv::countNonZero(black_frame) > 0) {
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(black_frame, black_frame, kernel);
    }
    return black_frame;
}

float TailMean(std::vector<float> values, ContrastPolarity polarity, float background) {
    if (values.empty()) return InvalidContrastRatio();
    if (polarity == ContrastPolarity::Auto) {
        std::vector<float> copy = values;
        polarity = Median(copy) <= background ? ContrastPolarity::Dark
                                              : ContrastPolarity::Bright;
    }
    std::sort(values.begin(), values.end());
    const size_t tail_count = std::max<size_t>(
        1, static_cast<size_t>(std::ceil(values.size() * 0.10)));
    const size_t begin = polarity == ContrastPolarity::Bright
                             ? values.size() - tail_count
                             : 0;
    const size_t end = polarity == ContrastPolarity::Bright
                           ? values.size()
                           : tail_count;
    double sum = 0.0;
    for (size_t i = begin; i < end; ++i) sum += values[i];
    return static_cast<float>(sum / static_cast<double>(end - begin));
}

}  // namespace

float CalculateContrastRatioGlobalFov(const cv::Mat& gray_img,
                                      const cv::Mat& binary_mask,
                                      int x,
                                      int y,
                                      int w,
                                      int h,
                                      ContrastPolarity polarity) {
    if (gray_img.empty() || gray_img.channels() != 1 || binary_mask.empty()) {
        return InvalidContrastRatio();
    }
    const cv::Rect detection = ClampRect(x, y, w, h, gray_img.size());
    if (detection.empty()) return InvalidContrastRatio();
    const cv::Mat mask = BuildDetectionMask(binary_mask, detection, gray_img.size());
    if (mask.empty() || cv::countNonZero(mask) == 0) return InvalidContrastRatio();

    cv::Mat gray_f32;
    gray_img.convertTo(gray_f32, CV_32F);
    const cv::Mat black_frame = FindBlackFrame(gray_f32);

    std::vector<float> background_values;
    background_values.reserve(gray_f32.total() -
                              static_cast<size_t>(cv::countNonZero(black_frame)));
    for (int yy = 0; yy < gray_f32.rows; ++yy) {
        const float* gray_row = gray_f32.ptr<float>(yy);
        const uchar* frame_row = black_frame.ptr<uchar>(yy);
        for (int xx = 0; xx < gray_f32.cols; ++xx) {
            if (!frame_row[xx] && std::isfinite(gray_row[xx])) {
                background_values.push_back(gray_row[xx]);
            }
        }
    }
    if (background_values.empty()) return InvalidContrastRatio();
    const float background = Median(background_values);
    if (!std::isfinite(background) || background <= 1e-6f) return InvalidContrastRatio();

    std::vector<float> defect_values;
    defect_values.reserve(static_cast<size_t>(cv::countNonZero(mask)));
    const cv::Mat defect_gray = gray_f32(detection);
    for (int yy = 0; yy < defect_gray.rows; ++yy) {
        const float* gray_row = defect_gray.ptr<float>(yy);
        const uchar* mask_row = mask.ptr<uchar>(yy);
        for (int xx = 0; xx < defect_gray.cols; ++xx) {
            if (mask_row[xx] && std::isfinite(gray_row[xx])) {
                defect_values.push_back(gray_row[xx]);
            }
        }
    }
    const float defect = TailMean(std::move(defect_values), polarity, background);
    const float ratio = defect / background;
    return IsContrastRatioValid(ratio) ? ratio : InvalidContrastRatio();
}

float CalculateContrastRatioByMode(const cv::Mat& gray_img,
                                   const cv::Mat& binary_mask,
                                   int x,
                                   int y,
                                   int w,
                                   int h,
                                   ContrastPolarity polarity,
                                   ContrastCalculationMode mode) {
    switch (mode) {
        case ContrastCalculationMode::LocalRing:
            return CalculateContrastRatio(gray_img, binary_mask, x, y, w, h, polarity);
        case ContrastCalculationMode::GlobalFovMedian:
            return CalculateContrastRatioGlobalFov(
                gray_img, binary_mask, x, y, w, h, polarity);
        case ContrastCalculationMode::AdaptiveLocalPlane:
        default:
            return CalculateContrastRatioAdaptive(
                gray_img, binary_mask, x, y, w, h, polarity);
    }
}

}  // namespace InspectionDLL::Internal
