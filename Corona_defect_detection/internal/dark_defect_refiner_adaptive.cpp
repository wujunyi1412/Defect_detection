#include "dark_defect_refiner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "image_utils.h"

namespace InspectionDLL::Internal {
namespace {

// 根据检测框的坐标和宽高，计算出一个 ROI 的矩形框，确保其在图像内
cv::Rect ClampDetectionRect(const DetectionResult& detail, const cv::Size& image_size) {
    if (image_size.width <= 0 || image_size.height <= 0 ||
        !std::isfinite(detail.x) || !std::isfinite(detail.y) ||
        !std::isfinite(detail.w) || !std::isfinite(detail.h) ||
        detail.w <= 0.0f || detail.h <= 0.0f) {
        return {};
    }

    int x1 = static_cast<int>(std::floor(detail.x));
    int y1 = static_cast<int>(std::floor(detail.y));
    int x2 = static_cast<int>(std::ceil(detail.x + detail.w));
    int y2 = static_cast<int>(std::ceil(detail.y + detail.h));
    x1 = std::clamp(x1, 0, image_size.width);
    y1 = std::clamp(y1, 0, image_size.height);
    x2 = std::clamp(x2, 0, image_size.width);
    y2 = std::clamp(y2, 0, image_size.height);
    if (x2 <= x1 || y2 <= y1) return {};
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

// 根据检测框的坐标和宽高，计算出一个扩展后的矩形框，确保其在图像内
cv::Rect ExpandRect(const cv::Rect& rect, int margin, const cv::Size& image_size) {
    const int x1 = std::max(0, rect.x - margin);
    const int y1 = std::max(0, rect.y - margin);
    const int x2 = std::min(image_size.width, rect.x + rect.width + margin);
    const int y2 = std::min(image_size.height, rect.y + rect.height + margin);
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

// 根据图像和掩码，计算掩码区域内的中值
double MaskedMedian(const cv::Mat& image, const cv::Mat& mask) {
    std::vector<float> values;
    values.reserve(static_cast<size_t>(cv::countNonZero(mask)));
    for (int y = 0; y < image.rows; ++y) {
        // 遍历图像的每一行，只处理掩码为非零的像素
        const uchar* image_row = image.ptr<uchar>(y);
        const uchar* mask_row = mask.ptr<uchar>(y);
        for (int x = 0; x < image.cols; ++x) {
            if (mask_row[x]) values.push_back(static_cast<float>(image_row[x]));
        }
    }
    return static_cast<double>(Median(values));
}

// 根据图像和掩码，拟合一个背景平面，返回平面的系数和残差标准差
bool FitBackgroundPlane(const cv::Mat& image, const cv::Mat& mask,
                        cv::Vec3d& coefficients, double& residual_sigma) {
    cv::Matx33d normal = cv::Matx33d::zeros();
    cv::Vec3d rhs(0.0, 0.0, 0.0);
    int sample_count = 0;
    for (int y = 0; y < image.rows; ++y) {
        const uchar* image_row = image.ptr<uchar>(y);
        const uchar* mask_row = mask.ptr<uchar>(y);
        for (int x = 0; x < image.cols; ++x) {
            if (!mask_row[x]) continue;
            // 遍历掩码为非零的像素，将其坐标变成 (x, y, 1) 的形式
            const cv::Vec3d sample(static_cast<double>(x), static_cast<double>(y), 1.0);
            // 同时得到像素值
            const double value = image_row[x];
            // 计算坐标的外积，得到一个 3x3 的矩阵
            normal += sample * sample.t();
            // 计算像素值与坐标的乘积，得到一个 3x1 的向量
            rhs += sample * value;
            ++sample_count;
        }
    }
    // 如果样本数量不足 3 个，或者无法求解线性方程组(三点不共线)，返回 false
    if (sample_count < 3 || !cv::solve(normal, rhs, coefficients, cv::DECOMP_SVD)) return false;

    std::vector<float> residuals;
    // 计算每个像素的预测值与实际值的差，得到残差向量
    residuals.reserve(static_cast<size_t>(sample_count));
    for (int y = 0; y < image.rows; ++y) {
        const uchar* image_row = image.ptr<uchar>(y);
        const uchar* mask_row = mask.ptr<uchar>(y);
        for (int x = 0; x < image.cols; ++x) {
            if (!mask_row[x]) continue;
            // 计算预测值，即系数与坐标的点积
            const double predicted = coefficients[0] * x + coefficients[1] * y + coefficients[2];
            residuals.push_back(static_cast<float>(image_row[x] - predicted));
        }
    }

    // 对残差向量取中值，将其加到背景系数的第三项上，作为背景的基线
    const double residual_median = Median(residuals);
    coefficients[2] += residual_median;
    for (float& residual : residuals) {
        residual = static_cast<float>(std::abs(residual - residual_median));
    }
    residual_sigma = std::max(0.8, 1.4826 * static_cast<double>(Median(residuals)));
    return true;
}

// 根据掩码，计算非零像素的最小外接矩形
cv::Rect NonZeroBoundingRect(const cv::Mat& mask) {
    std::vector<cv::Point> points;
    cv::findNonZero(mask, points);
    return points.empty() ? cv::Rect() : cv::boundingRect(points);
}

// 找到图像中最外层的黑色区域(掩码)
cv::Mat FindOuterBlackMask(const cv::Mat& context_u8, int transition_guard) {
    cv::Mat result(context_u8.size(), CV_8U, cv::Scalar(0));

    std::vector<float> values;
    values.reserve(context_u8.total());
    for (int y = 0; y < context_u8.rows; ++y) {
        const uchar* row = context_u8.ptr<uchar>(y);
        for (int x = 0; x < context_u8.cols; ++x) values.push_back(row[x]);
    }
    // 计算图像像素值的 80% 分位数，作为前景的参考值
    const double foreground_reference = Percentile(values, 80.0f);
    // 计算黑色阈值，确保其在 [6.0, 64.0] 范围内
    const double black_threshold = std::clamp(foreground_reference * 0.30, 6.0, 64.0);

    cv::Mat black_candidates;
    // 找到所有像素值小于等于黑色阈值的候选区域
    cv::compare(context_u8, black_threshold, black_candidates, cv::CMP_LE);
    cv::Mat labels;
    // 对候选区域进行连通域分析，得到不同的黑色区域(每个区域用不同的标签表示，最后一共有 label_count + 1 个区域)
    const int label_count = cv::connectedComponents(black_candidates, labels, 8, CV_32S);
    if (label_count <= 1) return result;

    // 构建边界标签，用于标记黑色区域的边界像素
    std::vector<uchar> border_labels(static_cast<size_t>(label_count), 0);
    for (int y = 0; y < labels.rows; ++y) {
        border_labels[labels.at<int>(y, 0)] = 1;
        border_labels[labels.at<int>(y, labels.cols - 1)] = 1;
    }
    for (int x = 0; x < labels.cols; ++x) {
        border_labels[labels.at<int>(0, x)] = 1;
        border_labels[labels.at<int>(labels.rows - 1, x)] = 1;
    }
    border_labels[0] = 0;

    // 遍历所有像素，将边界像素设为白色，其他像素设为黑色
    for (int y = 0; y < labels.rows; ++y) {
        const int* label_row = labels.ptr<int>(y);
        uchar* result_row = result.ptr<uchar>(y);
        for (int x = 0; x < labels.cols; ++x) {
            if (border_labels[label_row[x]]) result_row[x] = 255;
        }
    }
    // 如果指定了过渡守卫值，且结果中仍有非零像素，则对结果进行膨胀操作，以扩展黑色区域的边界
    if (transition_guard > 0 && cv::countNonZero(result) > 0) {
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::dilate(result, result, kernel, cv::Point(-1, -1), transition_guard);
    }
    return result;
}

// 判断检测框内的最外层黑色区域是否为长边界
bool IsLongProductBoundary(const cv::Mat& outer_black_in_detection) {
    if (outer_black_in_detection.empty() || cv::countNonZero(outer_black_in_detection) == 0) {
        return false;
    }
    const int width = outer_black_in_detection.cols;
    const int height = outer_black_in_detection.rows;
    const double aspect = static_cast<double>(std::max(width, height)) /
                          std::max(1, std::min(width, height));
    if (aspect < 6.0) return false;

    const cv::Rect black_extent = NonZeroBoundingRect(outer_black_in_detection);
    if (width >= height) {
        return black_extent.width >= static_cast<int>(std::ceil(width * 0.75)) &&
               black_extent.height <= static_cast<int>(std::ceil(height * 0.60));
    }
    return black_extent.height >= static_cast<int>(std::ceil(height * 0.75)) &&
           black_extent.width <= static_cast<int>(std::ceil(width * 0.60));
}

}  // namespace

bool RefineDarkDefectGeometryAdaptive(const cv::Mat& gray_yolo, DetectionResult& detail) {
    if (gray_yolo.empty() || gray_yolo.channels() != 1) return false;

    // 计算检测框的坐标和宽高，确保其在图像内
    const cv::Rect detection = ClampDetectionRect(detail, gray_yolo.size());
    if (detection.empty()) return false;

    // 可以二次处理的最大检测框边长
    constexpr int kMaxRefinedBoxSide = 30;
    if (detection.width > kMaxRefinedBoxSide || detection.height > kMaxRefinedBoxSide) {
        return false;
    }

    // 扩充检测框的上下文区域，并确保其在图像内
    const int short_side = std::min(detection.width, detection.height);
    const int margin = std::clamp(static_cast<int>(std::ceil(short_side * 0.30)), 4, 24);
    const cv::Rect context = ExpandRect(detection, margin, gray_yolo.size());
    cv::Mat context_u8 = ConvertGrayToU8Normalized(gray_yolo(context));
    if (context_u8.empty()) return false;

    // 计算检测框在上下文区域内的坐标和宽高
    const cv::Rect search(detection.x - context.x, detection.y - context.y,
                          detection.width, detection.height);
    const int transition_guard = std::clamp(short_side / 12, 1, 3);
    // 找到上下文区域内的最外层黑色区域(掩码)
    const cv::Mat image_border_black = FindOuterBlackMask(context_u8, transition_guard);
    if (IsLongProductBoundary(image_border_black(search))) return false;

    // 背景掩码，缺陷区域和最外层黑色区域置零
    cv::Mat background_mask(context.size(), CV_8U, cv::Scalar(255));
    background_mask(search).setTo(0);
    background_mask.setTo(0, image_border_black);

    // At an image edge there may be too little outer context. Fall back to a thin
    // band inside the detection, which is still more robust than the ROI mean.
    const int min_background_pixels = std::max(12, detection.area() / 8);
    if (cv::countNonZero(background_mask) < min_background_pixels) {
        background_mask.setTo(0);
        const int band = std::max(1, std::min(short_side / 6, 4));
        cv::Mat search_background = background_mask(search);
        search_background.rowRange(0, band).setTo(255);
        search_background.rowRange(search.height - band, search.height).setTo(255);
        search_background.colRange(0, band).setTo(255);
        search_background.colRange(search.width - band, search.width).setTo(255);
        background_mask.setTo(0, image_border_black);
    }
    if (cv::countNonZero(background_mask) == 0) return false;

    cv::Vec3d background_plane(0.0, 0.0, MaskedMedian(context_u8, background_mask));
    double noise_sigma = 0.8;
    FitBackgroundPlane(context_u8, background_mask, background_plane, noise_sigma);

    cv::Mat smoothed_background;
    int blur_size = std::clamp((short_side / 3) | 1, 3, 31);
    cv::GaussianBlur(context_u8, smoothed_background,
                     cv::Size(blur_size, blur_size), 0.0, 0.0, cv::BORDER_REPLICATE);

    cv::Mat context_f32;
    cv::Mat smooth_f32;
    context_u8.convertTo(context_f32, CV_32F);
    smoothed_background.convertTo(smooth_f32, CV_32F);

    cv::Mat fitted_background(context.size(), CV_32F);
    for (int y = 0; y < fitted_background.rows; ++y) {
        float* row = fitted_background.ptr<float>(y);
        for (int x = 0; x < fitted_background.cols; ++x) {
            row[x] = static_cast<float>(background_plane[0] * x +
                                        background_plane[1] * y + background_plane[2]);
        }
    }

    cv::Mat global_dark = fitted_background - context_f32;
    cv::Mat local_dark = smooth_f32 - context_f32;
    cv::max(global_dark, local_dark, global_dark);
    cv::max(global_dark, 0.0, global_dark);
    const cv::Mat response = global_dark(search);

    double max_response = 0.0;
    cv::minMaxLoc(response, nullptr, &max_response);
    const double seed_threshold = std::max(2.0, 2.5 * noise_sigma);
    if (max_response < seed_threshold) return false;

    // Hysteresis keeps weak edges belonging to a strong stain while rejecting
    // isolated low-contrast background fluctuations.
    const bool is_small_detection = short_side <= 30;
    const double grow_threshold = is_small_detection
                                      ? std::max(1.5, std::max(seed_threshold * 0.65,
                                                               1.75 * noise_sigma))
                                      : std::max(1.0, std::min(seed_threshold * 0.55,
                                                               1.25 * noise_sigma));
    cv::Mat seed_mask;
    cv::Mat grow_mask;
    cv::compare(response, seed_threshold, seed_mask, cv::CMP_GE);
    cv::compare(response, grow_threshold, grow_mask, cv::CMP_GE);
    const cv::Mat search_border_black = image_border_black(search);
    seed_mask.setTo(0, search_border_black);
    grow_mask.setTo(0, search_border_black);

    if (!is_small_detection && short_side >= 12) {
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(grow_mask, grow_mask, cv::MORPH_CLOSE, kernel);
        grow_mask.setTo(0, search_border_black);
    }

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int label_count = cv::connectedComponentsWithStats(
        grow_mask, labels, stats, centroids, is_small_detection ? 4 : 8, CV_32S);
    if (label_count <= 1) return false;

    const cv::Point2d detection_center((search.width - 1) * 0.5, (search.height - 1) * 0.5);
    const double center_scale = std::max(1.0, std::hypot(search.width, search.height));
    double best_score = -std::numeric_limits<double>::infinity();
    int best_label = -1;

    for (int label = 1; label < label_count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area <= 0) continue;
        const cv::Rect component(stats.at<int>(label, cv::CC_STAT_LEFT),
                                 stats.at<int>(label, cv::CC_STAT_TOP),
                                 stats.at<int>(label, cv::CC_STAT_WIDTH),
                                 stats.at<int>(label, cv::CC_STAT_HEIGHT));
        const cv::Mat component_mask = labels(component) == label;
        const int seed_overlap = cv::countNonZero(seed_mask(component) & component_mask);
        if (seed_overlap == 0) continue;

        const double mean_response = cv::mean(response(component), component_mask)[0];
        const double fill = static_cast<double>(area) / std::max(1, component.area());
        const cv::Point2d center(centroids.at<double>(label, 0), centroids.at<double>(label, 1));
        const double center_distance = cv::norm(center - detection_center) / center_scale;
        const double seed_fraction = static_cast<double>(seed_overlap) / area;
        const double score = mean_response + 8.0 * seed_fraction + 3.0 * fill +
                             2.0 * std::log1p(static_cast<double>(area)) - 2.0 * center_distance;
        if (score > best_score) {
            best_score = score;
            best_label = label;
        }
    }
    if (best_label < 0) return false;

    cv::Mat final_mask = labels == best_label;
    const cv::Rect refined = NonZeroBoundingRect(final_mask);
    const int refined_area = cv::countNonZero(final_mask);
    if (refined.empty() || refined_area <= 0) return false;

    const int refined_x = detection.x + refined.x;
    const int refined_y = detection.y + refined.y;
    const cv::Mat refined_mask = final_mask(refined);
    const float refined_contrast = CalculateContrastRatioAdaptive(
        gray_yolo, refined_mask, refined_x, refined_y, refined.width, refined.height,
        ContrastPolarity::Dark);

    detail.x = static_cast<float>(refined_x);
    detail.y = static_cast<float>(refined_y);
    detail.w = static_cast<float>(refined.width);
    detail.h = static_cast<float>(refined.height);
    detail.area = refined_area;
    detail.mask = refined_mask.clone();
    if (IsContrastRatioValid(refined_contrast)) detail.contrast = refined_contrast;
    return true;
}

}  // namespace InspectionDLL::Internal
