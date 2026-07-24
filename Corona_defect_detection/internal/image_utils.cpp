#include "image_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "image_process.h"

namespace InspectionDLL::Internal {
namespace {

float InvalidContrastRatio() {
    return std::numeric_limits<float>::quiet_NaN();
}

}  // namespace

bool IsContrastRatioValid(float contrast_ratio) {
    return std::isfinite(contrast_ratio) && contrast_ratio >= 0.0f;
}

InferenceImages PrepareInferenceImages(const cv::Mat& img, int max_border) {
    InferenceImages result;

    cv::Mat img_normalize = ImageProcess::ImageProcessor::Cvmat2Uint8(img);
    result.yolo_bgr = ImageProcess::ImageProcessor::Cvmat2BGR(img_normalize);
    if (result.yolo_bgr.empty()) return result;

    if (result.yolo_bgr.channels() == 3) {
        cv::cvtColor(result.yolo_bgr, result.yolo_gray, cv::COLOR_BGR2GRAY);
    } else {
        result.yolo_gray = result.yolo_bgr;
    }

    cv::Mat border_mask =
        ImageProcess::ImageProcessor::GetSafeBorderMask(result.yolo_gray, max_border, 5);

    result.patchcore_bgr = result.yolo_bgr.clone();
    result.patchcore_gray = result.yolo_gray.clone();
    result.patchcore_bgr.setTo(cv::Scalar(0, 0, 0), border_mask);
    result.patchcore_gray.setTo(cv::Scalar(0), border_mask);
    return result;
}

cv::Mat ProcessTIF32ForPatchcore(const cv::Mat& img, int max_border) {
    cv::Mat img_normalize = ImageProcess::ImageProcessor::Cvmat2Uint8(img);
    cv::Mat bgr = ImageProcess::ImageProcessor::Cvmat2BGR(img_normalize);
    if (bgr.empty()) return cv::Mat();

    cv::Mat gray;
    if (bgr.channels() == 3) {
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = bgr;
    }

    cv::Mat border_mask = ImageProcess::ImageProcessor::GetSafeBorderMask(gray, max_border, 5);
    bgr.setTo(cv::Scalar(0, 0, 0), border_mask);
    return bgr;
}

cv::Mat ProcessForYolo(const cv::Mat& img) {
    cv::Mat img_normalize = ImageProcess::ImageProcessor::Cvmat2Uint8(img);
    cv::Mat bgr = ImageProcess::ImageProcessor::Cvmat2BGR(img_normalize);
    if (bgr.empty()) return cv::Mat();
    return bgr;
}

float Median(std::vector<float>& values) {
    if (values.empty()) return 0.0f;
    const size_t n = values.size();
    const size_t mid = n / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    float med = values[mid];
    if (n % 2 == 0) {
        std::nth_element(values.begin(), values.begin() + (mid - 1), values.end());
        med = 0.5f * (med + values[mid - 1]);
    }
    return med;
}

float Percentile(std::vector<float> values, float p) {
    if (values.empty()) return 0.0f;
    if (p <= 0.0f) {
        return *std::min_element(values.begin(), values.end());
    }
    if (p >= 100.0f) {
        return *std::max_element(values.begin(), values.end());
    }
    std::sort(values.begin(), values.end());
    const float pos = (p / 100.0f) * static_cast<float>(values.size() - 1);
    const int lo = static_cast<int>(std::floor(pos));
    const int hi = static_cast<int>(std::ceil(pos));
    const float a = values[static_cast<size_t>(lo)];
    const float b = values[static_cast<size_t>(hi)];
    const float t = pos - static_cast<float>(lo);
    return a + (b - a) * t;
}

float CalculateContrastRatio(const cv::Mat& gray_img,
                             const cv::Mat& binary_mask,
                             int x,
                             int y,
                             int w,
                             int h,
                             ContrastPolarity polarity) {
    if (gray_img.empty() || binary_mask.empty()) return InvalidContrastRatio();

    cv::Mat g = gray_img;
    cv::Mat m = binary_mask;

    if (w > 0 && h > 0) {
        const int H = gray_img.rows;
        const int W = gray_img.cols;
        const int x0 = static_cast<int>(std::max(0, std::min(x, W - 1)));
        const int y0 = static_cast<int>(std::max(0, std::min(y, H - 1)));
        const int x1 = static_cast<int>(std::max(x0 + 1, std::min(x + w, W)));
        const int y1 = static_cast<int>(std::max(y0 + 1, std::min(y + h, H)));
        if (x1 <= x0 || y1 <= y0) return InvalidContrastRatio();
        g = gray_img(cv::Rect(x0, y0, x1 - x0, y1 - y0));
        m = binary_mask(cv::Rect(x0, y0, x1 - x0, y1 - y0));
    }

    if (g.empty()) return InvalidContrastRatio();

    cv::Mat g32;
    if (g.type() == CV_32F) {
        g32 = g;
    } else {
        g.convertTo(g32, CV_32F);
    }

    cv::Mat m_u8;
    if (m.type() == CV_8U) {
        m_u8 = m;
    } else {
        m.convertTo(m_u8, CV_8U);
    }
    cv::threshold(m_u8, m_u8, 0, 255, cv::THRESH_BINARY);
    if (cv::countNonZero(m_u8) == 0) return InvalidContrastRatio();

    const int min_dim = std::min(g32.rows, g32.cols);
    if (min_dim <= 0) return InvalidContrastRatio();

    const int r2 = static_cast<int>(std::clamp(static_cast<int>(std::round(static_cast<float>(min_dim) * 0.12f)), 6, 24));
    const int r1 = static_cast<int>(std::clamp(static_cast<int>(std::round(static_cast<float>(r2) * 0.25f)), 2, std::max(2, r2 - 2)));

    cv::Mat k1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * r1 + 1, 2 * r1 + 1));
    cv::Mat k2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * r2 + 1, 2 * r2 + 1));
    cv::Mat dil1, dil2;
    cv::dilate(m_u8, dil1, k1, cv::Point(-1, -1), 1);
    cv::dilate(m_u8, dil2, k2, cv::Point(-1, -1), 1);
    cv::Mat ring;
    cv::bitwise_and(dil2, ~dil1, ring);
    cv::Mat bg = ring;
    if (cv::countNonZero(bg) < 16) {
        bg = cv::Scalar(255) - m_u8;
    }
    cv::threshold(bg, bg, 0, 255, cv::THRESH_BINARY);
    if (cv::countNonZero(bg) == 0) return InvalidContrastRatio();

    std::vector<float> defect_vals;
    std::vector<float> bg_vals;
    defect_vals.reserve(static_cast<size_t>(cv::countNonZero(m_u8)));
    bg_vals.reserve(static_cast<size_t>(cv::countNonZero(bg)));
    for (int yy = 0; yy < g32.rows; ++yy) {
        const float* pg = g32.ptr<float>(yy);
        const uchar* pm = m_u8.ptr<uchar>(yy);
        const uchar* pb = bg.ptr<uchar>(yy);
        for (int xx = 0; xx < g32.cols; ++xx) {
            const float v = pg[xx];
            if (pm[xx]) defect_vals.push_back(v);
            if (pb[xx]) bg_vals.push_back(v);
        }
    }
    if (defect_vals.empty() || bg_vals.empty()) return InvalidContrastRatio();

    constexpr float eps = 1e-6f;
    constexpr float low_clip = 8.0f;

    std::vector<float> bg_vals_valid;
    bg_vals_valid.reserve(bg_vals.size());
    for (float v : bg_vals) {
        if (v >= low_clip) bg_vals_valid.push_back(v);
    }

    float bg_med = 0.0f;
    if (bg_vals_valid.size() >= 16) {
        bg_med = Median(bg_vals_valid);
    } else {
        std::vector<float> tmp = bg_vals;
        bg_med = Median(tmp);
    }

    std::vector<float> bg_dev;
    bg_dev.reserve(bg_vals.size());
    for (float v : bg_vals) bg_dev.push_back(std::abs(v - bg_med));
    const float bg_mad = Median(bg_dev) + eps;

    std::vector<float> bg_inliers;
    bg_inliers.reserve(bg_vals.size());
    for (size_t i = 0; i < bg_vals.size(); ++i) {
        if (bg_dev[i] <= 3.0f * bg_mad && bg_vals[i] >= low_clip) {
            bg_inliers.push_back(bg_vals[i]);
        }
    }

    float bg_mean = bg_med;
    if (bg_inliers.size() >= 16) {
        bg_mean = Median(bg_inliers);
    }

    std::vector<float> diff;
    diff.reserve(defect_vals.size());
    for (float v : defect_vals) diff.push_back(v - bg_mean);

    float defect_mean = 0.0f;
    if (diff.size() >= 32) {
        bool use_bright_tail = false;
        if (polarity == ContrastPolarity::Bright) {
            use_bright_tail = true;
        } else if (polarity == ContrastPolarity::Dark) {
            use_bright_tail = false;
        } else {
            std::vector<float> tmp = diff;
            use_bright_tail = Median(tmp) >= 0.0f;
        }

        if (use_bright_tail) {
            const float thr = Percentile(diff, 70.0f);
            double sum = 0.0;
            size_t cnt = 0;
            for (size_t i = 0; i < diff.size(); ++i) {
                if (diff[i] >= thr) {
                    sum += defect_vals[i];
                    ++cnt;
                }
            }
            if (cnt > 0) defect_mean = static_cast<float>(sum / static_cast<double>(cnt));
        } else {
            const float thr = Percentile(diff, 30.0f);
            double sum = 0.0;
            size_t cnt = 0;
            for (size_t i = 0; i < diff.size(); ++i) {
                if (diff[i] <= thr) {
                    sum += defect_vals[i];
                    ++cnt;
                }
            }
            if (cnt > 0) defect_mean = static_cast<float>(sum / static_cast<double>(cnt));
        }
    }

    if (defect_mean == 0.0f) {
        double sum = 0.0;
        for (float v : defect_vals) sum += v;
        defect_mean = static_cast<float>(sum / static_cast<double>(defect_vals.size()));
    }

    const float denom = std::max(bg_mean, low_clip) + eps;
    const float ratio = defect_mean / denom;
    if (!IsContrastRatioValid(ratio)) return InvalidContrastRatio();
    return ratio;
}

cv::Mat ConvertGrayToU8Normalized(const cv::Mat& gray) {
    if (gray.empty()) return cv::Mat();
    if (gray.type() == CV_8U) return gray;

    cv::Mat gray32;
    if (gray.depth() == CV_32F) {
        gray32 = gray;
    } else {
        gray.convertTo(gray32, CV_32F);
    }

    double min_v = 0.0;
    double max_v = 0.0;
    cv::minMaxLoc(gray32, &min_v, &max_v);
    if (!std::isfinite(min_v) || !std::isfinite(max_v)) return cv::Mat();
    if (max_v - min_v < 1e-6) {
        return cv::Mat(gray32.size(), CV_8U, cv::Scalar(0));
    }

    cv::Mat gray_u8;
    gray32.convertTo(gray_u8, CV_8U, 255.0 / (max_v - min_v), -min_v * 255.0 / (max_v - min_v));
    return gray_u8;
}

}  // namespace InspectionDLL::Internal
