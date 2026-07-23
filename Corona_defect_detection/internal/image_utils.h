#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

namespace InspectionDLL::Internal {

cv::Mat ProcessTIF32ForPatchcore(const cv::Mat& img, int max_border = 100);
cv::Mat ProcessForYolo(const cv::Mat& img);
cv::Mat ConvertGrayToU8Normalized(const cv::Mat& gray);

enum class ContrastPolarity {
    Auto,
    Dark,
    Bright,
};

// A finite value, including 0.0f, is a successfully calculated contrast ratio.
// NaN represents a calculation failure.
bool IsContrastRatioValid(float contrast_ratio);

float Median(std::vector<float>& values);
float Percentile(std::vector<float> values, float p);
float CalculateContrastRatio(const cv::Mat& gray_img,
                             const cv::Mat& binary_mask,
                             int x,
                             int y,
                             int w,
                             int h,
                             ContrastPolarity polarity = ContrastPolarity::Auto);
float CalculateContrastRatioAdaptive(const cv::Mat& gray_img,
                                     const cv::Mat& binary_mask,
                                     int x,
                                     int y,
                                     int w,
                                     int h,
                                     ContrastPolarity polarity = ContrastPolarity::Auto);

}  // namespace InspectionDLL::Internal
