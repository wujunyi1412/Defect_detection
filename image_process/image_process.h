#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

namespace ImageProcess {

class ImageProcessor {
public:
    static constexpr float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float IMAGENET_STD[3]  = {0.229f, 0.224f, 0.225f};

    static cv::Mat LetterboxResize(const cv::Mat& img,
                                   int target_h,
                                   int target_w,
                                   float& ratio,
                                   std::vector<float>& pad,
                                   cv::Scalar fill = cv::Scalar(114, 114, 114));


    static std::vector<float> HWCToCHW(const cv::Mat& img);
    static std::vector<float> HWCToCHW_normalize(const cv::Mat& img, float scale);
    static std::vector<float> HWCToCHW_normalize_gaussian(const cv::Mat& img, 
        const float mean[3], const float std[3], float scale);
    
    static cv::Mat GetSafeBorderMask(const cv::Mat& gray, int max_border_inner, int step);

    static void NormalizeNCHW(std::vector<float>& nchw,
                              int h,
                              int w,
                              const float mean[3],
                              const float scale[3]);

    static void ResizeBilinear3D(const float* in,
                                 int in_h,
                                 int in_w,
                                 int depth,
                                 float* out,
                                 int out_h,
                                 int out_w);

    static cv::Mat ResizeBilinearPillowResampleRGBU8(const cv::Mat& src, int dst_w, int dst_h);

    static cv::Mat Cvmat2BGR(const cv::Mat& img);
    static cv::Mat Cvmat2RGB(const cv::Mat& img);
    static cv::Mat Cvmat2Uint8(const cv::Mat& img);
};

} // namespace ImageProcess
