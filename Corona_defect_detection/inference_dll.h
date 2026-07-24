#pragma once

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT __attribute__((visibility("default")))
#endif

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace InspectionDLL {

struct DetectionResult {
    std::string name;
    int area;
    float x, y, w, h;
    float contrast;
    float score;
    cv::Mat mask;
};

struct MaskOverlayOptions {
    bool enabled = false;
    int color_r = 255;
    int color_g = 0;
    int color_b = 0;
    float alpha = 0.35f;
};

struct InferenceResult {
    std::string result; // "OK" or "NG"
    std::vector<DetectionResult> details;
    float yolo_score;
    float patchcore_score;
    float patchcore_area_ratio;
};

class InspectionEngine {
public:
    DLL_EXPORT InspectionEngine();
    DLL_EXPORT ~InspectionEngine();

    DLL_EXPORT bool Initialize(const std::string& config_path);

    DLL_EXPORT bool ProcessImage(const cv::Mat& input_image, InferenceResult& output);
    
    DLL_EXPORT bool ProcessImagePath(const std::string& image_path, InferenceResult& output);

    DLL_EXPORT bool ProcessFloatArry(const float* image_arry, InferenceResult& output, int width, int height);
    
    DLL_EXPORT void SetThresholds(float score_thresh, float area_thresh, float mask_area_thresh);

    DLL_EXPORT void SetDarkClustersThreshold(float dark_clusters_thresh);

    DLL_EXPORT void SetYoloNmsMode(bool class_aware);

    DLL_EXPORT bool ShouldDrawDefectBox() const;

    DLL_EXPORT bool ShouldDrawBoxDetails() const;

    DLL_EXPORT bool ShouldConcatOriginalImage() const;

    DLL_EXPORT MaskOverlayOptions GetMaskOverlayOptions() const;

    DLL_EXPORT const std::string& GetLastError() const;
    
    DLL_EXPORT void Release();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace InspectionDLL
