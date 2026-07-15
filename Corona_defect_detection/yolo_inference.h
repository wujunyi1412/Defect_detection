#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"

namespace YOLO {

struct Detection {
    int class_id;
    float confidence;
    float x1, y1, x2, y2;
    cv::Mat mask; // binary mask
};

class YOLOv8Segmentor {
public:
    enum class NmsMode {
        Global = 0,
        ClassAware = 1,
    };

    YOLOv8Segmentor();
    ~YOLOv8Segmentor();

    bool Initialize(
        const std::string& model_path,
        float score_threshold = 0.25f,
        float iou_threshold = 0.2f,
        const std::vector<int>& img_size = {640, 640},
        int ort_intra_threads = 4
    );

    bool Infer(const cv::Mat& image,
               std::vector<Detection>& detections);

    void SetNmsMode(NmsMode mode);

private:
    cv::Mat Preprocess(const cv::Mat& img, float& ratio, std::vector<float>& pad);
    
    void Postprocess(
        const std::vector<Ort::Value>& outputs,
        const cv::Mat& original_img,
        float ratio,
        const std::vector<float>& pad,
        std::vector<Detection>& detections
    );

    Ort::Env env_;
    Ort::Session* session_ = nullptr;
    Ort::RunOptions run_options_;
    
    std::vector<std::string> input_names_storage_;
    std::vector<std::string> output_names_storage_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    
    std::vector<int> img_size_;
    float score_threshold_;
    float iou_threshold_;
    NmsMode nms_mode_ = NmsMode::Global;

    static constexpr int NUM_CLASSES = 3;
};

} // namespace YOLO
