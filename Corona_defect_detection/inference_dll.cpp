#include "inference_dll.h"

#include <iostream>
#include <memory>

#include <opencv2/opencv.hpp>

#include "config.h"
#include "patchcore_inference.h"
#include "yolo_inference.h"
#include "internal/image_utils.h"
#include "internal/patchcore_postprocess.h"
#include "internal/result_builder.h"
#include "internal/yolo_postprocess.h"

namespace InspectionDLL {

class InspectionEngine::Impl {
public:
    using CategoryFilter = InspectionConfig::CategoryFilterConfig;
    using AbnormalFilter = InspectionConfig::AbnormalFilterConfig;

    YOLO::YOLOv8Segmentor yolo_detector;
    PatchCore::PatchCoreDetector patchcore_detector;

    float score_threshold_ = 1.4f;
    float area_threshold_ = 1.4f;
    float mask_area_threshold_ = 0.01f;
    float dark_clusters_threshold_ = 0.8f;
    bool draw_box_details_ = true;

    AbnormalFilter abnormal_filter_;
    CategoryFilter stain_filter_;
    CategoryFilter darkclusters_filter_;
    CategoryFilter brightstripes_filter_;
    CategoryFilter lineartifacts_filter_;

    bool initialized_ = false;

    Internal::ResultComposeContext BuildComposeContext() const {
        Internal::ResultComposeContext context;
        context.score_threshold = score_threshold_;
        context.dark_clusters_threshold = dark_clusters_threshold_;
        context.abnormal_filter = abnormal_filter_;
        context.stain_filter = stain_filter_;
        context.darkclusters_filter = darkclusters_filter_;
        context.brightstripes_filter = brightstripes_filter_;
        context.lineartifacts_filter = lineartifacts_filter_;
        return context;
    }
};

InspectionEngine::InspectionEngine() : pImpl(std::make_unique<Impl>()) {}

InspectionEngine::~InspectionEngine() {
    Release();
}

bool InspectionEngine::Initialize(const std::string& config_path) {
    InspectionConfig::InspectionConfigData config;
    std::string err;
    if (!InspectionConfig::LoadInspectionConfig(config_path, config, err)) {
        std::cerr << "[DLL] " << err << std::endl;
        return false;
    }

    pImpl->score_threshold_ = config.patchcore_score_threshold;
    pImpl->area_threshold_ = config.patchcore_area_threshold;
    pImpl->mask_area_threshold_ = config.patchcore_mask_area_threshold;
    pImpl->dark_clusters_threshold_ = config.dark_clusters_threshold;
    pImpl->draw_box_details_ = config.draw_box_details;

    pImpl->abnormal_filter_ = config.abnormal_filter;
    pImpl->stain_filter_ = config.stain_filter;
    pImpl->darkclusters_filter_ = config.darkclusters_filter;
    pImpl->brightstripes_filter_ = config.brightstripes_filter;
    pImpl->lineartifacts_filter_ = config.lineartifacts_filter;

    pImpl->yolo_detector.SetNmsMode(
        config.yolo_nms_class_aware ? YOLO::YOLOv8Segmentor::NmsMode::ClassAware : YOLO::YOLOv8Segmentor::NmsMode::Global
    );

    if (!pImpl->yolo_detector.Initialize(config.yolo_model_path, config.yolo_score_threshold, config.yolo_iou_threshold, {640, 640}, config.ort_intra_threads)) {
        std::cerr << "[DLL] Failed to initialize YOLO detector" << std::endl;
        return false;
    }

    if (!pImpl->patchcore_detector.Initialize(config.patchcore_model_path, config.faiss_index_path, config.metadata_path, config.ort_intra_threads)) {
        std::cerr << "[DLL] Failed to initialize PatchCore detector" << std::endl;
        return false;
    }

    pImpl->initialized_ = true;
    return true;
}

bool InspectionEngine::ProcessImage(const cv::Mat& input_image, InferenceResult& output) {
    if (!pImpl->initialized_) {
        std::cerr << "[DLL] Engine not initialized" << std::endl;
        return false;
    }

    if (input_image.empty()) {
        std::cerr << "[DLL] Input image is empty" << std::endl;
        return false;
    }

    try {
        cv::Mat img_patchcore = Internal::ProcessTIF32ForPatchcore(input_image);
        if (img_patchcore.empty()) {
            std::cerr << "[DLL] Image preprocessing failed(seg)" << std::endl;
            return false;
        }

        cv::Mat img_yolo = Internal::ProcessForYolo(input_image);
        if (img_yolo.empty()) {
            std::cerr << "[DLL] Image preprocessing failed(cls)" << std::endl;
            return false;
        }

        const cv::Size img_shape = img_patchcore.size();

        std::vector<YOLO::Detection> yolo_detections;
        const bool yolo_success = pImpl->yolo_detector.Infer(img_yolo, yolo_detections);

        PatchCore::PatchCoreResult patchcore_result;
        const bool patchcore_success = pImpl->patchcore_detector.Infer(img_patchcore, patchcore_result);

        if (!yolo_success && !patchcore_success) {
            std::cerr << "[DLL] Both models failed" << std::endl;
            return false;
        }

        cv::Mat gray_patchcore;
        cv::Mat gray_yolo;
        cv::cvtColor(img_patchcore, gray_patchcore, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img_yolo, gray_yolo, cv::COLOR_BGR2GRAY);

        Internal::PatchCoreDerived pc = Internal::AnalyzePatchCore(
            patchcore_result,
            img_shape,
            gray_patchcore,
            pImpl->score_threshold_,
            pImpl->area_threshold_,
            pImpl->mask_area_threshold_);
        Internal::YoloDerived yolo = Internal::AnalyzeYolo(
            yolo_detections,
            gray_yolo,
            pImpl->dark_clusters_threshold_);

        Internal::ComposeOutputWithDefectFilter(
            pc,
            yolo,
            img_shape,
            gray_patchcore,
            gray_yolo,
            output,
            pImpl->BuildComposeContext());
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[DLL] Processing error: " << e.what() << std::endl;
        return false;
    }
}

bool InspectionEngine::ProcessImagePath(const std::string& image_path, InferenceResult& output) {
    cv::Mat img = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "[DLL] Failed to read image: " << image_path << std::endl;
        return false;
    }

    return ProcessImage(img, output);
}

bool InspectionEngine::ProcessFloatArry(const float* image_arry, InferenceResult& output, int width, int height) {
    if (image_arry == nullptr || width <= 0 || height <= 0) {
        return false;
    }

    cv::Mat ori_img(height, width, CV_32FC1, const_cast<float*>(image_arry));
    if (!ProcessImage(ori_img, output)) {
        return false;
    }

    return true;
}

void InspectionEngine::SetThresholds(float score_thresh, float area_thresh, float mask_area_thresh) {
    pImpl->score_threshold_ = score_thresh;
    pImpl->area_threshold_ = area_thresh;
    pImpl->mask_area_threshold_ = mask_area_thresh;
}

void InspectionEngine::SetDarkClustersThreshold(float dark_clusters_thresh) {
    pImpl->dark_clusters_threshold_ = dark_clusters_thresh;
}

void InspectionEngine::SetYoloNmsMode(bool class_aware) {
    pImpl->yolo_detector.SetNmsMode(
        class_aware ? YOLO::YOLOv8Segmentor::NmsMode::ClassAware
                    : YOLO::YOLOv8Segmentor::NmsMode::Global
    );
}

bool InspectionEngine::ShouldDrawBoxDetails() const {
    return pImpl->draw_box_details_;
}

void InspectionEngine::Release() {
    pImpl->initialized_ = false;
}

}  // namespace InspectionDLL
