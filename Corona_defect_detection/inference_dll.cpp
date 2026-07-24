#include "inference_dll.h"

#include <exception>
#include <memory>

#include <opencv2/opencv.hpp>

#include "config.h"
#include "logger.h"
#include "patchcore_inference.h"
#include "yolo_inference.h"
#include "internal/image_utils.h"
#include "internal/patchcore_postprocess.h"
#include "internal/result_builder.h"
#include "internal/yolo_postprocess.h"

namespace InspectionDLL {

// InspectionEngine 实现类
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
    bool draw_defect_box_ = true;
    bool draw_box_details_ = true;
    MaskOverlayOptions mask_overlay_options_;
    std::string last_error_;

    AbnormalFilter abnormal_filter_;
    CategoryFilter stain_filter_;
    CategoryFilter darkclusters_filter_;
    CategoryFilter brightstripes_filter_;
    CategoryFilter lineartifacts_filter_;

    bool initialized_ = false;

    void SetLastError(const std::string& message, InspectionLogging::LogLevel level = InspectionLogging::LogLevel::Error) {
        last_error_ = message;
        InspectionLogging::LogMessage(level, message);
    }

    void ClearLastError() {
        last_error_.clear();
    }

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

// InspectionEngine 构造函数
InspectionEngine::InspectionEngine() : pImpl(std::make_unique<Impl>()) {}

// InspectionEngine 析构函数
InspectionEngine::~InspectionEngine() {
    Release();
}

// InspectionEngine 初始化函数
bool InspectionEngine::Initialize(const std::string& config_path) {
    pImpl->ClearLastError();
    InspectionConfig::InspectionConfigData config;
    std::string err;
    if (!InspectionConfig::LoadInspectionConfig(config_path, config, err)) {
        pImpl->SetLastError("config load failed: " + err);
        return false;
    }

    InspectionLogging::LoggerConfig logger_config;
    logger_config.enabled = config.log_enabled;
    logger_config.log_to_stderr = config.log_to_stderr;
    logger_config.log_to_file = config.log_to_file;
    logger_config.file_path = config.log_file_path;
    if (!InspectionLogging::TryParseLogLevel(config.log_level, logger_config.min_level)) {
        pImpl->SetLastError("invalid log level after config parse: " + config.log_level);
        return false;
    }
    InspectionLogging::SetLoggerConfig(logger_config);
    InspectionLogging::LogMessage(InspectionLogging::LogLevel::Info, "initializing inspection engine with config: " + config_path);

    pImpl->score_threshold_ = config.patchcore_score_threshold;
    pImpl->area_threshold_ = config.patchcore_area_threshold;
    pImpl->mask_area_threshold_ = config.patchcore_mask_area_threshold;
    pImpl->dark_clusters_threshold_ = config.dark_clusters_threshold;
    pImpl->draw_defect_box_ = config.draw_defect_box;
    pImpl->draw_box_details_ = config.draw_box_details;
    pImpl->mask_overlay_options_.enabled = config.draw_defect_mask;
    pImpl->mask_overlay_options_.color_r = config.defect_mask_color_r;
    pImpl->mask_overlay_options_.color_g = config.defect_mask_color_g;
    pImpl->mask_overlay_options_.color_b = config.defect_mask_color_b;
    pImpl->mask_overlay_options_.alpha = config.defect_mask_alpha;

    pImpl->abnormal_filter_ = config.abnormal_filter;
    pImpl->stain_filter_ = config.stain_filter;
    pImpl->darkclusters_filter_ = config.darkclusters_filter;
    pImpl->brightstripes_filter_ = config.brightstripes_filter;
    pImpl->lineartifacts_filter_ = config.lineartifacts_filter;

    pImpl->yolo_detector.SetNmsMode(
        config.yolo_nms_class_aware ? YOLO::YOLOv8Segmentor::NmsMode::ClassAware : YOLO::YOLOv8Segmentor::NmsMode::Global
    );

    if (!pImpl->yolo_detector.Initialize(config.yolo_model_path, config.yolo_score_threshold, config.yolo_iou_threshold, {640, 640}, config.ort_intra_threads)) {
        pImpl->SetLastError("failed to initialize YOLO detector");
        return false;
    }

    if (!pImpl->patchcore_detector.Initialize(config.patchcore_model_path, config.faiss_index_path, config.metadata_path, config.ort_intra_threads)) {
        pImpl->SetLastError("failed to initialize PatchCore detector");
        return false;
    }

    pImpl->initialized_ = true;
    InspectionLogging::LogMessage(InspectionLogging::LogLevel::Info, "inspection engine initialized successfully");
    return true;
}

// InspectionEngine 推理函数
bool InspectionEngine::ProcessImage(const cv::Mat& input_image, InferenceResult& output) {
    pImpl->ClearLastError();
    if (!pImpl->initialized_) {
        pImpl->SetLastError("engine not initialized");
        return false;
    }

    if (input_image.empty()) {
        pImpl->SetLastError("input image is empty");
        return false;
    }

    try {
        Internal::InferenceImages images = Internal::PrepareInferenceImages(input_image);
        if (images.patchcore_bgr.empty() || images.patchcore_gray.empty()) {
            pImpl->SetLastError("image preprocessing failed for patchcore");
            return false;
        }

        if (images.yolo_bgr.empty() || images.yolo_gray.empty()) {
            pImpl->SetLastError("image preprocessing failed for yolo");
            return false;
        }

        const cv::Size img_shape = images.patchcore_bgr.size();

        std::vector<YOLO::Detection> yolo_detections;
        const bool yolo_success = pImpl->yolo_detector.Infer(images.yolo_bgr, yolo_detections);

        PatchCore::PatchCoreResult patchcore_result;
        const bool patchcore_success = pImpl->patchcore_detector.Infer(images.patchcore_bgr, patchcore_result);

        if (!yolo_success && !patchcore_success) {
            pImpl->SetLastError("both models failed");
            return false;
        }

        Internal::PatchCoreDerived pc = Internal::AnalyzePatchCore(
            patchcore_result,
            img_shape,
            images.patchcore_gray,
            pImpl->score_threshold_,
            pImpl->area_threshold_,
            pImpl->mask_area_threshold_);
        Internal::YoloDerived yolo = Internal::AnalyzeYolo(
            yolo_detections,
            images.yolo_gray,
            pImpl->dark_clusters_threshold_);

        Internal::ComposeOutputWithDefectFilter(
            pc,
            yolo,
            img_shape,
            images.patchcore_gray,
            images.yolo_gray,
            output,
            pImpl->BuildComposeContext());
        InspectionLogging::LogMessage(InspectionLogging::LogLevel::Debug, "image processed successfully");
        return true;
    } catch (const std::exception& e) {
        pImpl->SetLastError(std::string("processing error: ") + e.what());
        return false;
    } catch (...) {
        pImpl->SetLastError("processing error: unknown exception");
        return false;
    }
}

// InspectionEngine 推理函数（图像路径）
bool InspectionEngine::ProcessImagePath(const std::string& image_path, InferenceResult& output) {
    pImpl->ClearLastError();
    cv::Mat img = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        pImpl->SetLastError("failed to read image: " + image_path);
        return false;
    }

    return ProcessImage(img, output);
}

// 处理单张图像数组
bool InspectionEngine::ProcessFloatArry(const float* image_arry, InferenceResult& output, int width, int height) {
    pImpl->ClearLastError();
    if (image_arry == nullptr || width <= 0 || height <= 0) {
        pImpl->SetLastError("invalid input array or dimensions");
        return false;
    }

    cv::Mat ori_img(height, width, CV_32FC1, const_cast<float*>(image_arry));
    if (!ProcessImage(ori_img, output)) {
        return false;
    }

    return true;
}

// 设置运行时阈值
void InspectionEngine::SetThresholds(float score_thresh, float area_thresh, float mask_area_thresh) {
    pImpl->score_threshold_ = score_thresh;
    pImpl->area_threshold_ = area_thresh;
    pImpl->mask_area_threshold_ = mask_area_thresh;
    InspectionLogging::LogMessage(InspectionLogging::LogLevel::Info, "runtime thresholds updated");
}

// 设置暗区聚类阈值
void InspectionEngine::SetDarkClustersThreshold(float dark_clusters_thresh) {
    pImpl->dark_clusters_threshold_ = dark_clusters_thresh;
    InspectionLogging::LogMessage(InspectionLogging::LogLevel::Info, "runtime dark_clusters_threshold updated");
}

// 设置YOLO NMS模式（是否类别感知）
void InspectionEngine::SetYoloNmsMode(bool class_aware) {
    pImpl->yolo_detector.SetNmsMode(
        class_aware ? YOLO::YOLOv8Segmentor::NmsMode::ClassAware
                    : YOLO::YOLOv8Segmentor::NmsMode::Global
    );
    InspectionLogging::LogMessage(InspectionLogging::LogLevel::Info, "runtime yolo nms mode updated");
}

// 是否绘制框详情
bool InspectionEngine::ShouldDrawBoxDetails() const {
    return pImpl->draw_box_details_;
}

bool InspectionEngine::ShouldDrawDefectBox() const {
    return pImpl->draw_defect_box_;
}

MaskOverlayOptions InspectionEngine::GetMaskOverlayOptions() const {
    return pImpl->mask_overlay_options_;
}

// 获取最后一次错误信息
const std::string& InspectionEngine::GetLastError() const {
    return pImpl->last_error_;
}

// 释放 inspection 引擎资源
void InspectionEngine::Release() {
    pImpl->initialized_ = false;
    pImpl->ClearLastError();
    InspectionLogging::LogMessage(InspectionLogging::LogLevel::Info, "inspection engine released");
}

}  // namespace InspectionDLL
