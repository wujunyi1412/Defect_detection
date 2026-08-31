#include "inference_dll.h"

#include <exception>
#include <future>
#include <memory>

#include <opencv2/opencv.hpp>

#include "config.h"
#include "logger.h"
#include "patchcore_inference.h"
#include "performance_timer.h"
#include "yolo_inference.h"
#include "internal/image_utils.h"
#include "internal/patchcore_postprocess.h"
#include "internal/result_builder.h"
#include "internal/yolo_postprocess.h"

namespace InspectionDLL {

namespace {

struct ModelRunResult {
    bool success = false;
    double elapsed_ms = 0.0;
};

}  // namespace

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
    float patchcore_yolo_iou_threshold_ = 0.0f;
    bool draw_defect_box_ = true;
    bool expand_defect_box_ = false;
    bool draw_box_details_ = true;
    bool concat_original_image_ = false;
    bool yolo_enabled_ = true;
    bool patchcore_enabled_ = true;
    MaskOverlayOptions mask_overlay_options_;
    std::string last_error_;

    AbnormalFilter abnormal_filter_;
    CategoryFilter stain_filter_;
    CategoryFilter brightstripes_filter_;
    CategoryFilter lineartifacts_filter_;
    InspectionConfig::AaFilterConfig aa_filter_;

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
        context.patchcore_yolo_iou_threshold = patchcore_yolo_iou_threshold_;
        context.abnormal_filter = abnormal_filter_;
        context.stain_filter = stain_filter_;
        context.brightstripes_filter = brightstripes_filter_;
        context.lineartifacts_filter = lineartifacts_filter_;
        context.aa_filter = aa_filter_;
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
    pImpl->initialized_ = false;
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
    pImpl->patchcore_yolo_iou_threshold_ = config.patchcore_yolo_iou_threshold;
    pImpl->draw_defect_box_ = config.draw_defect_box;
    pImpl->expand_defect_box_ = config.expand_defect_box;
    pImpl->draw_box_details_ = config.draw_box_details;
    pImpl->concat_original_image_ = config.concat_original_image;
    pImpl->yolo_enabled_ = config.yolo_enabled;
    pImpl->patchcore_enabled_ = config.patchcore_enabled;
    pImpl->mask_overlay_options_.enabled = config.draw_defect_mask;
    pImpl->mask_overlay_options_.color_r = config.defect_mask_color_r;
    pImpl->mask_overlay_options_.color_g = config.defect_mask_color_g;
    pImpl->mask_overlay_options_.color_b = config.defect_mask_color_b;
    pImpl->mask_overlay_options_.alpha = config.defect_mask_alpha;

    pImpl->abnormal_filter_ = config.abnormal_filter;
    pImpl->stain_filter_ = config.stain_filter;
    pImpl->brightstripes_filter_ = config.brightstripes_filter;
    pImpl->lineartifacts_filter_ = config.lineartifacts_filter;
    pImpl->aa_filter_ = config.aa_filter;

    if (pImpl->yolo_enabled_) {
        pImpl->yolo_detector.SetNmsMode(
            config.yolo_nms_class_aware ? YOLO::YOLOv8Segmentor::NmsMode::ClassAware : YOLO::YOLOv8Segmentor::NmsMode::Global
        );

        if (!pImpl->yolo_detector.Initialize(config.yolo_model_path, config.yolo_score_threshold, config.yolo_iou_threshold, {640, 640}, config.ort_intra_threads)) {
            pImpl->SetLastError("failed to initialize YOLO detector");
            return false;
        }
    }

    if (pImpl->patchcore_enabled_) {
        if (!pImpl->patchcore_detector.Initialize(config.patchcore_model_path, config.faiss_index_path, config.metadata_path, config.ort_intra_threads)) {
            pImpl->SetLastError("failed to initialize PatchCore detector");
            return false;
        }
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

    if (!pImpl->yolo_enabled_ && !pImpl->patchcore_enabled_) {
        output.result = "OK";
        output.details.clear();
        output.yolo_score = 0.0f;
        output.patchcore_score = 0.0f;
        output.patchcore_area_ratio = 0.0f;
        InspectionLogging::LogMessage(InspectionLogging::LogLevel::Debug, "image processing skipped because all models are disabled");
        return true;
    }

    try {
        const bool timing_enabled =
            InspectionLogging::IsLogEnabled(InspectionLogging::LogLevel::Debug);
        InspectionProfiling::PerformanceTimer total_timer(timing_enabled);
        InspectionProfiling::InferenceTiming timing;

        Internal::InferenceImages images = Internal::PrepareInferenceImages(input_image);
        timing.preprocess_ms = total_timer.ElapsedMilliseconds();
        if (pImpl->patchcore_enabled_ && (images.patchcore_bgr.empty() || images.patchcore_gray.empty())) {
            pImpl->SetLastError("image preprocessing failed for patchcore");
            return false;
        }

        if (pImpl->yolo_enabled_ && (images.yolo_bgr.empty() || images.yolo_gray.empty())) {
            pImpl->SetLastError("image preprocessing failed for yolo");
            return false;
        }

        const cv::Size img_shape = pImpl->patchcore_enabled_
            ? images.patchcore_bgr.size()
            : images.yolo_bgr.size();

        InspectionProfiling::PerformanceTimer models_timer(timing_enabled);
        std::vector<YOLO::Detection> yolo_detections;
        std::future<ModelRunResult> yolo_future;
        if (pImpl->yolo_enabled_) {
            yolo_future = std::async(
                std::launch::async,
                [&]() {
                    InspectionProfiling::PerformanceTimer timer(timing_enabled);
                    ModelRunResult run;
                    run.success = pImpl->yolo_detector.Infer(images.yolo_bgr, yolo_detections);
                    run.elapsed_ms = timer.ElapsedMilliseconds();
                    return run;
                });
        }

        PatchCore::PatchCoreResult patchcore_result;
        bool patchcore_success = false;
        if (pImpl->patchcore_enabled_) {
            InspectionProfiling::PerformanceTimer patchcore_timer(timing_enabled);
            patchcore_success = pImpl->patchcore_detector.Infer(images.patchcore_bgr, patchcore_result);
            timing.patchcore_infer_ms = patchcore_timer.ElapsedMilliseconds();
        }
        const ModelRunResult yolo_run = pImpl->yolo_enabled_ ? yolo_future.get() : ModelRunResult{};
        const bool yolo_success = yolo_run.success;
        timing.yolo_infer_ms = yolo_run.elapsed_ms;
        timing.models_wall_ms = models_timer.ElapsedMilliseconds();

        if (!yolo_success && !patchcore_success) {
            pImpl->SetLastError("all enabled models failed");
            return false;
        }

        InspectionProfiling::PerformanceTimer postprocess_timer(timing_enabled);
        Internal::PatchCoreDerived pc;
        if (patchcore_success) {
            pc = Internal::AnalyzePatchCore(
                patchcore_result,
                img_shape,
                images.patchcore_gray,
                pImpl->score_threshold_,
                pImpl->area_threshold_,
                pImpl->mask_area_threshold_);
        }
        timing.patchcore_post_ms = postprocess_timer.RestartMilliseconds();
        Internal::YoloDerived yolo;
        if (yolo_success) {
            yolo = Internal::AnalyzeYolo(yolo_detections, images.yolo_gray);
        }
        timing.yolo_post_ms = postprocess_timer.RestartMilliseconds();

        Internal::ComposeOutputWithDefectFilter(
            pc,
            yolo,
            img_shape,
            images.patchcore_gray,
            images.yolo_gray,
            output,
            pImpl->BuildComposeContext());
        timing.compose_ms = postprocess_timer.ElapsedMilliseconds();
        timing.total_ms = total_timer.ElapsedMilliseconds();
        if (timing_enabled) {
            InspectionLogging::LogMessage(
                InspectionLogging::LogLevel::Debug,
                timing.ToLogString());
        }
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

bool InspectionEngine::ShouldExpandDefectBox() const {
    return pImpl->expand_defect_box_;
}

bool InspectionEngine::ShouldConcatOriginalImage() const {
    return pImpl->concat_original_image_;
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
