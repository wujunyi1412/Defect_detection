#include "yolo_inference.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include "image_process.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace YOLO {

struct BoxF {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

static float IoU(const BoxF& a, const BoxF& b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float inter_w = std::max(0.0f, x2 - x1);
    const float inter_h = std::max(0.0f, y2 - y1);
    const float inter = inter_w * inter_h;
    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float uni = area_a + area_b - inter;
    if (uni <= 0.0f) return 0.0f;
    return inter / uni;
}


static void NMSBoxes(
    const std::vector<BoxF>& boxes,
    const std::vector<float>& scores,
    float score_threshold,
    float iou_threshold,
    std::vector<int>& indices)
{
    indices.clear();

    const int n = static_cast<int>(boxes.size());

    if (n == 0 || scores.size() != boxes.size())
    {
        return;
    }

    std::vector<int> order;
    order.reserve(n);

    for (int i = 0; i < n; ++i)
    {
        if (scores[i] < score_threshold)
            continue;
        order.push_back(i);
    }

    std::sort(order.begin(),
              order.end(),
              [&](int a, int b)
              {
                  return scores[a] > scores[b];
              });

    indices.reserve(order.size());
    for (int i : order)
    {
        bool keep = true;
        for (int j : indices)
        {
            if (IoU(boxes[i], boxes[j]) > iou_threshold)
            {
                keep = false;
                break;
            }
        }
        if (keep)
        {
            indices.push_back(i);
        }
    }
}

static void NMSBoxesByClass(
    const std::vector<BoxF>& boxes,
    const std::vector<float>& scores,
    const std::vector<int>& class_ids,
    int num_classes,
    float score_threshold,
    float iou_threshold,
    std::vector<int>& indices)
{
    indices.clear();

    const int n = static_cast<int>(boxes.size());
    if (n == 0 || num_classes <= 0 || scores.size() != boxes.size() || class_ids.size() != boxes.size())
    {
        return;
    }

    for (int cls = 0; cls < num_classes; ++cls)
    {
        std::vector<BoxF> cls_boxes;
        std::vector<float> cls_scores;
        std::vector<int> cls_map;
        cls_boxes.reserve(n);
        cls_scores.reserve(n);
        cls_map.reserve(n);

        for (int i = 0; i < n; ++i)
        {
            if (class_ids[i] != cls) continue;
            cls_boxes.push_back(boxes[i]);
            cls_scores.push_back(scores[i]);
            cls_map.push_back(i);
        }

        std::vector<int> cls_keep;
        NMSBoxes(cls_boxes, cls_scores, score_threshold, iou_threshold, cls_keep);
        for (int local_idx : cls_keep)
        {
            indices.push_back(cls_map[local_idx]);
        }
    }
}

YOLOv8Segmentor::YOLOv8Segmentor() {}

YOLOv8Segmentor::~YOLOv8Segmentor() {
    if (session_) {
        delete session_;
        session_ = nullptr;
    }
}

void YOLOv8Segmentor::SetNmsMode(NmsMode mode) {
    nms_mode_ = mode;
}

bool YOLOv8Segmentor::Initialize(
    const std::string& model_path,
    float score_threshold,
    float iou_threshold,
    const std::vector<int>& img_size,
    int ort_intra_threads) {
    
    score_threshold_ = score_threshold;
    iou_threshold_ = iou_threshold;
    img_size_ = img_size;

    try {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8-Seg");
        
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.SetIntraOpNumThreads(ort_intra_threads);

#ifdef _WIN32
        int path_size = MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, nullptr, 0);
        wchar_t* wide_path = new wchar_t[path_size];
        MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, wide_path, path_size);
        const wchar_t* model_path_w = wide_path;
#else
        const char* model_path_w = model_path.c_str();
#endif

        session_ = new Ort::Session(env_, model_path_w, session_options);

#ifdef _WIN32
        delete[] wide_path;
#endif
        
        Ort::AllocatorWithDefaultOptions allocator;
        
        size_t num_inputs = session_->GetInputCount();
        input_names_storage_.clear();
        input_names_.clear();
        for (size_t i = 0; i < num_inputs; i++) {
            auto name_ptr = session_->GetInputNameAllocated(i, allocator);
            input_names_storage_.push_back(std::string(name_ptr.get()));
        }
        for (auto& s : input_names_storage_) input_names_.push_back(s.c_str());

        size_t num_outputs = session_->GetOutputCount();
        output_names_storage_.clear();
        output_names_.clear();
        for (size_t i = 0; i < num_outputs; i++) {
            auto name_ptr = session_->GetOutputNameAllocated(i, allocator);
            output_names_storage_.push_back(std::string(name_ptr.get()));
        }
        for (auto& s : output_names_storage_) output_names_.push_back(s.c_str());

        run_options_ = Ort::RunOptions{nullptr};

        // Warm up
        cv::Mat warmup_img(img_size_[0], img_size_[1], CV_8UC3, cv::Scalar(114, 114, 114));
        std::vector<Detection> dummy;
        Infer(warmup_img, dummy);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[YOLO] Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat YOLOv8Segmentor::Preprocess(const cv::Mat& img, float& ratio, std::vector<float>& pad) {
    cv::Mat img_3 = ImageProcess::ImageProcessor::Cvmat2RGB(img);
    cv::Mat img_letterbox = ImageProcess::ImageProcessor::LetterboxResize(img_3, img_size_[0], img_size_[1], ratio, pad, cv::Scalar(114, 114, 114));
    return img_letterbox;
}

void YOLOv8Segmentor::Postprocess(
    const std::vector<Ort::Value>& outputs,
    const cv::Mat& original_img,
    float ratio,
    const std::vector<float>& pad,
    std::vector<Detection>& detections) {

    if (outputs.size() < 2) return;

    const Ort::Value* det_tensor = &outputs[0];
    const Ort::Value* proto_tensor = &outputs[1];

    {
        auto tsi0 = outputs[0].GetTensorTypeAndShapeInfo();
        auto d0 = tsi0.GetShape();
        auto tsi1 = outputs[1].GetTensorTypeAndShapeInfo();
        auto d1 = tsi1.GetShape();
        if (d0.size() == 4 && d1.size() != 4) {
            det_tensor = &outputs[1];
            proto_tensor = &outputs[0];
        }
    }

    auto det_info = det_tensor->GetTensorTypeAndShapeInfo();
    auto det_dims = det_info.GetShape();

    auto proto_info = proto_tensor->GetTensorTypeAndShapeInfo();
    auto proto_dims = proto_info.GetShape();
    if (proto_dims.size() != 4) return;

    const int num_masks = static_cast<int>(proto_dims[1]);
    const int mask_h = static_cast<int>(proto_dims[2]);
    const int mask_w = static_cast<int>(proto_dims[3]);
    const float* proto_data = proto_tensor->GetTensorData<float>();

    const int inp_h = img_size_[0];
    const int inp_w = img_size_[1];
    const int expected_feat = 4 + NUM_CLASSES + num_masks;

    bool transposed = false;
    int num_detections = 0;
    int features_per_det = 0;
    if (det_dims.size() == 3) {
        if (static_cast<int>(det_dims[2]) == expected_feat) {
            transposed = false;
            num_detections = static_cast<int>(det_dims[1]);
            features_per_det = static_cast<int>(det_dims[2]);
        } else if (static_cast<int>(det_dims[1]) == expected_feat) {
            transposed = true;
            num_detections = static_cast<int>(det_dims[2]);
            features_per_det = static_cast<int>(det_dims[1]);
        } else {
            transposed = false;
            num_detections = static_cast<int>(det_dims[1]);
            features_per_det = static_cast<int>(det_dims[2]);
        }
    } else if (det_dims.size() == 2) {
        if (static_cast<int>(det_dims[1]) == expected_feat) {
            transposed = false;
            num_detections = static_cast<int>(det_dims[0]);
            features_per_det = static_cast<int>(det_dims[1]);
        } else if (static_cast<int>(det_dims[0]) == expected_feat) {
            transposed = true;
            num_detections = static_cast<int>(det_dims[1]);
            features_per_det = static_cast<int>(det_dims[0]);
        } else {
            transposed = false;
            num_detections = static_cast<int>(det_dims[0]);
            features_per_det = static_cast<int>(det_dims[1]);
        }
    } else {
        return;
    }

    const float* output_data0 = det_tensor->GetTensorData<float>();

    auto get_feat = [&](int det_i, int feat_j) -> float {
        if (!transposed) {
            return output_data0[static_cast<size_t>(det_i) * features_per_det + feat_j];
        }
        return output_data0[static_cast<size_t>(feat_j) * num_detections + det_i];
    };

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<BoxF> boxes_orig;
    std::vector<BoxF> boxes_letterbox;
    std::vector<std::vector<float>> mask_coeffs;
    //detection[1, 39, 8400]
    //proto[1, 32, 160, 160]

    // Parse detections
    for (int i = 0; i < num_detections; ++i) {
        float cx = get_feat(i, 0);
        float cy = get_feat(i, 1);
        float w = get_feat(i, 2);
        float h = get_feat(i, 3);

        // Get max class score
        float max_score = 0.0f;
        int best_class = 0;
        for (int c = 0; c < NUM_CLASSES; c++) {
            const float v = get_feat(i, 4 + c);
            if (v > max_score) {
                max_score = v;
                best_class = c;
            }
        }

        if (max_score > score_threshold_) {
            // Convert from center format to corner format
            float x1 = cx - w / 2.0f;
            float y1 = cy - h / 2.0f;
            float x2 = cx + w / 2.0f;
            float y2 = cy + h / 2.0f;
            boxes_letterbox.push_back(BoxF{x1, y1, x2, y2});

            x1 = (x1 - pad[0]) / ratio;
            y1 = (y1 - pad[1]) / ratio;
            x2 = (x2 - pad[0]) / ratio;
            y2 = (y2 - pad[1]) / ratio;

            // Clip to image bounds
            x1 = std::max(0.0f, std::min(x1, (float)original_img.cols));
            y1 = std::max(0.0f, std::min(y1, (float)original_img.rows));
            x2 = std::max(0.0f, std::min(x2, (float)original_img.cols));
            y2 = std::max(0.0f, std::min(y2, (float)original_img.rows));

            confidences.push_back(max_score);
            class_ids.push_back(best_class);
            boxes_orig.push_back(BoxF{x1, y1, x2, y2});

            // Extract mask coefficients
            std::vector<float> coeffs;
            for (int m = 0; m < num_masks; m++) {
                coeffs.push_back(get_feat(i, 4 + NUM_CLASSES + m));
            }
            mask_coeffs.push_back(coeffs);
        }
    }

    // NMS
    std::vector<int> indices;
    if (nms_mode_ == NmsMode::ClassAware) {
        NMSBoxesByClass(
            boxes_orig,
            confidences,
            class_ids,
            NUM_CLASSES,
            score_threshold_,
            iou_threshold_,
            indices);
    } else {
        NMSBoxes(boxes_orig, confidences, score_threshold_, iou_threshold_, indices);
    }

    // Process masks for kept detections
    cv::Mat proto(num_masks, mask_h * mask_w, CV_32F, const_cast<float*>(proto_data));

    const int pad_x = static_cast<int>(pad.size() > 0 ? pad[0] : 0.0f);
    const int pad_y = static_cast<int>(pad.size() > 1 ? pad[1] : 0.0f);

    for (int idx : indices) {
        Detection det;
        det.class_id = class_ids[idx];
        det.confidence = confidences[idx];
        det.x1 = boxes_orig[idx].x1;
        det.y1 = boxes_orig[idx].y1;
        det.x2 = boxes_orig[idx].x2;
        det.y2 = boxes_orig[idx].y2;

        cv::Mat coeff_mat(1, num_masks, CV_32F, mask_coeffs[idx].data());
        cv::Mat mask_mat = (coeff_mat * proto);
        mask_mat = mask_mat.reshape(1, mask_h);

        // Apply sigmoid
        cv::exp(-mask_mat, mask_mat);
        mask_mat = 1.0 / (1.0 + mask_mat);

        const BoxF& bl = boxes_letterbox[idx];
        const float sx = static_cast<float>(mask_w) / static_cast<float>(inp_w);
        const float sy = static_cast<float>(mask_h) / static_cast<float>(inp_h);
        int x1m = static_cast<int>(std::ceil(static_cast<double>(bl.x1) * sx));
        int y1m = static_cast<int>(std::ceil(static_cast<double>(bl.y1) * sy));
        int x2m = static_cast<int>(std::ceil(static_cast<double>(bl.x2) * sx));
        int y2m = static_cast<int>(std::ceil(static_cast<double>(bl.y2) * sy));
        x1m = std::max(0, std::min(x1m, mask_w));
        y1m = std::max(0, std::min(y1m, mask_h));
        x2m = std::max(x1m + 1, std::min(x2m, mask_w));
        y2m = std::max(y1m + 1, std::min(y2m, mask_h));

        cv::Rect roi_mask(x1m, y1m, x2m - x1m, y2m - y1m);
        cv::Mat mask_cropped = cv::Mat::zeros(mask_h, mask_w, CV_32F);
        mask_mat(roi_mask).copyTo(mask_cropped(roi_mask));

        cv::Mat mask_640;
        cv::resize(mask_cropped, mask_640, cv::Size(inp_w, inp_h), 0, 0, cv::INTER_LINEAR);

        int x0 = pad_x;
        int y0 = pad_y;
        int x1 = inp_w - pad_x;
        int y1 = inp_h - pad_y;
        x0 = std::max(0, std::min(x0, std::max(0, inp_w - 1)));
        y0 = std::max(0, std::min(y0, std::max(0, inp_h - 1)));
        x1 = std::max(x0 + 1, std::min(x1, inp_w));
        y1 = std::max(y0 + 1, std::min(y1, inp_h));
        cv::Rect roi_unpad(x0, y0, x1 - x0, y1 - y0);
        roi_unpad &= cv::Rect(0, 0, inp_w, inp_h);
        cv::Mat mask_unpad = mask_640(roi_unpad);

        cv::Mat mask_orig_f;
        cv::resize(mask_unpad, mask_orig_f, original_img.size(), 0, 0, cv::INTER_LINEAR);
        det.mask = mask_orig_f > 0.25f;
        detections.push_back(det);
    }
}

bool YOLOv8Segmentor::Infer(const cv::Mat& image,
                            std::vector<Detection>& detections) {
    if (!session_) {
        std::cerr << "[YOLO] Session not initialized" << std::endl;
        return false;
    }

    try {
        float ratio;
        std::vector<float> pad;
        cv::Mat processed = Preprocess(image, ratio, pad);

        // Convert to blob
        int channels = processed.channels();
        int height = processed.rows;
        int width = processed.cols;
        
        std::vector<float> blob(channels * height * width);

        blob = ImageProcess::ImageProcessor::HWCToCHW_normalize(processed, 1.0f / 255.0f);

        std::vector<int64_t> input_dims = {1, 3, img_size_[0], img_size_[1]};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
            blob.data(), blob.size(),
            input_dims.data(), input_dims.size()
        );

        // Run inference
        auto output_tensors = session_->Run(
            run_options_,
            input_names_.data(), &input_tensor, 1,
            output_names_.data(), output_names_.size()
        );

        // Post-process
        Postprocess(output_tensors, image, ratio, pad, detections);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[YOLO] Inference error: " << e.what() << std::endl;
        return false;
    }
}

} // namespace YOLO
