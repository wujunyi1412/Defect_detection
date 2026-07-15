#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"

namespace PatchCore {

struct PatchCoreResult {
    float image_score;
    cv::Mat patch_scores; // 2D map of anomaly scores
    struct MetaData {
        int target_size; // e.g., 224
        int new_w, new_h;
        int pad_left, pad_top;
    } meta;
};

class PatchCoreDetector {
public:
    PatchCoreDetector();
    ~PatchCoreDetector();

    bool Initialize(
        const std::string& onnx_model_path,
        const std::string& faiss_index_path,
        const std::string& metadata_path,
        int ort_intra_threads = 4
    );

    bool Infer(const cv::Mat& image, PatchCoreResult& result);

private:
    struct PoolPlan {
        std::string mode;
        int kernel = 0;
    };

    bool LoadMetadata(const std::string& metadata_path);
    void PreprocessToNCHW(const cv::Mat& img_bgr_like, std::vector<float>& nchw, PatchCoreResult::MetaData& meta) const;
    bool RunBackbone(const std::vector<float>& input_nchw, std::vector<Ort::Value>& outputs);
    bool ExtractEmbeddings(const std::vector<Ort::Value>& outputs, std::vector<float>& embeddings);
    std::vector<float> ComputeAnomalyScores(const std::vector<float>& embeddings, int n, int d) const;

    static void ResizeBilinear3D(const float* in, int in_h, int in_w, int depth, float* out, int out_h, int out_w);
    static float Sigmoid(float x);

    Ort::Env env_;
    Ort::Session* session_ = nullptr;
    Ort::RunOptions run_options_;
    
    struct FaissIndexDeleter {
        void operator()(void* p) const noexcept;
    };

    std::unique_ptr<void, FaissIndexDeleter> faiss_index_{nullptr};
    
    std::vector<std::string> input_names_storage_;
    std::vector<std::string> output_names_storage_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    
    // Model metadata
    int imagesize_ = 224;
    int patchsize_ = 3;
    int patchstride_ = 1;
    int pretrain_embed_dim_ = 0;
    int target_embed_dim_ = 0;
    int num_nn_ = 1;
    std::pair<int, int> ref_patch_shape_{0, 0};
    
    std::vector<std::string> layer_names_;
    std::vector<PoolPlan> layer_pool_;
    PoolPlan agg_pool_;

    static constexpr float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float IMAGENET_STD[3] = {0.229f, 0.224f, 0.225f};
};

} // namespace PatchCore
