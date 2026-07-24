#include "patchcore_inference.h"

#include <faiss/Index.h>
#include <faiss/index_io.h>
#include "image_process.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace PatchCore {

void PatchCoreDetector::FaissIndexDeleter::operator()(void* p) const noexcept {
    delete static_cast<faiss::Index*>(p);
}


static bool FileExists(const std::string& path) {
    if (path.empty()) return false;
#ifdef _WIN32
    const DWORD attrs = GetFileAttributesA(path.c_str());
    return attrs != INVALID_FILE_ATTRIBUTES && !(attrs & FILE_ATTRIBUTE_DIRECTORY);
#else
    std::ifstream f(path);
    return f.good();
#endif
}

static std::string ReplaceExt(const std::string& path, const std::string& ext_with_dot) {
    const auto pos = path.find_last_of('.');
    if (pos == std::string::npos) return path + ext_with_dot;
    return path.substr(0, pos) + ext_with_dot;
}

static bool EndsWithIcase(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) return false;
    const size_t off = s.size() - suffix.size();
    for (size_t i = 0; i < suffix.size(); ++i) {
        const unsigned char a = static_cast<unsigned char>(s[off + i]);
        const unsigned char b = static_cast<unsigned char>(suffix[i]);
        if (std::tolower(a) != std::tolower(b)) return false;
    }
    return true;
}

static std::string ReplaceJsonNullWithZero(const std::string& json) {
    std::string out;
    out.reserve(json.size());
    bool in_string = false;
    bool esc = false;
    for (size_t i = 0; i < json.size(); ++i) {
        const char ch = json[i];
        if (in_string) {
            out.push_back(ch);
            if (esc) {
                esc = false;
            } else if (ch == '\\') {
                esc = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }
        if (ch == '"') {
            in_string = true;
            out.push_back(ch);
            continue;
        }
        if (i + 3 < json.size() && json.compare(i, 4, "null") == 0) {
            const auto is_word = [](char c) -> bool {
                const unsigned char uc = static_cast<unsigned char>(c);
                return std::isalnum(uc) || c == '_';
            };
            const bool left_word = (i > 0) ? is_word(json[i - 1]) : false;
            const bool right_word = (i + 4 < json.size()) ? is_word(json[i + 4]) : false;
            if (!left_word && !right_word) {
                out.push_back('0');
                i += 3;
                continue;
            }
        }
        out.push_back(ch);
    }
    return out;
}

PatchCoreDetector::PatchCoreDetector() {}

PatchCoreDetector::~PatchCoreDetector() {
    if (session_) {
        delete session_;
        session_ = nullptr;
    }
}

float PatchCoreDetector::Sigmoid(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

void PatchCoreDetector::ResizeBilinear3D(
    const float* in, int in_h, int in_w, int depth, float* out, int out_h, int out_w) {
    if (in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0 || depth <= 0) return;
    if (in_h == out_h && in_w == out_w) {
        std::memcpy(out, in, sizeof(float) * static_cast<size_t>(out_h) * out_w * depth);
        return;
    }

    const float scale_y = static_cast<float>(in_h) / static_cast<float>(out_h);
    const float scale_x = static_cast<float>(in_w) / static_cast<float>(out_w);

    for (int oy = 0; oy < out_h; ++oy) {
        const float fy = (static_cast<float>(oy) + 0.5f) * scale_y - 0.5f;
        int y0 = static_cast<int>(std::floor(fy));
        int y1 = y0 + 1;
        const float wy = fy - static_cast<float>(y0);
        y0 = std::max(0, std::min(y0, in_h - 1));
        y1 = std::max(0, std::min(y1, in_h - 1));

        for (int ox = 0; ox < out_w; ++ox) {
            const float fx = (static_cast<float>(ox) + 0.5f) * scale_x - 0.5f;
            int x0 = static_cast<int>(std::floor(fx));
            int x1 = x0 + 1;
            const float wx = fx - static_cast<float>(x0);
            x0 = std::max(0, std::min(x0, in_w - 1));
            x1 = std::max(0, std::min(x1, in_w - 1));

            const float w00 = (1.0f - wy) * (1.0f - wx);
            const float w01 = (1.0f - wy) * wx;
            const float w10 = wy * (1.0f - wx);
            const float w11 = wy * wx;

            const size_t out_base = (static_cast<size_t>(oy) * out_w + ox) * depth;
            const size_t in00 = (static_cast<size_t>(y0) * in_w + x0) * depth;
            const size_t in01 = (static_cast<size_t>(y0) * in_w + x1) * depth;
            const size_t in10 = (static_cast<size_t>(y1) * in_w + x0) * depth;
            const size_t in11 = (static_cast<size_t>(y1) * in_w + x1) * depth;

            for (int d = 0; d < depth; ++d) {
                out[out_base + d] =
                    in[in00 + d] * w00 + in[in01 + d] * w01 + in[in10 + d] * w10 + in[in11 + d] * w11;
            }
        }
    }
}

bool PatchCoreDetector::LoadMetadata(const std::string& metadata_path) {
    std::string path = metadata_path;
    if (path.size() >= 4) {
        const std::string lower = path.substr(path.size() - 4);
        if (lower == ".pkl") {
            const std::string json_path = ReplaceExt(path, ".json");
            if (FileExists(json_path)) path = json_path;
        }
    }
    if (!FileExists(path)) {
        std::cerr << "[PatchCore] metadata not found: " << path << std::endl;
        return false;
    }

    cv::FileStorage fs;
    if (EndsWithIcase(path, ".json")) {
        std::ifstream ifs(path, std::ios::in | std::ios::binary);
        std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        const std::string sanitized = ReplaceJsonNullWithZero(content);
        fs.open(sanitized, cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
    } else {
        fs.open(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_AUTO);
    }
    if (!fs.isOpened()) {
        std::cerr << "[PatchCore] failed to open metadata: " << path << std::endl;
        return false;
    }

    cv::FileNode layers_node = fs["layers_to_extract_from"];
    if (layers_node.type() != cv::FileNode::SEQ) {
        std::cerr << "[PatchCore] metadata missing layers_to_extract_from" << std::endl;
        return false;
    }
    layer_names_.clear();
    for (auto it = layers_node.begin(); it != layers_node.end(); ++it) {
        layer_names_.push_back(static_cast<std::string>(*it));
    }
    if (layer_names_.empty()) {
        std::cerr << "[PatchCore] layers_to_extract_from is empty" << std::endl;
        return false;
    }

    num_nn_ = static_cast<int>(fs["anomaly_scorer_num_nn"]);
    if (num_nn_ <= 0) num_nn_ = 1;

    {
        cv::FileNode in_shape = fs["input_shape"];
        if (in_shape.type() == cv::FileNode::SEQ && in_shape.size() > 0) {
            imagesize_ = static_cast<int>(in_shape[in_shape.size() - 1]);
        } else {
            imagesize_ = static_cast<int>(in_shape);
        }
    }
    if (imagesize_ <= 0) imagesize_ = 224;

    patchsize_ = static_cast<int>(fs["patchsize"]);
    patchstride_ = static_cast<int>(fs["patchstride"]);
    if (patchsize_ <= 0) patchsize_ = 3;
    if (patchstride_ <= 0) patchstride_ = 1;

    pretrain_embed_dim_ = static_cast<int>(fs["pretrain_embed_dimension"]);
    target_embed_dim_ = static_cast<int>(fs["target_embed_dimension"]);

    cv::FileNode ref_ps = fs["ref_patch_shape"];
    if (ref_ps.type() == cv::FileNode::SEQ && ref_ps.size() >= 2) {
        ref_patch_shape_.first = static_cast<int>(ref_ps[0]);
        ref_patch_shape_.second = static_cast<int>(ref_ps[1]);
    } else {
        ref_patch_shape_ = {28, 28};
    }

    layer_pool_.clear();
    cv::FileNode layer_pool_node = fs["layer_pool"];
    if (layer_pool_node.type() == cv::FileNode::SEQ) {
        for (auto it = layer_pool_node.begin(); it != layer_pool_node.end(); ++it) {
            PoolPlan p;
            cv::FileNode mode_n = (*it)["mode"];
            p.mode = mode_n.empty() ? std::string() : static_cast<std::string>(mode_n);
            cv::FileNode k_n = (*it)["kernel"];
            p.kernel = k_n.empty() ? 0 : static_cast<int>(k_n);
            layer_pool_.push_back(p);
        }
    } else {
        cv::FileNode kernels = fs["layer_pool_kernels"];
        if (kernels.type() == cv::FileNode::SEQ) {
            for (auto it = kernels.begin(); it != kernels.end(); ++it) {
                PoolPlan p;
                p.mode = "avgpool";
                p.kernel = static_cast<int>(*it);
                layer_pool_.push_back(p);
            }
        }
    }

    if (static_cast<int>(layer_pool_.size()) != static_cast<int>(layer_names_.size())) {
        std::cerr << "[PatchCore] layer_pool length mismatch: " << layer_pool_.size() << " vs " << layer_names_.size()
                  << std::endl;
        return false;
    }

    cv::FileNode agg_pool_node = fs["agg_pool"];
    if (!agg_pool_node.empty()) {
        cv::FileNode mode_n = agg_pool_node["mode"];
        agg_pool_.mode = mode_n.empty() ? std::string() : static_cast<std::string>(mode_n);
        cv::FileNode k_n = agg_pool_node["kernel"];
        agg_pool_.kernel = k_n.empty() ? 0 : static_cast<int>(k_n);
    } else {
        agg_pool_.mode = "avgpool";
        agg_pool_.kernel = static_cast<int>(fs["agg_pool_kernel"]);
    }

    if (pretrain_embed_dim_ <= 0 || target_embed_dim_ <= 0) {
        std::cerr << "[PatchCore] invalid embed dims: pretrain=" << pretrain_embed_dim_
                  << " target=" << target_embed_dim_ << std::endl;
        return false;
    }

    return true;
}

void PatchCoreDetector::PreprocessToNCHW(
    const cv::Mat& img_bgr_like, std::vector<float>& nchw, PatchCoreResult::MetaData& meta) const {
    cv::Mat img_u8 = img_bgr_like.depth() == CV_8U
                         ? img_bgr_like
                         : ImageProcess::ImageProcessor::Cvmat2Uint8(img_bgr_like);
    cv::Mat img3 = ImageProcess::ImageProcessor::Cvmat2RGB(img_u8);

    int img_w = img3.cols;
    int img_h = img3.rows;
    if (img_w <= 0 || img_h <= 0) {
        nchw.clear();
        meta = PatchCoreResult::MetaData{};
        return;
    }

    int new_w = 0;
    int new_h = 0;
    if (img_w >= img_h) {
        new_w = imagesize_;
        new_h = static_cast<int>(std::floor(static_cast<double>(img_h) * imagesize_ / static_cast<double>(img_w)));
    } else {
        new_h = imagesize_;
        new_w = static_cast<int>(std::floor(static_cast<double>(img_w) * imagesize_ / static_cast<double>(img_h)));
    }
    new_w = std::max(1, std::min(imagesize_, new_w));
    new_h = std::max(1, std::min(imagesize_, new_h));

    cv::Mat resized;
    resized = ImageProcess::ImageProcessor::ResizeBilinearPillowResampleRGBU8(img3, new_w, new_h);
    if (resized.empty()) {
        cv::resize(img3, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    }

    const int left = (imagesize_ - new_w) / 2;
    const int top = (imagesize_ - new_h) / 2;

    cv::Mat square(imagesize_, imagesize_, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Rect roi(left, top, new_w, new_h);
    resized.copyTo(square(roi));

    meta.target_size = imagesize_;
    meta.new_w = new_w;
    meta.new_h = new_h;
    meta.pad_left = left;
    meta.pad_top = top;

    nchw = ImageProcess::ImageProcessor::HWCToCHW_normalize_gaussian(square, IMAGENET_MEAN, IMAGENET_STD, 1.0f / 255.0f);
}

bool PatchCoreDetector::RunBackbone(const std::vector<float>& input_nchw, std::vector<Ort::Value>& outputs) {
    const std::vector<int64_t> input_dims = {1, 3, imagesize_, imagesize_};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
        const_cast<float*>(input_nchw.data()), input_nchw.size(),
        input_dims.data(), input_dims.size());

    outputs = session_->Run(
        run_options_,
        input_names_.data(), &input_tensor, 1,
        output_names_.data(), output_names_.size());

    return !outputs.empty();
}

bool PatchCoreDetector::ExtractEmbeddings(const std::vector<Ort::Value>& outputs, std::vector<float>& embeddings) {
    const int ref_h = ref_patch_shape_.first;
    const int ref_w = ref_patch_shape_.second;
    const int P = ref_h * ref_w;
    const int L = static_cast<int>(outputs.size());

    std::vector<std::vector<float>> per_layer(L);

    auto prepare_adaptive_pool = [](int flat_len, int out_dim, std::vector<int>& starts, std::vector<int>& ends) {
        starts.resize(static_cast<size_t>(out_dim));
        ends.resize(static_cast<size_t>(out_dim));
        for (int i = 0; i < out_dim; ++i) {
            int start = static_cast<int>(std::floor(static_cast<double>(i) * flat_len / out_dim));
            int end = static_cast<int>(std::ceil(static_cast<double>(i + 1) * flat_len / out_dim));
            start = std::max(0, std::min(start, flat_len));
            end = std::max(0, std::min(end, flat_len));
            end = std::max(end, start + 1);
            end = std::max(0, std::min(end, flat_len));
            starts[static_cast<size_t>(i)] = start;
            ends[static_cast<size_t>(i)] = end;
        }
    };

    auto adaptive_mean_pool_1d = [](
        const float* flat, int flat_len,
        int out_dim,
        const std::vector<int>& starts,
        const std::vector<int>& ends,
        std::vector<double>& cs,
        float* out)
    {
        if (out_dim <= 0) return;
        if (flat_len <= 0) {
            std::fill(out, out + out_dim, 0.0f);
            return;
        }
        if (flat_len == out_dim) {
            std::memcpy(out, flat, sizeof(float) * static_cast<size_t>(out_dim));
            return;
        }

        cs.resize(static_cast<size_t>(flat_len) + 1);
        cs[0] = 0.0;
        for (int i = 0; i < flat_len; ++i) {
            cs[static_cast<size_t>(i) + 1] = cs[static_cast<size_t>(i)] + static_cast<double>(flat[i]);
        }

        for (int i = 0; i < out_dim; ++i) {
            const int start = starts[static_cast<size_t>(i)];
            const int end = ends[static_cast<size_t>(i)];
            const double sum = cs[static_cast<size_t>(end)] - cs[static_cast<size_t>(start)];
            const float sum_f = static_cast<float>(sum);
            const float len_f = static_cast<float>(end - start);
            out[i] = sum_f / len_f;
        }
    };

    for (int li = 0; li < L; ++li) {
        Ort::TypeInfo ti = outputs[li].GetTypeInfo();
        auto tsi = ti.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> dims = tsi.GetShape();
        if (dims.size() != 4 || dims[0] != 1) {
            std::cerr << "[PatchCore] unexpected output dims for layer " << li << std::endl;
            return false;
        }

        const int C = static_cast<int>(dims[1]);
        const int H = static_cast<int>(dims[2]);
        const int W = static_cast<int>(dims[3]);
        const float* feat = outputs[li].GetTensorData<float>();

        const int padding = (patchsize_ - 1) / 2;
        const int ph = (H + 2 * padding - patchsize_) / patchstride_ + 1;
        const int pw = (W + 2 * padding - patchsize_) / patchstride_ + 1;
        if (ph <= 0 || pw <= 0) {
            std::cerr << "[PatchCore] invalid patch grid for layer " << li << std::endl;
            return false;
        }

        const int patch_count = ph * pw;
        const int flat_len = C * patchsize_ * patchsize_;
        const bool is_avgpool = (layer_pool_[li].mode == "avgpool");
        int pool_kernel = 0;
        if (is_avgpool) {
            pool_kernel = flat_len / pretrain_embed_dim_;
            if (pretrain_embed_dim_ <= 0 || flat_len % pretrain_embed_dim_ != 0 || pool_kernel != layer_pool_[li].kernel) {
                std::cerr << "[PatchCore] mean-mapper mismatch at layer " << li << std::endl;
                return false;
            }
        } else {
            if (pretrain_embed_dim_ <= 0) {
                std::cerr << "[PatchCore] invalid pretrain_embed_dim at layer " << li << std::endl;
                return false;
            }
        }

        std::vector<float> layer_grid(static_cast<size_t>(patch_count) * pretrain_embed_dim_);
        std::vector<float> flat(static_cast<size_t>(flat_len));
        std::vector<int> adaptive_starts;
        std::vector<int> adaptive_ends;
        std::vector<double> adaptive_cs;
        if (!is_avgpool && flat_len != pretrain_embed_dim_) {
            prepare_adaptive_pool(flat_len, pretrain_embed_dim_, adaptive_starts, adaptive_ends);
        }

        for (int gy = 0; gy < ph; ++gy) {
            for (int gx = 0; gx < pw; ++gx) {
                int idx_flat = 0;
                for (int c = 0; c < C; ++c) {
                    const size_t c_base = static_cast<size_t>(c) * H * W;
                    for (int dy = 0; dy < patchsize_; ++dy) {
                        const int sy = gy * patchstride_ + dy - padding;
                        for (int dx = 0; dx < patchsize_; ++dx) {
                            const int sx = gx * patchstride_ + dx - padding;
                            float v = 0.0f;
                            if (sy >= 0 && sy < H && sx >= 0 && sx < W) {
                                v = feat[c_base + static_cast<size_t>(sy) * W + sx];
                            }
                            flat[idx_flat++] = v;
                        }
                    }
                }

                const int patch_id = gy * pw + gx;
                float* out_vec = layer_grid.data() + static_cast<size_t>(patch_id) * pretrain_embed_dim_;
                if (is_avgpool) {
                    for (int od = 0; od < pretrain_embed_dim_; ++od) {
                        const int start = od * pool_kernel;
                        double sum = 0.0;
                        for (int k = 0; k < pool_kernel; ++k) sum += static_cast<double>(flat[start + k]);
                        out_vec[od] = static_cast<float>(sum / static_cast<double>(pool_kernel));
                    }
                } else {
                    adaptive_mean_pool_1d(
                        flat.data(), flat_len,
                        pretrain_embed_dim_,
                        adaptive_starts, adaptive_ends,
                        adaptive_cs,
                        out_vec);
                }
            }
        }

        if (ph != ref_h || pw != ref_w) {
            std::vector<float> aligned(static_cast<size_t>(P) * pretrain_embed_dim_);
            ResizeBilinear3D(layer_grid.data(), ph, pw, pretrain_embed_dim_, aligned.data(), ref_h, ref_w);
            per_layer[li] = std::move(aligned);
        } else {
            per_layer[li] = std::move(layer_grid);
        }
    }

    const int concat_len = static_cast<int>(per_layer.size()) * pretrain_embed_dim_;
    const bool agg_avgpool = (agg_pool_.mode == "avgpool");
    int agg_kernel = 0;
    if (agg_avgpool) {
        agg_kernel = concat_len / target_embed_dim_;
        if (target_embed_dim_ <= 0 || concat_len % target_embed_dim_ != 0 || agg_kernel != agg_pool_.kernel) {
            std::cerr << "[PatchCore] aggregator mismatch" << std::endl;
            return false;
        }
    } else {
        if (target_embed_dim_ <= 0) {
            std::cerr << "[PatchCore] invalid target_embed_dim" << std::endl;
            return false;
        }
    }

    embeddings.resize(static_cast<size_t>(P) * target_embed_dim_);
    std::vector<int> agg_adaptive_starts;
    std::vector<int> agg_adaptive_ends;
    std::vector<double> agg_adaptive_cs;
    std::vector<float> concat;
    if (!agg_avgpool && concat_len != target_embed_dim_) {
        prepare_adaptive_pool(concat_len, target_embed_dim_, agg_adaptive_starts, agg_adaptive_ends);
        concat.resize(static_cast<size_t>(concat_len));
    }

    for (int p = 0; p < P; ++p) {
        float* out_vec = embeddings.data() + static_cast<size_t>(p) * target_embed_dim_;
        if (agg_avgpool) {
            for (int od = 0; od < target_embed_dim_; ++od) {
                double sum = 0.0;
                const int start = od * agg_kernel;
                for (int k = 0; k < agg_kernel; ++k) {
                    const int lin = start + k;
                    const int li = lin / pretrain_embed_dim_;
                    const int di = lin % pretrain_embed_dim_;
                    sum += static_cast<double>(per_layer[li][static_cast<size_t>(p) * pretrain_embed_dim_ + di]);
                }
                out_vec[od] = static_cast<float>(sum / static_cast<double>(agg_kernel));
            }
        } else {
            for (int li = 0; li < L; ++li) {
                const float* src = per_layer[li].data() + static_cast<size_t>(p) * pretrain_embed_dim_;
                std::memcpy(concat.data() + static_cast<size_t>(li) * pretrain_embed_dim_, src,
                            sizeof(float) * static_cast<size_t>(pretrain_embed_dim_));
            }
            adaptive_mean_pool_1d(
                concat.data(), concat_len,
                target_embed_dim_,
                agg_adaptive_starts, agg_adaptive_ends,
                agg_adaptive_cs,
                out_vec);
        }
    }

    return true;
}

std::vector<float> PatchCoreDetector::ComputeAnomalyScores(const std::vector<float>& embeddings, int n, int d) const {
    std::vector<float> scores;
    faiss::Index* index = static_cast<faiss::Index*>(faiss_index_.get());
    if (!index) return scores;
    if (n <= 0 || d <= 0) return scores;

    std::vector<float> distances(static_cast<size_t>(n) * num_nn_);
    std::vector<faiss::idx_t> labels(static_cast<size_t>(n) * num_nn_);
    index->search(n, embeddings.data(), num_nn_, distances.data(), labels.data());

    scores.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < num_nn_; ++j) sum += distances[static_cast<size_t>(i) * num_nn_ + j];
        scores[i] = sum / static_cast<float>(num_nn_);
    }

    return scores;
}

bool PatchCoreDetector::Initialize(
    const std::string& onnx_model_path,
    const std::string& faiss_index_path,
    const std::string& metadata_path,
    int ort_intra_threads) {
    try {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "PatchCore");

        if (!LoadMetadata(metadata_path)) return false;

        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.SetIntraOpNumThreads(ort_intra_threads);

#ifdef _WIN32
        const int path_size = MultiByteToWideChar(CP_UTF8, 0, onnx_model_path.c_str(), -1, nullptr, 0);
        std::wstring wide_path;
        wide_path.resize(static_cast<size_t>(path_size));
        MultiByteToWideChar(CP_UTF8, 0, onnx_model_path.c_str(), -1, &wide_path[0], path_size);
        session_ = new Ort::Session(env_, wide_path.c_str(), session_options);
#else
        session_ = new Ort::Session(env_, onnx_model_path.c_str(), session_options);
#endif

        Ort::AllocatorWithDefaultOptions allocator;
        input_names_storage_.clear();
        output_names_storage_.clear();
        input_names_.clear();
        output_names_.clear();

        const size_t num_inputs = session_->GetInputCount();
        if (num_inputs == 0) return false;
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name_ptr = session_->GetInputNameAllocated(i, allocator);
            input_names_storage_.push_back(std::string(name_ptr.get()));
        }
        for (auto& s : input_names_storage_) input_names_.push_back(s.c_str());

        const size_t num_outputs = session_->GetOutputCount();
        std::vector<std::string> available_outputs;
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name_ptr = session_->GetOutputNameAllocated(i, allocator);
            available_outputs.push_back(std::string(name_ptr.get()));
        }

        for (const auto& n : layer_names_) {
            if (std::find(available_outputs.begin(), available_outputs.end(), n) == available_outputs.end()) {
                std::cerr << "[PatchCore] missing onnx output: " << n << std::endl;
                return false;
            }
            output_names_storage_.push_back(n);
        }
        for (auto& s : output_names_storage_) output_names_.push_back(s.c_str());

        run_options_ = Ort::RunOptions{nullptr};

        faiss::Index* index = faiss::read_index(faiss_index_path.c_str());
        if (!index) {
            std::cerr << "[PatchCore] failed to load faiss index: " << faiss_index_path << std::endl;
            return false;
        }
        faiss_index_.reset(index);

        if (index->d != target_embed_dim_) {
            std::cerr << "[PatchCore] faiss d mismatch: index.d=" << index->d << " target_embed_dim=" << target_embed_dim_
                      << std::endl;
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[PatchCore] Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

bool PatchCoreDetector::Infer(const cv::Mat& image, PatchCoreResult& result) {
    if (!session_) {
        std::cerr << "[PatchCore] Session not initialized" << std::endl;
        return false;
    }

    try {
        std::vector<float> input_nchw;
        PatchCoreResult::MetaData meta{};
        PreprocessToNCHW(image, input_nchw, meta);

        std::vector<Ort::Value> outputs;
        if (!RunBackbone(input_nchw, outputs)) return false;

        std::vector<float> embeddings;
        if (!ExtractEmbeddings(outputs, embeddings)) return false;

        const int ref_h = ref_patch_shape_.first;
        const int ref_w = ref_patch_shape_.second;
        const int P = ref_h * ref_w;

        std::vector<float> patch_scores = ComputeAnomalyScores(embeddings, P, target_embed_dim_);
        if (patch_scores.size() != static_cast<size_t>(P)) {
            std::cerr << "[PatchCore] patch_scores size mismatch" << std::endl;
            return false;
        }

        const float max_score = *std::max_element(patch_scores.begin(), patch_scores.end());

        result.image_score = max_score;
        result.meta = meta;

        cv::Mat score_map(ref_h, ref_w, CV_32F, patch_scores.data());
        result.patch_scores = score_map.clone();

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[PatchCore] Inference error: " << e.what() << std::endl;
        return false;
    }
}

} // namespace PatchCore
