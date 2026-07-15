#include "image_process.h"
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>


namespace ImageProcess {

cv::Mat ImageProcessor::Cvmat2Uint8(const cv::Mat& img)
{
    if (img.empty()) return cv::Mat();

    if (img.depth() == CV_8U)
        return img.clone();

    cv::Mat img_u8;

    int depth = img.depth();

    if (depth == CV_32F)
    {
        double min_val, max_val;
        cv::minMaxLoc(img, &min_val, &max_val);

        cv::Mat norm;

        if (max_val > min_val + 1e-8)
        {
            norm = (img - min_val) / (max_val - min_val);
        }
        else
        {
            norm = cv::Mat::zeros(img.size(), CV_32F);
        }

        norm.convertTo(img_u8, CV_8U, 255.0);
    }

    else if (depth == CV_16U)
    {
        // 0~65535 → 0~255
        img.convertTo(img_u8, CV_8U, 1.0 / 256.0);
    }

    else if (depth == CV_16S)
    {
        // 有符号范围压缩
        double min_val, max_val;
        cv::minMaxLoc(img, &min_val, &max_val);

        if (max_val > min_val + 1e-8)
        {
            cv::Mat norm = (img - min_val) / (max_val - min_val);
            norm.convertTo(img_u8, CV_8U, 255.0);
        }
        else
        {
            img_u8 = cv::Mat::zeros(img.size(), CV_8U);
        }
    }

    else
    {
        double min_val, max_val;
        cv::minMaxLoc(img, &min_val, &max_val);

        if (max_val > min_val + 1e-8)
        {
            cv::Mat norm = (img - min_val) / (max_val - min_val);
            norm.convertTo(img_u8, CV_8U, 255.0);
        }
        else
        {
            img.convertTo(img_u8, CV_8U);
        }
    }

    return img_u8;
}

cv::Mat ImageProcessor::Cvmat2BGR(const cv::Mat& img) {
    if (img.channels() == 3) return img;

    cv::Mat img3;
    if (img.channels() == 1) {
        cv::cvtColor(img, img3, cv::COLOR_GRAY2BGR);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, img3, cv::COLOR_BGRA2BGR);
    } else {
        img3 = img;
    }
    return img3;
}

cv::Mat ImageProcessor::Cvmat2RGB(const cv::Mat& img) {
    
    cv::Mat img3;
    if (img.channels() == 3) {
        cv::cvtColor(img, img3, cv::COLOR_BGR2RGB);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, img3, cv::COLOR_BGRA2RGB);
    } else if (img.channels() == 1){
        cv::cvtColor(img, img3, cv::COLOR_GRAY2RGB);
    }
    return img3;
}

cv::Mat ImageProcessor::LetterboxResize(const cv::Mat& img,
                                        int target_h,
                                        int target_w,
                                        float& ratio,
                                        std::vector<float>& pad,
                                        cv::Scalar fill) {
  

    const int img_h = img.rows;
    const int img_w = img.cols;

    ratio = std::min(static_cast<float>(target_w) / img_w,
                     static_cast<float>(target_h) / img_h);
    const int new_w = static_cast<int>(std::round(img_w * ratio));
    const int new_h = static_cast<int>(std::round(img_h * ratio));

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    pad.resize(2);
    pad[0] = (target_w - new_w) / 2.0f;
    pad[1] = (target_h - new_h) / 2.0f;

    const int top    = static_cast<int>(std::round(pad[1] - 0.1f));
    const int bottom = static_cast<int>(std::round(pad[1] + 0.1f));
    const int left   = static_cast<int>(std::round(pad[0] - 0.1f));
    const int right  = static_cast<int>(std::round(pad[0] + 0.1f));

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                       cv::BORDER_CONSTANT, fill);

    return padded;
}


std::vector<float> ImageProcessor::HWCToCHW(const cv::Mat& img)
{
    CV_Assert(!img.empty());
    CV_Assert(img.depth() == CV_8U || img.depth() == CV_32F);
    CV_Assert(img.isContinuous());

    const int h = img.rows;
    const int w = img.cols;
    const int c = img.channels();
    CV_Assert(c > 0);

    const size_t hw = static_cast<size_t>(h) * w;

    std::vector<float> out(hw * c);
    float* dst = out.data();

    std::vector<float*> planes(c);
    for (int i = 0; i < c; ++i)
        planes[i] = dst + i * hw;

    if (img.depth() == CV_8U)
    {
        const uchar* data = img.ptr<uchar>(0);

        for (size_t i = 0; i < hw; ++i)
        {
            const uchar* src = data + i * c;
            for (int k = 0; k < c; ++k)
                planes[k][i] = static_cast<float>(src[k]);
        }
    }
    else
    {
        const float* data = img.ptr<float>(0);

        for (size_t i = 0; i < hw; ++i)
        {
            const float* src = data + i * c;
            for (int k = 0; k < c; ++k)
                planes[k][i] = src[k];
        }
    }

    return out;
}

std::vector<float> ImageProcessor::HWCToCHW_normalize(
    const cv::Mat& img,
    float scale = 1.0f / 255.0f)
{
    CV_Assert(!img.empty());
    CV_Assert(img.depth() == CV_8U || img.depth() == CV_32F);
    CV_Assert(img.isContinuous());

    const int h = img.rows;
    const int w = img.cols;
    const int c = img.channels();

    const size_t hw = static_cast<size_t>(h) * w;

    std::vector<float> out(hw * c);
    float* dst = out.data();

    std::vector<float*> planes(c);
    for (int i = 0; i < c; ++i)
        planes[i] = dst + i * hw;

    if (img.depth() == CV_8U)
    {
        const uchar* data = img.ptr<uchar>(0);

        for (size_t i = 0; i < hw; ++i)
        {
            const uchar* src = data + i * c;
            for (int k = 0; k < c; ++k)
            {
                planes[k][i] = static_cast<float>(src[k]) * scale;
            }
        }
    }
    else // CV_32F
    {
        const float* data = img.ptr<float>(0);

        for (size_t i = 0; i < hw; ++i)
        {
            const float* src = data + i * c;
            for (int k = 0; k < c; ++k)
            {
                planes[k][i] = src[k] * scale;
            }
        }
    }

    return out;
}


std::vector<float> ImageProcessor::HWCToCHW_normalize_gaussian(
    const cv::Mat& img,
    const float mean[3],
    const float std[3],
    float scale = 1.0f / 255.0f)
{
    CV_Assert(!img.empty());
    CV_Assert(img.depth() == CV_8U || img.depth() == CV_32F);
    CV_Assert(img.isContinuous());

    const int h = img.rows;
    const int w = img.cols;
    const int c = img.channels();
    CV_Assert(c == 3); 

    const size_t hw = static_cast<size_t>(h) * w;

    std::vector<float> out(hw * c);
    float* dst = out.data();

    std::vector<float*> planes(c);
    for (int i = 0; i < c; ++i)
        planes[i] = dst + i * hw;

    if (img.depth() == CV_8U)
    {
        const uchar* data = img.ptr<uchar>(0);

        for (size_t i = 0; i < hw; ++i)
        {
            const uchar* src = data + i * c;

            for (int k = 0; k < c; ++k)
            {
                float v = static_cast<float>(src[k]) * scale;
                planes[k][i] = (v - mean[k]) / std[k];
            }
        }
    }
    else // CV_32F
    {
        const float* data = img.ptr<float>(0);

        for (size_t i = 0; i < hw; ++i)
        {
            const float* src = data + i * c;

            for (int k = 0; k < c; ++k)
            {
                float v = src[k] * scale;
                planes[k][i] = (v - mean[k]) / std[k];
            }
        }
    }

    return out;
}

void ImageProcessor::NormalizeNCHW(std::vector<float>& nchw,
                                   int h,
                                   int w,
                                   const float mean[3],
                                   const float scale[3])
{
    const size_t plane = static_cast<size_t>(h) * w;

    CV_Assert(nchw.size() == plane * 3);

    float* base = nchw.data();

    for (int c = 0; c < 3; ++c)
    {
        float* ptr = base + c * plane;
        const float m = mean[c];
        const float s = scale[c];

        for (size_t i = 0; i < plane; ++i)
            ptr[i] = (ptr[i] - m) / s;
    }
}


void ImageProcessor::ResizeBilinear3D(const float* in,
                                      int in_h,
                                      int in_w,
                                      int depth,
                                      float* out,
                                      int out_h,
                                      int out_w) {
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
                out[out_base + d] = in[in00 + d] * w00 +
                                    in[in01 + d] * w01 +
                                    in[in10 + d] * w10 +
                                    in[in11 + d] * w11;
            }
        }
    }
}


cv::Mat ImageProcessor::ResizeBilinearPillowResampleRGBU8(const cv::Mat& src, int dst_w, int dst_h) {
    if (dst_w <= 0 || dst_h <= 0) return cv::Mat();
    CV_Assert(!src.empty());
    CV_Assert(src.type() == CV_8UC3);

    const int src_h = src.rows;
    const int src_w = src.cols;
    CV_Assert(src_h > 0 && src_w > 0);

    constexpr int PRECISION_BITS = (32 - 8 - 2);
    const int32_t ROUNDING = 1 << (PRECISION_BITS - 1);

    const int precision_bits = PRECISION_BITS;
    auto clip8 = [precision_bits](int32_t in) -> uint8_t {
        int32_t v = in >> precision_bits;
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        return static_cast<uint8_t>(v);
    };

    auto bilinear_filter = [](double x) -> double {
        x = std::abs(x);
        if (x < 1.0) return 1.0 - x;
        return 0.0;
    };

    struct Coeffs {
        int ksize = 0;
        std::vector<int> bounds;
        std::vector<int32_t> kk;
    };

    auto precompute = [&](int inSize, int outSize) -> Coeffs {
        Coeffs c;
        const double scale = static_cast<double>(inSize) / static_cast<double>(outSize);
        double filterscale = scale;
        if (filterscale < 1.0) filterscale = 1.0;
        const double support = 1.0 * filterscale;
        const int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
        c.ksize = ksize;
        c.bounds.assign(static_cast<size_t>(outSize) * 2, 0);
        c.kk.assign(static_cast<size_t>(outSize) * static_cast<size_t>(ksize), 0);

        const double ss = 1.0 / filterscale;

        for (int xx = 0; xx < outSize; ++xx) {
            const double center = (static_cast<double>(xx) + 0.5) * scale;
            double ww = 0.0;

            int xmin = static_cast<int>(center - support + 0.5);
            if (xmin < 0) xmin = 0;
            int xmax = static_cast<int>(center + support + 0.5);
            if (xmax > inSize) xmax = inSize;
            xmax -= xmin;
            if (xmax < 0) xmax = 0;

            std::vector<double> wbuf(static_cast<size_t>(xmax), 0.0);
            for (int x = 0; x < xmax; ++x) {
                const double w = bilinear_filter((static_cast<double>(x + xmin) - center + 0.5) * ss);
                wbuf[static_cast<size_t>(x)] = w;
                ww += w;
            }
            if (ww != 0.0) {
                for (int x = 0; x < xmax; ++x) wbuf[static_cast<size_t>(x)] /= ww;
            }

            c.bounds[static_cast<size_t>(xx) * 2 + 0] = xmin;
            c.bounds[static_cast<size_t>(xx) * 2 + 1] = xmax;

            int32_t* outk = c.kk.data() + static_cast<size_t>(xx) * static_cast<size_t>(ksize);
            for (int x = 0; x < xmax; ++x) {
                const double w = wbuf[static_cast<size_t>(x)];
                outk[x] = (w < 0.0)
                              ? static_cast<int32_t>(-0.5 + w * static_cast<double>(1 << PRECISION_BITS))
                              : static_cast<int32_t>(0.5 + w * static_cast<double>(1 << PRECISION_BITS));
            }
            for (int x = xmax; x < ksize; ++x) outk[x] = 0;
        }
        return c;
    };

    const Coeffs cx = precompute(src_w, dst_w);
    const Coeffs cy = precompute(src_h, dst_h);

    cv::Mat tmp(src_h, dst_w, CV_8UC3);
    for (int yy = 0; yy < src_h; ++yy) {
        const uint8_t* row = src.ptr<uint8_t>(yy);
        uint8_t* out = tmp.ptr<uint8_t>(yy);
        for (int xx = 0; xx < dst_w; ++xx) {
            const int xmin = cx.bounds[static_cast<size_t>(xx) * 2 + 0];
            const int xmax = cx.bounds[static_cast<size_t>(xx) * 2 + 1];
            const int32_t* k = cx.kk.data() + static_cast<size_t>(xx) * static_cast<size_t>(cx.ksize);

            int32_t ss0 = ROUNDING;
            int32_t ss1 = ROUNDING;
            int32_t ss2 = ROUNDING;
            for (int x = 0; x < xmax; ++x) {
                const int ix = x + xmin;
                const uint8_t* p = row + ix * 3;
                const int32_t kk = k[x];
                ss0 += static_cast<int32_t>(p[0]) * kk;
                ss1 += static_cast<int32_t>(p[1]) * kk;
                ss2 += static_cast<int32_t>(p[2]) * kk;
            }
            out[xx * 3 + 0] = clip8(ss0);
            out[xx * 3 + 1] = clip8(ss1);
            out[xx * 3 + 2] = clip8(ss2);
        }
    }

    cv::Mat dst(dst_h, dst_w, CV_8UC3);
    for (int yy = 0; yy < dst_h; ++yy) {
        const int ymin = cy.bounds[static_cast<size_t>(yy) * 2 + 0];
        const int ymax = cy.bounds[static_cast<size_t>(yy) * 2 + 1];
        const int32_t* k = cy.kk.data() + static_cast<size_t>(yy) * static_cast<size_t>(cy.ksize);

        uint8_t* out = dst.ptr<uint8_t>(yy);
        for (int xx = 0; xx < dst_w; ++xx) {
            int32_t ss0 = ROUNDING;
            int32_t ss1 = ROUNDING;
            int32_t ss2 = ROUNDING;
            for (int y = 0; y < ymax; ++y) {
                const uint8_t* row = tmp.ptr<uint8_t>(ymin + y);
                const uint8_t* p = row + xx * 3;
                const int32_t kk = k[y];
                ss0 += static_cast<int32_t>(p[0]) * kk;
                ss1 += static_cast<int32_t>(p[1]) * kk;
                ss2 += static_cast<int32_t>(p[2]) * kk;
            }
            out[xx * 3 + 0] = clip8(ss0);
            out[xx * 3 + 1] = clip8(ss1);
            out[xx * 3 + 2] = clip8(ss2);
        }
    }

    return dst;
}

cv::Mat ImageProcessor::GetSafeBorderMask(const cv::Mat& gray, int max_border_inner, int step) {
    const int h = gray.rows;
    const int w = gray.cols;
    cv::Mat mask = cv::Mat::zeros(h, w, CV_8U);
    if (h <= 0 || w <= 0) return mask;

    const int max_inner = std::min(max_border_inner, std::min(h, w) / 2);
    auto mean_strip = [&](int x0, int y0, int x1, int y1) -> double {
        x0 = std::max(0, std::min(x0, w));
        x1 = std::max(0, std::min(x1, w));
        y0 = std::max(0, std::min(y0, h));
        y1 = std::max(0, std::min(y1, h));
        if (x1 <= x0 || y1 <= y0) return 0.0;
        return cv::mean(gray(cv::Rect(x0, y0, x1 - x0, y1 - y0)))[0];
    };

    int edge_top = 0;
    for (int y = 0; y < std::min(max_inner, h / 2); y += step) {
        const double diff = mean_strip(0, y + step, w, y + 2 * step) - mean_strip(0, y, w, y + step);
        if (diff > 8.0) {
            edge_top = y + step;
            break;
        }
    }
    edge_top = std::min(edge_top, max_inner);
    if (edge_top > 0) mask.rowRange(0, edge_top).setTo(255);

    int edge_bot = h;
    for (int y = h - 1; y > h - std::min(max_inner, h / 2); y -= step) {
        const double diff = mean_strip(0, y - step, w, y) - mean_strip(0, y, w, y + step);
        if (diff > 8.0) {
            edge_bot = y - step;
            break;
        }
    }
    edge_bot = std::max(edge_bot, h - max_inner);
    if (edge_bot < h) mask.rowRange(edge_bot, h).setTo(255);

    int edge_left = 0;
    for (int x = 0; x < std::min(max_inner, w / 2); x += step) {
        const double diff = mean_strip(x + step, 0, x + 2 * step, h) - mean_strip(x, 0, x + step, h);
        if (diff > 8.0) {
            edge_left = x + step;
            break;
        }
    }
    edge_left = std::min(edge_left, max_inner);
    if (edge_left > 0) mask.colRange(0, edge_left).setTo(255);

    int edge_right = w;
    for (int x = w - 1; x > w - std::min(max_inner, w / 2); x -= step) {
        const double diff = mean_strip(x - step, 0, x, h) - mean_strip(x, 0, x + step, h);
        if (diff > 8.0) {
            edge_right = x - step;
            break;
        }
    }
    edge_right = std::max(edge_right, w - max_inner);
    if (edge_right < w) mask.colRange(edge_right, w).setTo(255);

    return mask;
}

} // namespace ImageProcess
