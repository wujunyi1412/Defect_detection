#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "inference_dll.h"

namespace fs = std::filesystem;

namespace {

constexpr std::array<const char*, 5> kCategoryNames = {
    "Abnormal",
    "Stain",
    "BrightStripes",
    "LineArtifacts",
    "DarkClusters",
};

struct BatchOptions {
    std::string config_path = "config.ini";
    std::string input_dir;
    std::string output_dir;
    bool recurse = false;
    bool save_overlay = true;
};

struct FileResult {
    std::string file_name;
    std::string file_path;
    std::string status;
    std::string message;
    std::string overlay_path;
    InspectionDLL::InferenceResult result;
};

std::string ToLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool IsPngPath(const fs::path& path) {
    return ToLower(path.extension().string()) == ".png";
}

std::string CsvEscape(const std::string& value) {
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += "\"";
    return out;
}

int CategoryIndex(const std::string& name) {
    if (name == "Abnormal") return 0;
    if (name == "Stain") return 1;
    if (name == "BrightStripes") return 2;
    if (name == "LineArtifacts") return 3;
    if (name == "DarkClusters") return 4;
    return -1;
}

cv::Scalar CategoryColor(const std::string& name) {
    const int idx = CategoryIndex(name);
    if (idx == 0) return cv::Scalar(0, 0, 255);
    if (idx == 1) return cv::Scalar(0, 165, 255);
    if (idx == 2) return cv::Scalar(0, 255, 255);
    if (idx == 3) return cv::Scalar(255, 0, 0);
    if (idx == 4) return cv::Scalar(255, 0, 255);
    return cv::Scalar(0, 255, 0);
}

std::array<int, kCategoryNames.size()> CountByCategory(const InspectionDLL::InferenceResult& result) {
    std::array<int, kCategoryNames.size()> counts{};
    for (const auto& detail : result.details) {
        const int idx = CategoryIndex(detail.name);
        if (idx >= 0) {
            ++counts[static_cast<size_t>(idx)];
        }
    }
    return counts;
}

void PrintResult(const InspectionDLL::InferenceResult& r) {
    const auto counts = CountByCategory(r);

    std::cout << "result: " << r.result << "\n";
    std::cout << "details_count: " << r.details.size() << "\n";
    std::cout << "yolo_score: " << r.yolo_score << "\n";
    std::cout << "patchcore_score: " << r.patchcore_score << "\n";
    std::cout << "patchcore_area_ratio: " << r.patchcore_area_ratio << "\n";

    for (size_t c = 0; c < kCategoryNames.size(); ++c) {
        std::cout << kCategoryNames[c] << " count: " << counts[c] << "\n";
    }

    for (size_t i = 0; i < r.details.size(); ++i) {
        const auto& d = r.details[i];
        std::cout << "  [" << i << "] "
                  << "name=" << d.name
                  << ", area=" << d.area
                  << ", x=" << d.x << ", y=" << d.y << ", w=" << d.w << ", h=" << d.h
                  << ", contrast=" << d.contrast
                  << ", score=" << d.score
                  << "\n";
    }
}

bool SaveDetectionOverlay(
    const std::string& image_path,
    const InspectionDLL::InferenceResult& result,
    const fs::path& output_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        std::cerr << "OpenCV imread failed when saving overlay: " << image_path << "\n";
        return false;
    }

    cv::Mat vis;
    if (image.channels() == 1) {
        cv::normalize(image, vis, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::cvtColor(vis, vis, cv::COLOR_GRAY2BGR);
    } else if (image.channels() == 3) {
        if (image.depth() == CV_8U) {
            vis = image.clone();
        } else {
            cv::normalize(image, vis, 0, 255, cv::NORM_MINMAX, CV_8U);
        }
    } else if (image.channels() == 4) {
        cv::Mat bgr;
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
        if (bgr.depth() == CV_8U) {
            vis = bgr;
        } else {
            cv::normalize(bgr, vis, 0, 255, cv::NORM_MINMAX, CV_8U);
        }
    } else {
        std::cerr << "Unsupported image channel count for overlay: " << image.channels() << "\n";
        return false;
    }

    for (const auto& d : result.details) {
        cv::Rect rect(
            static_cast<int>(std::round(d.x)),
            static_cast<int>(std::round(d.y)),
            static_cast<int>(std::round(d.w)),
            static_cast<int>(std::round(d.h)));
        rect &= cv::Rect(0, 0, vis.cols, vis.rows);
        if (rect.empty()) {
            continue;
        }

        const cv::Scalar color = CategoryColor(d.name);
        cv::rectangle(vis, rect, color, 2);

        std::ostringstream label;
        label << d.name << " area=" << d.area << " score=" << std::fixed << std::setprecision(3) << d.score;
        const std::string text = label.str();

        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const int label_x = std::max(0, rect.x);
        const int label_y = std::max(text_size.height + baseline + 2, rect.y);
        cv::rectangle(
            vis,
            cv::Rect(label_x, label_y - text_size.height - baseline - 2,
                     std::min(text_size.width + 6, vis.cols - label_x),
                     text_size.height + baseline + 4),
            color,
            cv::FILLED);
        cv::putText(vis, text, cv::Point(label_x + 3, label_y - baseline - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    fs::create_directories(output_path.parent_path());
    if (!cv::imwrite(output_path.string(), vis)) {
        std::cerr << "cv::imwrite failed: " << output_path.string() << "\n";
        return false;
    }
    return true;
}

bool ExampleProcessPngPath(
    InspectionDLL::InspectionEngine& engine,
    const std::string& image_path,
    const std::string& overlay_path) {
    InspectionDLL::InferenceResult out;
    if (!engine.ProcessImagePath(image_path, out)) {
        std::cerr << "ProcessImagePath failed: " << image_path
                  << ", error=" << engine.GetLastError() << "\n";
        return false;
    }

    PrintResult(out);
    if (!overlay_path.empty() && SaveDetectionOverlay(image_path, out, overlay_path)) {
        std::cout << "overlay: " << overlay_path << "\n";
    }
    return true;
}

bool ExampleProcessPngMat(
    InspectionDLL::InspectionEngine& engine,
    const std::string& image_path,
    const std::string& overlay_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        std::cerr << "OpenCV imread failed: " << image_path << "\n";
        return false;
    }

    InspectionDLL::InferenceResult out;
    if (!engine.ProcessImage(image, out)) {
        std::cerr << "ProcessImage(cv::Mat) failed: " << image_path
                  << ", error=" << engine.GetLastError() << "\n";
        return false;
    }

    PrintResult(out);
    if (!overlay_path.empty() && SaveDetectionOverlay(image_path, out, overlay_path)) {
        std::cout << "overlay: " << overlay_path << "\n";
    }
    return true;
}

std::vector<fs::path> FindPngFiles(const fs::path& input_dir, bool recurse) {
    std::vector<fs::path> files;

    if (recurse) {
        for (const auto& entry : fs::recursive_directory_iterator(input_dir)) {
            if (entry.is_regular_file() && IsPngPath(entry.path())) {
                files.push_back(entry.path());
            }
        }
    } else {
        for (const auto& entry : fs::directory_iterator(input_dir)) {
            if (entry.is_regular_file() && IsPngPath(entry.path())) {
                files.push_back(entry.path());
            }
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

FileResult ProcessOnePng(
    InspectionDLL::InspectionEngine& engine,
    const fs::path& image_path,
    const fs::path& overlay_dir,
    bool save_overlay) {
    FileResult item;
    item.file_name = image_path.filename().string();
    item.file_path = fs::absolute(image_path).string();

    if (engine.ProcessImagePath(item.file_path, item.result)) {
        item.status = "Success";
    } else {
        item.status = "Failed";
        item.message = engine.GetLastError();
    }

    if (save_overlay && item.status == "Success") {
        const fs::path overlay_path =
            overlay_dir / (image_path.stem().string() + "_detect.png");
        if (SaveDetectionOverlay(item.file_path, item.result, overlay_path)) {
            item.overlay_path = overlay_path.string();
        } else {
            item.message = item.message.empty() ? "save overlay failed" : item.message + "; save overlay failed";
        }
    }

    return item;
}

void WriteSummaryCsv(const fs::path& summary_path, const std::vector<FileResult>& results) {
    std::ofstream writer(summary_path, std::ios::binary);
    writer << "FileName,FilePath,Status,Result,AbnormalCount,StainCount,"
              "BrightStripesCount,LineArtifactsCount,DarkClustersCount,TotalDetectCount,"
              "YoloScore,PatchcoreScore,PatchcoreAreaRatio,Message,OverlayPath\n";

    for (const auto& item : results) {
        const auto counts = CountByCategory(item.result);
        const int total = std::accumulate(counts.begin(), counts.end(), 0);
        writer << CsvEscape(item.file_name) << ','
               << CsvEscape(item.file_path) << ','
               << CsvEscape(item.status) << ','
               << CsvEscape(item.result.result) << ','
               << counts[0] << ','
               << counts[1] << ','
               << counts[2] << ','
               << counts[3] << ','
               << counts[4] << ','
               << total << ','
               << std::setprecision(8) << item.result.yolo_score << ','
               << std::setprecision(8) << item.result.patchcore_score << ','
               << std::setprecision(8) << item.result.patchcore_area_ratio << ','
               << CsvEscape(item.message) << ','
               << CsvEscape(item.overlay_path) << '\n';
    }
}

int ExampleBatchPngDirectory(InspectionDLL::InspectionEngine& engine, const BatchOptions& options) {
    const fs::path input_dir = fs::absolute(options.input_dir);
    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        std::cerr << "Input directory not found: " << input_dir.string() << "\n";
        return 3;
    }

    const auto files = FindPngFiles(input_dir, options.recurse);
    if (files.empty()) {
        std::cerr << "No .png files found under: " << input_dir.string() << "\n";
        return 4;
    }

    fs::path output_dir = options.output_dir.empty()
        ? input_dir
        : fs::absolute(options.output_dir);
    fs::create_directories(output_dir);
    const fs::path overlay_dir = output_dir / "detect_images";

    std::cout << "InputDir  : " << input_dir.string() << "\n";
    std::cout << "PngCount  : " << files.size() << "\n";
    std::cout << "OutputDir : " << output_dir.string() << "\n";
    std::cout << "OverlayDir: " << (options.save_overlay ? overlay_dir.string() : std::string("<disabled>")) << "\n";
    std::cout << "InitMode  : single engine + single initialize\n\n";

    std::vector<FileResult> results;
    results.reserve(files.size());

    for (size_t i = 0; i < files.size(); ++i) {
        std::cout << "[" << (i + 1) << "/" << files.size() << "] "
                  << files[i].string() << "\n";

        auto item = ProcessOnePng(engine, files[i], overlay_dir, options.save_overlay);
        const auto counts = CountByCategory(item.result);
        const int total = std::accumulate(counts.begin(), counts.end(), 0);

        std::cout << "  Status     : " << item.status << "\n";
        std::cout << "  Result     : " << item.result.result << "\n";
        std::cout << "  TotalCount : " << total << "\n";
        if (!item.message.empty()) {
            std::cout << "  Message    : " << item.message << "\n";
        }
        if (!item.overlay_path.empty()) {
            std::cout << "  Overlay    : " << item.overlay_path << "\n";
        }

        results.push_back(std::move(item));
    }

    const fs::path summary_path = output_dir / "summary.csv";
    WriteSummaryCsv(summary_path, results);

    const auto failed = std::count_if(results.begin(), results.end(), [](const FileResult& item) {
        return item.status != "Success";
    });

    std::cout << "\nBatch inference completed.\n";
    std::cout << "Success    : " << (results.size() - static_cast<size_t>(failed)) << "\n";
    std::cout << "Failed     : " << failed << "\n";
    std::cout << "SummaryCsv : " << summary_path.string() << "\n";

    return failed == 0 ? 0 : 5;
}

void PrintUsage() {
    std::cout
        << "Usage:\n"
        << "  use_examples [config.ini] [image.png]\n"
        << "  use_examples --single-png <image.png> [--config <config.ini>]\n"
        << "  use_examples --single-png-mat <image.png> [--config <config.ini>]\n"
        << "  use_examples --batch-png --input-dir <png_folder> [--config <config.ini>] [--output-dir <dir>] [--recurse] [--no-save-overlay]\n\n"
        << "Examples:\n"
        << "  use_examples config.ini test.png\n"
        << "  use_examples --single-png test.png --config config.ini\n"
        << "  use_examples --single-png-mat test.png --config config.ini\n"
        << "  use_examples --batch-png --input-dir E:\\\\images --output-dir E:\\\\out --config config.ini --recurse\n";
}

std::string RequireValue(int argc, char** argv, int& index, const std::string& option) {
    if (index + 1 >= argc) {
        throw std::runtime_error("Missing value for " + option);
    }
    ++index;
    return argv[index];
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string config_path = "E:/onnx_infer_code/cpp_version_C#/Use_DLL/config.ini";
        std::string single_png;
        std::string single_png_mat;
        std::string overlay_path;
        BatchOptions batch;
        bool batch_mode = false;

        if (argc == 1) {
            single_png = "test.png";
        } else if (argc <= 3 && argv[1][0] != '-') {
            config_path = argv[1];
            single_png = (argc >= 3) ? argv[2] : "test.png";
        } else {
            for (int i = 1; i < argc; ++i) {
                const std::string arg = argv[i];
                if (arg == "--config") {
                    config_path = RequireValue(argc, argv, i, arg);
                } else if (arg == "--single-png") {
                    single_png = RequireValue(argc, argv, i, arg);
                } else if (arg == "--single-png-mat") {
                    single_png_mat = RequireValue(argc, argv, i, arg);
                } else if (arg == "--overlay") {
                    overlay_path = RequireValue(argc, argv, i, arg);
                } else if (arg == "--batch-png") {
                    batch_mode = true;
                } else if (arg == "--input-dir") {
                    batch.input_dir = RequireValue(argc, argv, i, arg);
                } else if (arg == "--output-dir") {
                    batch.output_dir = RequireValue(argc, argv, i, arg);
                } else if (arg == "--recurse") {
                    batch.recurse = true;
                } else if (arg == "--no-save-overlay") {
                    batch.save_overlay = false;
                } else if (arg == "--help" || arg == "-h") {
                    PrintUsage();
                    return 0;
                } else {
                    throw std::runtime_error("Unknown argument: " + arg);
                }
            }
        }

        InspectionDLL::InspectionEngine engine;
        if (!engine.Initialize(config_path)) {
            std::cerr << "InspectionEngine::Initialize failed: " << config_path
                      << ", error=" << engine.GetLastError() << "\n";
            return 1;
        }
        std::cout << "Initialize succeeded.\n";

        if (batch_mode) {
            batch.config_path = config_path;
            if (batch.input_dir.empty()) {
                throw std::runtime_error("--input-dir is required for --batch-png");
            }
            return ExampleBatchPngDirectory(engine, batch);
        }

        if (!single_png_mat.empty()) {
            if (overlay_path.empty()) {
                overlay_path = (fs::path(single_png_mat).stem().string() + "_detect.png");
            }
            return ExampleProcessPngMat(engine, single_png_mat, overlay_path) ? 0 : 2;
        }

        if (single_png.empty()) {
            PrintUsage();
            return 2;
        }
        if (overlay_path.empty()) {
            overlay_path = (fs::path(single_png).stem().string() + "_detect.png");
        }
        return ExampleProcessPngPath(engine, single_png, overlay_path) ? 0 : 2;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        PrintUsage();
        return 2;
    }
}
