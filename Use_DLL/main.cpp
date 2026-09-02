#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "inference_dll.h"

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::array<const char*, 9> kCategoryNames = {
    "Abnormal",
    "Glue_overflow",
    "Decolorization",
    "Stain",
    "Stripes",
    "BrightStripes",
    "Bright_clusters",
    "Line_artifacts",
    "LineArtifacts",
};

enum class RunMode {
    Single,
    Batch,
};

enum class InputType {
    Auto,
    Png,
    Tiff,
    Bin,
};

struct Options {
    RunMode mode = RunMode::Single;
    InputType input_type = InputType::Auto;
    fs::path input;
    fs::path config_path = "config.ini";
    fs::path output_dir;
    int width = 1000;
    int height = 800;
    int repeat_count = 1;
    bool recurse = false;
    bool save_images = true;
};

struct LoadedInput {
    cv::Mat image;
    std::vector<float> float_data;
};

struct FileResult {
    int repeat_index = 1;
    fs::path file_path;
    std::string status = "Failed";
    int return_code = -1;
    InspectionDLL::InferenceResult inference;
    std::string message;
    fs::path output_image_path;
    double inference_ms = 0.0;
    double save_ms = 0.0;
    double total_ms = 0.0;
    bool image_save_attempted = false;
};

double Milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string ToLower(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

const char* InputTypeName(InputType type) {
    switch (type) {
        case InputType::Png: return "png";
        case InputType::Tiff: return "tiff";
        case InputType::Bin: return "bin";
        case InputType::Auto: return "auto";
    }
    return "unknown";
}

InputType ParseInputType(const std::string& value) {
    const std::string lower = ToLower(value);
    if (lower == "auto") return InputType::Auto;
    if (lower == "png") return InputType::Png;
    if (lower == "tif" || lower == "tiff") return InputType::Tiff;
    if (lower == "bin") return InputType::Bin;
    throw std::runtime_error(
        "Invalid --input-type: " + value + " (expected auto, png, tiff, or bin)");
}

InputType DetectInputType(const fs::path& path) {
    const std::string extension = ToLower(path.extension().string());
    if (extension == ".png") return InputType::Png;
    if (extension == ".tif" || extension == ".tiff") return InputType::Tiff;
    if (extension == ".bin") return InputType::Bin;
    return InputType::Auto;
}

bool MatchesInputType(const fs::path& path, InputType requested) {
    const InputType detected = DetectInputType(path);
    return detected != InputType::Auto &&
           (requested == InputType::Auto || requested == detected);
}

int CategoryIndex(const std::string& name) {
    if (name == "Abnormal") return 0;
    for (size_t index = 1; index < kCategoryNames.size(); ++index) {
        if (name == kCategoryNames[index]) return static_cast<int>(index);
    }
    return -1;
}

std::array<int, kCategoryNames.size()> CountByCategory(
    const InspectionDLL::InferenceResult& result) {
    std::array<int, kCategoryNames.size()> counts{};
    for (const auto& detail : result.details) {
        const int index = CategoryIndex(detail.name);
        if (index >= 0) {
            ++counts[static_cast<size_t>(index)];
        }
    }
    return counts;
}

cv::Scalar CategoryColor(const std::string& name) {
    switch (CategoryIndex(name)) {
        case 0: return cv::Scalar(0, 0, 255);
        case 1: return cv::Scalar(0, 165, 255);
        case 2: return cv::Scalar(0, 255, 255);
        case 3: return cv::Scalar(255, 0, 0);
        case 4: return cv::Scalar(255, 0, 255);
        default: return cv::Scalar(0, 255, 0);
    }
}

std::string Csv(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (char c : value) {
        if (c == '"') escaped.push_back('"');
        escaped.push_back(c);
    }
    escaped.push_back('"');
    return escaped;
}

std::string RequireValue(int argc, char** argv, int& index, const std::string& option) {
    if (index + 1 >= argc) {
        throw std::runtime_error("Missing value for " + option);
    }
    return argv[++index];
}

int ParsePositiveInt(const std::string& value, const std::string& option) {
    size_t parsed = 0;
    int result = 0;
    try {
        result = std::stoi(value, &parsed);
    } catch (...) {
        throw std::runtime_error("Invalid integer for " + option + ": " + value);
    }
    if (parsed != value.size() || result <= 0) {
        throw std::runtime_error(option + " must be a positive integer");
    }
    return result;
}

Options ParseArgs(int argc, char** argv) {
    Options options;
    if (argc <= 1) {
        throw std::runtime_error("Missing arguments");
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--mode") {
            const std::string mode = ToLower(RequireValue(argc, argv, i, arg));
            if (mode == "single") {
                options.mode = RunMode::Single;
            } else if (mode == "batch") {
                options.mode = RunMode::Batch;
            } else {
                throw std::runtime_error("--mode must be single or batch");
            }
        } else if (arg == "--single") {
            options.mode = RunMode::Single;
        } else if (arg == "--batch") {
            options.mode = RunMode::Batch;
        } else if (arg == "--input") {
            options.input = RequireValue(argc, argv, i, arg);
        } else if (arg == "--input-file") {
            options.mode = RunMode::Single;
            options.input = RequireValue(argc, argv, i, arg);
        } else if (arg == "--input-dir") {
            options.mode = RunMode::Batch;
            options.input = RequireValue(argc, argv, i, arg);
        } else if (arg == "--input-type" || arg == "--format") {
            options.input_type = ParseInputType(RequireValue(argc, argv, i, arg));
        } else if (arg == "--config") {
            options.config_path = RequireValue(argc, argv, i, arg);
        } else if (arg == "--output-dir") {
            options.output_dir = RequireValue(argc, argv, i, arg);
        } else if (arg == "--width") {
            options.width = ParsePositiveInt(RequireValue(argc, argv, i, arg), arg);
        } else if (arg == "--height") {
            options.height = ParsePositiveInt(RequireValue(argc, argv, i, arg), arg);
        } else if (arg == "--repeat") {
            options.repeat_count = ParsePositiveInt(RequireValue(argc, argv, i, arg), arg);
        } else if (arg == "--recurse") {
            options.recurse = true;
        } else if (arg == "--save-images" || arg == "--export-halcon-png") {
            options.save_images = true;
        } else if (arg == "--no-save-images" ||
                   arg == "--no-save-overlay" ||
                   arg == "--no-export-halcon-png") {
            options.save_images = false;
        } else if (arg == "--single-png" || arg == "--single-png-mat") {
            options.mode = RunMode::Single;
            options.input_type = InputType::Png;
            options.input = RequireValue(argc, argv, i, arg);
        } else if (arg == "--batch-png") {
            options.mode = RunMode::Batch;
            options.input_type = InputType::Png;
        } else if (arg == "--help" || arg == "-h") {
            throw std::runtime_error("help");
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (options.input.empty()) {
        throw std::runtime_error("--input is required");
    }
    return options;
}

fs::path ResolveExistingPath(const fs::path& path, bool expect_directory) {
    const fs::path absolute = fs::absolute(path);
    if (!fs::exists(absolute)) {
        throw std::runtime_error("Input path not found: " + absolute.string());
    }
    if (expect_directory != fs::is_directory(absolute)) {
        throw std::runtime_error(
            expect_directory ? "Input is not a directory: " + absolute.string()
                             : "Input is not a file: " + absolute.string());
    }
    return absolute;
}

fs::path ResolveConfigPath(const fs::path& path) {
    const fs::path absolute = fs::absolute(path);
    if (!fs::exists(absolute) || !fs::is_regular_file(absolute)) {
        throw std::runtime_error("Config file not found: " + absolute.string());
    }
    return absolute;
}

std::vector<fs::path> FindInputFiles(
    const fs::path& input_dir,
    InputType requested_type,
    bool recurse) {
    std::vector<fs::path> files;
    auto add_file = [&](const fs::directory_entry& entry) {
        if (entry.is_regular_file() && MatchesInputType(entry.path(), requested_type)) {
            files.push_back(entry.path());
        }
    };

    if (recurse) {
        for (const auto& entry : fs::recursive_directory_iterator(input_dir)) {
            add_file(entry);
        }
    } else {
        for (const auto& entry : fs::directory_iterator(input_dir)) {
            add_file(entry);
        }
    }

    std::sort(files.begin(), files.end(), [](const fs::path& lhs, const fs::path& rhs) {
        return ToLower(lhs.string()) < ToLower(rhs.string());
    });
    return files;
}

LoadedInput LoadInput(
    const fs::path& path,
    InputType type,
    int width,
    int height) {
    LoadedInput loaded;
    if (type == InputType::Bin) {
        if (width > std::numeric_limits<int>::max() / height) {
            throw std::runtime_error("BIN width * height is too large");
        }
        const size_t element_count =
            static_cast<size_t>(width) * static_cast<size_t>(height);
        if (element_count > std::numeric_limits<size_t>::max() / sizeof(float)) {
            throw std::runtime_error("BIN dimensions overflow");
        }
        const uintmax_t expected_bytes =
            static_cast<uintmax_t>(element_count * sizeof(float));
        const uintmax_t actual_bytes = fs::file_size(path);
        if (actual_bytes != expected_bytes) {
            std::ostringstream error;
            error << "BIN size mismatch: expected " << expected_bytes
                  << " bytes (" << width << "x" << height << " float32), got "
                  << actual_bytes;
            throw std::runtime_error(error.str());
        }

        loaded.float_data.resize(element_count);
        std::ifstream stream(path, std::ios::binary);
        if (!stream.read(
                reinterpret_cast<char*>(loaded.float_data.data()),
                static_cast<std::streamsize>(expected_bytes))) {
            throw std::runtime_error("Failed to read BIN file: " + path.string());
        }
        loaded.image = cv::Mat(
            height,
            width,
            CV_32FC1,
            loaded.float_data.data());
    } else {
        loaded.image = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
        if (loaded.image.empty()) {
            throw std::runtime_error("OpenCV failed to read image: " + path.string());
        }
    }
    return loaded;
}

cv::Mat ToDisplayBgr(const cv::Mat& source) {
    if (source.empty()) return {};

    cv::Mat normalized;
    if (source.depth() == CV_8U) {
        normalized = source.clone();
    } else {
        double min_value = 0.0;
        double max_value = 0.0;
        cv::minMaxLoc(source.reshape(1), &min_value, &max_value);
        if (max_value > min_value + 1e-12) {
            source.convertTo(
                normalized,
                CV_MAKETYPE(CV_8U, source.channels()),
                255.0 / (max_value - min_value),
                -min_value * 255.0 / (max_value - min_value));
        } else {
            normalized = cv::Mat::zeros(source.size(), CV_MAKETYPE(CV_8U, source.channels()));
        }
    }

    cv::Mat bgr;
    if (normalized.channels() == 1) {
        cv::cvtColor(normalized, bgr, cv::COLOR_GRAY2BGR);
    } else if (normalized.channels() == 3) {
        bgr = normalized;
    } else if (normalized.channels() == 4) {
        cv::cvtColor(normalized, bgr, cv::COLOR_BGRA2BGR);
    } else {
        throw std::runtime_error(
            "Unsupported input channel count: " + std::to_string(normalized.channels()));
    }
    return bgr;
}

void DrawMask(
    cv::Mat& image,
    const InspectionDLL::DetectionResult& detection,
    const InspectionDLL::MaskOverlayOptions& options) {
    if (detection.mask.empty() || options.alpha <= 0.0f) return;

    cv::Rect rect(
        static_cast<int>(std::round(detection.x)),
        static_cast<int>(std::round(detection.y)),
        static_cast<int>(std::round(detection.w)),
        static_cast<int>(std::round(detection.h)));
    rect &= cv::Rect(0, 0, image.cols, image.rows);
    if (rect.empty()) return;

    cv::Mat binary;
    detection.mask.convertTo(binary, CV_8U);
    cv::threshold(binary, binary, 0, 255, cv::THRESH_BINARY);

    cv::Mat full_mask(image.size(), CV_8U, cv::Scalar(0));
    if (binary.size() == image.size()) {
        binary.copyTo(full_mask);
    } else {
        cv::Mat local;
        cv::resize(binary, local, rect.size(), 0.0, 0.0, cv::INTER_NEAREST);
        local.copyTo(full_mask(rect));
    }

    cv::Mat color_layer(
        image.size(),
        image.type(),
        cv::Scalar(options.color_b, options.color_g, options.color_r));
    cv::Mat blended;
    cv::addWeighted(
        image,
        1.0 - options.alpha,
        color_layer,
        options.alpha,
        0.0,
        blended);
    blended.copyTo(image, full_mask);
}

bool SaveResultImage(
    const cv::Mat& source,
    const InspectionDLL::InferenceResult& result,
    const InspectionDLL::InspectionEngine& engine,
    const fs::path& output_path,
    std::string& error) {
    try {
        cv::Mat original = ToDisplayBgr(source);
        cv::Mat detected = original.clone();
        const auto mask_options = engine.GetMaskOverlayOptions();

        if (mask_options.enabled) {
            for (const auto& detail : result.details) {
                DrawMask(detected, detail, mask_options);
            }
        }

        for (const auto& detail : result.details) {
            cv::Rect rect(
                static_cast<int>(std::round(detail.x)),
                static_cast<int>(std::round(detail.y)),
                static_cast<int>(std::round(detail.w)),
                static_cast<int>(std::round(detail.h)));
            rect &= cv::Rect(0, 0, detected.cols, detected.rows);
            if (rect.empty()) continue;

            const cv::Scalar color = CategoryColor(detail.name);
            if (engine.ShouldDrawDefectBox()) {
                cv::rectangle(detected, rect, color, 2);
            }
            if (engine.ShouldDrawBoxDetails()) {
                std::ostringstream label;
                label << detail.name << " area=" << detail.area
                      << " contrast=" << std::fixed << std::setprecision(2)
                      << detail.contrast;
                const int text_y = std::max(18, rect.y);
                cv::putText(
                    detected,
                    label.str(),
                    cv::Point(rect.x, text_y),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    1,
                    cv::LINE_AA);
            }
        }

        cv::putText(
            detected,
            result.result.empty() ? "UNKNOWN" : result.result,
            cv::Point(6, 26),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            result.result == "OK" ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255),
            2,
            cv::LINE_AA);

        cv::Mat output;
        if (engine.ShouldConcatOriginalImage()) {
            cv::hconcat(original, detected, output);
        } else {
            output = detected;
        }

        fs::create_directories(output_path.parent_path());
        if (!cv::imwrite(output_path.string(), output)) {
            error = "cv::imwrite returned false";
            return false;
        }
        return true;
    } catch (const std::exception& ex) {
        error = ex.what();
        return false;
    }
}

fs::path MakeOutputImagePath(
    const fs::path& output_dir,
    const fs::path& input_path,
    const fs::path& input_root,
    bool batch_mode,
    int repeat_index,
    int repeat_count) {
    fs::path image_dir = output_dir / "detect_images";
    if (batch_mode) {
        const fs::path relative_dir =
            fs::relative(input_path.parent_path(), input_root);
        if (!relative_dir.empty() && relative_dir != ".") {
            image_dir /= relative_dir;
        }
    }

    std::string name = input_path.stem().string() + "_" +
                       InputTypeName(DetectInputType(input_path)) + "_detect";
    if (repeat_count > 1) {
        name += "_r" + std::to_string(repeat_index);
    }
    return image_dir / (name + ".png");
}

void PrintDetectionDetails(const InspectionDLL::InferenceResult& result) {
    const auto counts = CountByCategory(result);
    for (size_t category = 0; category < counts.size(); ++category) {
        std::cout << "  " << kCategoryNames[category]
                  << " count: " << counts[category] << '\n';
    }
    for (size_t i = 0; i < result.details.size(); ++i) {
        const auto& detail = result.details[i];
        std::cout << "    [" << i << "] name=" << detail.name
                  << ", area=" << detail.area
                  << ", contrast=" << std::fixed << std::setprecision(4)
                  << detail.contrast
                  << ", score=" << detail.score << '\n';
    }
}

FileResult ProcessOneFile(
    InspectionDLL::InspectionEngine& engine,
    const Options& options,
    const fs::path& file_path,
    const fs::path& output_dir,
    int repeat_index) {
    FileResult item;
    item.repeat_index = repeat_index;
    item.file_path = fs::absolute(file_path);
    const auto total_start = Clock::now();

    try {
        InputType actual_type = options.input_type;
        if (actual_type == InputType::Auto) {
            actual_type = DetectInputType(file_path);
        }
        if (actual_type == InputType::Auto) {
            throw std::runtime_error(
                "Unsupported input extension: " + file_path.extension().string());
        }

        LoadedInput loaded =
            LoadInput(file_path, actual_type, options.width, options.height);
        const auto inference_start = Clock::now();
        bool success = false;
        if (actual_type == InputType::Bin) {
            success = engine.ProcessFloatArry(
                loaded.float_data.data(),
                item.inference,
                options.width,
                options.height);
        } else {
            success = engine.ProcessImage(loaded.image, item.inference);
        }
        item.inference_ms = Milliseconds(inference_start, Clock::now());

        if (!success) {
            item.message = engine.GetLastError();
        } else {
            item.status = "Success";
            item.return_code = 0;
            if (options.save_images) {
                item.image_save_attempted = true;
                item.output_image_path = MakeOutputImagePath(
                    output_dir,
                    file_path,
                    options.input,
                    options.mode == RunMode::Batch,
                    repeat_index,
                    options.repeat_count);
                const auto save_start = Clock::now();
                std::string save_error;
                if (!SaveResultImage(
                        loaded.image,
                        item.inference,
                        engine,
                        item.output_image_path,
                        save_error)) {
                    item.output_image_path.clear();
                    item.message = "Image export failed: " + save_error;
                }
                item.save_ms = Milliseconds(save_start, Clock::now());
            }
        }
    } catch (const std::exception& ex) {
        item.message = ex.what();
    }

    item.total_ms = Milliseconds(total_start, Clock::now());
    return item;
}

void WriteSummaryCsv(const fs::path& path, const std::vector<FileResult>& results) {
    std::ofstream writer(path, std::ios::binary);
    if (!writer) {
        throw std::runtime_error("Failed to create summary CSV: " + path.string());
    }

    const unsigned char utf8_bom[] = {0xEF, 0xBB, 0xBF};
    writer.write(reinterpret_cast<const char*>(utf8_bom), sizeof(utf8_bom));
    writer
        << "RepeatIndex,FileName,FilePath,InputType,Status,ReturnCode,Result,"
           "AbnormalCount,GlueOverflowCount,DecolorizationCount,StainCount,StripesCount,"
           "BrightStripesCount,BrightClustersCount,LineArtifactsUnderscoreCount,LineArtifactsCount,"
           "TotalDetectCount,InferenceTimeMs,SaveTimeMs,"
           "TotalTimeMs,Message,OutputImagePath\n";

    for (const auto& item : results) {
        const auto counts = CountByCategory(item.inference);
        const int total_count =
            std::accumulate(counts.begin(), counts.end(), 0);
        writer << item.repeat_index << ','
               << Csv(item.file_path.filename().string()) << ','
               << Csv(item.file_path.string()) << ','
               << Csv(InputTypeName(DetectInputType(item.file_path))) << ','
               << Csv(item.status) << ','
               << item.return_code << ','
               << Csv(item.inference.result) << ','
               << counts[0] << ','
               << counts[1] << ','
               << counts[2] << ','
               << counts[3] << ','
               << counts[4] << ','
               << counts[5] << ','
               << counts[6] << ','
               << counts[7] << ','
               << counts[8] << ','
               << total_count << ','
               << std::fixed << std::setprecision(3)
               << item.inference_ms << ',';
        if (item.image_save_attempted) {
            writer << item.save_ms;
        }
        writer << ',' << item.total_ms << ','
               << Csv(item.message) << ','
               << Csv(item.output_image_path.string()) << '\n';
    }
}

void PrintUsage() {
    std::cout
        << "Usage:\n"
        << "  use_examples --mode single --input <file> [options]\n"
        << "  use_examples --mode batch  --input <folder> [options]\n\n"
        << "Options:\n"
        << "  --mode <single|batch>       Process one file or a directory\n"
        << "  --input <path>              Input file or directory\n"
        << "  --input-type <type>         auto, png, tiff, or bin (default: auto)\n"
        << "  --width <int>               BIN image width (default: 1000)\n"
        << "  --height <int>              BIN image height (default: 800)\n"
        << "  --repeat <int>              Repeat all selected inputs (default: 1)\n"
        << "  --config <path>             Config file (default: config.ini)\n"
        << "  --output-dir <path>         Images and summary.csv output directory\n"
        << "  --recurse                   Search batch subdirectories\n"
        << "  --save-images               Save detection PNGs (default)\n"
        << "  --no-save-images            Do not save detection PNGs\n"
        << "  --help, -h                  Show this help\n\n"
        << "Examples:\n"
        << "  use_examples --mode single --input test.png --input-type png --config config.ini\n"
        << "  use_examples --mode single --input test.tiff --input-type tiff --config config.ini\n"
        << "  use_examples --mode single --input test.bin --input-type bin --width 1000 --height 800 --config config.ini\n"
        << "  use_examples --mode batch --input E:\\\\images --input-type auto --output-dir E:\\\\out --recurse --config config.ini\n";
}

int Run(const Options& original_options) {
    Options options = original_options;
    options.config_path = ResolveConfigPath(options.config_path);
    const bool batch_mode = options.mode == RunMode::Batch;
    options.input = ResolveExistingPath(options.input, batch_mode);

    std::vector<fs::path> files;
    if (batch_mode) {
        files = FindInputFiles(options.input, options.input_type, options.recurse);
    } else {
        if (!MatchesInputType(options.input, options.input_type)) {
            throw std::runtime_error(
                "Input file type does not match --input-type: " +
                options.input.string());
        }
        files.push_back(options.input);
    }

    if (files.empty()) {
        throw std::runtime_error(
            "No matching png, tiff, or bin files found under: " +
            options.input.string());
    }

    if (options.output_dir.empty()) {
        options.output_dir =
            (batch_mode ? options.input : options.input.parent_path()) /
            "inspection_output";
    } else {
        options.output_dir = fs::absolute(options.output_dir);
    }
    fs::create_directories(options.output_dir);

    std::cout << "Mode       : " << (batch_mode ? "batch" : "single") << '\n';
    std::cout << "Input      : " << options.input.string() << '\n';
    std::cout << "InputType  : " << InputTypeName(options.input_type) << '\n';
    std::cout << "FileCount  : " << files.size() << '\n';
    std::cout << "RepeatCount: " << options.repeat_count << '\n';
    std::cout << "TotalRuns  : "
              << static_cast<uint64_t>(files.size()) *
                     static_cast<uint64_t>(options.repeat_count)
              << '\n';
    std::cout << "OutputDir  : " << options.output_dir.string() << '\n';
    std::cout << "SaveImages : " << (options.save_images ? "true" : "false") << '\n';
    if (options.input_type == InputType::Bin ||
        (options.input_type == InputType::Auto &&
         std::any_of(files.begin(), files.end(), [](const fs::path& path) {
             return DetectInputType(path) == InputType::Bin;
         }))) {
        std::cout << "BinShape   : " << options.width << 'x' << options.height << '\n';
    }
    std::cout << "InitMode   : single engine + single initialize\n\n";

    InspectionDLL::InspectionEngine engine;
    if (!engine.Initialize(options.config_path.string())) {
        std::cerr << "InspectionEngine::Initialize failed: "
                  << engine.GetLastError() << '\n';
        return 4;
    }

    std::vector<FileResult> results;
    results.reserve(files.size() * static_cast<size_t>(options.repeat_count));
    const size_t total_runs =
        files.size() * static_cast<size_t>(options.repeat_count);
    size_t completed = 0;

    for (int repeat = 1; repeat <= options.repeat_count; ++repeat) {
        std::cout << "===== Repeat " << repeat << '/'
                  << options.repeat_count << " =====\n";
        for (size_t i = 0; i < files.size(); ++i) {
            ++completed;
            std::cout << '[' << completed << '/' << total_runs << "] File "
                      << (i + 1) << '/' << files.size() << ": "
                      << files[i].string() << '\n';

            FileResult item = ProcessOneFile(
                engine,
                options,
                files[i],
                options.output_dir,
                repeat);
            const auto counts = CountByCategory(item.inference);
            const int total_count =
                std::accumulate(counts.begin(), counts.end(), 0);

            std::cout << "  Status       : " << item.status << '\n';
            std::cout << "  ReturnCode   : " << item.return_code << '\n';
            std::cout << "  Result       : " << item.inference.result << '\n';
            std::cout << "  TotalDetect  : " << total_count << '\n';
            std::cout << "  InferenceTime: " << std::fixed
                      << std::setprecision(3) << item.inference_ms << " ms\n";
            std::cout << "  SaveTime     : ";
            if (item.image_save_attempted) {
                std::cout << item.save_ms << " ms\n";
            } else {
                std::cout << "N/A\n";
            }
            std::cout << "  TotalTime    : " << item.total_ms << " ms\n";
            if (!item.message.empty()) {
                std::cout << "  Message      : " << item.message << '\n';
            }
            if (!item.output_image_path.empty()) {
                std::cout << "  OutputImage  : "
                          << item.output_image_path.string() << '\n';
            }
            PrintDetectionDetails(item.inference);
            std::cout << '\n';
            results.push_back(std::move(item));
        }
    }

    const fs::path summary_path = options.output_dir / "summary.csv";
    WriteSummaryCsv(summary_path, results);
    const size_t failed = static_cast<size_t>(std::count_if(
        results.begin(),
        results.end(),
        [](const FileResult& item) { return item.return_code != 0; }));

    std::cout << "Processing completed.\n";
    std::cout << "Success    : " << results.size() - failed << '\n';
    std::cout << "Failed     : " << failed << '\n';
    std::cout << "SummaryCsv : " << summary_path.string() << '\n';
    return failed == 0 ? 0 : 5;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseArgs(argc, argv);
        return Run(options);
    } catch (const std::exception& ex) {
        if (std::string(ex.what()) != "help") {
            std::cerr << ex.what() << "\n\n";
        }
        PrintUsage();
        return std::string(ex.what()) == "help" ? 0 : 2;
    }
}
