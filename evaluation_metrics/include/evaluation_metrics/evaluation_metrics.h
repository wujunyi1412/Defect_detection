#pragma once

#include <vector>

namespace evaluation_metrics {

struct Box {
    double x{};
    double y{};
    double width{};
    double height{};
};

struct Match {
    int ground_truth_index{-1};
    int prediction_index{-1};
    double iou{};
};

struct DetectionMetricInput {
    int ground_truth_count{};
    int prediction_count{};
    int true_positive_count{};
    int missed_count{};
    int false_positive_count{};
    int misclassified_count{};
    double matched_iou_sum{};
    int matched_iou_count{};
    double undefined_precision_value{};
    double undefined_recall_value{};
};

struct DetectionMetrics {
    double precision{};
    double recall{};
    double f1{};
    double localization_recall{};
    double matched_class_accuracy{};
    double mean_iou{};
};

double calculate_iou(const Box& left, const Box& right) noexcept;

// Maximizes qualified match count first, then the sum of IoUs.
std::vector<Match> match_boxes(
    const std::vector<Box>& ground_truth,
    const std::vector<Box>& predictions,
    double iou_threshold);

DetectionMetrics calculate_detection_metrics(const DetectionMetricInput& input);

}  // namespace evaluation_metrics

