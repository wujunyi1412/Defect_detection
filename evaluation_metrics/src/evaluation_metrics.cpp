#include "evaluation_metrics/evaluation_metrics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace evaluation_metrics {
namespace {

double ratio(int numerator, int denominator, double undefined_value = 0.0) noexcept {
    return denominator == 0
        ? undefined_value
        : static_cast<double>(numerator) / static_cast<double>(denominator);
}

std::vector<int> maximize_assignment(
    const std::vector<std::vector<double>>& weights,
    double maximum_weight) {
    const int size = static_cast<int>(weights.size());
    std::vector<double> u(size + 1);
    std::vector<double> v(size + 1);
    std::vector<int> p(size + 1);
    std::vector<int> way(size + 1);

    for (int row = 1; row <= size; ++row) {
        p[0] = row;
        int column0 = 0;
        std::vector<double> minimum(size + 1, std::numeric_limits<double>::infinity());
        std::vector<bool> used(size + 1);
        do {
            used[column0] = true;
            const int current_row = p[column0];
            double delta = std::numeric_limits<double>::infinity();
            int column1 = 0;
            for (int column = 1; column <= size; ++column) {
                if (used[column]) continue;
                const double cost = maximum_weight - weights[current_row - 1][column - 1];
                const double current = cost - u[current_row] - v[column];
                if (current < minimum[column]) {
                    minimum[column] = current;
                    way[column] = column0;
                }
                if (minimum[column] < delta) {
                    delta = minimum[column];
                    column1 = column;
                }
            }
            for (int column = 0; column <= size; ++column) {
                if (used[column]) {
                    u[p[column]] += delta;
                    v[column] -= delta;
                } else {
                    minimum[column] -= delta;
                }
            }
            column0 = column1;
        } while (p[column0] != 0);

        do {
            const int column1 = way[column0];
            p[column0] = p[column1];
            column0 = column1;
        } while (column0 != 0);
    }

    std::vector<int> assignment(size, -1);
    for (int column = 1; column <= size; ++column) {
        if (p[column] > 0) assignment[p[column] - 1] = column - 1;
    }
    return assignment;
}

void validate_non_negative(int value, const char* name) {
    if (value < 0) throw std::invalid_argument(name);
}

}  // namespace

double calculate_iou(const Box& left, const Box& right) noexcept {
    if (left.width <= 0.0 || left.height <= 0.0 ||
        right.width <= 0.0 || right.height <= 0.0) return 0.0;
    const double intersection_width = std::max(
        0.0, std::min(left.x + left.width, right.x + right.width) - std::max(left.x, right.x));
    const double intersection_height = std::max(
        0.0, std::min(left.y + left.height, right.y + right.height) - std::max(left.y, right.y));
    const double intersection = intersection_width * intersection_height;
    const double union_area = left.width * left.height + right.width * right.height - intersection;
    return union_area > 0.0 ? intersection / union_area : 0.0;
}

std::vector<Match> match_boxes(
    const std::vector<Box>& ground_truth,
    const std::vector<Box>& predictions,
    double iou_threshold) {
    if (!(iou_threshold > 0.0 && iou_threshold <= 1.0))
        throw std::invalid_argument("iou_threshold");
    if (ground_truth.empty() || predictions.empty()) return {};

    const int size = static_cast<int>(std::max(ground_truth.size(), predictions.size()));
    const double cardinality_bonus = static_cast<double>(size) + 1.0;
    std::vector<std::vector<double>> weights(size, std::vector<double>(size));
    std::vector<std::vector<double>> ious(
        ground_truth.size(), std::vector<double>(predictions.size()));
    for (size_t gt = 0; gt < ground_truth.size(); ++gt) {
        for (size_t prediction = 0; prediction < predictions.size(); ++prediction) {
            const double iou = calculate_iou(ground_truth[gt], predictions[prediction]);
            ious[gt][prediction] = iou;
            if (iou >= iou_threshold) weights[gt][prediction] = cardinality_bonus + iou;
        }
    }

    const auto assignment = maximize_assignment(weights, cardinality_bonus + 1.0);
    std::vector<Match> matches;
    matches.reserve(std::min(ground_truth.size(), predictions.size()));
    for (size_t gt = 0; gt < ground_truth.size(); ++gt) {
        const int prediction = assignment[gt];
        if (prediction < 0 || prediction >= static_cast<int>(predictions.size()) ||
            ious[gt][prediction] < iou_threshold) continue;
        matches.push_back({static_cast<int>(gt), prediction, ious[gt][prediction]});
    }
    return matches;
}

DetectionMetrics calculate_detection_metrics(const DetectionMetricInput& input) {
    validate_non_negative(input.ground_truth_count, "ground_truth_count");
    validate_non_negative(input.prediction_count, "prediction_count");
    validate_non_negative(input.true_positive_count, "true_positive_count");
    validate_non_negative(input.missed_count, "missed_count");
    validate_non_negative(input.false_positive_count, "false_positive_count");
    validate_non_negative(input.misclassified_count, "misclassified_count");
    validate_non_negative(input.matched_iou_count, "matched_iou_count");
    if (!std::isfinite(input.matched_iou_sum) || input.matched_iou_sum < 0.0)
        throw std::invalid_argument("matched_iou_sum");

    DetectionMetrics result;
    const int precision_denominator = input.true_positive_count +
        input.false_positive_count + input.misclassified_count;
    const int recall_denominator = input.true_positive_count +
        input.missed_count + input.misclassified_count;
    result.precision = ratio(
        input.true_positive_count, precision_denominator, input.undefined_precision_value);
    result.recall = ratio(
        input.true_positive_count, recall_denominator, input.undefined_recall_value);
    result.f1 = result.precision + result.recall == 0.0
        ? 0.0
        : 2.0 * result.precision * result.recall / (result.precision + result.recall);
    const int localized = input.true_positive_count + input.misclassified_count;
    result.localization_recall = ratio(localized, input.ground_truth_count);
    result.matched_class_accuracy = ratio(input.true_positive_count, localized);
    result.mean_iou = input.matched_iou_count == 0
        ? 0.0
        : input.matched_iou_sum / static_cast<double>(input.matched_iou_count);
    return result;
}

}  // namespace evaluation_metrics

