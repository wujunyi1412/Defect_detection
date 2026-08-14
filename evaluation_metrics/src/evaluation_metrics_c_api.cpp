#include "evaluation_metrics/evaluation_metrics_c_api.h"
#include "evaluation_metrics/evaluation_metrics.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <vector>

using evaluation_metrics::Box;

extern "C" {

int EVALUATION_METRICS_CALL EvaluationMetrics_MatchBoxes(
    const EvaluationMetricsBox* ground_truth,
    int ground_truth_count,
    const EvaluationMetricsBox* predictions,
    int prediction_count,
    double iou_threshold,
    EvaluationMetricsMatch* output_matches,
    int output_capacity,
    int* output_count) {
    if (!output_count || ground_truth_count < 0 || prediction_count < 0 || output_capacity < 0 ||
        (ground_truth_count > 0 && !ground_truth) ||
        (prediction_count > 0 && !predictions) ||
        (output_capacity > 0 && !output_matches)) return EVALUATION_METRICS_INVALID_ARGUMENT;
    try {
        std::vector<Box> gt;
        std::vector<Box> pred;
        gt.reserve(static_cast<size_t>(ground_truth_count));
        pred.reserve(static_cast<size_t>(prediction_count));
        for (int index = 0; index < ground_truth_count; ++index)
            gt.push_back({ground_truth[index].x, ground_truth[index].y,
                          ground_truth[index].width, ground_truth[index].height});
        for (int index = 0; index < prediction_count; ++index)
            pred.push_back({predictions[index].x, predictions[index].y,
                            predictions[index].width, predictions[index].height});
        const auto matches = evaluation_metrics::match_boxes(gt, pred, iou_threshold);
        *output_count = static_cast<int>(matches.size());
        if (output_capacity < *output_count) return EVALUATION_METRICS_OUTPUT_TOO_SMALL;
        for (size_t index = 0; index < matches.size(); ++index) {
            output_matches[index] = {
                matches[index].ground_truth_index,
                matches[index].prediction_index,
                matches[index].iou};
        }
        return EVALUATION_METRICS_OK;
    } catch (const std::invalid_argument&) {
        return EVALUATION_METRICS_INVALID_ARGUMENT;
    } catch (...) {
        return EVALUATION_METRICS_INTERNAL_ERROR;
    }
}

int EVALUATION_METRICS_CALL EvaluationMetrics_Calculate(
    const EvaluationMetricsInput* input,
    EvaluationMetricsResult* output) {
    if (!input || !output) return EVALUATION_METRICS_INVALID_ARGUMENT;
    try {
        const auto result = evaluation_metrics::calculate_detection_metrics({
            input->ground_truth_count,
            input->prediction_count,
            input->true_positive_count,
            input->missed_count,
            input->false_positive_count,
            input->misclassified_count,
            input->matched_iou_sum,
            input->matched_iou_count,
            input->undefined_precision_value,
            input->undefined_recall_value});
        *output = {result.precision, result.recall, result.f1,
                   result.localization_recall, result.matched_class_accuracy, result.mean_iou};
        return EVALUATION_METRICS_OK;
    } catch (const std::invalid_argument&) {
        return EVALUATION_METRICS_INVALID_ARGUMENT;
    } catch (...) {
        return EVALUATION_METRICS_INTERNAL_ERROR;
    }
}

const char* EVALUATION_METRICS_CALL EvaluationMetrics_Version(void) {
    return "1.0.0";
}

}  // extern "C"
