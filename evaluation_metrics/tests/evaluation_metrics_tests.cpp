#include "evaluation_metrics/evaluation_metrics.h"
#include "evaluation_metrics/evaluation_metrics_c_api.h"

#include <cassert>
#include <cmath>
#include <iostream>

namespace {
bool near(double left, double right) { return std::abs(left - right) < 1e-9; }
}

int main() {
    using namespace evaluation_metrics;

    assert(near(calculate_iou({0, 0, 10, 10}, {5, 5, 10, 10}), 25.0 / 175.0));
    assert(near(calculate_iou({0, 0, 0, 10}, {0, 0, 10, 10}), 0.0));

    const auto matches = match_boxes(
        {{0, 0, 10, 10}, {20, 20, 10, 10}},
        {{20, 20, 10, 10}, {0, 0, 10, 10}}, 0.5);
    assert(matches.size() == 2);
    assert(matches[0].ground_truth_index == 0 && matches[0].prediction_index == 1);
    assert(matches[1].ground_truth_index == 1 && matches[1].prediction_index == 0);

    const auto metrics = calculate_detection_metrics({10, 9, 7, 2, 1, 1, 7.2, 8, 0.0, 0.0});
    assert(near(metrics.precision, 7.0 / 9.0));
    assert(near(metrics.recall, 7.0 / 10.0));
    assert(near(metrics.localization_recall, 0.8));
    assert(near(metrics.matched_class_accuracy, 7.0 / 8.0));
    assert(near(metrics.mean_iou, 0.9));

    const auto empty = calculate_detection_metrics({0, 0, 0, 0, 0, 0, 0.0, 0, 1.0, 1.0});
    assert(near(empty.precision, 1.0));
    assert(near(empty.recall, 1.0));
    assert(near(empty.f1, 1.0));

    const EvaluationMetricsBox c_gt[] = {{0, 0, 10, 10}};
    const EvaluationMetricsBox c_pred[] = {{0, 0, 10, 10}};
    EvaluationMetricsMatch c_match{};
    int c_match_count = 0;
    assert(EvaluationMetrics_MatchBoxes(
        c_gt, 1, c_pred, 1, 0.5, &c_match, 1, &c_match_count) == EVALUATION_METRICS_OK);
    assert(c_match_count == 1 && near(c_match.iou, 1.0));
    const EvaluationMetricsInput c_input = {1, 1, 1, 0, 0, 0, 1.0, 1, 0.0, 0.0};
    EvaluationMetricsResult c_result{};
    assert(EvaluationMetrics_Calculate(&c_input, &c_result) == EVALUATION_METRICS_OK);
    assert(near(c_result.f1, 1.0) && near(c_result.mean_iou, 1.0));

    std::cout << "evaluation_metrics_tests passed\n";
    return 0;
}
