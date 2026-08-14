#pragma once

#include <stddef.h>

#if defined(_WIN32)
#  if defined(EVALUATION_METRICS_EXPORTS)
#    define EVALUATION_METRICS_API __declspec(dllexport)
#  else
#    define EVALUATION_METRICS_API __declspec(dllimport)
#  endif
#  define EVALUATION_METRICS_CALL __cdecl
#else
#  define EVALUATION_METRICS_API
#  define EVALUATION_METRICS_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum EvaluationMetricsStatus {
    EVALUATION_METRICS_OK = 0,
    EVALUATION_METRICS_INVALID_ARGUMENT = 1,
    EVALUATION_METRICS_OUTPUT_TOO_SMALL = 2,
    EVALUATION_METRICS_INTERNAL_ERROR = 3
};

typedef struct EvaluationMetricsBox {
    double x;
    double y;
    double width;
    double height;
} EvaluationMetricsBox;

typedef struct EvaluationMetricsMatch {
    int ground_truth_index;
    int prediction_index;
    double iou;
} EvaluationMetricsMatch;

typedef struct EvaluationMetricsInput {
    int ground_truth_count;
    int prediction_count;
    int true_positive_count;
    int missed_count;
    int false_positive_count;
    int misclassified_count;
    double matched_iou_sum;
    int matched_iou_count;
    double undefined_precision_value;
    double undefined_recall_value;
} EvaluationMetricsInput;

typedef struct EvaluationMetricsResult {
    double precision;
    double recall;
    double f1;
    double localization_recall;
    double matched_class_accuracy;
    double mean_iou;
} EvaluationMetricsResult;

EVALUATION_METRICS_API int EVALUATION_METRICS_CALL EvaluationMetrics_MatchBoxes(
    const EvaluationMetricsBox* ground_truth,
    int ground_truth_count,
    const EvaluationMetricsBox* predictions,
    int prediction_count,
    double iou_threshold,
    EvaluationMetricsMatch* output_matches,
    int output_capacity,
    int* output_count);

EVALUATION_METRICS_API int EVALUATION_METRICS_CALL EvaluationMetrics_Calculate(
    const EvaluationMetricsInput* input,
    EvaluationMetricsResult* output);

EVALUATION_METRICS_API const char* EVALUATION_METRICS_CALL EvaluationMetrics_Version(void);

#ifdef __cplusplus
}
#endif

