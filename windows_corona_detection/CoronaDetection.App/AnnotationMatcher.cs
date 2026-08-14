namespace CoronaDetection;

// The matching algorithm lives in the reusable C++ evaluation_metrics library.
// This class only maps domain models to/from the native ABI.
internal static class AnnotationMatcher
{
    public static IReadOnlyList<AnnotationMatch> Match(
        IReadOnlyList<AnnotationBox> groundTruth,
        IReadOnlyList<AnnotationBox> predictions,
        double iouThreshold)
    {
        EvaluationMetricsNative.Match[] nativeMatches = EvaluationMetricsNative.MatchBoxes(
            groundTruth.Select(annotation => annotation.Box).ToArray(),
            predictions.Select(annotation => annotation.Box).ToArray(),
            iouThreshold);
        return nativeMatches.Select(match =>
        {
            AnnotationBox gt = groundTruth[match.GroundTruthIndex];
            AnnotationBox prediction = predictions[match.PredictionIndex];
            return new AnnotationMatch(
                gt,
                prediction,
                match.Iou,
                gt.Label.Equals(prediction.Label, StringComparison.OrdinalIgnoreCase));
        }).ToList();
    }
}
