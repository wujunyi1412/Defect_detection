using System.Runtime.InteropServices;

namespace CoronaDetection;

internal static class EvaluationMetricsNative
{
    private const string LibraryName = "evaluation_metrics.dll";
    private const int StatusOk = 0;

    [StructLayout(LayoutKind.Sequential)]
    private struct NativeBox(double x, double y, double width, double height)
    {
        public double X = x;
        public double Y = y;
        public double Width = width;
        public double Height = height;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct Match
    {
        public int GroundTruthIndex;
        public int PredictionIndex;
        public double Iou;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MetricInput
    {
        public int GroundTruthCount;
        public int PredictionCount;
        public int TruePositiveCount;
        public int MissedCount;
        public int FalsePositiveCount;
        public int MisclassifiedCount;
        public double MatchedIouSum;
        public int MatchedIouCount;
        public double UndefinedPrecisionValue;
        public double UndefinedRecallValue;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct MetricResult
    {
        public double Precision;
        public double Recall;
        public double F1;
        public double LocalizationRecall;
        public double MatchedClassAccuracy;
        public double MeanIou;
    }

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int EvaluationMetrics_MatchBoxes(
        [In] NativeBox[] groundTruth,
        int groundTruthCount,
        [In] NativeBox[] predictions,
        int predictionCount,
        double iouThreshold,
        [Out] Match[] outputMatches,
        int outputCapacity,
        out int outputCount);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int EvaluationMetrics_Calculate(
        in MetricInput input,
        out MetricResult output);

    internal static Match[] MatchBoxes(
        IReadOnlyList<BoundingBox> groundTruth,
        IReadOnlyList<BoundingBox> predictions,
        double iouThreshold)
    {
        if (groundTruth.Count == 0 || predictions.Count == 0) return [];
        NativeBox[] gt = groundTruth.Select(ToNative).ToArray();
        NativeBox[] pred = predictions.Select(ToNative).ToArray();
        var matches = new Match[Math.Min(gt.Length, pred.Length)];
        int status = EvaluationMetrics_MatchBoxes(
            gt, gt.Length, pred, pred.Length, iouThreshold,
            matches, matches.Length, out int matchCount);
        EnsureSuccess(status, nameof(EvaluationMetrics_MatchBoxes));
        return matchCount == matches.Length ? matches : matches[..matchCount];
    }

    internal static MetricResult Calculate(
        int groundTruthCount,
        int predictionCount,
        int truePositiveCount,
        int missedCount,
        int falsePositiveCount,
        int misclassifiedCount,
        double matchedIouSum,
        int matchedIouCount,
        double undefinedPrecisionValue = 0.0,
        double undefinedRecallValue = 0.0)
    {
        var input = new MetricInput
        {
            GroundTruthCount = groundTruthCount,
            PredictionCount = predictionCount,
            TruePositiveCount = truePositiveCount,
            MissedCount = missedCount,
            FalsePositiveCount = falsePositiveCount,
            MisclassifiedCount = misclassifiedCount,
            MatchedIouSum = matchedIouSum,
            MatchedIouCount = matchedIouCount,
            UndefinedPrecisionValue = undefinedPrecisionValue,
            UndefinedRecallValue = undefinedRecallValue
        };
        int status = EvaluationMetrics_Calculate(in input, out MetricResult result);
        EnsureSuccess(status, nameof(EvaluationMetrics_Calculate));
        return result;
    }

    private static NativeBox ToNative(BoundingBox box) =>
        new(box.X, box.Y, box.Width, box.Height);

    private static void EnsureSuccess(int status, string operation)
    {
        if (status != StatusOk)
            throw new InvalidOperationException($"C++ 评价指标模块调用失败：{operation}，状态码 {status}。");
    }
}
