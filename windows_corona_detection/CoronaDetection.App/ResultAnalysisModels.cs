using System.IO;

namespace CoronaDetection;

internal readonly record struct BoundingBox(double X, double Y, double Width, double Height)
{
    public double Right => X + Width;
    public double Bottom => Y + Height;
    public bool IsValid => Width > 0.0 && Height > 0.0;
}

internal sealed record AnnotationBox(string Label, BoundingBox Box);

internal sealed record AnnotationReadResult(
    IReadOnlyList<AnnotationBox> Annotations,
    IReadOnlyList<string> Warnings);

internal sealed record AnnotationMatch(
    AnnotationBox GroundTruth,
    AnnotationBox Prediction,
    double Iou,
    bool ClassCorrect);

internal enum AnalysisErrorType
{
    Missed,
    FalsePositive,
    Misclassified
}

internal sealed record AnalysisError(
    AnalysisErrorType Type,
    string ActualClass,
    string PredictedClass,
    BoundingBox Box,
    double Iou,
    string DisplayText);

internal sealed record AnalyzedImage(
    string Prefix,
    string ImagePath,
    string ImageName,
    IReadOnlyList<AnnotationBox> GroundTruth,
    IReadOnlyList<AnnotationBox> Predictions,
    IReadOnlyList<AnnotationMatch> Matches,
    IReadOnlyList<AnalysisError> Errors,
    IReadOnlyList<string> Warnings);

public sealed record AnalysisImageRow(
    int Index,
    string ImageName,
    int GroundTruthCount,
    int PredictionCount,
    int TruePositiveCount,
    int MissedCount,
    int FalsePositiveCount,
    int MisclassifiedCount,
    double Precision,
    double Recall,
    double F1,
    double MeanIou,
    string VisualizationPath,
    string Status);

internal sealed record ResultAnalysisOptions(
    double IouThreshold = 0.5,
    bool Recursive = true,
    bool OverwriteOutputs = false,
    PredictionLabelRules? PredictionRules = null,
    bool EvaluateByClass = true);

internal sealed class PredictionLabelRules(
    IEnumerable<string>? ignoredLabels = null,
    IEnumerable<KeyValuePair<string, string>>? labelMappings = null)
{
    private readonly HashSet<string> _ignoredLabels =
        (ignoredLabels ?? []).ToHashSet(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string> _labelMappings =
        (labelMappings ?? []).ToDictionary(
            pair => pair.Key,
            pair => pair.Value,
            StringComparer.OrdinalIgnoreCase);

    public IReadOnlyCollection<string> IgnoredLabels => _ignoredLabels;
    public IReadOnlyDictionary<string, string> LabelMappings => _labelMappings;

    public IReadOnlyList<AnnotationBox> Apply(IReadOnlyList<AnnotationBox> predictions) =>
        predictions
            .Where(prediction => !_ignoredLabels.Contains(prediction.Label))
            .Select(prediction => _labelMappings.TryGetValue(prediction.Label, out string? mapped)
                ? prediction with { Label = mapped }
                : prediction)
            .ToList();

    public static PredictionLabelRules FromUiOptions(
        bool ignoreAbnormal,
        bool mapDarkClustersToStain)
    {
        string[] ignored = ignoreAbnormal ? ["Abnormal"] : [];
        KeyValuePair<string, string>[] mappings = mapDarkClustersToStain
            ? [new KeyValuePair<string, string>("DarkClusters", "Stain")]
            : [];
        return new PredictionLabelRules(ignored, mappings);
    }
}

internal sealed record ResultAnalysisProgress(
    int Completed,
    int Total,
    AnalysisImageRow Row);

internal sealed record ResultAnalysisSummary(
    int ImageCount,
    int ErrorImageCount,
    int TruePositiveCount,
    int MissedCount,
    int FalsePositiveCount,
    int MisclassifiedCount,
    double Precision,
    double Recall,
    double F1,
    double MeanIou,
    IReadOnlyList<AnalysisImageRow> Rows);

internal sealed class AnalysisOutputConflictException(IReadOnlyList<string> conflictingPaths)
    : IOException("分析输出目录中存在同名结果文件。")
{
    public IReadOnlyList<string> ConflictingPaths { get; } = conflictingPaths;
}

internal sealed class DuplicateDatasetPrefixException(
    string datasetName,
    IReadOnlyList<string> prefixes)
    : InvalidOperationException($"{datasetName}中存在重名文件前缀，无法建立唯一对应关系。")
{
    public string DatasetName { get; } = datasetName;
    public IReadOnlyList<string> Prefixes { get; } = prefixes;
}
