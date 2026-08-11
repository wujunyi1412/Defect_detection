using System.Globalization;
using System.IO;
using System.Text;

namespace CoronaDetection;

internal sealed class ResultAnalysisService
{
    private static readonly HashSet<string> ImageExtensions =
        DetectionFiles.FormatExtensions.Values
            .SelectMany(extensions => extensions)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

    public async Task<ResultAnalysisSummary> AnalyzeAsync(
        string imageDirectory,
        string predictionDirectory,
        string groundTruthDirectory,
        string outputDirectory,
        ResultAnalysisOptions options,
        IProgress<ResultAnalysisProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        return await Task.Run(
            () => Analyze(
                imageDirectory, predictionDirectory, groundTruthDirectory,
                outputDirectory, options, progress, cancellationToken),
            cancellationToken);
    }

    private static ResultAnalysisSummary Analyze(
        string imageDirectory,
        string predictionDirectory,
        string groundTruthDirectory,
        string outputDirectory,
        ResultAnalysisOptions options,
        IProgress<ResultAnalysisProgress>? progress,
        CancellationToken cancellationToken)
    {
        ValidateInputs(
            imageDirectory, predictionDirectory, groundTruthDirectory,
            outputDirectory, options);
        SearchOption searchOption = options.Recursive
            ? SearchOption.AllDirectories
            : SearchOption.TopDirectoryOnly;
        Dictionary<string, string> images = BuildUniquePrefixMap(
            Directory.EnumerateFiles(imageDirectory, "*", searchOption)
                .Where(path => ImageExtensions.Contains(Path.GetExtension(path))),
            "原始图片文件夹");
        if (images.Count == 0)
            throw new InvalidOperationException("原始图片文件夹中没有找到支持的图片。");
        Dictionary<string, string> predictions = BuildUniquePrefixMap(
            Directory.EnumerateFiles(predictionDirectory, "*.json", searchOption),
            "预测 JSON 文件夹");
        Dictionary<string, string> groundTruth = BuildUniquePrefixMap(
            Directory.EnumerateFiles(groundTruthDirectory, "*.json", searchOption),
            "GT JSON 文件夹");

        string visualizationDirectory = Path.Combine(outputDirectory, "visualizations");
        string[] reportPaths =
        [
            Path.Combine(outputDirectory, "image_summary.csv"),
            Path.Combine(outputDirectory, "error_details.csv"),
            Path.Combine(outputDirectory, "overall_metrics.csv"),
            Path.Combine(outputDirectory, "class_metrics.csv"),
            Path.Combine(outputDirectory, "confusion_matrix.csv")
        ];
        string[] visualizationPaths = images.Keys
            .Select(prefix => Path.Combine(visualizationDirectory, prefix + ".png"))
            .ToArray();
        EnsureOutputsAvailable(
            reportPaths.Concat(visualizationPaths), options.OverwriteOutputs);

        var analyzedImages = new List<AnalyzedImage>(images.Count);
        foreach ((string prefix, string imagePath) in images)
        {
            cancellationToken.ThrowIfCancellationRequested();
            predictions.TryGetValue(prefix, out string? predictionPath);
            groundTruth.TryGetValue(prefix, out string? groundTruthPath);
            AnnotationReadResult gt = LabelMeAnnotationReader.Read(groundTruthPath);
            AnnotationReadResult predRaw = LabelMeAnnotationReader.Read(predictionPath);
            IReadOnlyList<AnnotationBox> normalizedPredictions =
                (options.PredictionRules ?? new PredictionLabelRules()).Apply(predRaw.Annotations);
            IReadOnlyList<AnnotationMatch> matches = AnnotationMatcher.Match(
                gt.Annotations, normalizedPredictions, options.IouThreshold);

            var errors = new List<AnalysisError>();
            foreach (AnnotationMatch match in matches.Where(match => !match.ClassCorrect))
            {
                errors.Add(new AnalysisError(
                    AnalysisErrorType.Misclassified,
                    match.GroundTruth.Label,
                    match.Prediction.Label,
                    match.Prediction.Box,
                    match.Iou,
                    $"{match.GroundTruth.Label} 误检成 {match.Prediction.Label}"));
            }
            foreach (AnnotationBox annotation in gt.Annotations.Where(
                         annotation => !matches.Any(match => ReferenceEquals(match.GroundTruth, annotation))))
            {
                errors.Add(new AnalysisError(
                    AnalysisErrorType.Missed,
                    annotation.Label,
                    string.Empty,
                    annotation.Box,
                    0.0,
                    $"{annotation.Label} 漏检"));
            }
            foreach (AnnotationBox annotation in normalizedPredictions.Where(
                         annotation => !matches.Any(match => ReferenceEquals(match.Prediction, annotation))))
            {
                errors.Add(new AnalysisError(
                    AnalysisErrorType.FalsePositive,
                    string.Empty,
                    annotation.Label,
                    annotation.Box,
                    0.0,
                    $"{annotation.Label} 误检"));
            }

            var warnings = gt.Warnings.Select(warning => "GT: " + warning)
                .Concat(predRaw.Warnings.Select(warning => "预测: " + warning))
                .ToList();
            if (groundTruthPath is null) warnings.Add("缺少 GT JSON，按空标注处理");
            if (predictionPath is null) warnings.Add("缺少预测 JSON，按空标注处理");
            analyzedImages.Add(new AnalyzedImage(
                prefix,
                imagePath,
                Path.GetFileName(imagePath),
                gt.Annotations,
                normalizedPredictions,
                matches,
                errors,
                warnings));
        }

        Directory.CreateDirectory(outputDirectory);
        Directory.CreateDirectory(visualizationDirectory);
        var rows = new List<AnalysisImageRow>(analyzedImages.Count);
        for (int index = 0; index < analyzedImages.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            AnalyzedImage image = analyzedImages[index];
            int truePositive = image.Matches.Count(match => match.ClassCorrect);
            int misclassified = image.Matches.Count - truePositive;
            int missed = image.Errors.Count(error => error.Type == AnalysisErrorType.Missed);
            int falsePositive = image.Errors.Count(error => error.Type == AnalysisErrorType.FalsePositive);
            int metricFalseNegative = missed + misclassified;
            int metricFalsePositive = falsePositive + misclassified;
            double precision = Score(truePositive, truePositive + metricFalsePositive,
                image.GroundTruth.Count == 0);
            double recall = Score(truePositive, truePositive + metricFalseNegative,
                image.Predictions.Count == 0);
            double f1 = F1(precision, recall);
            double meanIou = image.Matches.Count == 0
                ? 0.0
                : image.Matches.Average(match => match.Iou);
            string visualizationPath = string.Empty;
            if (image.Errors.Count > 0)
            {
                visualizationPath = Path.Combine(
                    visualizationDirectory, image.Prefix + ".png");
                AnalysisOverlayRenderer.Render(
                    image.ImagePath,
                    visualizationPath,
                    image.Errors,
                    options.OverwriteOutputs);
            }
            string status = image.Errors.Count == 0 ? "正确" : "存在错误";
            if (image.Warnings.Count > 0) status += "；" + string.Join("；", image.Warnings);
            var row = new AnalysisImageRow(
                index + 1,
                image.ImageName,
                image.GroundTruth.Count,
                image.Predictions.Count,
                truePositive,
                missed,
                falsePositive,
                misclassified,
                precision,
                recall,
                f1,
                meanIou,
                visualizationPath,
                status);
            rows.Add(row);
            progress?.Report(new ResultAnalysisProgress(index + 1, analyzedImages.Count, row));
        }

        WriteReports(outputDirectory, analyzedImages, rows, options);
        int totalTp = rows.Sum(row => row.TruePositiveCount);
        int totalMissed = rows.Sum(row => row.MissedCount);
        int totalFalsePositive = rows.Sum(row => row.FalsePositiveCount);
        int totalMisclassified = rows.Sum(row => row.MisclassifiedCount);
        int totalMetricFp = totalFalsePositive + totalMisclassified;
        int totalMetricFn = totalMissed + totalMisclassified;
        double overallPrecision = Ratio(totalTp, totalTp + totalMetricFp);
        double overallRecall = Ratio(totalTp, totalTp + totalMetricFn);
        return new ResultAnalysisSummary(
            rows.Count,
            rows.Count(row => row.MissedCount + row.FalsePositiveCount + row.MisclassifiedCount > 0),
            totalTp,
            totalMissed,
            totalFalsePositive,
            totalMisclassified,
            overallPrecision,
            overallRecall,
            F1(overallPrecision, overallRecall),
            analyzedImages.SelectMany(image => image.Matches).Any()
                ? analyzedImages.SelectMany(image => image.Matches).Average(match => match.Iou)
                : 0.0,
            rows);
    }

    private static void ValidateInputs(
        string imageDirectory,
        string predictionDirectory,
        string groundTruthDirectory,
        string outputDirectory,
        ResultAnalysisOptions options)
    {
        if (!Directory.Exists(imageDirectory))
            throw new DirectoryNotFoundException($"原始图片文件夹不存在：{imageDirectory}");
        if (!Directory.Exists(predictionDirectory))
            throw new DirectoryNotFoundException($"预测 JSON 文件夹不存在：{predictionDirectory}");
        if (!Directory.Exists(groundTruthDirectory))
            throw new DirectoryNotFoundException($"GT JSON 文件夹不存在：{groundTruthDirectory}");
        if (string.IsNullOrWhiteSpace(outputDirectory))
            throw new ArgumentException("请选择分析输出目录。", nameof(outputDirectory));
        if (options.IouThreshold <= 0.0 || options.IouThreshold > 1.0)
            throw new ArgumentOutOfRangeException(nameof(options), "IoU 阈值必须在 (0, 1] 范围内。");
    }

    private static Dictionary<string, string> BuildUniquePrefixMap(
        IEnumerable<string> paths,
        string datasetName)
    {
        var groups = paths
            .GroupBy(
                path => Path.GetFileNameWithoutExtension(path) ?? string.Empty,
                StringComparer.OrdinalIgnoreCase)
            .ToList();
        string[] duplicates = groups
            .Where(group => group.Count() > 1)
            .Select(group => group.Key)
            .ToArray();
        if (duplicates.Length > 0)
            throw new DuplicateDatasetPrefixException(datasetName, duplicates);
        return groups.ToDictionary(
            group => group.Key,
            group => group.Single(),
            StringComparer.OrdinalIgnoreCase);
    }

    private static void EnsureOutputsAvailable(
        IEnumerable<string> outputPaths,
        bool overwrite)
    {
        if (overwrite) return;
        string[] conflicts = outputPaths
            .Where(File.Exists)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (conflicts.Length > 0)
            throw new AnalysisOutputConflictException(conflicts);
    }

    private static void WriteReports(
        string outputDirectory,
        IReadOnlyList<AnalyzedImage> images,
        IReadOnlyList<AnalysisImageRow> rows,
        ResultAnalysisOptions options)
    {
        WriteImageSummary(Path.Combine(outputDirectory, "image_summary.csv"), rows, options.OverwriteOutputs);
        WriteErrorDetails(Path.Combine(outputDirectory, "error_details.csv"), images, options.OverwriteOutputs);
        WriteOverallMetrics(Path.Combine(outputDirectory, "overall_metrics.csv"), images, rows, options);
        WriteClassMetrics(Path.Combine(outputDirectory, "class_metrics.csv"), images, options.OverwriteOutputs);
        WriteConfusionMatrix(Path.Combine(outputDirectory, "confusion_matrix.csv"), images, options.OverwriteOutputs);
    }

    private static void WriteImageSummary(string path, IReadOnlyList<AnalysisImageRow> rows, bool overwrite)
    {
        var csv = CsvBuilder.Create(
            "序号", "图片名称", "GT数", "预测数", "正确检出", "漏检", "误检", "类别错误",
            "Precision", "Recall", "F1", "平均匹配IoU", "错误图路径", "状态");
        foreach (AnalysisImageRow row in rows)
            csv.Add(
                row.Index, row.ImageName, row.GroundTruthCount, row.PredictionCount,
                row.TruePositiveCount, row.MissedCount, row.FalsePositiveCount,
                row.MisclassifiedCount, F(row.Precision), F(row.Recall), F(row.F1),
                F(row.MeanIou), row.VisualizationPath, row.Status);
        csv.Write(path, overwrite);
    }

    private static void WriteErrorDetails(string path, IReadOnlyList<AnalyzedImage> images, bool overwrite)
    {
        var csv = CsvBuilder.Create(
            "图片名称", "错误类型", "实际类别", "预测类别", "X", "Y", "宽", "高", "IoU", "显示文本");
        foreach (AnalyzedImage image in images)
        {
            foreach (AnalysisError error in image.Errors)
            {
                csv.Add(
                    image.ImageName,
                    error.Type switch
                    {
                        AnalysisErrorType.Missed => "漏检",
                        AnalysisErrorType.FalsePositive => "误检",
                        _ => "类别错误"
                    },
                    error.ActualClass,
                    error.PredictedClass,
                    F(error.Box.X), F(error.Box.Y), F(error.Box.Width), F(error.Box.Height),
                    F(error.Iou), error.DisplayText);
            }
        }
        csv.Write(path, overwrite);
    }

    private static void WriteOverallMetrics(
        string path,
        IReadOnlyList<AnalyzedImage> images,
        IReadOnlyList<AnalysisImageRow> rows,
        ResultAnalysisOptions options)
    {
        int gtCount = images.Sum(image => image.GroundTruth.Count);
        int predCount = images.Sum(image => image.Predictions.Count);
        int tp = rows.Sum(row => row.TruePositiveCount);
        int missed = rows.Sum(row => row.MissedCount);
        int falsePositive = rows.Sum(row => row.FalsePositiveCount);
        int misclassified = rows.Sum(row => row.MisclassifiedCount);
        double precision = Ratio(tp, tp + falsePositive + misclassified);
        double recall = Ratio(tp, tp + missed + misclassified);
        int localized = tp + misclassified;
        double meanIou = images.SelectMany(image => image.Matches).Any()
            ? images.SelectMany(image => image.Matches).Average(match => match.Iou)
            : 0.0;
        var csv = CsvBuilder.Create("指标", "值", "说明");
        PredictionLabelRules rules = options.PredictionRules ?? new PredictionLabelRules();
        csv.Add("IoU匹配阈值", F(options.IouThreshold), "预测框与GT外接矩形的一对一匹配阈值");
        csv.Add(
            "忽略的预测标签",
            rules.IgnoredLabels.Count == 0 ? "无" : string.Join("; ", rules.IgnoredLabels),
            "这些预测标签在匹配和指标统计前被移除");
        csv.Add(
            "预测标签映射",
            rules.LabelMappings.Count == 0
                ? "无"
                : string.Join("; ", rules.LabelMappings.Select(pair => $"{pair.Key}->{pair.Value}")),
            "映射在匹配和指标统计前执行");
        csv.Add("图片数", rows.Count, "以原始图片文件夹为准");
        csv.Add("错误图片数", rows.Count(row => row.MissedCount + row.FalsePositiveCount + row.MisclassifiedCount > 0), "存在漏检、误检或类别错误的图片");
        csv.Add("GT实例数", gtCount, "GT polygon 外接矩形数量");
        csv.Add("预测实例数", predCount, "预测 rectangle 数量");
        csv.Add("正确检出TP", tp, "IoU达标且类别正确");
        csv.Add("漏检", missed, "未匹配GT");
        csv.Add("误检", falsePositive, "未匹配预测");
        csv.Add("类别错误", misclassified, "IoU达标但类别不同");
        csv.Add("Precision", F(precision), "TP/(TP+误检+类别错误)");
        csv.Add("Recall", F(recall), "TP/(TP+漏检+类别错误)");
        csv.Add("F1", F(F1(precision, recall)), "Precision与Recall调和平均");
        csv.Add("定位召回率", F(Ratio(localized, gtCount)), "IoU达标的匹配数/GT数，不考虑类别");
        csv.Add("匹配后分类正确率", F(Ratio(tp, localized)), "类别正确匹配数/全部定位匹配数");
        csv.Add("平均匹配IoU", F(meanIou), "全部IoU达标匹配的平均值");
        csv.Write(path, options.OverwriteOutputs);
    }

    private static void WriteClassMetrics(string path, IReadOnlyList<AnalyzedImage> images, bool overwrite)
    {
        string[] classes = images
            .SelectMany(image => image.GroundTruth.Concat(image.Predictions))
            .Select(annotation => annotation.Label)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(label => label, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        var csv = CsvBuilder.Create(
            "类别", "GT数", "预测数", "TP", "FN", "FP", "Precision", "Recall", "F1", "平均正确匹配IoU");
        foreach (string className in classes)
        {
            int gt = images.Sum(image => image.GroundTruth.Count(annotation => SameClass(annotation.Label, className)));
            int pred = images.Sum(image => image.Predictions.Count(annotation => SameClass(annotation.Label, className)));
            AnnotationMatch[] correct = images.SelectMany(image => image.Matches)
                .Where(match => match.ClassCorrect && SameClass(match.GroundTruth.Label, className))
                .ToArray();
            int tp = correct.Length;
            double precision = Ratio(tp, pred);
            double recall = Ratio(tp, gt);
            csv.Add(
                className, gt, pred, tp, gt - tp, pred - tp,
                F(precision), F(recall), F(F1(precision, recall)),
                F(correct.Length == 0 ? 0.0 : correct.Average(match => match.Iou)));
        }
        csv.Write(path, overwrite);
    }

    private static void WriteConfusionMatrix(string path, IReadOnlyList<AnalyzedImage> images, bool overwrite)
    {
        string[] classes = images
            .SelectMany(image => image.GroundTruth.Concat(image.Predictions))
            .Select(annotation => annotation.Label)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(label => label, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        string[] columns = classes.Concat(["<漏检>"]).ToArray();
        string[] rows = classes.Concat(["<背景>"]).ToArray();
        var counts = new Dictionary<(string Actual, string Predicted), int>(new ClassPairComparer());
        foreach (AnalyzedImage image in images)
        {
            foreach (AnnotationMatch match in image.Matches)
                Increment(counts, match.GroundTruth.Label, match.Prediction.Label);
            foreach (AnalysisError error in image.Errors)
            {
                if (error.Type == AnalysisErrorType.Missed)
                    Increment(counts, error.ActualClass, "<漏检>");
                else if (error.Type == AnalysisErrorType.FalsePositive)
                    Increment(counts, "<背景>", error.PredictedClass);
            }
        }
        var csv = CsvBuilder.Create(["实际\\预测", .. columns]);
        foreach (string actual in rows)
            csv.Add([actual, .. columns.Select(predicted => GetCount(counts, actual, predicted))]);
        csv.Write(path, overwrite);
    }

    private static void Increment(Dictionary<(string, string), int> counts, string actual, string predicted)
    {
        var key = (actual, predicted);
        counts[key] = counts.GetValueOrDefault(key) + 1;
    }

    private static int GetCount(Dictionary<(string, string), int> counts, string actual, string predicted) =>
        counts.GetValueOrDefault((actual, predicted));

    private static bool SameClass(string left, string right) =>
        left.Equals(right, StringComparison.OrdinalIgnoreCase);

    private static double Score(int numerator, int denominator, bool oppositeSetEmpty) =>
        denominator == 0 ? (oppositeSetEmpty ? 1.0 : 0.0) : (double)numerator / denominator;

    private static double Ratio(int numerator, int denominator) =>
        denominator == 0 ? 0.0 : (double)numerator / denominator;

    private static double F1(double precision, double recall) =>
        precision + recall == 0.0 ? 0.0 : 2.0 * precision * recall / (precision + recall);

    private static string F(double value) => value.ToString("0.######", CultureInfo.InvariantCulture);

    private sealed class ClassPairComparer : IEqualityComparer<(string Actual, string Predicted)>
    {
        public bool Equals((string Actual, string Predicted) left, (string Actual, string Predicted) right) =>
            SameClass(left.Actual, right.Actual) && SameClass(left.Predicted, right.Predicted);

        public int GetHashCode((string Actual, string Predicted) pair) =>
            HashCode.Combine(
                StringComparer.OrdinalIgnoreCase.GetHashCode(pair.Actual),
                StringComparer.OrdinalIgnoreCase.GetHashCode(pair.Predicted));
    }
}

internal sealed class CsvBuilder
{
    private readonly StringBuilder _content = new("\uFEFF");

    private CsvBuilder(IEnumerable<object?> headers) => Add(headers);

    public static CsvBuilder Create(params object?[] headers) => new(headers);

    public void Add(params object?[] values) => Add((IEnumerable<object?>)values);

    public void Add(IEnumerable<object?> values)
    {
        _content.AppendLine(string.Join(",", values.Select(Escape)));
    }

    public void Write(string path, bool overwrite)
    {
        FileMode mode = overwrite ? FileMode.Create : FileMode.CreateNew;
        using var stream = new FileStream(path, mode, FileAccess.Write, FileShare.None);
        using var writer = new StreamWriter(stream, new UTF8Encoding(false));
        writer.Write(_content.ToString());
    }

    private static string Escape(object? value)
    {
        string text = Convert.ToString(value, CultureInfo.InvariantCulture) ?? string.Empty;
        return "\"" + text.Replace("\"", "\"\"") + "\"";
    }
}
