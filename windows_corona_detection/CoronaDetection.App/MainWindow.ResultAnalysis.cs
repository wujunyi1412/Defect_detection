using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using Microsoft.Win32;

namespace CoronaDetection;

public partial class MainWindow
{
    private readonly ResultAnalysisService _resultAnalysisService = new();
    private CancellationTokenSource? _analysisCancellation;
    private bool _analysisBusy;

    public ObservableCollection<AnalysisImageRow> AnalysisResults { get; } = [];

    private void BrowseAnalysisImage_Click(object sender, RoutedEventArgs e) =>
        SelectAnalysisFolder(AnalysisImagePathText, "选择原始图片文件夹");

    private void BrowseAnalysisPrediction_Click(object sender, RoutedEventArgs e) =>
        SelectAnalysisFolder(AnalysisPredictionPathText, "选择预测 LabelMe JSON 文件夹");

    private void BrowseAnalysisGroundTruth_Click(object sender, RoutedEventArgs e) =>
        SelectAnalysisFolder(AnalysisGroundTruthPathText, "选择 GT LabelMe JSON 文件夹");

    private void BrowseAnalysisOutput_Click(object sender, RoutedEventArgs e) =>
        SelectAnalysisFolder(AnalysisOutputPathText, "选择分析报告和错误图输出文件夹");

    private void SelectAnalysisFolder(TextBox target, string title)
    {
        var dialog = new OpenFolderDialog
        {
            Title = title,
            InitialDirectory = Directory.Exists(target.Text)
                ? target.Text
                : Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments)
        };
        if (dialog.ShowDialog(this) == true)
            target.Text = dialog.FolderName;
    }

    private async void StartAnalysis_Click(object sender, RoutedEventArgs e)
    {
        if (_analysisBusy || _busy || _resultConversionBusy || _annotationCompletionBusy) return;
        string imageDirectory = AnalysisImagePathText.Text.Trim();
        string predictionDirectory = AnalysisPredictionPathText.Text.Trim();
        string groundTruthDirectory = AnalysisGroundTruthPathText.Text.Trim();
        string outputDirectory = AnalysisOutputPathText.Text.Trim();
        if (!TryValidateAnalysisInputs(
                imageDirectory, predictionDirectory, groundTruthDirectory,
                outputDirectory, out double iouThreshold))
            return;

        AnalysisResults.Clear();
        AnalysisProgress.Minimum = 0;
        AnalysisProgress.Maximum = 1;
        AnalysisProgress.Value = 0;
        _analysisCancellation = new CancellationTokenSource();
        SetAnalysisBusy(true, "正在读取预测和 GT 标注…");
        var progress = new Progress<ResultAnalysisProgress>(update =>
        {
            AnalysisProgress.Maximum = Math.Max(1, update.Total);
            AnalysisProgress.Value = update.Completed;
            AnalysisProgressText.Text =
                $"{update.Completed} / {update.Total}（{update.Completed * 100 / Math.Max(1, update.Total)}%）";
            AnalysisStatusText.Text = $"正在分析：{update.Row.ImageName}";
            AnalysisResults.Add(update.Row);
            if (AnalysisResults.Count == 1)
                AnalysisResultGrid.SelectedIndex = 0;
        });

        try
        {
            ResultAnalysisSummary summary = await _resultAnalysisService.AnalyzeAsync(
                imageDirectory,
                predictionDirectory,
                groundTruthDirectory,
                outputDirectory,
                new ResultAnalysisOptions(
                    iouThreshold,
                    RecursiveAnalysisCheck.IsChecked == true,
                    OverwriteAnalysisCheck.IsChecked == true),
                progress,
                _analysisCancellation.Token);
            AnalysisProgress.Maximum = Math.Max(1, summary.ImageCount);
            AnalysisProgress.Value = summary.ImageCount;
            AnalysisProgressText.Text = $"{summary.ImageCount} / {summary.ImageCount}（100%）";
            AnalysisStatusText.Text =
                $"分析完成：图片 {summary.ImageCount} 张，错误图片 {summary.ErrorImageCount} 张；" +
                $"TP {summary.TruePositiveCount}，漏检 {summary.MissedCount}，" +
                $"误检 {summary.FalsePositiveCount}，类别错误 {summary.MisclassifiedCount}；" +
                $"P={summary.Precision:0.###}，R={summary.Recall:0.###}，F1={summary.F1:0.###}";
        }
        catch (OperationCanceledException)
        {
            AnalysisStatusText.Text = $"分析已取消，已处理 {AnalysisResults.Count} 张图片";
        }
        catch (AnalysisOutputConflictException ex)
        {
            AnalysisStatusText.Text = "未执行：分析输出文件冲突";
            ShowAnalysisConflict(
                "发现已有分析结果，未写入任何新报告。请先更换输出目录，或勾选“覆盖已有分析结果”后重新执行。",
                ex.ConflictingPaths);
        }
        catch (DuplicateDatasetPrefixException ex)
        {
            AnalysisStatusText.Text = "未执行：数据集存在重名文件";
            ShowAnalysisConflict(
                $"{ex.DatasetName}中存在相同的不含扩展名文件名，无法唯一配对，未执行分析。",
                ex.Prefixes);
        }
        catch (Exception ex)
        {
            AnalysisStatusText.Text = "结果分析失败";
            MessageBox.Show(this, ex.Message, "结果分析失败",
                MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            _analysisCancellation.Dispose();
            _analysisCancellation = null;
            SetAnalysisBusy(false, AnalysisStatusText.Text);
        }
    }

    private bool TryValidateAnalysisInputs(
        string imageDirectory,
        string predictionDirectory,
        string groundTruthDirectory,
        string outputDirectory,
        out double iouThreshold)
    {
        iouThreshold = 0.0;
        if (!Directory.Exists(imageDirectory) ||
            !Directory.Exists(predictionDirectory) ||
            !Directory.Exists(groundTruthDirectory))
        {
            MessageBox.Show(this, "请选择有效的原始图片、预测 JSON 和 GT JSON 文件夹。", "结果分析",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return false;
        }
        if (string.IsNullOrWhiteSpace(outputDirectory))
        {
            MessageBox.Show(this, "请选择分析输出文件夹。", "结果分析",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return false;
        }
        if (!double.TryParse(
                AnalysisIouThresholdText.Text.Trim(),
                NumberStyles.Float,
                CultureInfo.InvariantCulture,
                out iouThreshold) ||
            iouThreshold <= 0.0 || iouThreshold > 1.0)
        {
            MessageBox.Show(this, "IoU 阈值必须是 (0, 1] 范围内的数字，例如 0.5。", "结果分析",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return false;
        }
        return true;
    }

    private void ShowAnalysisConflict(string message, IReadOnlyList<string> values)
    {
        string preview = string.Join(Environment.NewLine, values.Take(8));
        string remaining = values.Count > 8
            ? $"{Environment.NewLine}……另有 {values.Count - 8} 项"
            : string.Empty;
        MessageBox.Show(
            this,
            message + Environment.NewLine + Environment.NewLine + preview + remaining,
            "结果分析预检失败",
            MessageBoxButton.OK,
            MessageBoxImage.Warning);
    }

    private void CancelAnalysis_Click(object sender, RoutedEventArgs e) =>
        _analysisCancellation?.Cancel();

    private void OpenAnalysisOutput_Click(object sender, RoutedEventArgs e)
    {
        string outputDirectory = AnalysisOutputPathText.Text.Trim();
        if (!Directory.Exists(outputDirectory))
        {
            MessageBox.Show(this, "分析输出文件夹尚不存在。", "结果分析",
                MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }
        Process.Start(new ProcessStartInfo("explorer.exe", outputDirectory)
        {
            UseShellExecute = true
        });
    }

    private void SetAnalysisBusy(bool busy, string status)
    {
        _analysisBusy = busy;
        StartAnalysisButton.IsEnabled = !busy;
        CancelAnalysisButton.IsEnabled = busy;
        BrowseAnalysisImageButton.IsEnabled = !busy;
        BrowseAnalysisPredictionButton.IsEnabled = !busy;
        BrowseAnalysisGroundTruthButton.IsEnabled = !busy;
        BrowseAnalysisOutputButton.IsEnabled = !busy;
        AnalysisIouThresholdText.IsEnabled = !busy;
        RecursiveAnalysisCheck.IsEnabled = !busy;
        OverwriteAnalysisCheck.IsEnabled = !busy;
        OpenAnalysisOutputButton.IsEnabled = !busy;
        StartButton.IsEnabled = !busy && !_busy;
        StartConversionButton.IsEnabled = !busy && !_resultConversionBusy;
        StartCompletionButton.IsEnabled = !busy && !_annotationCompletionBusy;
        AnalysisStatusText.Text = status;
    }
}
