using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Windows;
using Microsoft.Win32;

namespace CoronaDetection;

public partial class MainWindow
{
    private readonly AnnotationCompletionService _annotationCompletionService = new();
    private CancellationTokenSource? _annotationCompletionCancellation;
    private bool _annotationCompletionBusy;

    public ObservableCollection<AnnotationCompletionItem> AnnotationCompletionResults { get; } = [];

    private void BrowseCompletionImage_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFolderDialog
        {
            Title = "选择原始图片文件夹",
            InitialDirectory = Directory.Exists(CompletionImagePathText.Text)
                ? CompletionImagePathText.Text
                : Environment.GetFolderPath(Environment.SpecialFolder.MyPictures)
        };
        if (dialog.ShowDialog(this) != true) return;

        CompletionImagePathText.Text = dialog.FolderName;
        if (string.IsNullOrWhiteSpace(CompletionJsonPathText.Text))
            CompletionJsonPathText.Text = Path.Combine(dialog.FolderName, "jsons");
    }

    private void BrowseCompletionJson_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFolderDialog
        {
            Title = "选择需要检查或补齐的 LabelMe JSON 文件夹",
            InitialDirectory = Directory.Exists(CompletionJsonPathText.Text)
                ? CompletionJsonPathText.Text
                : CompletionImagePathText.Text
        };
        if (dialog.ShowDialog(this) == true)
            CompletionJsonPathText.Text = dialog.FolderName;
    }

    private async void StartCompletion_Click(object sender, RoutedEventArgs e)
    {
        if (_annotationCompletionBusy || _resultConversionBusy || _busy || _analysisBusy) return;
        string imageDirectory = CompletionImagePathText.Text.Trim();
        string jsonDirectory = CompletionJsonPathText.Text.Trim();
        if (!Directory.Exists(imageDirectory))
        {
            MessageBox.Show(this, "请选择有效的原始图片文件夹。", "补齐空标注",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        if (string.IsNullOrWhiteSpace(jsonDirectory))
        {
            MessageBox.Show(this, "请选择 LabelMe JSON 文件夹。", "补齐空标注",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        AnnotationCompletionResults.Clear();
        CompletionProgress.Minimum = 0;
        CompletionProgress.Maximum = 1;
        CompletionProgress.Value = 0;
        _annotationCompletionCancellation = new CancellationTokenSource();
        SetAnnotationCompletionBusy(true, "正在扫描图片和 JSON…");

        var progress = new Progress<AnnotationCompletionProgress>(update =>
        {
            CompletionProgress.Maximum = Math.Max(1, update.Total);
            CompletionProgress.Value = update.Completed;
            CompletionProgressText.Text =
                $"{update.Completed} / {update.Total}（{update.Completed * 100 / Math.Max(1, update.Total)}%）";
            CompletionStatusText.Text = $"正在补齐：{update.Item.ImageName}";
            AnnotationCompletionResults.Add(update.Item);
            if (AnnotationCompletionResults.Count == 1)
                CompletionResultGrid.SelectedIndex = 0;
        });

        try
        {
            AnnotationCompletionSummary summary = await _annotationCompletionService.CompleteAsync(
                imageDirectory,
                jsonDirectory,
                RecursiveCompletionCheck.IsChecked == true,
                progress,
                _annotationCompletionCancellation.Token);
            CompletionProgress.Maximum = Math.Max(1, summary.CreatedCount);
            CompletionProgress.Value = summary.CreatedCount;
            CompletionProgressText.Text = summary.CreatedCount == 0
                ? "无需补齐"
                : $"{summary.CreatedCount} / {summary.CreatedCount}（100%）";
            CompletionStatusText.Text =
                $"检查完成：图片 {summary.ImageCount} 张，已有标注 {summary.ExistingCount} 个，补齐空标注 {summary.CreatedCount} 个";
        }
        catch (OperationCanceledException)
        {
            CompletionStatusText.Text = $"补齐任务已取消，已生成 {AnnotationCompletionResults.Count} 个空标注";
        }
        catch (DuplicateImagePrefixException ex)
        {
            CompletionStatusText.Text = "未执行：发现重名图片";
            string preview = string.Join(Environment.NewLine, ex.Prefixes.Take(8));
            string remaining = ex.Prefixes.Count > 8
                ? $"{Environment.NewLine}……另有 {ex.Prefixes.Count - 8} 个重名前缀"
                : string.Empty;
            MessageBox.Show(
                this,
                $"发现 {ex.Prefixes.Count} 个重名图片前缀，无法确定 JSON 对应关系，未生成任何文件：" +
                $"{Environment.NewLine}{Environment.NewLine}{preview}{remaining}",
                "图片名称冲突",
                MessageBoxButton.OK,
                MessageBoxImage.Warning);
        }
        catch (Exception ex)
        {
            CompletionStatusText.Text = "补齐空标注失败";
            MessageBox.Show(this, ex.Message, "补齐空标注失败",
                MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            _annotationCompletionCancellation.Dispose();
            _annotationCompletionCancellation = null;
            SetAnnotationCompletionBusy(false, CompletionStatusText.Text);
        }
    }

    private void CancelCompletion_Click(object sender, RoutedEventArgs e) =>
        _annotationCompletionCancellation?.Cancel();

    private void OpenCompletionOutput_Click(object sender, RoutedEventArgs e)
    {
        string jsonDirectory = CompletionJsonPathText.Text.Trim();
        if (!Directory.Exists(jsonDirectory))
        {
            MessageBox.Show(this, "JSON 文件夹尚不存在。", "补齐空标注",
                MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }
        Process.Start(new ProcessStartInfo("explorer.exe", jsonDirectory)
        {
            UseShellExecute = true
        });
    }

    private void SetAnnotationCompletionBusy(bool busy, string status)
    {
        _annotationCompletionBusy = busy;
        StartCompletionButton.IsEnabled = !busy;
        CancelCompletionButton.IsEnabled = busy;
        BrowseCompletionImageButton.IsEnabled = !busy;
        BrowseCompletionJsonButton.IsEnabled = !busy;
        OpenCompletionOutputButton.IsEnabled = !busy;
        RecursiveCompletionCheck.IsEnabled = !busy;
        StartConversionButton.IsEnabled = !busy && !_resultConversionBusy;
        StartAnalysisButton.IsEnabled = !busy && !_analysisBusy;
        CompletionStatusText.Text = status;
    }
}
