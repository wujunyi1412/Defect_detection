using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Windows;
using Microsoft.Win32;

namespace CoronaDetection;

public partial class MainWindow
{
    private readonly ResultConversionService _resultConversionService = new();
    private CancellationTokenSource? _resultConversionCancellation;
    private bool _resultConversionBusy;

    public ObservableCollection<ResultConversionItem> ConversionResults { get; } = [];

    private void BrowseConversionCsv_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Title = "选择检测生成的缺陷明细 CSV",
            Filter = "缺陷明细 CSV|details.csv|CSV 文件|*.csv|所有文件|*.*"
        };
        if (dialog.ShowDialog(this) != true) return;

        ConversionCsvPathText.Text = dialog.FileName;
        if (string.IsNullOrWhiteSpace(ConversionOutputPathText.Text))
            ConversionOutputPathText.Text = Path.Combine(
                Path.GetDirectoryName(dialog.FileName)!, "jsons");
    }

    private void BrowseConversionOutput_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFolderDialog
        {
            Title = "选择 LabelMe JSON 输出文件夹",
            InitialDirectory = Directory.Exists(ConversionOutputPathText.Text)
                ? ConversionOutputPathText.Text
                : Path.GetDirectoryName(ConversionCsvPathText.Text)
        };
        if (dialog.ShowDialog(this) == true)
            ConversionOutputPathText.Text = dialog.FolderName;
    }

    private async void StartConversion_Click(object sender, RoutedEventArgs e)
    {
        if (_resultConversionBusy || _annotationCompletionBusy || _busy || _analysisBusy) return;
        string csvPath = ConversionCsvPathText.Text.Trim();
        string outputDirectory = ConversionOutputPathText.Text.Trim();
        if (!File.Exists(csvPath))
        {
            MessageBox.Show(this, "请选择有效的 details.csv 文件。", "结果处理",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        if (string.IsNullOrWhiteSpace(outputDirectory))
        {
            MessageBox.Show(this, "请选择 JSON 输出目录。", "结果处理",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        ConversionResults.Clear();
        ConversionProgress.Minimum = 0;
        ConversionProgress.Maximum = 1;
        ConversionProgress.Value = 0;
        _resultConversionCancellation = new CancellationTokenSource();
        SetResultConversionBusy(true, "正在读取 CSV…");

        var progress = new Progress<ResultConversionProgress>(update =>
        {
            if (ConversionProgress.Maximum != update.Total)
                ConversionProgress.Maximum = Math.Max(1, update.Total);
            ConversionProgress.Value = update.Completed;
            ConversionProgressText.Text =
                $"{update.Completed} / {update.Total}（{update.Completed * 100 / Math.Max(1, update.Total)}%）";
            ConversionStatusText.Text = $"正在转换：{update.Item.ImageName}";
            ConversionResults.Add(update.Item);
            if (ConversionResults.Count == 1)
                ConversionResultGrid.SelectedIndex = 0;
        });

        try
        {
            ResultConversionSummary summary = await _resultConversionService.ConvertAsync(
                csvPath,
                outputDirectory,
                new ResultConversionOptions(OverwriteConversionFilesCheck.IsChecked == true),
                progress,
                _resultConversionCancellation.Token);
            ConversionProgress.Maximum = Math.Max(1, summary.ImageCount);
            ConversionProgress.Value = summary.ImageCount;
            ConversionProgressText.Text = $"{summary.ImageCount} / {summary.ImageCount}（100%）";
            ConversionStatusText.Text =
                $"转换完成：图片 {summary.ImageCount} 张，缺陷 {summary.DefectCount} 个，CSV 编码 {summary.CsvEncoding}";
        }
        catch (OperationCanceledException)
        {
            ConversionStatusText.Text = $"转换已取消，已完成 {ConversionResults.Count} 张图片";
        }
        catch (OutputFileConflictException ex)
        {
            ConversionStatusText.Text = "未执行：发现同名 JSON 文件";
            string preview = string.Join(Environment.NewLine, ex.ConflictingPaths.Take(8));
            string remaining = ex.ConflictingPaths.Count > 8
                ? $"{Environment.NewLine}……另有 {ex.ConflictingPaths.Count - 8} 个冲突文件"
                : string.Empty;
            MessageBox.Show(
                this,
                $"发现 {ex.ConflictingPaths.Count} 个同名 JSON，转换未执行：{Environment.NewLine}{Environment.NewLine}" +
                preview + remaining +
                $"{Environment.NewLine}{Environment.NewLine}如需覆盖，请先勾选“覆盖同名 JSON”，然后重新开始转换。",
                "输出文件冲突",
                MessageBoxButton.OK,
                MessageBoxImage.Warning);
        }
        catch (Exception ex)
        {
            ConversionStatusText.Text = "转换失败";
            MessageBox.Show(this, ex.Message, "结果处理失败",
                MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            _resultConversionCancellation.Dispose();
            _resultConversionCancellation = null;
            SetResultConversionBusy(false, ConversionStatusText.Text);
        }
    }

    private void CancelConversion_Click(object sender, RoutedEventArgs e) =>
        _resultConversionCancellation?.Cancel();

    private void OpenConversionOutput_Click(object sender, RoutedEventArgs e)
    {
        string outputDirectory = ConversionOutputPathText.Text.Trim();
        if (!Directory.Exists(outputDirectory))
        {
            MessageBox.Show(this, "输出目录尚不存在。", "结果处理",
                MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }
        Process.Start(new ProcessStartInfo("explorer.exe", outputDirectory)
        {
            UseShellExecute = true
        });
    }

    private void SetResultConversionBusy(bool busy, string status)
    {
        _resultConversionBusy = busy;
        StartConversionButton.IsEnabled = !busy;
        CancelConversionButton.IsEnabled = busy;
        BrowseConversionCsvButton.IsEnabled = !busy;
        BrowseConversionOutputButton.IsEnabled = !busy;
        OpenConversionOutputButton.IsEnabled = !busy;
        OverwriteConversionFilesCheck.IsEnabled = !busy;
        StartCompletionButton.IsEnabled = !busy && !_annotationCompletionBusy;
        StartAnalysisButton.IsEnabled = !busy && !_analysisBusy;
        ConversionStatusText.Text = status;
    }
}
