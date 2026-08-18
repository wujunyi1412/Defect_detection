using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Microsoft.Win32;

namespace CoronaDetection;

public partial class MainWindow : Window
{
    private readonly InspectionEngine _engine = new();
    private readonly List<DetailRow> _details = [];
    private readonly string _configPath = Path.Combine(AppContext.BaseDirectory, "config.ini");
    private IniDocument? _ini;
    private CancellationTokenSource? _cancellation;
    private bool _configDirty;
    private bool _busy;
    private bool _loaded;

    public ObservableCollection<ResultRow> Results { get; } = [];

    public MainWindow()
    {
        InitializeComponent();
        DataContext = this;
        string version = typeof(MainWindow).Assembly.GetName().Version?.ToString(3) ?? "1.0.0";
        Title = $"Corona 瑕疵检测工作站 v{version}";
        VersionText.Text = $"版本 {version}";
        FormatCombo.ItemsSource = DetectionFiles.FormatExtensions.Keys;
        FormatCombo.SelectedIndex = 0;
    }

    private async void Window_Loaded(object sender, RoutedEventArgs e)
    {
        _loaded = true;
        LoadConfiguration();

        // Let WPF render the window before the heavy model initialization starts.
        await Task.Yield();
        await RestartModelAsync();
    }

    private void Window_Closing(object? sender, CancelEventArgs e)
    {
        if (_busy || _resultConversionBusy || _annotationCompletionBusy || _analysisBusy)
        {
            _cancellation?.Cancel();
            _resultConversionCancellation?.Cancel();
            _annotationCompletionCancellation?.Cancel();
            _analysisCancellation?.Cancel();
            if (_busy) TaskStatusText.Text = "正在取消，请稍候…";
            if (_resultConversionBusy) ConversionStatusText.Text = "正在取消，请稍候…";
            if (_annotationCompletionBusy) CompletionStatusText.Text = "正在取消，请稍候…";
            if (_analysisBusy) AnalysisStatusText.Text = "正在取消，请稍候…";
            e.Cancel = true;
            return;
        }
        _engine.Dispose();
    }

    private void ModeRadio_Checked(object sender, RoutedEventArgs e)
    {
        if (!_loaded) return;
        bool batch = BatchModeRadio.IsChecked == true;
        BrowseInputButton.Content = batch ? "选择文件夹…" : "选择图片…";
        RecursiveCheck.IsEnabled = batch;
        PreserveTreeCheck.IsEnabled = batch;
        InputPathText.Clear();
    }

    private void BrowseInput_Click(object sender, RoutedEventArgs e)
    {
        if (BatchModeRadio.IsChecked == true)
        {
            var dialog = new OpenFolderDialog { Title = "选择待检测图片文件夹" };
            if (dialog.ShowDialog(this) == true)
                InputPathText.Text = dialog.FolderName;
        }
        else
        {
            var dialog = new OpenFileDialog
            {
                Title = "选择待检测图片",
                Filter = "所有支持图片|*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff|PNG|*.png|JPEG|*.jpg;*.jpeg|BMP|*.bmp|TIFF|*.tif;*.tiff"
            };
            if (dialog.ShowDialog(this) == true)
                InputPathText.Text = dialog.FileName;
        }
    }

    private void BrowseOutput_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFolderDialog
        {
            Title = "选择检测结果保存文件夹",
            InitialDirectory = Directory.Exists(OutputPathText.Text)
                ? OutputPathText.Text
                : Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments)
        };
        if (dialog.ShowDialog(this) == true)
            OutputPathText.Text = dialog.FolderName;
    }

    private void LoadConfiguration()
    {
        try
        {
            if (!File.Exists(_configPath))
                throw new FileNotFoundException("程序目录中找不到 config.ini。", _configPath);
            _ini = IniDocument.Load(_configPath);
            ConfigGrid.ItemsSource = _ini.Settings;
            _configDirty = false;
            UpdateEngineStatus();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "读取配置失败", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private bool SaveConfiguration()
    {
        try
        {
            ConfigGrid.CommitEdit(DataGridEditingUnit.Cell, true);
            ConfigGrid.CommitEdit(DataGridEditingUnit.Row, true);
            _ini?.Save(_configPath);
            _configDirty = true;
            UpdateEngineStatus();
            TaskStatusText.Text = "配置已保存，等待模型重启";
            return true;
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "保存配置失败", MessageBoxButton.OK, MessageBoxImage.Error);
            return false;
        }
    }

    private void ConfigGrid_CellEditEnding(object sender, DataGridCellEditEndingEventArgs e)
    {
        Dispatcher.BeginInvoke(() =>
        {
            _configDirty = true;
            UpdateEngineStatus();
        });
    }

    private void SaveConfig_Click(object sender, RoutedEventArgs e) => SaveConfiguration();

    private async void RestartModel_Click(object sender, RoutedEventArgs e) =>
        await RestartModelAsync();

    private void ReloadConfig_Click(object sender, RoutedEventArgs e) => LoadConfiguration();

    private void SelectModelPath_Click(object sender, RoutedEventArgs e)
    {
        if (ConfigGrid.SelectedItem is not IniSetting setting ||
            !setting.Section.Equals("models", StringComparison.OrdinalIgnoreCase))
        {
            MessageBox.Show(this, "请先在 models 分组中选中一个模型路径参数。", "选择模型文件",
                MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        var dialog = new OpenFileDialog
        {
            Title = $"选择 {setting.Key}",
            Filter = "模型或数据文件|*.onnx;*.faiss;*.json|所有文件|*.*"
        };
        if (dialog.ShowDialog(this) != true) return;
        setting.Value = Path.GetRelativePath(Path.GetDirectoryName(_configPath)!, dialog.FileName);
        _configDirty = true;
        UpdateEngineStatus();
    }

    private async Task<bool> RestartModelAsync()
    {
        if (_busy) return false;
        if (_configDirty && !SaveConfiguration()) return false;
        SetBusy(true, "正在加载模型…", true);
        try
        {
            string message = await Task.Run(() => _engine.Initialize(_configPath));
            _configDirty = false;
            TaskStatusText.Text = string.IsNullOrWhiteSpace(message) ? "模型初始化成功" : message;
            UpdateEngineStatus();
            return true;
        }
        catch (Exception ex)
        {
            TaskStatusText.Text = "模型初始化失败";
            UpdateEngineStatus();
            MessageBox.Show(this, ex.Message, "模型初始化失败", MessageBoxButton.OK, MessageBoxImage.Error);
            return false;
        }
        finally
        {
            SetBusy(false, TaskStatusText.Text, true);
        }
    }

    private async void Start_Click(object sender, RoutedEventArgs e)
    {
        if (_busy || _resultConversionBusy || _annotationCompletionBusy || _analysisBusy) return;
        List<string> files;
        string inputRoot = InputPathText.Text.Trim();
        string outputRoot = OutputPathText.Text.Trim();
        try
        {
            if (string.IsNullOrWhiteSpace(outputRoot))
                throw new InvalidOperationException("请选择检测结果保存文件夹。");
            files = DetectionFiles.Enumerate(
                inputRoot, BatchModeRadio.IsChecked == true,
                RecursiveCheck.IsChecked == true, FormatCombo.Text);
            if (files.Count == 0)
                throw new InvalidOperationException("没有找到符合格式筛选条件的图片。");
            Directory.CreateDirectory(outputRoot);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "任务设置有误", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        if (_configDirty || !_engine.IsInitialized)
        {
            bool ready = await RestartModelAsync();
            if (!ready) return;
        }

        try
        {
            DetectionFiles.CopyConfigSnapshot(_configPath, outputRoot);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "复制配置文件失败",
                MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        Results.Clear();
        _details.Clear();
        TaskProgress.Minimum = 0;
        TaskProgress.Maximum = files.Count;
        TaskProgress.Value = 0;
        _cancellation = new CancellationTokenSource();
        SetBusy(true, $"准备检测，共 {files.Count} 张");

        try
        {
            for (int i = 0; i < files.Count; i++)
            {
                _cancellation.Token.ThrowIfCancellationRequested();
                string file = files[i];
                TaskStatusText.Text = $"正在检测：{Path.GetFileName(file)}";
                var stopwatch = Stopwatch.StartNew();
                ResultRow row;
                try
                {
                    string outputImage = string.Empty;
                    string saveError = string.Empty;
                    DetectionResult result;
                    double inferenceMs;
                    double saveMs;
                    if (SaveImagesCheck.IsChecked == true)
                    {
                        outputImage = DetectionFiles.BuildOutputImagePath(
                            outputRoot, inputRoot, file, BatchModeRadio.IsChecked == true,
                            PreserveTreeCheck.IsChecked == true);
                        NativeOverlayResult nativeOverlay = await Task.Run(
                            () => ProcessWithNativeOverlay(file, outputImage),
                            _cancellation.Token);
                        result = nativeOverlay.Result;
                        saveError = nativeOverlay.ExportError;
                        inferenceMs = nativeOverlay.InferenceMs;
                        saveMs = nativeOverlay.SaveMs;
                        if (!string.IsNullOrEmpty(saveError))
                            outputImage = string.Empty;
                    }
                    else
                    {
                        TimedDetectionResult timedResult = await Task.Run(
                            () => ProcessWithAsciiPath(file), _cancellation.Token);
                        result = timedResult.Result;
                        inferenceMs = timedResult.InferenceMs;
                        saveMs = 0.0;
                    }
                    stopwatch.Stop();
                    row = new ResultRow
                    {
                        Index = i + 1,
                        FileName = Path.GetFileName(file),
                        InputPath = file,
                        Verdict = result.Verdict,
                        DefectCount = result.Details.Count,
                        InferenceMs = inferenceMs,
                        SaveMs = saveMs,
                        TotalMs = stopwatch.Elapsed.TotalMilliseconds,
                        Status = string.IsNullOrEmpty(saveError) ? "完成" : "部分完成",
                        OutputPath = outputImage,
                        Error = saveError
                    };
                    for (int d = 0; d < result.Details.Count; d++)
                    {
                        DefectDetail detail = result.Details[d];
                        _details.Add(new DetailRow(
                            i + 1, file, d + 1, detail.Name, detail.Area,
                            detail.X, detail.Y, detail.W, detail.H, detail.Contrast));
                    }
                }
                catch (Exception ex)
                {
                    stopwatch.Stop();
                    row = new ResultRow
                    {
                        Index = i + 1,
                        FileName = Path.GetFileName(file),
                        InputPath = file,
                        TotalMs = stopwatch.Elapsed.TotalMilliseconds,
                        Status = "失败",
                        Error = ex.Message
                    };
                }
                Results.Add(row);
                TaskProgress.Value = i + 1;
                ProgressText.Text = $"{i + 1} / {files.Count}（{(i + 1) * 100 / files.Count}%）";
                if (Results.Count == 1)
                    ResultGrid.SelectedIndex = 0;
            }
            TaskStatusText.Text =
                $"检测完成：成功 {Results.Count(r => r.Status != "失败")}，失败 {Results.Count(r => r.Status == "失败")}";
        }
        catch (OperationCanceledException)
        {
            TaskStatusText.Text = $"任务已取消，已完成 {Results.Count} / {files.Count}";
        }
        finally
        {
            try
            {
                DetectionFiles.SaveCsv(outputRoot, Results, _details);
                TaskStatusText.Text += "；汇总表已保存";
            }
            catch (Exception ex)
            {
                TaskStatusText.Text += "；汇总保存失败";
                MessageBox.Show(this, ex.Message, "保存汇总失败", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            _cancellation.Dispose();
            _cancellation = null;
            SetBusy(false, TaskStatusText.Text);
        }
    }

    private void Cancel_Click(object sender, RoutedEventArgs e) => _cancellation?.Cancel();

    private TimedDetectionResult ProcessWithAsciiPath(string originalPath)
    {
        if (originalPath.All(c => c <= 127))
            return _engine.ProcessTimed(originalPath);

        string tempRoot = Path.Combine(Path.GetTempPath(), "corona_detection_ascii");
        Directory.CreateDirectory(tempRoot);
        string tempFile = Path.Combine(
            tempRoot, Guid.NewGuid().ToString("N") + Path.GetExtension(originalPath).ToLowerInvariant());
        try
        {
            File.Copy(originalPath, tempFile, true);
            return _engine.ProcessTimed(tempFile);
        }
        finally
        {
            try { File.Delete(tempFile); } catch { }
        }
    }

    private NativeOverlayResult ProcessWithNativeOverlay(
        string originalPath,
        string destinationPath)
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "corona_detection_ascii");
        Directory.CreateDirectory(tempRoot);
        bool copyInput = !originalPath.All(c => c <= 127);
        string nativeInputPath = copyInput
            ? Path.Combine(
                tempRoot,
                Guid.NewGuid().ToString("N") + Path.GetExtension(originalPath).ToLowerInvariant())
            : originalPath;
        string temporaryOutput = Path.Combine(
            tempRoot, Guid.NewGuid().ToString("N") + ".png");

        try
        {
            if (copyInput)
                File.Copy(originalPath, nativeInputPath, true);

            NativeOverlayResult native = _engine.ProcessToOverlayFile(
                nativeInputPath, temporaryOutput);
            if (!string.IsNullOrEmpty(native.ExportError))
                return native;
            if (!File.Exists(temporaryOutput))
                return new NativeOverlayResult(
                    native.Result,
                    "DLL 未生成可视化图片。",
                    native.InferenceMs,
                    native.SaveMs);

            Directory.CreateDirectory(Path.GetDirectoryName(destinationPath)!);
            var moveTimer = Stopwatch.StartNew();
            File.Move(temporaryOutput, destinationPath, true);
            moveTimer.Stop();
            return native with { SaveMs = native.SaveMs + moveTimer.Elapsed.TotalMilliseconds };
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                "DLL 推理或 8 位可视化图片保存失败：" + ex.Message, ex);
        }
        finally
        {
            if (copyInput)
            {
                try { File.Delete(nativeInputPath); } catch { }
            }
            try { File.Delete(temporaryOutput); } catch { }
        }
    }

    private void SetBusy(bool busy, string status, bool modelOperation = false)
    {
        _busy = busy;
        StartButton.IsEnabled = !busy;
        CancelButton.IsEnabled = busy && !modelOperation;
        SaveConfigButton.IsEnabled = !busy;
        RestartModelButton.IsEnabled = !busy;
        BrowseInputButton.IsEnabled = !busy;
        StartConversionButton.IsEnabled = !busy && !_resultConversionBusy;
        StartCompletionButton.IsEnabled = !busy && !_annotationCompletionBusy;
        StartAnalysisButton.IsEnabled = !busy && !_analysisBusy;
        TaskStatusText.Text = status;
        if (!busy) UpdateEngineStatus();
    }

    private void UpdateEngineStatus()
    {
        if (_configDirty)
        {
            EngineStatusText.Text = "● 配置已修改，待重启";
            EngineStatusText.Foreground = new SolidColorBrush(Color.FromRgb(255, 193, 7));
        }
        else if (_engine.IsInitialized)
        {
            EngineStatusText.Text = "● 模型已就绪";
            EngineStatusText.Foreground = new SolidColorBrush(Color.FromRgb(76, 217, 100));
        }
        else
        {
            EngineStatusText.Text = "● 模型未初始化";
            EngineStatusText.Foreground = new SolidColorBrush(Color.FromRgb(255, 193, 7));
        }
    }
}
