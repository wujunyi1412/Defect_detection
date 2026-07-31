using System.Text;

namespace CoronaDetection;

internal sealed class InspectionEngine : IDisposable
{
    private readonly object _sync = new();
    private IntPtr _handle;

    public bool IsInitialized { get; private set; }

    public string Initialize(string configPath)
    {
        lock (_sync)
        {
            CloseHandle();
            _handle = NativeMethods.Inspection_Create();
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException("Inspection_Create 创建句柄失败。");

            var message = new StringBuilder(100);
            int code = NativeMethods.Inspection_Initialize(_handle, configPath, message);
            if (code != NativeMethods.StatusOk)
            {
                string detail = message.ToString();
                CloseHandle();
                throw new InvalidOperationException($"模型初始化失败（状态码 {code}）：{detail}");
            }

            IsInitialized = true;
            return message.ToString();
        }
    }

    public DetectionResult Process(string imagePath)
        => ProcessTimed(imagePath).Result;

    public TimedDetectionResult ProcessTimed(string imagePath)
    {
        lock (_sync)
        {
            if (!IsInitialized || _handle == IntPtr.Zero)
                throw new InvalidOperationException("模型尚未初始化。");

            var native = NativeMethods.InspectionResult.Create();
            int code = NativeMethods.Inspection_ProcessImagePathTimed(
                _handle, imagePath, ref native, out double inferenceMs);
            if (code != NativeMethods.StatusOk)
                throw new InvalidOperationException($"图片推理失败（状态码 {code}）。");

            return new TimedDetectionResult(
                ConvertResult(native), inferenceMs);
        }
    }

    public NativeOverlayResult ProcessToOverlayFile(string imagePath, string outputPath)
    {
        lock (_sync)
        {
            if (!IsInitialized || _handle == IntPtr.Zero)
                throw new InvalidOperationException("模型尚未初始化。");

            var native = NativeMethods.InspectionResult.Create();
            var message = new StringBuilder(100);
            int code = NativeMethods.Inspection_ProcessImagePathToOverlayFile(
                _handle, imagePath, outputPath, ref native, message,
                out double inferenceMs, out double saveMs);
            DetectionResult result = ConvertResult(native);
            if (code == NativeMethods.StatusOk)
                return new NativeOverlayResult(
                    result, string.Empty, inferenceMs, saveMs);
            if (code == 5)
                return new NativeOverlayResult(
                    result,
                    $"DLL 生成可视化图片失败（状态码 {code}）：{message}",
                    inferenceMs,
                    saveMs);
            throw new InvalidOperationException(
                $"图片推理失败（状态码 {code}）：{message}");
        }
    }

    private static DetectionResult ConvertResult(NativeMethods.InspectionResult native)
    {
        int count = Math.Clamp(native.DetailsCount, 0, NativeMethods.MaxDetections);
        var details = native.Details.Take(count)
            .Select(d => new DefectDetail(
                d.Name ?? string.Empty, d.Area, d.X, d.Y, d.W, d.H, d.Contrast))
            .ToArray();
        return new DetectionResult(
            native.Result == 0 ? "OK" : "NG",
            native.YoloScore,
            native.PatchcoreScore,
            native.PatchcoreAreaRatio,
            details);
    }

    private void CloseHandle()
    {
        if (_handle == IntPtr.Zero)
            return;
        if (IsInitialized)
            NativeMethods.Inspection_Release(_handle);
        NativeMethods.Inspection_Destroy(_handle);
        _handle = IntPtr.Zero;
        IsInitialized = false;
    }

    public void Dispose()
    {
        lock (_sync)
            CloseHandle();
    }
}

internal sealed record DefectDetail(
    string Name, int Area, float X, float Y, float W, float H, float Contrast);

internal sealed record DetectionResult(
    string Verdict,
    float YoloScore,
    float PatchcoreScore,
    float PatchcoreAreaRatio,
    IReadOnlyList<DefectDetail> Details);

internal sealed record NativeOverlayResult(
    DetectionResult Result,
    string ExportError,
    double InferenceMs,
    double SaveMs);

internal sealed record TimedDetectionResult(
    DetectionResult Result,
    double InferenceMs);
