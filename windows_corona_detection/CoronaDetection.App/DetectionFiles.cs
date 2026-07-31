using System.Globalization;
using System.IO;
using System.Text;

namespace CoronaDetection;

internal static class DetectionFiles
{
    internal static readonly IReadOnlyDictionary<string, string[]> FormatExtensions =
        new Dictionary<string, string[]>
        {
            ["所有图片"] = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"],
            ["PNG"] = [".png"],
            ["JPEG"] = [".jpg", ".jpeg"],
            ["BMP"] = [".bmp"],
            ["TIFF"] = [".tif", ".tiff"]
        };

    public static List<string> Enumerate(
        string inputPath, bool batch, bool recursive, string format)
    {
        string[] extensions = FormatExtensions.TryGetValue(format, out var selected)
            ? selected : FormatExtensions["所有图片"];
        var allowed = extensions.ToHashSet(StringComparer.OrdinalIgnoreCase);

        if (!batch)
        {
            if (!File.Exists(inputPath))
                throw new FileNotFoundException("输入图片不存在。", inputPath);
            if (!allowed.Contains(Path.GetExtension(inputPath)))
                throw new InvalidOperationException("单张图片与当前格式筛选不匹配。");
            return [Path.GetFullPath(inputPath)];
        }

        if (!Directory.Exists(inputPath))
            throw new DirectoryNotFoundException($"输入文件夹不存在：{inputPath}");
        var option = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
        return Directory.EnumerateFiles(inputPath, "*", option)
            .Where(p => allowed.Contains(Path.GetExtension(p)))
            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
            .Select(Path.GetFullPath)
            .ToList();
    }

    public static string BuildOutputImagePath(
        string outputRoot, string inputRoot, string inputFile, bool batch, bool preserveTree)
    {
        string directory = outputRoot;
        if (batch && preserveTree)
        {
            string relative = Path.GetRelativePath(inputRoot, Path.GetDirectoryName(inputFile)!);
            if (relative != ".")
                directory = Path.Combine(outputRoot, relative);
        }
        Directory.CreateDirectory(directory);
        string baseName = Path.GetFileNameWithoutExtension(inputFile) + "_detect";
        string candidate = Path.Combine(directory, baseName + ".png");
        for (int i = 2; File.Exists(candidate); i++)
            candidate = Path.Combine(directory, $"{baseName}_{i}.png");
        return candidate;
    }

    public static void SaveCsv(
        string outputRoot, IEnumerable<ResultRow> rows, IEnumerable<DetailRow> details)
    {
        Directory.CreateDirectory(outputRoot);
        var summary = new StringBuilder("\uFEFF");
        summary.AppendLine("序号,文件名,原图路径,判定,缺陷数,推理耗时ms,图片保存耗时ms,总耗时ms,状态,结果图路径,错误");
        foreach (var row in rows)
        {
            summary.AppendLine(string.Join(",",
                row.Index,
                Csv(row.FileName), Csv(row.InputPath), Csv(row.Verdict),
                row.DefectCount,
                row.InferenceMs.ToString("0.###", CultureInfo.InvariantCulture),
                row.SaveMs.ToString("0.###", CultureInfo.InvariantCulture),
                row.TotalMs.ToString("0.###", CultureInfo.InvariantCulture),
                Csv(row.Status), Csv(row.OutputPath), Csv(row.Error)));
        }
        File.WriteAllText(Path.Combine(outputRoot, "summary.csv"), summary.ToString(), new UTF8Encoding(false));

        var detailCsv = new StringBuilder("\uFEFF");
        detailCsv.AppendLine("序号,原图路径,缺陷序号,类别,面积,X,Y,宽,高,对比度");
        foreach (var row in details)
        {
            detailCsv.AppendLine(string.Join(",",
                row.ResultIndex, Csv(row.InputPath), row.DetailIndex, Csv(row.Name), row.Area,
                F(row.X), F(row.Y), F(row.Width), F(row.Height), F(row.Contrast)));
        }
        File.WriteAllText(Path.Combine(outputRoot, "details.csv"), detailCsv.ToString(), new UTF8Encoding(false));
    }

    private static string F(float value) => value.ToString("0.####", CultureInfo.InvariantCulture);
    private static string Csv(string? value) => "\"" + (value ?? string.Empty).Replace("\"", "\"\"") + "\"";
}

public sealed class ResultRow
{
    public int Index { get; init; }
    public string FileName { get; init; } = string.Empty;
    public string InputPath { get; init; } = string.Empty;
    public string Verdict { get; init; } = string.Empty;
    public int DefectCount { get; init; }
    public double InferenceMs { get; init; }
    public double SaveMs { get; init; }
    public double TotalMs { get; init; }
    public string Status { get; init; } = string.Empty;
    public string OutputPath { get; init; } = string.Empty;
    public string Error { get; init; } = string.Empty;
}

internal sealed record DetailRow(
    int ResultIndex, string InputPath, int DetailIndex, string Name, int Area,
    float X, float Y, float Width, float Height, float Contrast);
