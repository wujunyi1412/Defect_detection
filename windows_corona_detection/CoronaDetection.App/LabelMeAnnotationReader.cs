using System.Globalization;
using System.IO;
using System.Text.Json;

namespace CoronaDetection;

internal static class LabelMeAnnotationReader
{
    public static AnnotationReadResult Read(string? jsonPath)
    {
        if (string.IsNullOrEmpty(jsonPath) || !File.Exists(jsonPath))
            return new AnnotationReadResult([], []);

        using JsonDocument document = JsonDocument.Parse(
            File.ReadAllText(jsonPath),
            new JsonDocumentOptions { AllowTrailingCommas = true });
        if (!document.RootElement.TryGetProperty("shapes", out JsonElement shapes) ||
            shapes.ValueKind == JsonValueKind.Null)
            return new AnnotationReadResult([], []);
        if (shapes.ValueKind != JsonValueKind.Array)
            throw new InvalidDataException($"LabelMe JSON 的 shapes 不是数组：{jsonPath}");

        var annotations = new List<AnnotationBox>();
        var warnings = new List<string>();
        int shapeIndex = 0;
        foreach (JsonElement shape in shapes.EnumerateArray())
        {
            shapeIndex++;
            string label = shape.TryGetProperty("label", out JsonElement labelElement)
                ? labelElement.GetString()?.Trim() ?? string.Empty
                : string.Empty;
            if (string.IsNullOrEmpty(label))
            {
                warnings.Add($"第 {shapeIndex} 个标注缺少类别，已忽略");
                continue;
            }
            if (!shape.TryGetProperty("points", out JsonElement pointsElement) ||
                pointsElement.ValueKind != JsonValueKind.Array)
            {
                warnings.Add($"第 {shapeIndex} 个 {label} 标注缺少 points，已忽略");
                continue;
            }

            var points = new List<(double X, double Y)>();
            foreach (JsonElement point in pointsElement.EnumerateArray())
            {
                if (point.ValueKind != JsonValueKind.Array || point.GetArrayLength() < 2)
                    continue;
                JsonElement.ArrayEnumerator coordinates = point.EnumerateArray();
                coordinates.MoveNext();
                double x = coordinates.Current.GetDouble();
                coordinates.MoveNext();
                double y = coordinates.Current.GetDouble();
                if (double.IsFinite(x) && double.IsFinite(y))
                    points.Add((x, y));
            }
            if (points.Count < 2)
            {
                warnings.Add($"第 {shapeIndex} 个 {label} 标注有效点不足，已忽略");
                continue;
            }

            double minX = points.Min(point => point.X);
            double minY = points.Min(point => point.Y);
            double maxX = points.Max(point => point.X);
            double maxY = points.Max(point => point.Y);
            var box = new BoundingBox(minX, minY, maxX - minX, maxY - minY);
            if (!box.IsValid)
            {
                warnings.Add($"第 {shapeIndex} 个 {label} 标注外接矩形无效，已忽略");
                continue;
            }
            annotations.Add(new AnnotationBox(label, box));
        }
        return new AnnotationReadResult(annotations, warnings);
    }

    public static string BoxToString(BoundingBox box) => string.Create(
        CultureInfo.InvariantCulture,
        $"{box.X:0.####},{box.Y:0.####},{box.Width:0.####},{box.Height:0.####}");
}
