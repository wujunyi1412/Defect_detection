using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Windows.Media.Imaging;

namespace CoronaDetection;

internal sealed class ResultConversionService
{
    private static readonly string[] RequiredColumns =
    [
        "原图路径", "缺陷序号", "类别", "面积", "X", "Y", "宽", "高", "对比度"
    ];

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };

    public async Task<ResultConversionSummary> ConvertAsync(
        string csvPath,
        string outputDirectory,
        IProgress<ResultConversionProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        return await Task.Run(
            () => Convert(csvPath, outputDirectory, progress, cancellationToken),
            cancellationToken);
    }

    private static ResultConversionSummary Convert(
        string csvPath,
        string outputDirectory,
        IProgress<ResultConversionProgress>? progress,
        CancellationToken cancellationToken)
    {
        if (!File.Exists(csvPath))
            throw new FileNotFoundException("CSV 文件不存在。", csvPath);
        if (string.IsNullOrWhiteSpace(outputDirectory))
            throw new ArgumentException("请选择 JSON 输出目录。", nameof(outputDirectory));

        string csvText = ReadCsvText(csvPath, out string encodingName);
        List<string[]> records = CsvParser.Parse(csvText);
        if (records.Count == 0)
            throw new InvalidDataException("CSV 文件为空。");

        string[] headers = records[0].Select(value => value.Trim()).ToArray();
        var columns = headers
            .Select((name, index) => (name, index))
            .GroupBy(item => item.name, StringComparer.Ordinal)
            .ToDictionary(group => group.Key, group => group.First().index, StringComparer.Ordinal);
        string[] missing = RequiredColumns.Where(column => !columns.ContainsKey(column)).ToArray();
        if (missing.Length > 0)
            throw new InvalidDataException("CSV 缺少字段：" + string.Join("、", missing));

        var groups = new List<ImageDefectGroup>();
        var groupLookup = new Dictionary<string, ImageDefectGroup>(StringComparer.OrdinalIgnoreCase);
        for (int recordIndex = 1; recordIndex < records.Count; recordIndex++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string[] record = records[recordIndex];
            if (record.All(string.IsNullOrWhiteSpace)) continue;

            string imagePath = GetValue(record, columns["原图路径"], recordIndex, "原图路径").Trim();
            if (string.IsNullOrWhiteSpace(imagePath))
                throw new InvalidDataException($"CSV 第 {recordIndex + 1} 行的原图路径为空。");

            if (!groupLookup.TryGetValue(imagePath, out ImageDefectGroup? group))
            {
                group = new ImageDefectGroup(imagePath);
                groupLookup.Add(imagePath, group);
                groups.Add(group);
            }

            group.Defects.Add(new DefectRecord(
                ParseInt(record, columns["缺陷序号"], recordIndex, "缺陷序号"),
                GetValue(record, columns["类别"], recordIndex, "类别"),
                ParseDouble(record, columns["面积"], recordIndex, "面积"),
                ParseDouble(record, columns["X"], recordIndex, "X"),
                ParseDouble(record, columns["Y"], recordIndex, "Y"),
                ParseDouble(record, columns["宽"], recordIndex, "宽"),
                ParseDouble(record, columns["高"], recordIndex, "高"),
                ParseDouble(record, columns["对比度"], recordIndex, "对比度")));
        }

        Directory.CreateDirectory(outputDirectory);
        var items = new List<ResultConversionItem>(groups.Count);
        int totalDefects = 0;
        for (int index = 0; index < groups.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ImageDefectGroup group = groups[index];
            string resolvedImagePath = Path.IsPathRooted(group.ImagePath)
                ? group.ImagePath
                : Path.GetFullPath(Path.Combine(Path.GetDirectoryName(csvPath)!, group.ImagePath));
            (int? width, int? height, string warning) = ReadImageSize(resolvedImagePath);

            string imageName = Path.GetFileName(group.ImagePath);
            string jsonPath = Path.Combine(
                outputDirectory,
                Path.GetFileNameWithoutExtension(imageName) + ".json");
            var document = new LabelMeDocument
            {
                Shapes = group.Defects.Select(ToLabelMeShape).ToList(),
                ImagePath = imageName,
                ImageHeight = height,
                ImageWidth = width
            };
            File.WriteAllText(
                jsonPath,
                JsonSerializer.Serialize(document, JsonOptions),
                new UTF8Encoding(false));

            totalDefects += group.Defects.Count;
            var item = new ResultConversionItem(
                index + 1, imageName, group.ImagePath, group.Defects.Count,
                width, height, jsonPath, warning);
            items.Add(item);
            progress?.Report(new ResultConversionProgress(index + 1, groups.Count, item));
        }

        return new ResultConversionSummary(encodingName, groups.Count, totalDefects, items);
    }

    private static LabelMeShape ToLabelMeShape(DefectRecord defect) => new()
    {
        Label = defect.ClassName,
        Points =
        [
            [defect.X, defect.Y],
            [defect.X + defect.Width, defect.Y + defect.Height]
        ],
        Description = string.Create(
            CultureInfo.InvariantCulture,
            $"defect_id={defect.Id}; area={defect.Area}; contrast={defect.Contrast}")
    };

    private static string ReadCsvText(string path, out string encodingName)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.AsSpan().StartsWith([(byte)0xEF, (byte)0xBB, (byte)0xBF]))
        {
            encodingName = "UTF-8 BOM";
            return new UTF8Encoding(false, true).GetString(bytes, 3, bytes.Length - 3);
        }

        try
        {
            encodingName = "UTF-8";
            return new UTF8Encoding(false, true).GetString(bytes);
        }
        catch (DecoderFallbackException)
        {
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        }

        foreach ((int codePage, string name) in new[] { (936, "GBK"), (54936, "GB18030") })
        {
            try
            {
                Encoding encoding = Encoding.GetEncoding(
                    codePage, EncoderFallback.ExceptionFallback, DecoderFallback.ExceptionFallback);
                encodingName = name;
                return encoding.GetString(bytes);
            }
            catch (DecoderFallbackException)
            {
            }
        }

        throw new InvalidDataException("无法识别 CSV 编码（支持 UTF-8、GBK、GB18030）。");
    }

    private static (int? Width, int? Height, string Warning) ReadImageSize(string imagePath)
    {
        if (!File.Exists(imagePath))
            return (null, null, "原图不存在，JSON 中尺寸已留空");
        try
        {
            using var stream = new FileStream(imagePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            BitmapFrame frame = BitmapDecoder.Create(
                stream, BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.OnLoad).Frames[0];
            return (frame.PixelWidth, frame.PixelHeight, string.Empty);
        }
        catch (Exception ex)
        {
            return (null, null, "无法读取原图尺寸：" + ex.Message);
        }
    }

    private static string GetValue(string[] record, int column, int recordIndex, string columnName)
    {
        if (column >= record.Length)
            throw new InvalidDataException($"CSV 第 {recordIndex + 1} 行缺少“{columnName}”字段。");
        return record[column];
    }

    private static int ParseInt(string[] record, int column, int recordIndex, string columnName)
    {
        string value = GetValue(record, column, recordIndex, columnName);
        if (int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int result))
            return result;
        throw new InvalidDataException($"CSV 第 {recordIndex + 1} 行的“{columnName}”不是有效整数：{value}");
    }

    private static double ParseDouble(string[] record, int column, int recordIndex, string columnName)
    {
        string value = GetValue(record, column, recordIndex, columnName);
        if (double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double result))
            return result;
        throw new InvalidDataException($"CSV 第 {recordIndex + 1} 行的“{columnName}”不是有效数字：{value}");
    }

    private sealed class ImageDefectGroup(string imagePath)
    {
        public string ImagePath { get; } = imagePath;
        public List<DefectRecord> Defects { get; } = [];
    }

    private sealed record DefectRecord(
        int Id, string ClassName, double Area, double X, double Y,
        double Width, double Height, double Contrast);

    private sealed class LabelMeDocument
    {
        [JsonPropertyName("version")]
        public string Version { get; init; } = "5.0.1";
        [JsonPropertyName("flags")]
        public Dictionary<string, object> Flags { get; init; } = [];
        [JsonPropertyName("shapes")]
        public required List<LabelMeShape> Shapes { get; init; }
        [JsonPropertyName("imagePath")]
        public required string ImagePath { get; init; }
        [JsonPropertyName("imageData")]
        public string? ImageData { get; init; }
        [JsonPropertyName("imageHeight")]
        public int? ImageHeight { get; init; }
        [JsonPropertyName("imageWidth")]
        public int? ImageWidth { get; init; }
    }

    private sealed class LabelMeShape
    {
        [JsonPropertyName("label")]
        public required string Label { get; init; }
        [JsonPropertyName("points")]
        public required List<List<double>> Points { get; init; }
        [JsonPropertyName("group_id")]
        public int? GroupId { get; init; }
        [JsonPropertyName("description")]
        public required string Description { get; init; }
        [JsonPropertyName("shape_type")]
        public string ShapeType { get; init; } = "rectangle";
        [JsonPropertyName("flags")]
        public Dictionary<string, object> Flags { get; init; } = [];
    }
}

public sealed record ResultConversionItem(
    int Index,
    string ImageName,
    string ImagePath,
    int DefectCount,
    int? ImageWidth,
    int? ImageHeight,
    string JsonPath,
    string Warning);

internal sealed record ResultConversionProgress(
    int Completed,
    int Total,
    ResultConversionItem Item);

internal sealed record ResultConversionSummary(
    string CsvEncoding,
    int ImageCount,
    int DefectCount,
    IReadOnlyList<ResultConversionItem> Items);

internal static class CsvParser
{
    public static List<string[]> Parse(string text)
    {
        var records = new List<string[]>();
        var record = new List<string>();
        var field = new StringBuilder();
        bool quoted = false;

        for (int i = 0; i < text.Length; i++)
        {
            char current = text[i];
            if (quoted)
            {
                if (current == '"')
                {
                    if (i + 1 < text.Length && text[i + 1] == '"')
                    {
                        field.Append('"');
                        i++;
                    }
                    else
                    {
                        quoted = false;
                    }
                }
                else
                {
                    field.Append(current);
                }
                continue;
            }

            switch (current)
            {
                case '"' when field.Length == 0:
                    quoted = true;
                    break;
                case ',':
                    record.Add(field.ToString());
                    field.Clear();
                    break;
                case '\r':
                    if (i + 1 < text.Length && text[i + 1] == '\n') i++;
                    AddRecord(records, record, field);
                    break;
                case '\n':
                    AddRecord(records, record, field);
                    break;
                default:
                    field.Append(current);
                    break;
            }
        }

        if (quoted)
            throw new InvalidDataException("CSV 中存在未闭合的引号。");
        if (field.Length > 0 || record.Count > 0)
            AddRecord(records, record, field);
        return records;
    }

    private static void AddRecord(List<string[]> records, List<string> record, StringBuilder field)
    {
        record.Add(field.ToString());
        field.Clear();
        records.Add(record.ToArray());
        record.Clear();
    }
}
