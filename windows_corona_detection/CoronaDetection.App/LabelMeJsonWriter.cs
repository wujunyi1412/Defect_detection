using System.IO;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Windows.Media.Imaging;

namespace CoronaDetection;

internal static class LabelMeJsonWriter
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };

    public static ImageMetadata Write(
        string sourceImagePath,
        string imageName,
        string outputPath,
        IReadOnlyList<LabelMeShapeData> shapes,
        bool overwrite)
    {
        ImageMetadata metadata = ReadImageMetadata(sourceImagePath);
        var document = new LabelMeDocument
        {
            Shapes = shapes.Select(ToDocumentShape).ToList(),
            ImagePath = imageName,
            ImageHeight = metadata.Height,
            ImageWidth = metadata.Width
        };

        string json = JsonSerializer.Serialize(document, JsonOptions);
        FileMode mode = overwrite ? FileMode.Create : FileMode.CreateNew;
        using var stream = new FileStream(outputPath, mode, FileAccess.Write, FileShare.None);
        using var writer = new StreamWriter(stream, new UTF8Encoding(false));
        writer.Write(json);
        return metadata;
    }

    public static ImageMetadata ReadImageMetadata(string imagePath)
    {
        if (!File.Exists(imagePath))
            return new ImageMetadata(null, null, "原图不存在，JSON 中尺寸已留空");
        try
        {
            using var stream = new FileStream(imagePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            BitmapFrame frame = BitmapDecoder.Create(
                stream, BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.OnLoad).Frames[0];
            return new ImageMetadata(frame.PixelWidth, frame.PixelHeight, string.Empty);
        }
        catch (Exception ex)
        {
            return new ImageMetadata(null, null, "无法读取原图尺寸：" + ex.Message);
        }
    }

    private static LabelMeShape ToDocumentShape(LabelMeShapeData shape) => new()
    {
        Label = shape.Label,
        Points = shape.Points,
        Description = shape.Description
    };

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

internal sealed record LabelMeShapeData(
    string Label,
    List<List<double>> Points,
    string Description);

internal sealed record ImageMetadata(int? Width, int? Height, string Warning);
