using System.IO;

namespace CoronaDetection;

internal sealed class AnnotationCompletionService
{
    private static readonly HashSet<string> ImageExtensions =
        DetectionFiles.FormatExtensions.Values
            .SelectMany(extensions => extensions)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

    public async Task<AnnotationCompletionSummary> CompleteAsync(
        string imageDirectory,
        string jsonDirectory,
        bool recursive,
        IProgress<AnnotationCompletionProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        return await Task.Run(
            () => Complete(
                imageDirectory, jsonDirectory, recursive, progress, cancellationToken),
            cancellationToken);
    }

    private static AnnotationCompletionSummary Complete(
        string imageDirectory,
        string jsonDirectory,
        bool recursive,
        IProgress<AnnotationCompletionProgress>? progress,
        CancellationToken cancellationToken)
    {
        if (!Directory.Exists(imageDirectory))
            throw new DirectoryNotFoundException($"原始图片文件夹不存在：{imageDirectory}");
        if (string.IsNullOrWhiteSpace(jsonDirectory))
            throw new ArgumentException("请选择 JSON 文件夹。", nameof(jsonDirectory));

        SearchOption searchOption = recursive
            ? SearchOption.AllDirectories
            : SearchOption.TopDirectoryOnly;
        string[] imagePaths = Directory
            .EnumerateFiles(imageDirectory, "*", searchOption)
            .Where(path => ImageExtensions.Contains(Path.GetExtension(path)))
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (imagePaths.Length == 0)
            throw new InvalidOperationException("原始图片文件夹中没有找到支持的图片。");

        string[] duplicatePrefixes = imagePaths
            .GroupBy(GetFilePrefix, StringComparer.OrdinalIgnoreCase)
            .Where(group => group.Count() > 1)
            .Select(group => group.Key)
            .ToArray();
        if (duplicatePrefixes.Length > 0)
            throw new DuplicateImagePrefixException(duplicatePrefixes);

        var existingPrefixes = Directory.Exists(jsonDirectory)
            ? Directory
                .EnumerateFiles(jsonDirectory, "*.json", searchOption)
                .Select(GetFilePrefix)
                .ToHashSet(StringComparer.OrdinalIgnoreCase)
            : new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        string[] missingImages = imagePaths
            .Where(imagePath => !existingPrefixes.Contains(GetFilePrefix(imagePath)))
            .ToArray();

        // Everything above is a read-only preflight. Start writing only after the
        // image/JSON mapping is known to be unambiguous.
        Directory.CreateDirectory(jsonDirectory);
        var createdItems = new List<AnnotationCompletionItem>(missingImages.Length);
        for (int index = 0; index < missingImages.Length; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string imagePath = missingImages[index];
            string jsonPath = Path.Combine(jsonDirectory, GetFilePrefix(imagePath) + ".json");
            ImageMetadata metadata = LabelMeJsonWriter.Write(
                imagePath,
                Path.GetFileName(imagePath),
                jsonPath,
                [],
                overwrite: false);
            var item = new AnnotationCompletionItem(
                index + 1,
                Path.GetFileName(imagePath),
                imagePath,
                metadata.Width,
                metadata.Height,
                jsonPath,
                metadata.Warning);
            createdItems.Add(item);
            progress?.Report(new AnnotationCompletionProgress(
                index + 1, missingImages.Length, item));
        }

        return new AnnotationCompletionSummary(
            imagePaths.Length,
            imagePaths.Length - missingImages.Length,
            missingImages.Length,
            createdItems);
    }

    private static string GetFilePrefix(string path) =>
        Path.GetFileNameWithoutExtension(path);
}

public sealed record AnnotationCompletionItem(
    int Index,
    string ImageName,
    string ImagePath,
    int? ImageWidth,
    int? ImageHeight,
    string JsonPath,
    string Warning);

internal sealed record AnnotationCompletionProgress(
    int Completed,
    int Total,
    AnnotationCompletionItem Item);

internal sealed record AnnotationCompletionSummary(
    int ImageCount,
    int ExistingCount,
    int CreatedCount,
    IReadOnlyList<AnnotationCompletionItem> CreatedItems);

internal sealed class DuplicateImagePrefixException(IReadOnlyList<string> prefixes)
    : InvalidOperationException("原始图片文件夹中存在重名图片，无法安全生成平铺 JSON。")
{
    public IReadOnlyList<string> Prefixes { get; } = prefixes;
}
