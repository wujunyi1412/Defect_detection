using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Security;
using System.Text;
using System.Xml;

namespace CoronaDetection;

internal sealed record ManifestMatchOptions(
    string CsvPath,
    string ImageRoot,
    int FolderLevel,
    string ImageFormat);

internal sealed record ManifestMatchRow(
    int CsvRow,
    string StartTimeText,
    DateTime? StartTime,
    string SerialNumber,
    string ExpectedKey,
    string Status,
    string MatchedFolder,
    int ImageCount,
    string ImagePaths,
    string Message);

internal sealed class ManifestMatchResult
{
    public required IReadOnlyList<ManifestMatchRow> Rows { get; init; }
    public required IReadOnlyList<string> ImageFiles { get; init; }
    public int RecordCount => Rows.Count;
    public int MatchedCount => Rows.Count(row => row.Status == ManifestMatchingService.MatchedStatus);
    public int UnmatchedCount => Rows.Count(row => row.Status == ManifestMatchingService.UnmatchedStatus);
    public int AmbiguousCount => Rows.Count(row => row.Status == ManifestMatchingService.AmbiguousStatus);
    public int InvalidCount => Rows.Count(row => row.Status == ManifestMatchingService.InvalidStatus);
    public int EmptyImageCount => Rows.Count(row => row.Status == ManifestMatchingService.EmptyImageStatus);
    public int ScannedImageCount => Rows.Sum(row => row.ImageCount);
}

internal static class ManifestMatchingService
{
    internal const string MatchedStatus = "已匹配";
    internal const string UnmatchedStatus = "未匹配";
    internal const string AmbiguousStatus = "多目录冲突";
    internal const string InvalidStatus = "清单数据无效";
    internal const string EmptyImageStatus = "已匹配但无图片";

    public static ManifestMatchResult Match(ManifestMatchOptions options)
    {
        if (!File.Exists(options.CsvPath))
            throw new FileNotFoundException("清单 CSV 不存在。", options.CsvPath);
        if (!Directory.Exists(options.ImageRoot))
            throw new DirectoryNotFoundException($"图片根目录不存在：{options.ImageRoot}");
        if (options.FolderLevel < 1)
            throw new InvalidOperationException("匹配文件夹层级必须大于或等于 1。根目录的直接子文件夹为第 1 级。");

        string[] extensions = DetectionFiles.FormatExtensions.TryGetValue(options.ImageFormat, out var selected)
            ? selected
            : DetectionFiles.FormatExtensions["所有图片"];
        var allowedExtensions = extensions.ToHashSet(StringComparer.OrdinalIgnoreCase);
        List<string> candidateDirectories = EnumerateDirectoriesAtLevel(options.ImageRoot, options.FolderLevel);
        List<string[]> csvRows = ReadCsv(options.CsvPath);
        if (csvRows.Count == 0)
            throw new InvalidOperationException("清单 CSV 为空。");

        int startTimeColumn = FindColumn(csvRows[0], "start_time");
        int serialNumberColumn = FindColumn(csvRows[0], "serial_number");
        if (startTimeColumn < 0 || serialNumberColumn < 0)
            throw new InvalidOperationException("清单 CSV 必须包含 start_time 和 serial_number 两列。");

        var matches = new List<ManifestMatchRow>();
        var uniqueImages = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        for (int index = 1; index < csvRows.Count; index++)
        {
            string[] csvRow = csvRows[index];
            if (csvRow.All(string.IsNullOrWhiteSpace))
                continue;

            int csvRowNumber = index + 1;
            string startTimeText = Cell(csvRow, startTimeColumn).Trim();
            string serialNumber = Cell(csvRow, serialNumberColumn).Trim();
            if (serialNumber.Length == 0 || !TryParseStartTime(startTimeText, out DateTime startTime))
            {
                matches.Add(new ManifestMatchRow(
                    csvRowNumber, startTimeText, null, serialNumber, string.Empty,
                    InvalidStatus, string.Empty, 0, string.Empty,
                    serialNumber.Length == 0 ? "serial_number 为空。" : "start_time 无法解析。"));
                continue;
            }

            string expectedKey = $"{serialNumber}_{startTime:yyyyMMdd'T'HHmmss}";
            List<string> folders = candidateDirectories
                .Where(path => Path.GetFileName(path).Contains(expectedKey, StringComparison.OrdinalIgnoreCase))
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
                .ToList();

            if (folders.Count == 0)
            {
                matches.Add(new ManifestMatchRow(
                    csvRowNumber, startTimeText, startTime, serialNumber, expectedKey,
                    UnmatchedStatus, string.Empty, 0, string.Empty,
                    $"第 {options.FolderLevel} 级目录中未找到包含匹配键的文件夹。"));
                continue;
            }
            if (folders.Count > 1)
            {
                matches.Add(new ManifestMatchRow(
                    csvRowNumber, startTimeText, startTime, serialNumber, expectedKey,
                    AmbiguousStatus, string.Join(" | ", folders), 0, string.Empty,
                    $"找到 {folders.Count} 个候选文件夹，已跳过检测。"));
                continue;
            }

            string matchedFolder = folders[0];
            List<string> images = Directory.EnumerateFiles(matchedFolder, "*", SearchOption.TopDirectoryOnly)
                .Where(path => allowedExtensions.Contains(Path.GetExtension(path)))
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
                .Select(Path.GetFullPath)
                .ToList();
            foreach (string image in images)
                uniqueImages.Add(image);

            matches.Add(new ManifestMatchRow(
                csvRowNumber, startTimeText, startTime, serialNumber, expectedKey,
                images.Count == 0 ? EmptyImageStatus : MatchedStatus,
                matchedFolder, images.Count, string.Join(Environment.NewLine, images),
                images.Count == 0 ? $"文件夹内没有 {options.ImageFormat} 格式图片。" : string.Empty));
        }

        return new ManifestMatchResult
        {
            Rows = matches,
            ImageFiles = uniqueImages.OrderBy(path => path, StringComparer.OrdinalIgnoreCase).ToList()
        };
    }

    private static List<string> EnumerateDirectoriesAtLevel(string root, int level)
    {
        var current = new List<string> { Path.GetFullPath(root) };
        for (int depth = 0; depth < level; depth++)
        {
            var next = new List<string>();
            foreach (string directory in current)
            {
                try
                {
                    next.AddRange(Directory.EnumerateDirectories(directory, "*", SearchOption.TopDirectoryOnly));
                }
                catch (UnauthorizedAccessException)
                {
                    // Inaccessible branches cannot contain usable task images, so skip them.
                }
            }
            current = next;
            if (current.Count == 0)
                break;
        }
        return current;
    }

    private static int FindColumn(string[] headers, string expected) =>
        Array.FindIndex(headers, header =>
            header.Trim().TrimStart('\uFEFF').Equals(expected, StringComparison.OrdinalIgnoreCase));

    private static string Cell(string[] row, int index) => index < row.Length ? row[index] : string.Empty;

    private static bool TryParseStartTime(string text, out DateTime value)
    {
        string[] exactFormats =
        [
            "yyyy/M/d H:mm:ss", "yyyy/M/d HH:mm:ss", "yyyy-MM-dd H:mm:ss", "yyyy-MM-dd HH:mm:ss",
            "yyyyMMdd'T'HHmmss"
        ];
        return DateTime.TryParseExact(text.Trim(), exactFormats, CultureInfo.InvariantCulture,
                   DateTimeStyles.AllowWhiteSpaces, out value)
               || DateTime.TryParse(text.Trim(), CultureInfo.CurrentCulture,
                   DateTimeStyles.AllowWhiteSpaces, out value);
    }

    private static List<string[]> ReadCsv(string path)
    {
        string text;
        try
        {
            text = File.ReadAllText(path, new UTF8Encoding(false, true));
        }
        catch (DecoderFallbackException)
        {
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
            text = File.ReadAllText(path, Encoding.GetEncoding(936));
        }

        var rows = new List<string[]>();
        var row = new List<string>();
        var field = new StringBuilder();
        bool quoted = false;
        for (int i = 0; i < text.Length; i++)
        {
            char ch = text[i];
            if (quoted)
            {
                if (ch == '"' && i + 1 < text.Length && text[i + 1] == '"')
                {
                    field.Append('"');
                    i++;
                }
                else if (ch == '"')
                {
                    quoted = false;
                }
                else
                {
                    field.Append(ch);
                }
                continue;
            }

            if (ch == '"') quoted = true;
            else if (ch == ',')
            {
                row.Add(field.ToString());
                field.Clear();
            }
            else if (ch == '\r' || ch == '\n')
            {
                if (ch == '\r' && i + 1 < text.Length && text[i + 1] == '\n') i++;
                row.Add(field.ToString());
                field.Clear();
                rows.Add(row.ToArray());
                row.Clear();
            }
            else field.Append(ch);
        }
        if (field.Length > 0 || row.Count > 0)
        {
            row.Add(field.ToString());
            rows.Add(row.ToArray());
        }
        return rows;
    }
}

internal static class ManifestAuditWorkbookWriter
{
    public static string Save(string outputRoot, ManifestMatchOptions options, ManifestMatchResult result)
    {
        Directory.CreateDirectory(outputRoot);
        string path = Path.Combine(outputRoot, "manifest_match_audit.xlsx");
        using FileStream stream = File.Create(path);
        using var archive = new ZipArchive(stream, ZipArchiveMode.Create);

        WriteEntry(archive, "[Content_Types].xml", ContentTypes());
        WriteEntry(archive, "_rels/.rels", PackageRelationships());
        WriteEntry(archive, "docProps/app.xml", AppProperties());
        WriteEntry(archive, "docProps/core.xml", CoreProperties());
        WriteEntry(archive, "xl/workbook.xml", Workbook());
        WriteEntry(archive, "xl/_rels/workbook.xml.rels", WorkbookRelationships());
        WriteEntry(archive, "xl/styles.xml", Styles());
        WriteEntry(archive, "xl/worksheets/sheet1.xml", SummarySheet(options, result));
        WriteEntry(archive, "xl/worksheets/sheet2.xml", DetailSheet(result));
        return path;
    }

    private static void WriteEntry(ZipArchive archive, string name, string content)
    {
        ZipArchiveEntry entry = archive.CreateEntry(name, CompressionLevel.Optimal);
        using StreamWriter writer = new(entry.Open(), new UTF8Encoding(false));
        writer.Write(content);
    }

    private static string SummarySheet(ManifestMatchOptions options, ManifestMatchResult result)
    {
        var rows = new List<string>
        {
            Row(1, TextCell("A1", "清单图片匹配审计", 1)),
            Row(3, TextCell("A3", "清单文件", 2), TextCell("B3", Path.GetFullPath(options.CsvPath))),
            Row(4, TextCell("A4", "图片根目录", 2), TextCell("B4", Path.GetFullPath(options.ImageRoot))),
            Row(5, TextCell("A5", "匹配文件夹层级", 2), NumberCell("B5", options.FolderLevel)),
            Row(6, TextCell("A6", "图片格式", 2), TextCell("B6", options.ImageFormat)),
            Row(8, TextCell("A8", "指标", 2), TextCell("B8", "数量", 2)),
            Row(9, TextCell("A9", "清单有效记录"), NumberCell("B9", result.RecordCount)),
            Row(10, TextCell("A10", "成功匹配记录"), NumberCell("B10", result.MatchedCount)),
            Row(11, TextCell("A11", "未匹配记录"), NumberCell("B11", result.UnmatchedCount)),
            Row(12, TextCell("A12", "多目录冲突记录"), NumberCell("B12", result.AmbiguousCount)),
            Row(13, TextCell("A13", "无效清单记录"), NumberCell("B13", result.InvalidCount)),
            Row(14, TextCell("A14", "匹配但无图片记录"), NumberCell("B14", result.EmptyImageCount)),
            Row(15, TextCell("A15", "扫描图片数（含清单重复）"), NumberCell("B15", result.ScannedImageCount)),
            Row(16, TextCell("A16", "实际待检测图片数（已去重）"), NumberCell("B16", result.ImageFiles.Count))
        };
        return WorksheetXml("<cols><col min=\"1\" max=\"1\" width=\"28\" customWidth=\"1\"/><col min=\"2\" max=\"2\" width=\"80\" customWidth=\"1\"/></cols>", rows, "A8:B16", "A9", merge: "A1:B1");
    }

    private static string DetailSheet(ManifestMatchResult result)
    {
        string[] headers = ["CSV行号", "start_time", "serial_number", "匹配键", "状态", "匹配文件夹", "图片数", "图片路径", "说明"];
        var rows = new List<string>
        {
            Row(1, headers.Select((header, i) => TextCell($"{Column(i + 1)}1", header, 2)).ToArray())
        };
        for (int index = 0; index < result.Rows.Count; index++)
        {
            ManifestMatchRow item = result.Rows[index];
            int rowNumber = index + 2;
            rows.Add(Row(rowNumber,
                NumberCell($"A{rowNumber}", item.CsvRow),
                item.StartTime is DateTime date
                    ? NumberCell($"B{rowNumber}", date.ToOADate(), 3)
                    : TextCell($"B{rowNumber}", item.StartTimeText),
                TextCell($"C{rowNumber}", item.SerialNumber),
                TextCell($"D{rowNumber}", item.ExpectedKey),
                TextCell($"E{rowNumber}", item.Status, StatusStyle(item.Status)),
                TextCell($"F{rowNumber}", item.MatchedFolder),
                NumberCell($"G{rowNumber}", item.ImageCount),
                TextCell($"H{rowNumber}", item.ImagePaths, 7),
                TextCell($"I{rowNumber}", item.Message)));
        }
        string columns = "<cols><col min=\"1\" max=\"1\" width=\"10\" customWidth=\"1\"/>" +
            "<col min=\"2\" max=\"2\" width=\"21\" customWidth=\"1\"/><col min=\"3\" max=\"5\" width=\"25\" customWidth=\"1\"/>" +
            "<col min=\"6\" max=\"6\" width=\"55\" customWidth=\"1\"/><col min=\"7\" max=\"7\" width=\"10\" customWidth=\"1\"/>" +
            "<col min=\"8\" max=\"9\" width=\"55\" customWidth=\"1\"/></cols>";
        string filter = result.Rows.Count == 0 ? string.Empty : $"A1:I{result.Rows.Count + 1}";
        return WorksheetXml(columns, rows, filter, "A2");
    }

    private static int StatusStyle(string status) => status switch
    {
        ManifestMatchingService.MatchedStatus => 4,
        ManifestMatchingService.UnmatchedStatus or ManifestMatchingService.AmbiguousStatus => 5,
        ManifestMatchingService.InvalidStatus or ManifestMatchingService.EmptyImageStatus => 6,
        _ => 0
    };

    private static string WorksheetXml(string columns, IEnumerable<string> rows, string autoFilter, string topLeftCell, string? merge = null) =>
        $"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">" +
        $"<sheetViews><sheetView showGridLines=\"0\" workbookViewId=\"0\"><pane ySplit=\"1\" topLeftCell=\"{topLeftCell}\" activePane=\"bottomLeft\" state=\"frozen\"/></sheetView></sheetViews>" +
        $"<sheetFormatPr defaultRowHeight=\"18\"/>{columns}<sheetData>{string.Concat(rows)}</sheetData>" +
        (merge is null ? string.Empty : $"<mergeCells count=\"1\"><mergeCell ref=\"{merge}\"/></mergeCells>") +
        (autoFilter.Length == 0 ? string.Empty : $"<autoFilter ref=\"{autoFilter}\"/>") + "</worksheet>";

    private static string Row(int number, params string[] cells) => $"<row r=\"{number}\">{string.Concat(cells)}</row>";
    private static string TextCell(string reference, string value, int style = 0) =>
        $"<c r=\"{reference}\" t=\"inlineStr\" s=\"{style}\"><is><t xml:space=\"preserve\">{Escape(value)}</t></is></c>";
    private static string NumberCell(string reference, double value, int style = 0) =>
        $"<c r=\"{reference}\" s=\"{style}\"><v>{value.ToString(CultureInfo.InvariantCulture)}</v></c>";
    private static string Escape(string value) => SecurityElement.Escape(value) ?? string.Empty;
    private static string Column(int number) => number <= 26 ? ((char)('A' + number - 1)).ToString() : throw new ArgumentOutOfRangeException(nameof(number));

    private static string ContentTypes() => "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"><Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/><Default Extension=\"xml\" ContentType=\"application/xml\"/><Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/><Override PartName=\"/xl/worksheets/sheet1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/><Override PartName=\"/xl/worksheets/sheet2.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/><Override PartName=\"/xl/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml\"/><Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/><Override PartName=\"/docProps/app.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.extended-properties+xml\"/></Types>";
    private static string PackageRelationships() => "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/><Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" Target=\"docProps/core.xml\"/><Relationship Id=\"rId3\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties\" Target=\"docProps/app.xml\"/></Relationships>";
    private static string AppProperties() => "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Properties xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties\" xmlns:vt=\"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes\"><Application>Corona Detection Workstation</Application></Properties>";
    private static string CoreProperties() => $"<?xml version=\"1.0\" encoding=\"UTF-8\"?><cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:dcterms=\"http://purl.org/dc/terms/\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"><dc:title>清单图片匹配审计</dc:title><dc:creator>Corona Detection Workstation</dc:creator><dcterms:created xsi:type=\"dcterms:W3CDTF\">{DateTime.UtcNow:yyyy-MM-dd'T'HH:mm:ss'Z'}</dcterms:created></cp:coreProperties>";
    private static string Workbook() => "<?xml version=\"1.0\" encoding=\"UTF-8\"?><workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\"><sheets><sheet name=\"匹配汇总\" sheetId=\"1\" r:id=\"rId1\"/><sheet name=\"匹配明细\" sheetId=\"2\" r:id=\"rId2\"/></sheets></workbook>";
    private static string WorkbookRelationships() => "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet1.xml\"/><Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet2.xml\"/><Relationship Id=\"rId3\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" Target=\"styles.xml\"/></Relationships>";
    private static string Styles() => "<?xml version=\"1.0\" encoding=\"UTF-8\"?><styleSheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><numFmts count=\"1\"><numFmt numFmtId=\"164\" formatCode=\"yyyy-mm-dd hh:mm:ss\"/></numFmts><fonts count=\"3\"><font><sz val=\"10\"/><name val=\"Microsoft YaHei\"/></font><font><b/><sz val=\"16\"/><color rgb=\"FFFFFFFF\"/><name val=\"Microsoft YaHei\"/></font><font><b/><sz val=\"10\"/><color rgb=\"FFFFFFFF\"/><name val=\"Microsoft YaHei\"/></font></fonts><fills count=\"6\"><fill><patternFill patternType=\"none\"/></fill><fill><patternFill patternType=\"gray125\"/></fill><fill><patternFill patternType=\"solid\"><fgColor rgb=\"FF1F4E78\"/></patternFill></fill><fill><patternFill patternType=\"solid\"><fgColor rgb=\"FF5B9BD5\"/></patternFill></fill><fill><patternFill patternType=\"solid\"><fgColor rgb=\"FFE2F0D9\"/></patternFill></fill><fill><patternFill patternType=\"solid\"><fgColor rgb=\"FFFFE699\"/></patternFill></fill></fills><borders count=\"1\"><border><left/><right/><top/><bottom/><diagonal/></border></borders><cellStyleXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\"/></cellStyleXfs><cellXfs count=\"8\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\"/><xf numFmtId=\"0\" fontId=\"1\" fillId=\"2\" borderId=\"0\" xfId=\"0\" applyFont=\"1\" applyFill=\"1\" applyAlignment=\"1\"><alignment horizontal=\"center\" vertical=\"center\"/></xf><xf numFmtId=\"0\" fontId=\"2\" fillId=\"3\" borderId=\"0\" xfId=\"0\" applyFont=\"1\" applyFill=\"1\"/><xf numFmtId=\"164\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\" applyNumberFormat=\"1\"/><xf numFmtId=\"0\" fontId=\"0\" fillId=\"4\" borderId=\"0\" xfId=\"0\" applyFill=\"1\"/><xf numFmtId=\"0\" fontId=\"0\" fillId=\"5\" borderId=\"0\" xfId=\"0\" applyFill=\"1\"/><xf numFmtId=\"0\" fontId=\"0\" fillId=\"5\" borderId=\"0\" xfId=\"0\" applyFill=\"1\"/><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\" applyAlignment=\"1\"><alignment wrapText=\"1\" vertical=\"top\"/></xf></cellXfs><cellStyles count=\"1\"><cellStyle name=\"Normal\" xfId=\"0\" builtinId=\"0\"/></cellStyles></styleSheet>";
}
