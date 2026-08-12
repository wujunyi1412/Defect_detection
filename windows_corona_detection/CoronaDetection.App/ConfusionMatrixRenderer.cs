using System.Globalization;
using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace CoronaDetection;

internal static class ConfusionMatrixRenderer
{
    public static void Render(string outputPath, ConfusionMatrixData matrix, bool overwrite)
    {
        const double titleHeight = 72.0;
        const double cellWidth = 160.0;
        const double cellHeight = 76.0;
        const double rowLabelWidth = 230.0;
        const double columnLabelHeight = 115.0;
        const double margin = 24.0;
        int width = (int)Math.Ceiling(margin * 2 + rowLabelWidth + matrix.PredictedLabels.Count * cellWidth);
        int height = (int)Math.Ceiling(
            margin * 2 + titleHeight + columnLabelHeight + matrix.ActualLabels.Count * cellHeight);
        int maxCount = 0;
        foreach (int count in matrix.Counts) maxCount = Math.Max(maxCount, count);

        var visual = new DrawingVisual();
        using (DrawingContext drawing = visual.RenderOpen())
        {
            drawing.DrawRectangle(Brushes.White, null, new Rect(0, 0, width, height));
            DrawText(drawing, "混淆矩阵", 25, FontWeights.Bold,
                Brushes.Black, new Rect(0, margin, width, titleHeight), TextAlignment.Center);

            double gridX = margin + rowLabelWidth;
            double gridY = margin + titleHeight + columnLabelHeight;
            DrawText(drawing, "实际类别 \\ 预测类别", 15, FontWeights.SemiBold,
                Brushes.Black, new Rect(margin, gridY - columnLabelHeight, rowLabelWidth, columnLabelHeight),
                TextAlignment.Center);
            for (int column = 0; column < matrix.PredictedLabels.Count; column++)
            {
                DrawText(drawing, matrix.PredictedLabels[column], 15, FontWeights.SemiBold,
                    Brushes.Black,
                    new Rect(gridX + column * cellWidth, gridY - columnLabelHeight, cellWidth, columnLabelHeight),
                    TextAlignment.Center);
            }

            var borderPen = new Pen(new SolidColorBrush(Color.FromRgb(205, 211, 220)), 1.0);
            borderPen.Freeze();
            for (int row = 0; row < matrix.ActualLabels.Count; row++)
            {
                DrawText(drawing, matrix.ActualLabels[row], 15, FontWeights.SemiBold,
                    Brushes.Black,
                    new Rect(margin, gridY + row * cellHeight, rowLabelWidth, cellHeight),
                    TextAlignment.Right);
                for (int column = 0; column < matrix.PredictedLabels.Count; column++)
                {
                    int count = matrix.Counts[row, column];
                    double intensity = maxCount == 0 ? 0.0 : (double)count / maxCount;
                    Color color = HeatColor(intensity);
                    var fill = new SolidColorBrush(color);
                    fill.Freeze();
                    Rect cell = new(
                        gridX + column * cellWidth,
                        gridY + row * cellHeight,
                        cellWidth,
                        cellHeight);
                    drawing.DrawRectangle(fill, borderPen, cell);
                    Brush textBrush = intensity >= 0.55 ? Brushes.White : Brushes.Black;
                    DrawText(drawing, count.ToString(CultureInfo.InvariantCulture), 20,
                        FontWeights.Bold, textBrush, cell, TextAlignment.Center);
                }
            }
        }

        var bitmap = new RenderTargetBitmap(width, height, 96, 96, PixelFormats.Pbgra32);
        bitmap.Render(visual);
        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(bitmap));
        FileMode mode = overwrite ? FileMode.Create : FileMode.CreateNew;
        using var output = new FileStream(outputPath, mode, FileAccess.Write, FileShare.None);
        encoder.Save(output);
    }

    private static Color HeatColor(double intensity)
    {
        intensity = Math.Clamp(intensity, 0.0, 1.0);
        byte red = (byte)Math.Round(240 - 195 * intensity);
        byte green = (byte)Math.Round(247 - 122 * intensity);
        byte blue = (byte)Math.Round(255 - 70 * intensity);
        return Color.FromRgb(red, green, blue);
    }

    private static void DrawText(
        DrawingContext drawing,
        string text,
        double fontSize,
        FontWeight fontWeight,
        Brush brush,
        Rect area,
        TextAlignment alignment)
    {
        var formatted = new FormattedText(
            text,
            CultureInfo.GetCultureInfo("zh-CN"),
            FlowDirection.LeftToRight,
            new Typeface(new FontFamily("Microsoft YaHei UI"), FontStyles.Normal, fontWeight, FontStretches.Normal),
            fontSize,
            brush,
            1.0)
        {
            MaxTextWidth = Math.Max(1.0, area.Width - 12.0),
            MaxTextHeight = Math.Max(1.0, area.Height - 8.0),
            TextAlignment = alignment,
            Trimming = TextTrimming.CharacterEllipsis
        };
        double x = area.X + 6.0;
        double y = area.Y + (area.Height - formatted.Height) / 2.0;
        drawing.DrawText(formatted, new Point(Math.Max(area.X, x), Math.Max(area.Y, y)));
    }
}
