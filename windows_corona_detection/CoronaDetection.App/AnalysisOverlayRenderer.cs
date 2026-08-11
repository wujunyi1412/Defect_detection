using System.Globalization;
using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace CoronaDetection;

internal static class AnalysisOverlayRenderer
{
    private static readonly Color MissedColor = Color.FromRgb(230, 35, 45);
    private static readonly Color FalsePositiveColor = Color.FromRgb(20, 180, 80);

    public static void Render(
        string imagePath,
        string outputPath,
        IReadOnlyList<AnalysisError> errors,
        bool overwrite)
    {
        using var input = new FileStream(imagePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        BitmapFrame frame = BitmapDecoder.Create(
            input, BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.OnLoad).Frames[0];
        int width = frame.PixelWidth;
        int height = frame.PixelHeight;
        double scale = Math.Max(1.0, Math.Min(width, height) / 800.0);
        double penThickness = Math.Max(2.0, 2.2 * scale);
        double fontSize = Math.Clamp(16.0 * scale, 16.0, 34.0);

        var visual = new DrawingVisual();
        using (DrawingContext drawing = visual.RenderOpen())
        {
            drawing.DrawImage(frame, new Rect(0, 0, width, height));
            foreach (AnalysisError error in errors)
            {
                Color color = error.Type == AnalysisErrorType.Missed
                    ? MissedColor
                    : FalsePositiveColor;
                var brush = new SolidColorBrush(color);
                brush.Freeze();
                var pen = new Pen(brush, penThickness);
                pen.Freeze();

                Rect box = Clamp(error.Box, width, height);
                drawing.DrawRectangle(null, pen, box);
                DrawLabel(drawing, error.DisplayText, box, brush, fontSize, width, height);
            }
        }

        var bitmap = new RenderTargetBitmap(
            width, height, 96.0, 96.0, PixelFormats.Pbgra32);
        bitmap.Render(visual);
        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(bitmap));

        FileMode mode = overwrite ? FileMode.Create : FileMode.CreateNew;
        using var output = new FileStream(outputPath, mode, FileAccess.Write, FileShare.None);
        encoder.Save(output);
    }

    private static void DrawLabel(
        DrawingContext drawing,
        string text,
        Rect box,
        Brush foreground,
        double fontSize,
        double imageWidth,
        double imageHeight)
    {
        var formatted = new FormattedText(
            text,
            CultureInfo.GetCultureInfo("zh-CN"),
            FlowDirection.LeftToRight,
            new Typeface("Microsoft YaHei UI"),
            fontSize,
            Brushes.White,
            1.0)
        {
            MaxTextWidth = Math.Max(1.0, imageWidth)
        };
        double padding = Math.Max(3.0, fontSize * 0.22);
        double labelWidth = Math.Min(imageWidth, formatted.Width + padding * 2.0);
        double labelHeight = formatted.Height + padding * 2.0;
        double x = Math.Clamp(box.X, 0.0, Math.Max(0.0, imageWidth - labelWidth));
        double y = box.Y >= labelHeight
            ? box.Y - labelHeight
            : Math.Min(imageHeight - labelHeight, box.Bottom);
        y = Math.Max(0.0, y);

        var background = foreground.Clone();
        background.Opacity = 0.86;
        background.Freeze();
        drawing.DrawRectangle(background, null, new Rect(x, y, labelWidth, labelHeight));
        drawing.DrawText(formatted, new Point(x + padding, y + padding));
    }

    private static Rect Clamp(BoundingBox box, double width, double height)
    {
        double x1 = Math.Clamp(box.X, 0.0, width);
        double y1 = Math.Clamp(box.Y, 0.0, height);
        double x2 = Math.Clamp(box.Right, 0.0, width);
        double y2 = Math.Clamp(box.Bottom, 0.0, height);
        return new Rect(x1, y1, Math.Max(1.0, x2 - x1), Math.Max(1.0, y2 - y1));
    }
}
