using System.Runtime.InteropServices;
using System.Text;

namespace CoronaDetection;

internal static class NativeMethods
{
    internal const int MaxDetections = 128;
    internal const int StatusOk = 0;

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    internal struct InspectionDetection
    {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
        public string Name;
        public int Area;
        public float X;
        public float Y;
        public float W;
        public float H;
        public float Contrast;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    internal struct InspectionResult
    {
        public int Result;
        public int DetailsCount;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = MaxDetections)]
        public InspectionDetection[] Details;
        public float YoloScore;
        public float PatchcoreScore;
        public float PatchcoreAreaRatio;

        public static InspectionResult Create() =>
            new() { Details = new InspectionDetection[MaxDetections] };
    }

    [DllImport("Corona_defect_detection.dll", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr Inspection_Create();

    [DllImport("Corona_defect_detection.dll", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void Inspection_Destroy(IntPtr handle);

    [DllImport("Corona_defect_detection.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern int Inspection_Initialize(
        IntPtr handle,
        [MarshalAs(UnmanagedType.LPStr)] string configPath,
        StringBuilder message);

    [DllImport("Corona_defect_detection.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern int Inspection_ProcessImagePath(
        IntPtr handle,
        [MarshalAs(UnmanagedType.LPStr)] string imagePath,
        ref InspectionResult result);

    [DllImport("Corona_defect_detection.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern int Inspection_ProcessImagePathTimed(
        IntPtr handle,
        [MarshalAs(UnmanagedType.LPStr)] string imagePath,
        ref InspectionResult result,
        out double inferenceMs);

    [DllImport("Corona_defect_detection.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern int Inspection_ProcessImagePathToOverlayFile(
        IntPtr handle,
        [MarshalAs(UnmanagedType.LPStr)] string imagePath,
        [MarshalAs(UnmanagedType.LPStr)] string outputPath,
        ref InspectionResult result,
        StringBuilder message,
        out double inferenceMs,
        out double saveMs);

    [DllImport("Corona_defect_detection.dll", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void Inspection_Release(IntPtr handle);
}
