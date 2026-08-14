# Evaluation Metrics

Standalone C++17 detection-evaluation library. It has no OpenCV, HALCON, ONNX Runtime, or .NET dependency.

The public C++ API is in `include/evaluation_metrics/evaluation_metrics.h`. A stable C ABI for P/Invoke and other languages is provided by `evaluation_metrics_c_api.h`.

Current modules:

- axis-aligned box IoU;
- one-to-one Hungarian matching (maximum qualified matches, then maximum total IoU);
- detection metrics: Precision, Recall, F1, localization recall, matched-class accuracy, and mean matched IoU.

New metrics should be added to the C++ input/result types first, then exposed through the C ABI. UI, JSON parsing, label normalization, report formatting, and rendering deliberately remain outside this library.

