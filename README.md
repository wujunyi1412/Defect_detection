# Defect Detection

基于 ONNX Runtime 的工业视觉缺陷检测 DLL 库，用于表面缺陷的自动检测与分类，并附带 C#/WPF 上位机应用。

## 架构概览

采用 **双模型协同** 策略：

- **YOLOv8-seg** — 检测 8 种已知缺陷（Glue_overflow、Decolorization、Stain、Stripes、BrightStripes、Bright_clusters、Line_artifacts、LineArtifacts），输出分割 mask 和边界框
- **PatchCore + Faiss** — 基于正常样本特征分布的异常检测，输出 Abnormal（泛化异常）

两者结果通过可配置的类别过滤器融合（对比度、置信度、面积、宽高、AA 区域规则等），输出结构化的检测结果。对外提供 C API，可通过 P/Invoke 由 C#/WPF 上位机调用。

```
输入图像 (CV_32FC1 灰度图)
    │
    ├─→ [YOLOv8-seg] 分割推理 → 缺陷分类 + mask
    │        └─→ 类别过滤 → 对比度测量 → 暗缺陷精修 → AA 区域过滤
    ├─→ [PatchCore] Backbone 特征提取 → Faiss KNN → 异常分数图
    │        └─→ 热图后处理 → 种子生长/自适应精修
    │
    └─→ [结果融合] IoU 去重 → OK/NG 判定 → InferenceResult
              └─→ 可视化渲染（缺陷框/标签/遮罩）→ Halcon 图像句柄
```

## 目录结构

```
.
├── Corona_defect_detection/        # 核心 DLL 模块
│   ├── inference_dll.h/.cpp        # 引擎入口（PIMPL 模式）
│   ├── inference_c_api.h/.cpp      # C API 接口
│   ├── yolo_inference.h/.cpp       # YOLOv8-seg ONNX 推理
│   ├── patchcore_inference.h/.cpp  # PatchCore 推理 + Faiss
│   ├── config.ini                  # 运行时配置文件
│   └── internal/                   # 内部后处理模块
│       ├── yolo_postprocess.*      # YOLO 后处理与类别过滤
│       ├── yolo_categories.h       # 类别表（极性/过滤方向）
│       ├── patchcore_postprocess.* # PatchCore 热图后处理
│       ├── dark_defect_refiner*.*  # 暗缺陷精修（传统测量/自适应）
│       ├── image_utils*.*          # 对比度测量（自适应版）
│       ├── aa_yolo_filter.*        # AA 区域/形状规则过滤
│       ├── iqt_yolo_filter.*       # IQT 独立区域/形状规则过滤
│       ├── patchcore_yolo_iou_filter.* # PatchCore 与 YOLO 框 IoU 去重
│       ├── result_builder.*        # 结果融合与输出构建
│       └── overlay_renderer.*      # 可视化渲染 → Halcon 图像句柄
├── config/                         # 配置加载模块
│   ├── config.h/.cpp               # 配置结构体 + 加载逻辑
│   └── ini_parser.h/.cpp           # INI 解析器
├── image_process/                  # 图像处理工具
│   └── image_process.h/.cpp        # Letterbox / HWC→CHW / 归一化 / 双线性插值
├── logging/                        # 日志模块
│   └── logger.h/.cpp
├── profiling/                      # 性能计时模块
│   └── performance_timer.h/.cpp
├── evaluation_metrics/             # 评测指标库（含 C API 与单元测试）
├── tests/                          # 冒烟测试
│   └── smoke_tests.cpp
├── Use_DLL/                        # DLL 调用示例（独立 CMake 项目）
├── windows_corona_detection/       # C#/WPF 上位机应用
├── onnx_model/                     # 模型文件（不纳入版本管理）
│   ├── yolov8_seg.onnx
│   ├── patchcore_backbone.onnx
│   ├── nnscorer_search_index.faiss
│   └── model_metadata.json
├── 3rdparty/                       # 第三方依赖（不纳入版本管理）
├── build_windows_app.bat           # 一键构建：原生库 + 测试 + WPF 应用
├── package_windows_app.bat         # 应用打包脚本
└── CMakeLists.txt
```

## 环境要求

- **编译器**: Visual Studio 2022，支持 C++17
- **CMake**: 3.16+
- **.NET SDK**: dotnet（构建 WPF 应用时需要）
- **第三方库**:
  - OpenCV 4.5.3（core, imgproc, imgcodecs）
  - ONNX Runtime 1.24.4
  - Faiss + OpenBLAS
  - HALCON 12（用于可视化输出；根目录通过 CMake 变量 `HALCON_ROOT` 配置，默认 `D:/halcon_12`）

## 构建

### 原生 DLL

```bash
# 生成项目
cmake -S . -B ./build -G "Visual Studio 17 2022" -A x64

# 编译 Release（DLL + 冒烟测试 + 评测库）
cmake --build ./build --config Release

# 运行冒烟测试
./build/bin/corona_smoke_tests.exe
```

构建后 DLL 和依赖库（onnxruntime、halcon、faiss、opencv 等）自动复制到 `build/bin/Release`。

### 一键构建完整应用（推荐）

```bash
build_windows_app.bat
```

该脚本自动完成：环境检查 → 编译原生库 → 运行评测指标测试 → 部署运行时 DLL 到 WPF Runtime 目录 → 编译发布 `CoronaDetection.exe`。

## 配置文件

```ini
[models]         # 模型路径（支持相对路径，相对 INI 文件所在目录解析）
yolo_model_path = yolov8_seg.onnx
patchcore_model_path = patchcore_backbone.onnx
faiss_index_path = nnscorer_search_index.faiss
metadata_path = model_metadata.json

[ort]            # ONNX Runtime
intra_threads = 4

[yolo]           # YOLO 推理参数
enabled = 1            # 是否启用 YOLO 模型：1=启用，0=禁用
score_threshold = 0.25
iou_threshold = 0.2
nms_class_aware = 1    # NMS 模式：0=全局，1=按类别

[patchcore]      # PatchCore 推理参数
enabled = 1            # 是否启用 PatchCore 模型：1=启用，0=禁用
score_threshold = 1.4
area_threshold = 1.4
mask_area_threshold = 0.3

[log]            # 日志
enabled = 1
level = debug          # debug / info / warning / error
log_to_stderr = 1
log_to_file = 0
file_path = logs\inspection.log

[post]           # 后处理与可视化
contrast_mode = 1                    # 0=局部环形背景，1=自适应局部平面，2=全 FOV 中位数背景（排除黑边）
patchcore_yolo_iou_threshold = 0.0  # PatchCore 与任一最终 YOLO 框的 IoU 严格大于此值时移除；设为 1 可关闭
draw_defect_box = 1                 # 是否绘制缺陷框
expand_defect_box = 1               # 缺陷框四边是否各外扩 2 像素
draw_box_details = 1                # 是否绘制缺陷信息标签
draw_defect_mask = 0                # 是否绘制缺陷遮罩
concat_original_image = 0           # 1：输出"左原图、右标注图"；0：仅输出标注图
defect_mask_color_r = 255           # 遮罩颜色（RGB）与透明度
defect_mask_color_g = 0
defect_mask_color_b = 0
defect_mask_alpha = 0.05

[AA]             # YOLO 最终结果的可配置区域/形状过滤
enabled = 1
center_y_min = 665
center_y_max = 715
min_width_height_ratio = 3.0
categories = Stain
position_filter_enabled = 1
position_categories = Stain
position_x_min = 430
position_x_max = 480
position_y_min = 280
position_y_max = 340
position_width_min = 145
position_width_max = 200
position_height_min = 175
position_height_max = 245

[IQT]            # 与 AA 同级且参数完全独立；在 PatchCore/YOLO IoU 过滤后执行
enabled = 0
center_y_min = 665
center_y_max = 715
min_width_height_ratio = 3.0
max_width = 0      # 宽度大于此值时也会过滤；0 表示关闭宽度条件
categories = Stain
```

每个缺陷类别有独立的过滤器小节，共 9 个：`[Abormal_config]`（PatchCore 异常）、`[Glue_overflow_config]`、`[Decolorization_config]`、`[Stain_config]`、`[Stripes_config]`、`[BrightStripes_config]`、`[Bright_clusters_config]`、`[Line_artifacts_config]`、`[LineArtifacts_config]`。

每个类别过滤器支持：启用/禁用、置信度阈值（`confidence_threshold`）、对比度阈值（`contrast_threshold`）、最小宽/高/面积；`use_traditional_measure=1` 时使用传统图像算法重算面积、宽高与对比度（当前仅 Stain 启用）。对比度过滤方向由类别极性决定：暗缺陷保留 `contrast < 阈值`，亮缺陷保留 `contrast >= 阈值`，详见 [yolo_categories.h](Corona_defect_detection/internal/yolo_categories.h)。

`contrast_mode=2` 时，背景亮度取整幅 FOV 的灰度中位数。统计前会剔除与图像四边连通的低灰度黑框；缺陷亮度按类别极性，取 mask 内最暗 10%（暗缺陷）或最亮 10%（亮缺陷）像素的平均值，最终对比度为 `缺陷亮度 / 背景亮度`。

`[Abormal_config]` 额外支持双分数阈值：`score >= score_threshold_2` 直接判定异常；`score >= score_threshold_1` 时需同时满足 `area_ratio >= area_threshold_1`。

## API 使用

### C++ 接口

```cpp
InspectionDLL::InspectionEngine engine;
if (!engine.Initialize("config.ini")) {
    // engine.GetLastError() 获取错误信息
}

InferenceResult result;
if (engine.ProcessImage(image, result)) {
    // result.result == "OK" / "NG"，result.details ...
}

engine.Release();  // 仅重置状态；引擎对象由析构自动释放
```

### C API（C# P/Invoke 友好）

```cpp
InspectionHandle handle = Inspection_Create();
char message[100] = {0};
if (Inspection_Initialize(handle, "config.ini", message) == 0) {
    InspectionResultC result;
    Inspection_ProcessFloatArray(handle, data, len, width, height, &result, imageHandle, message);
    // ...
}
Inspection_Destroy(handle);  // Create 与 Destroy 必须成对调用，否则内存泄漏
```

### 输出结构

```cpp
struct DetectionResult {
    std::string name;      // 缺陷名称
    int area;              // 面积（像素）
    float x, y, w, h;     // 边界框
    float contrast;        // 对比度
    float score;           // 置信度
    cv::Mat mask;          // 分割/异常遮罩
};

struct InferenceResult {
    std::string result;    // "OK" 或 "NG"
    std::vector<DetectionResult> details;
    float yolo_score;
    float patchcore_score;
    float patchcore_area_ratio;
};
```

## 缺陷类别

| 类别 | 检测模型 | 说明 |
|------|----------|------|
| Abnormal | PatchCore | 异常区域（泛化检测） |
| Glue_overflow | YOLOv8 | 胶溢出 |
| Decolorization | YOLOv8 | 脱色 |
| Stain | YOLOv8 | 污点 |
| Stripes | YOLOv8 | 条纹 |
| BrightStripes | YOLOv8 | 亮条纹 |
| Bright_clusters | YOLOv8 | 亮斑簇 |
| Line_artifacts | YOLOv8 | 线状伪影（下划线类别） |
| LineArtifacts | YOLOv8 | 线状伪影 |
