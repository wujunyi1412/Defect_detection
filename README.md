# Defect Detection

基于 ONNX Runtime 的工业视觉缺陷检测 DLL 库，用于表面缺陷的自动检测与分类。

## 架构概览

采用 **双模型协同** 策略：

- **YOLOv8-seg** — 检测已知缺陷类型（Stain 污点、BrightStripes 亮条纹、LineArtifacts 线状伪影），输出分割 mask 和边界框
- **PatchCore + Faiss** — 基于正常样本特征分布的异常检测，捕获未知异常模式

两者结果通过可配置的类别过滤器融合，输出结构化的检测结果。对外提供 C API，可通过 P/Invoke 由 C# 上位机调用。

```
输入图像 (CV_32FC1 灰度图)
    │
    ├─→ [YOLOv8-seg] 分割推理 → 缺陷分类 + mask
    ├─→ [PatchCore] Backbone 特征提取 → Faiss KNN → 异常分数图
    │
    └─→ [结果融合] 按类别过滤 → OK/NG 判定 → InferenceResult
```

## 目录结构

```
.
├── Corona_defect_detection/   # 核心 DLL 模块
│   ├── inference_dll.h/.cpp   # 引擎入口（PIMPL 模式）
│   ├── inference_c_api.h/.cpp # C API 接口
│   ├── yolo_inference.h/.cpp  # YOLOv8-seg ONNX 推理
│   ├── patchcore_inference.h/.cpp  # PatchCore 推理 + Faiss
│   ├── config.ini             # 运行时配置文件
│   └── internal/              # 内部后处理模块
│       ├── yolo_postprocess.h/.cpp
│       ├── patchcore_postprocess.h/.cpp
│       ├── dark_defect_refiner.h/.cpp
│       └── result_builder.h/.cpp
├── config/                    # 配置加载模块
│   ├── config.h/.cpp          # 配置结构体 + 加载逻辑
│   └── ini_parser.h/.cpp      # INI 解析器
├── image_process/             # 图像处理工具
│   └── image_process.h/.cpp   # Letterbox / HWC→CHW / 归一化 / 双线性插值
├── Use_DLL/                   # DLL 调用示例
│   ├── use_example.cpp        # 示例代码
│   └── test.tif               # 测试图像
├── onnx_model/                # 模型文件（不纳入版本管理）
│   ├── yolov8_seg.onnx
│   ├── patchcore_backbone.onnx
│   ├── nnscorer_search_index.faiss
│   └── model_metadata.json
├── 3rdparty/                  # 第三方依赖（不纳入版本管理）
│   ├── install_opencv/
│   ├── onnxruntime-win-x64-1.24.4/
│   ├── install_faiss/
│   └── install_openblas/
└── CMakeLists.txt
```

## 环境要求

- **编译器**: Visual Studio 2022，支持 C++17
- **CMake**: 3.20+
- **第三方库**:
  - OpenCV 4.5.3（core, imgproc, imgcodecs）
  - ONNX Runtime 1.24.4
  - Faiss + OpenBLAS
  - HALCON 12（可选，用于可视化输出）

## 构建

```bash
# 生成项目
cmake -S . -B ./build -G "Visual Studio 17 2022" -A x64

# 编译 Release
cmake --build ./build --config Release

# 编译示例程序（可选）
cmake -S . -B ./build -G "Visual Studio 17 2022" -A x64 -DBUILD_EXAMPLES=ON
cmake --build ./build --config Release
```

构建后 DLL 和依赖库自动复制到输出目录。

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
score_threshold = 0.25
iou_threshold = 0.2
nms_class_aware = 0

[patchcore]      # PatchCore 推理参数
score_threshold = 1.4
area_threshold = 1.4
mask_area_threshold = 0.3

[post]           # 后处理
patchcore_yolo_iou_threshold = 0.0  # PatchCore 与任一最终 YOLO 框的 IoU 严格大于此值时移除；范围 0~1，设为 1 可关闭
concat_original_image = 0  # 1：输出“左原图、右标注图”；0：仅输出标注图

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

[Abnormal_config]     # Abnormal 缺陷过滤器
[Stain_config]        # Stain 缺陷过滤器
[Brightstripes_config]# BrightStripes 缺陷过滤器
[Lineartifacts_config]# LineArtifacts 缺陷过滤器
```

每个缺陷过滤器支持：启用/禁用、置信度阈值、对比度阈值、最小宽/高/面积。

## API 使用

### C++ 接口

```cpp
InspectionDLL::InspectionEngine engine;
engine.Initialize("config.ini");

InferenceResult result;
engine.ProcessImage(image, result);
// result.ok, result.defects, result.yolo_score, result.patchcore_score ...

engine.Release();
```

### C API（C# P/Invoke 友好）

```cpp
InspectionHandle handle = Inspection_Create();
Inspection_Initialize(handle, "config.ini", message);
Inspection_ProcessFloatArray(handle, data, len, width, height, &result, &imageHandle, msg);
Inspection_Destroy(handle);
```

### 输出结构

```cpp
struct DetectionResult {
    std::string name;      // 缺陷名称
    float x, y, w, h;     // 边界框
    int area;              // 面积（像素）
    float contrast;        // 对比度
    float score;           // 置信度
};

struct InferenceResult {
    bool ok;               // OK / NG
    std::vector<DetectionResult> defects;
    float yolo_score;
    float patchcore_score;
    float patchcore_area_ratio;
};
```

## 缺陷类别

| 类别 | 检测模型 | 说明 |
|------|----------|------|
| Abnormal | PatchCore | 异常区域（泛化检测） |
| Stain | YOLOv8 | 污点 |
| BrightStripes | YOLOv8 | 亮条纹 |
| LineArtifacts | YOLOv8 | 线状伪影 |
