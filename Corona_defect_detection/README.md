# Corona 缺陷检测 DLL

基于 ONNX Runtime 的 Corona（电晕）缺陷检测推理库，输出为 Windows DLL，通过 C API 供外部（如 Halcon）调用。

---

## 目录结构

```
Corona_defect_detection/
├── inference_dll.h / .cpp      ← 主引擎（PIMPL 封装，对外 C++ 接口）
├── inference_c_api.h / .cpp    ← C API 层（extern "C"，Halcon 可视化）
├── patchcore_inference.h / .cpp ← PatchCore ONNX 推理器
├── yolo_inference.h / .cpp     ← YOLOv8-Seg ONNX 推理器
├── config.ini                  ← 配置文件（模型路径、阈值、过滤器）
├── config.h / .cpp             ← INI 解析
└── internal/                   ← 后处理模块（详见 internal/README.md）
    ├── image_utils             ← 图像预处理 & 统计工具
    ├── yolo_postprocess        ← YOLO 检测结果 → YoloDerived
    ├── patchcore_postprocess   ← PatchCore 异常图 → PatchCoreDerived
    ├── dark_defect_refiner     ← 暗缺陷几何精修
    └── result_builder          ← 融合 + 过滤 → InferenceResult
```

---

## 架构分层

```
┌─────────────────────────────────────────────┐
│  inference_c_api.cpp   (C API / extern "C") │  ← Halcon 调用入口
├─────────────────────────────────────────────┤
│  inference_dll.cpp     (InspectionEngine)   │  ← 主控 & 状态管理
├──────────────┬──────────────────────────────┤
│  yolo_inference      patchcore_inference    │  ← ONNX 推理层
│  (YOLOv8Segmentor)   (PatchCoreDetector)    │
├──────────────┴──────────────────────────────┤
│  internal/  (后处理 & 结果融合)              │  ← 传统 CV & 业务逻辑
├─────────────────────────────────────────────┤
│  config.h / config.ini  (配置)              │  ← INI 解析 & 参数管理
└─────────────────────────────────────────────┘
```

---

## 各文件说明

### 1. inference_dll — 主引擎

**头文件** (`inference_dll.h`) 定义对外 C++ 接口：

| 结构体 | 说明 |
|---|---|
| `DetectionResult` | 单个缺陷：名称、面积、框坐标、对比度、置信度 |
| `InferenceResult` | 整体结果：OK/NG、缺陷列表、YOLO/PatchCore 分数 |
| `InspectionEngine` | 主类，PIMPL 模式，隐藏内部实现 |

**实现** (`inference_dll.cpp`) 使用 PIMPL（Pointer to Implementation）：

- `Impl` 持有两个推理器（`YOLOv8Segmentor`、`PatchCoreDetector`）和所有阈值/过滤器配置
- `Initialize(config_path)` → 解析 INI → 初始化两个 ONNX 模型
- `ProcessImage()` 主流程：
  1. 图像预处理（`ProcessTIF32ForPatchcore` / `ProcessForYolo`）
  2. 双模型并行推理
  3. 后处理（`AnalyzePatchCore` / `AnalyzeYolo`）
  4. 结果融合（`ComposeOutputWithDefectFilter`）
- `SetThresholds` / `SetDarkClustersThreshold` / `SetYoloNmsMode` — 运行时动态调整参数
- `ProcessImagePath` — 从文件路径读取图像
- `ProcessFloatArry` — 从 float 数组构造 cv::Mat（兼容外部内存）

### 2. inference_c_api — C 语言接口

供 C 语言调用者（如 Halcon、C# P/Invoke）使用，`extern "C"` 导出：

| 函数 | 说明 |
|---|---|
| `Inspection_Create` / `Inspection_Destroy` | 创建/销毁引擎实例 |
| `Inspection_Initialize` | 加载配置 & 模型 |
| `Inspection_ProcessFloatArray` | 推理 float 数组图像，返回结果 + Halcon 可视化图像 |
| `Inspection_ProcessFloatArray_npy` | 同上，但按 5 个缺陷类别分数组输出 |
| `Inspection_ProcessImagePath` | 从文件路径推理 |
| `Inspection_SetThresholds` 等 | 运行时调参 |

内部还包含：
- **可视化** (`SaveResultOverlay`)：将检测框 + 标签绘制到图像上，序列化为 Halcon `HObject`
- **文本适配** (`FitTextToWidth` / `DrawLabelBlock`)：检测框旁的文字标签自动换行/截断

### 3. patchcore_inference — PatchCore 推理

`PatchCoreDetector` 实现 PatchCore 异常检测算法：

1. **预处理** (`PreprocessToNCHW`)：图像缩放 + padding 到 224×224，ImageNet 归一化
2. **主干推理** (`RunBackbone`)：ONNX backbone 提取多层特征
3. **Embedding 提取** (`ExtractEmbeddings`)：多层特征 → patch embedding（含 avgpool/adaptive pool、双线性 resize）
4. **异常评分** (`ComputeAnomalyScores`)：Faiss 检索 k-NN 距离 → patch-level anomaly score
5. **元数据加载** (`LoadMetadata`)：从 JSON/PKL 读取 backbone 配置（层名、pool 策略、embedding 维度等）

输出 `PatchCoreResult`：`image_score`（整图最高异常分）+ `patch_scores`（28×28 分数图）+ `meta`（padding/缩放元数据）。

### 4. yolo_inference — YOLOv8-Seg 推理

`YOLOv8Segmentor` 实现 YOLOv8 实例分割推理：

1. **预处理** (`Preprocess`)：Letterbox resize → HWC→CHW 归一化
2. **ONNX 推理** → 检测头 + proto mask
3. **后处理** (`Postprocess`)：
   - 解析检测框（cx/cy/w/h）→ 角点坐标
   - NMS（支持 Global / ClassAware 两种模式）
   - 实例分割 mask：`mask_coeff × proto → sigmoid → crop → resize → 阈值化`
4. **Warm-up**：初始化时跑一次空推理预热

输出 `Detection` 列表：`class_id`（0=Stain, 1=BrightStripes, 2=LineArtifacts）、置信度、边界框、二值 mask。

### 5. config.ini — 配置文件

```ini
[models]          # ONNX 模型 & Faiss 索引路径
[ort]             # ONNX Runtime 线程数
[yolo]            # YOLO 阈值 & NMS 模式
[patchcore]       # PatchCore 阈值
[post]            # 后处理与输出图配置（concat_original_image=1 时左原图、右标注图）
[AA]              # YOLO 最终结果的可配置区域/形状过滤
[Abormal_config]  # Abnormal 类别过滤器（双阈值：score + area）
[Stain_config]    # Stain 类别过滤器（含传统精修开关）
[Darkclusters_config]
[Brightstripes_config]
[Lineartifacts_config]
```

每个类别过滤器包含：`Enable`、`confidence_threshold`、`contrast_threshold`、`min_width/height/area`。Stain/DarkClusters 额外支持 `use_traditional_measure`（启用 `RefineDarkDefectGeometryOnly` 精修）。

### 6. internal/ — 后处理模块

详见 `internal/README.md`，负责：
- 图像预处理 & 统计工具（`image_utils`）
- YOLO/PatchCore 结果后处理（`yolo_postprocess` / `patchcore_postprocess`）
- 暗缺陷几何精修（`dark_defect_refiner`）
- 双模型结果融合 & 类别过滤（`result_builder`）

---

## 完整数据流

```
config.ini
    │
    ▼
Inspection_Initialize()  ──→  InspectionEngine::Initialize()
    │                              │
    │                    ┌─────────┴──────────┐
    │                    ▼                    ▼
    │            YOLOv8Segmentor      PatchCoreDetector
    │              (ONNX)               (ONNX + Faiss)
    │
    ▼
Inspection_ProcessFloatArray()
    │
    ├─→ ProcessTIF32ForPatchcore() ──→ PatchCoreDetector::Infer()
    │                                       │
    │                                       ▼
    │                              AnalyzePatchCore()
    │                              (十字线检测、对比度)
    │                                       │
    │                                       ▼
    │                              PatchCoreDerived
    │
    ├─→ ProcessForYolo() ──→ YOLOv8Segmentor::Infer()
    │                              │
    │                              ▼
    │                     AnalyzeYolo()
    │                     (Stain→DarkClusters 重分类)
    │                              │
    │                              ▼
    │                     YoloDerived
    │
    └─→ ComposeOutputWithDefectFilter()
              │
              ├── YOLO 过滤（按类别 filter，可选 RefineDarkDefectGeometryOnly）
              ├── PatchCore 过滤（按 AbnormalFilter）
              └── OK/NG 判定
              │
              ▼
         InferenceResult
              │
              ▼
    SaveResultOverlay()  ──→  Halcon HObject (可视化)
```
