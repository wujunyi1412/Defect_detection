# YOLO 输出类别修改指南

本文按当前 `Corona_defect_detection` 实现整理。当前 YOLOv8-Seg 模型输出 **10 类**，类别顺序必须与训练数据和 ONNX 模型完全一致。

## 1. 当前类别顺序

| ONNX `class_id` | YOLO 内部类别名 | 对比度极性 | 最终默认输出名 |
|---:|---|---|---|
| 0 | `Glue_overflow` | Dark | `GlueOverflow` |
| 1 | `Decolorization` | Dark | `Decolorization` |
| 2 | `Stain` | Dark | `Stain` |
| 3 | `Stripes` | Auto | `Abnormal` |
| 4 | `BrightStripes` | Bright | `Abnormal` |
| 5 | `Bright_clusters` | Bright | `BrightClusters` |
| 6 | `Line_artifacts` | Dark | `LineArtifacts` |
| 7 | `LineArtifacts` | Bright | `BrightClusters` |
| 8 | `Stripe_horizontal` | Auto | `Abnormal` |
| 9 | `Stripe_vertical` | Auto | `Abnormal` |

`Stripe_horizontal` 和 `Stripe_vertical` 在过滤阶段仍是两个独立类别，最终通过 `[category_mapping]` 归并为 `Abnormal`。这样可以分别设置置信度、尺寸和对比度阈值，同时保持外部业务接口的 `Abnormal` 分类不变。

新增两类目前使用 `Auto` 对比度极性，因为名称只描述方向，没有明确亮/暗属性。若模型定义明确，应在 `internal/yolo_categories.h` 中改为 `Dark` 或 `Bright`；其中暗缺陷还应把 `keep_contrast_below_threshold` 设为 `true`。

## 2. 本次从 8 类改为 10 类需要修改的位置

### 2.1 模型输出解析

文件：`yolo_inference.h`

```cpp
static constexpr int NUM_CLASSES = 10;
```

该常量同时控制类别分数遍历数量、mask 系数起始偏移、检测输出特征维校验和类别感知 NMS 的类别数量。

当前分割模型每个候选框的特征数应满足：

```text
features_per_detection = 4 + NUM_CLASSES + num_masks
```

若 `num_masks=32`，10 类模型的特征维应为 `46`。模型仍须采用当前解析器支持的 `[cx, cy, w, h, class_scores..., mask_coefficients...]` 布局，并提供对应的 proto 输出。

### 2.2 `class_id`、名称和对比度语义

文件：`internal/yolo_categories.h`

当前代码不再在 `yolo_postprocess.cpp` 中维护 `switch`。所有类别集中定义在 `kYoloCategories`，数组下标就是 ONNX `class_id`。增加、删除、改名或调整顺序时，必须修改该数组及其长度。

`internal/yolo_postprocess.cpp` 会通过 `FindYoloCategory(det.class_id)` 自动取得类别名和对比度极性；超出数组范围的类别会被忽略。

### 2.3 逐类别过滤配置注册

需要同步修改：

- `config/config.h`：`DefaultCategoryFilters()` 中登记所有类别；
- `config/config.cpp`：`kYoloCategorySections` 中登记“内部类别名 -> INI section”；
- 各运行环境的 `config.ini`：增加对应 section。

本次新增：

```ini
[Stripe_horizontal_config]
Enable=True
use_traditional_measure=0
confidence_threshold=0.25
contrast_threshold=0
min_width=0
min_height=0
min_area=0

[Stripe_vertical_config]
Enable=True
use_traditional_measure=0
confidence_threshold=0.25
contrast_threshold=0
min_width=0
min_height=0
min_area=0
```

如果类别没有注册到 `category_filters`，`ComposeOutputWithDefectFilter()` 会跳过该检测结果。

### 2.4 最终名称映射

文件：各运行环境的 `config.ini`

```ini
[category_mapping]
Stripe_horizontal=Abnormal
Stripe_vertical=Abnormal
```

名称映射在逐类别过滤、AA/IQT 过滤和 PatchCore/YOLO IoU 过滤之后执行。因此过滤配置必须使用内部名称 `Stripe_horizontal`、`Stripe_vertical`，外部 `InspectionResultC.details[].name` 最终收到 `Abnormal`。

`Inspection_ProcessFloatArray_npy` 已经包含 `Abnormal` 这一最终类别，所以此次归并不需要增加 `Defect_name` 枚举值，也不需要改变 `INSPECTION_DETECT_CATEGORIES`。若未来要求这两类在外部统计中保持独立，才需要扩展 C API 枚举、数组维度、名称索引映射及所有调用方。

### 2.5 配置文件副本

仓库中需要保持一致的配置包括：

- `Corona_defect_detection/config.ini`；
- `Use_DLL/config.ini`；
- `windows_corona_detection/Runtime/config.ini`。

部署时还要确认实际随 DLL/应用发布的 `config.ini` 来自已更新版本。

### 2.6 测试和说明文档

需要同步检查：

- `tests/smoke_tests.cpp`：10 个 class id 的名称和极性、默认过滤器数量、最终映射；
- `Corona_defect_detection/README.md`；
- `Corona_defect_detection/internal/README.md`；
- 外部评估脚本、标签文件或模型元数据中写死的类别列表。

## 3. 通常不需要修改的位置

- `yolo_inference.cpp`：NMS 调用已经传入 `NUM_CLASSES`；
- `internal/yolo_postprocess.cpp`：类别查询来自 `kYoloCategories`；
- `internal/result_builder.cpp`：使用通用的类别过滤表和名称映射表；
- `inference_c_api.h/.cpp`：新增两类最终都归并到现有 `Abnormal`；
- Windows C# 原生头文件副本：最终 C API 枚举没有变化。

## 4. 后续再次修改类别时的检查清单

- [ ] ONNX 类别数量、顺序和训练标签一致
- [ ] `yolo_inference.h::NUM_CLASSES` 已同步
- [ ] 输出特征维满足 `4 + NUM_CLASSES + num_masks`
- [ ] `internal/yolo_categories.h::kYoloCategories` 数量、顺序、名称和极性已同步
- [ ] `config.h` 默认过滤表已登记所有内部类别
- [ ] `config.cpp` 的类别 section 表已登记所有内部类别
- [ ] 所有运行配置包含所需过滤 section 和最终名称映射
- [ ] AA/IQT 类别列表仍使用内部类别名且符合业务规则
- [ ] 若最终类别发生变化，C API 枚举、数组维度和调用方已同步
- [ ] 单元/冒烟测试覆盖每个 class id 和最终名称映射
- [ ] DLL 与调用方已重新编译，并用新模型逐类回归
