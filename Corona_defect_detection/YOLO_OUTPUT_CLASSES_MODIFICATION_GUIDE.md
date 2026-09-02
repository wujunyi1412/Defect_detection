# YOLO 输出类别修改指南

> 当前实现已经切换为 8 类；本文后续展示的 3 类代码片段是修改前的历史示例，不再代表当前源码。当前顺序为 `Glue_overflow`、`Decolorization`、`Stain`、`Stripes`、`BrightStripes`、`Bright_clusters`、`Line_artifacts`、`LineArtifacts`。

本文依据当前仓库代码整理，适用于 `Corona_defect_detection` 中的 YOLOv8-Seg 推理流程。

## 1. 先明确当前类别体系

当前 ONNX 模型直接输出 **3 个 YOLO 类别**：

| ONNX `class_id` | 内部名称 |
|---:|---|
| 0 | `Stain` |
| 1 | `BrightStripes` |
| 2 | `LineArtifacts` |

但 C API 的分类数组包含 **4 个最终结果类别**：

| C API 索引 | 最终名称 | 来源 |
|---:|---|---|
| 0 | `Abnormal` | PatchCore |
| 1 | `Stain` | YOLO class 0 |
| 2 | `BrightStripes` | YOLO class 1 |
| 3 | `LineArtifacts` | YOLO class 2 |

因此，修改时必须区分：

- **YOLO 原始类别数**：决定 ONNX 输出张量如何解析。
- **最终结果类别数**：决定分类数组接口如何计数和排列，还包含 PatchCore 类别及派生类别。

## 2. 按修改类型判断范围

### 2.1 只替换模型，类别数量、顺序和含义均不变

只需修改 [`config.ini`](config.ini) 中的 `models.yolo_model_path`，其他类别代码通常不用改。

### 2.2 仍是 3 类，只修改名称或类别顺序

至少需要修改：

1. [`internal/yolo_postprocess.cpp`](internal/yolo_postprocess.cpp) 的 `class_id -> name` 映射。
2. [`internal/result_builder.cpp`](internal/result_builder.cpp) 的类别过滤器选择逻辑。
3. 对应的配置结构、配置加载代码和 `config.ini` 类别段。
4. 如果使用 `Inspection_ProcessFloatArray_npy`，还要修改 C API 枚举及名称到索引的映射。
5. 所有与旧类别语义绑定的特殊处理和测试。

类别数量仍为 3 时，`NUM_CLASSES` 和 NMS 的类别数可以保持 3，但必须确认 ONNX 的类别顺序与 `switch (det.class_id)` 完全一致。

### 2.3 增加或减少 YOLO 类别

需要完成本文第 3～8 节的检查。只改类别名称映射是不够的，否则模型输出会被错误解析，或者新类别在最终过滤阶段被静默丢弃。

## 3. 修改模型输出解析（必改）

### 3.1 `yolo_inference.h`：YOLO 原始类别数

文件：[`yolo_inference.h`](yolo_inference.h)

当前代码：

```cpp
static constexpr int NUM_CLASSES = 3;
```

将其改成 ONNX 模型实际类别数。例如新模型有 4 类：

```cpp
static constexpr int NUM_CLASSES = 4;
```

该常量影响：

- 每个候选框的类别分数遍历范围；
- mask 系数的起始偏移；
- 检测输出张量方向及尺寸判断。

当前 YOLOv8-Seg 输出特征数按下式计算：

```text
features_per_detection = 4 + NUM_CLASSES + num_masks
```

当前模型注释中的检测输出是 `[1, 39, 8400]`，proto 是 `[1, 32, 160, 160]`，即：

```text
39 = 4 个框坐标 + 3 个类别分数 + 32 个 mask 系数
```

如果新模型有 4 类且仍有 32 个 mask 通道，则检测输出特征维应为 `40`。必须用 Netron、ONNX Runtime 或导出脚本确认真实输出形状。

### 3.2 `yolo_inference.cpp`：类别感知 NMS 的类别数

文件：[`yolo_inference.cpp`](yolo_inference.cpp)

`NMSBoxesByClass()` 已通过参数接收类别数，调用处传入 `NUM_CLASSES`：

```cpp
NMSBoxesByClass(
    boxes_orig,
    confidences,
    class_ids,
    NUM_CLASSES,
    score_threshold_,
    iou_threshold_,
    indices);
```

因此以后修改模型类别数时，只需更新 `yolo_inference.h` 中的 `NUM_CLASSES`，类别感知 NMS 会自动使用相同数值。

### 3.3 检查模型格式是否仍与当前解析器一致

当前解析器面向 YOLOv8-Seg 风格输出：一个检测张量加一个 4 维 proto 张量，并假定每个检测为：

```text
[cx, cy, w, h, class_scores..., mask_coefficients...]
```

如果新模型改成检测模型（无 proto）、带独立 objectness、端到端 NMS 输出，或者输出节点布局不同，不能只修改类别数，需要同步重写 `Postprocess()` 的张量识别和字段偏移。

建议同时把“输出特征维不等于 `4 + NUM_CLASSES + num_masks`”改成明确报错，而不是继续按兜底尺寸解析，以便尽早发现模型与代码不匹配。

## 4. 修改 `class_id -> 类别名称` 映射（必改）

文件：[`internal/yolo_postprocess.cpp`](internal/yolo_postprocess.cpp)

当前映射：

```cpp
switch (det.class_id) {
    case 0: detail.name = "Stain"; break;
    case 1: detail.name = "BrightStripes"; break;
    case 2: detail.name = "LineArtifacts"; break;
    default: continue;
}
```

这里必须严格按照训练数据集/ONNX 元数据中的类别顺序修改。例如：

```cpp
case 0: detail.name = "Scratch"; break;
case 1: detail.name = "Spot"; break;
case 2: detail.name = "Crack"; break;
case 3: detail.name = "Contamination"; break;
```

未加入 `switch` 的 `class_id` 会执行 `continue`，检测结果会被直接丢弃。

## 5. 修改类别专用后处理（按语义必查）

类别名称不仅用于显示，当前还有多处按字符串决定算法行为。

### 5.1 对比度极性

文件：[`internal/yolo_postprocess.cpp`](internal/yolo_postprocess.cpp)

`GetYoloContrastPolarity()` 当前规则是：

- `Stain` 使用暗缺陷对比度；
- `BrightStripes` 使用亮缺陷对比度；
- 其他类别使用自动极性。

新增类别时，应根据缺陷是亮、暗还是不确定来更新该函数。否则面积框可能正常，但 `contrast` 和后续对比度过滤结果可能不符合预期。

### 5.2 `Stain` 类别保持不变

当前已取消基于对比度的派生分类，YOLO class 0 在后处理和传统图像细化后均保持为 `Stain`。

### 5.3 暗缺陷传统几何细化

文件：[`internal/result_builder.cpp`](internal/result_builder.cpp)

当前只有 `Stain` 在 `use_traditional_measure=1` 时调用 `RefineDarkDefectGeometryAdaptive()`。新增暗缺陷类别是否使用该算法，需要单独决定。

### 5.4 AA 区域过滤

文件：

- [`config.ini`](config.ini) 的 `[AA] categories` 和 `position_categories`
- [`config/config.h`](../config/config.h) 的默认类别列表
- [`config/config.cpp`](../config/config.cpp) 的 `CanonicalAaCategory()` 和配置校验
- [`internal/aa_yolo_filter.cpp`](internal/aa_yolo_filter.cpp)

如果 AA 规则需要作用于新类别，应把新类别加入允许列表和配置；当前配置校验只接受四个既有 YOLO 最终类别，直接在 INI 中填写新名称会导致初始化失败。

如果新类别不适用 AA 规则，则不要加入 `[AA]` 列表，但仍应评估旧名称是否需要删除。

## 6. 修改逐类别过滤配置（新增、删除或改名时必改）

这是最容易遗漏并导致“模型检测到了，但最终结果为空”的部分。

### 6.1 配置数据结构

文件：[`config/config.h`](../config/config.h)

`InspectionConfigData` 当前分别保存：

```cpp
CategoryFilterConfig stain_filter;
CategoryFilterConfig brightstripes_filter;
CategoryFilterConfig lineartifacts_filter;
```

为每个需要独立阈值的新最终类别新增字段，删除类别时也应清理无用字段。

### 6.2 配置加载

文件：[`config/config.cpp`](../config/config.cpp)

在 `LoadInspectionConfig()` 中增加或修改 `LoadCategoryFilter()` 调用，例如：

```cpp
out.scratch_filter = LoadCategoryFilter(ini, "scratch_config");
```

INI section 名称经过解析器统一为小写，因此代码中现有加载使用小写 section 名。

### 6.3 配置文件

文件：[`config.ini`](config.ini)

为新类别增加对应段，例如：

```ini
[Scratch_config]
Enable=True
confidence_threshold=0.25
contrast_threshold=0
min_width=0
min_height=0
min_area=0
use_traditional_measure=0
```

### 6.4 将配置传入后处理

文件：

- [`inference_dll.cpp`](inference_dll.cpp)：`Impl` 成员、初始化赋值及 `BuildComposeContext()`
- [`internal/result_builder.h`](internal/result_builder.h)：`ResultComposeContext`
- [`internal/result_builder.cpp`](internal/result_builder.cpp)：按 `d.name` 选择过滤器

当前 `result_builder.cpp` 对不认识的类别执行：

```cpp
if (!cf) continue;
```

所以新增类别如果没有在这里绑定过滤器，会在最终结果合成时被静默丢弃。

同时确认新类别的对比度阈值方向。当前暗缺陷保留 `contrast < threshold`，其他类别保留 `contrast >= threshold`；该方向由 `IsDarkDefectCategory()` 决定。

## 7. 修改 C API 分类数组（使用 `_npy` 接口时必改）

普通接口 `Inspection_ProcessImagePath` 和 `Inspection_ProcessFloatArray` 通过 `InspectionResultC.details[].name` 返回字符串；只要名称长度没有超过 `INSPECTION_MAX_NAME_LEN - 1`，新增名称本身不要求修改结构体布局。

但 `Inspection_ProcessFloatArray_npy` 使用固定类别维度的数组，必须同步修改以下位置。

### 7.1 C API 头文件

文件：[`inference_c_api.h`](inference_c_api.h)

- 更新 `Defect_name` 枚举，明确每个最终结果类别的稳定索引；
- 更新 `INSPECTION_DETECT_CATEGORIES`；
- 不要把它机械地设置为 `NUM_CLASSES`，因为这里还可能包含 `Abnormal` 和派生类别。

如果接口已经交付给外部程序，尽量不要改变现有枚举值。建议保留旧索引，在末尾追加新类别，否则调用方会把数组中的类别解释错。

### 7.2 名称到分类数组索引的映射

文件：[`inference_c_api.cpp`](inference_c_api.cpp)

更新 `DefectNameToIndex()`。没有映射的名称会得到 `-1`，随后被 `_npy` 接口跳过，不会进入 `o_detect_num / o_detect_area / o_detect_contrast`。

`FillResult()` 只复制名称和几何结果，一般不需要因类别数变化而修改。

### 7.3 同步外部头文件和调用方

仓库中还有需要同步的消费者：

- [`windows_corona_detection/Native/inference_c_api.h`](../windows_corona_detection/Native/inference_c_api.h)：C API 头文件副本；
- [`Use_DLL/main.cpp`](../Use_DLL/main.cpp)：类别名称数组、名称到索引映射和 CSV 表头；
- C# 或其他外部调用程序中任何固定类别列表、报表列、统计映射。

当前 Windows C# 的通用 `InspectionResult` 使用字符串名称，没有固定类别数组，但改名或删除类别时仍要检查结果分析和报表逻辑。

修改 `INSPECTION_DETECT_CATEGORIES` 会改变 `_npy` 调用方需要分配的数组尺寸，DLL 和调用方必须同时重新编译/发布，不能混用新旧头文件。

## 8. 文档和测试（建议同步）

需要检查：

- [`internal/README.md`](internal/README.md) 中的类别映射和后处理说明；
- [`README.md`](README.md) 中的类别配置段；
- [`tests/smoke_tests.cpp`](../tests/smoke_tests.cpp) 中写死的类别名、AA 允许类别和过滤行为；
- 外部评估脚本、CSV 表头、标签文件和前端显示逻辑。

建议至少增加以下测试：

1. 每个 ONNX `class_id` 都映射到预期名称；
2. 类别感知 NMS 能保留新增类别；
3. 每个新增类别能通过 `ComposeOutputWithDefectFilter()`，不会被 `if (!cf) continue` 丢弃；
4. `_npy` 接口的每个最终类别落入正确数组索引；
5. 新模型检测输出特征维满足 `4 + NUM_CLASSES + num_masks`；
6. 新类别的亮/暗对比度方向、AA 过滤和传统几何细化符合业务预期。

## 9. 推荐修改顺序

1. 确认训练类别列表、顺序、数量和 ONNX 输出形状。
2. 修改 `NUM_CLASSES`；类别感知 NMS 会自动使用该值。
3. 修改 `class_id -> name` 映射。
4. 修改对比度极性、派生分类、传统几何细化和 AA 规则。
5. 增加逐类别配置字段、INI 加载、上下文传递和结果过滤映射。
6. 如使用 `_npy` 接口，更新 C API 最终类别枚举、数量和名称索引。
7. 同步外部头文件、C#/C++ 调用方、CSV/统计逻辑和文档。
8. 重新编译 DLL 和调用方，并用每个类别各至少一张样本做回归测试。

## 10. 最小检查清单

- [ ] 新 ONNX 的类别数量和顺序已确认
- [ ] `yolo_inference.h::NUM_CLASSES` 已同步
- [ ] 类别感知 NMS 调用仍传入 `NUM_CLASSES`
- [ ] ONNX 检测张量特征维与 `4 + 类别数 + mask通道数` 一致
- [ ] `yolo_postprocess.cpp` 的 `class_id` 映射已同步
- [ ] 新类别的对比度极性已确认
- [ ] `Stain` 是否需要保持现有暗缺陷对比度和传统细化逻辑已确认
- [ ] `result_builder.cpp` 能为所有新类别找到过滤配置
- [ ] `config.h / config.cpp / config.ini` 已同步
- [ ] AA 类别允许列表和配置已同步
- [ ] C API `_npy` 的最终类别数、枚举和映射已同步
- [ ] 外部头文件、调用方、统计报表和测试已同步
- [ ] DLL 与调用方已重新编译并完成逐类别回归
