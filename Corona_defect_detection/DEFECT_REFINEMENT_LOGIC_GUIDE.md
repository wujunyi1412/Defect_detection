# 缺陷细化逻辑说明

本文说明 `Corona_defect_detection` 当前实现中的缺陷细化流程，包括触发条件、暗/亮缺陷细化算法、失败回退、对比度重算及后续过滤顺序。

## 1. 细化的作用

YOLO 首先提供：

- 缺陷类别；
- 置信度；
- 检测框；
- 分割 Mask；
- 初始面积和对比度。

细化不会重新判断类别，也不会修改 YOLO 置信度。它以 YOLO 检测框为搜索范围，从灰度图中重新找出更准确的缺陷区域，并尝试更新：

- `x / y`：缺陷框左上角；
- `w / h`：缺陷框宽高；
- `area`：缺陷像素面积；
- `mask`：细化后的二值 Mask；
- `contrast`：基于新 Mask 重新计算的对比度。

细化失败时不会删除检测结果，而是继续使用原始 YOLO 结果。

## 2. 当前实际启用情况

细化方式通过每个类别配置段中的 `traditional_measure_mode` 控制：

```ini
traditional_measure_mode=none    # 不细化
traditional_measure_mode=dark    # 按暗缺陷细化
traditional_measure_mode=bright  # 按亮缺陷细化
```

当前只有 `Stain` 开启暗缺陷细化：

```ini
[Stain_config]
traditional_measure_mode=dark
```

其他类别当前为 `none`。亮缺陷细化代码已经实现，但默认配置没有启用。

旧配置 `use_traditional_measure=1` 仍然兼容，并会被解释为 `dark`；新配置建议统一使用 `traditional_measure_mode`。

## 3. 总体处理顺序

```text
YOLO 检测结果
  ↓
生成内部 DetectionResult
  ↓
查询对应类别的过滤配置
  ↓
根据 traditional_measure_mode 选择细化方式
  ├─ none   → 不细化
  ├─ dark   → RefineDarkDefectGeometryAdaptive()
  └─ bright → RefineBrightDefectGeometryAdaptive()
  ↓
细化成功：更新框、面积、Mask 和对比度
细化失败：完整保留原始 YOLO 结果
  ↓
执行类别启用、置信度、对比度、宽高和面积过滤
  ↓
执行 PatchCore/YOLO IoU、AA、IQT 等过滤
  ↓
映射最终类别名并输出
```

主流程入口位于：

```text
internal/result_builder.cpp
ComposeOutputWithDefectFilter()
```

细化发生在阈值过滤之前，因此分段对比度阈值以及两组附加置信度、宽、高、面积阈值使用的是细化后的数据。

## 4. 细化的前置条件

暗缺陷和亮缺陷细化共用 `RefineDefectGeometryAdaptive()`，只有极性不同。

以下情况不会细化：

1. 灰度图为空；
2. 灰度图不是单通道；
3. YOLO 检测框存在非有限数值；
4. 检测框宽高无效；
5. 检测框裁剪到图像范围后为空；
6. 检测框宽度或高度超过 30 像素；
7. 检测框命中明显的长产品黑边；
8. 找不到可用的背景区域；
9. 缺陷响应没有达到强种子阈值；
10. 找不到包含强种子的有效连通域；
11. 最终 Mask 或外接框为空。

暗、亮缺陷当前都只处理最大 `30×30` 的 YOLO 框：

```cpp
constexpr int kMaxDarkRefinedBoxWidth = 30;
constexpr int kMaxDarkRefinedBoxHeight = 30;
constexpr int kMaxBrightRefinedBoxWidth = 30;
constexpr int kMaxBrightRefinedBoxHeight = 30;
```

因此，大于该尺寸的目标直接保留 YOLO 原结果。

## 5. 详细算法

### 5.1 整理 YOLO 检测框

YOLO 的浮点框按照以下方式转换成整数 ROI：

- 左上角 `x1/y1` 向下取整；
- 右下角 `x2/y2` 向上取整；
- 所有坐标裁剪到图像有效范围。

例如：

```text
YOLO 框：x=100.4, y=80.7, w=12.2, h=10.1
整数框：x1=100, y1=80, x2=113, y2=91
```

### 5.2 扩展局部上下文

为了估计缺陷周围的正常背景，算法会在检测框四周扩展：

```text
short_side = min(width, height)
margin = clamp(ceil(short_side × 0.30), 4, 24)
```

例如 `20×15` 的检测框：

```text
short_side = 15
margin = ceil(15 × 0.30) = 5
```

即上下左右各扩展 5 像素。扩展区域仅用于估计背景，最终细化区域仍限制在原始 YOLO 检测框内部。

### 5.3 转换为 8 位灰度图

如果局部灰度图已经是 `CV_8U`，直接使用。

如果是浮点或其他深度，则在当前上下文内执行 min-max 归一化：

```text
gray_u8 = (gray - min) × 255 / (max - min)
```

如果区域灰度完全相同，则输出全零图。若最小值或最大值不是有限数值，则细化失败。

浮点图是在每个缺陷的局部上下文内独立归一化，因此后续响应阈值表示局部灰度差，不是原图绝对灰度值。

### 5.4 查找并排除产品黑边

算法计算上下文像素的 80% 分位数：

```text
foreground_reference = P80(context)
black_threshold = clamp(foreground_reference × 0.30, 6, 64)
```

灰度小于等于 `black_threshold` 的像素成为黑色候选。随后执行 8 邻域连通域分析，只保留与上下文边缘连接的黑色区域。

这些区域通常对应：

- 产品外部黑色背景；
- FOV 黑边；
- 靠近图像边缘的无效区域。

边界黑色区域还会膨胀 1～3 次：

```text
transition_guard = clamp(short_side / 12, 1, 3)
```

这样可防止黑白过渡带被误识别成缺陷。

### 5.5 拒绝明显的长产品边界

如果 YOLO 框内的边界黑色区域满足：

- 长宽比至少为 6；
- 黑色区域沿长边覆盖至少 75%；
- 黑色区域沿短边覆盖不超过 60%；

则该区域被认为更像产品边缘，而不是点状缺陷，算法放弃细化并保留原 YOLO 结果。

### 5.6 构造背景采样区

正常情况下：

```text
背景区域 = 扩展上下文 - YOLO 检测框 - 产品黑边
```

背景像素最少要求：

```text
min_background_pixels = max(12, detection_area / 8)
```

如果检测框靠近图像边缘，导致框外背景不足，则退化为使用检测框内部四周的一条窄边带：

```text
band = clamp(short_side / 6, 1, 4)
```

内部边带中仍会排除产品黑边。如果最终没有可用背景像素，细化失败。

### 5.7 拟合局部背景平面

算法使用背景像素拟合线性平面：

```text
B(x, y) = a·x + b·y + c
```

这样可以适应：

- 左右或上下方向的光照渐变；
- 曲面造成的缓慢灰度变化；
- 缺陷框两侧背景亮度不同。

拟合后计算残差：

```text
residual = 实际背景灰度 - 拟合背景灰度
```

使用残差中位数修正平面的截距，并使用 MAD 估计背景噪声：

```text
noise_sigma = max(
    0.8,
    1.4826 × median(|residual - median(residual)|)
)
```

MAD 对少量异常亮点和暗点不敏感，比普通标准差更稳健。

如果平面拟合失败，则退化为使用背景灰度中位数构成水平平面，并使用默认噪声 `0.8`。

### 5.8 生成缺陷响应图

算法同时建立一个高斯平滑背景。高斯核大小为：

```text
blur_size = clamp((short_side / 3) | 1, 3, 31)
```

`| 1` 用于保证核尺寸为奇数。

暗缺陷响应：

```text
global_response = 拟合背景平面 - 原图
local_response  = 高斯平滑图 - 原图
```

亮缺陷响应：

```text
global_response = 原图 - 拟合背景平面
local_response  = 原图 - 高斯平滑图
```

最终响应为：

```text
response = max(global_response, local_response, 0)
```

其中：

- 全局响应适合发现相对整体背景明显的缺陷；
- 局部响应适合发现位于渐变背景上的小缺陷；
- 两者取最大值可降低背景不均匀造成的漏检。

### 5.9 强种子与弱生长区域

强种子阈值：

```text
seed_threshold = max(2.0, 2.5 × noise_sigma)
```

如果检测框内最大响应小于该值，则认为缺陷信号不足，细化失败。

当前细化框最大为 `30×30`，因此实际使用小目标生长阈值：

```text
grow_threshold = max(
    1.5,
    0.65 × seed_threshold,
    1.75 × noise_sigma
)
```

生成两个 Mask：

```text
seed_mask = response >= seed_threshold
grow_mask = response >= grow_threshold
```

- `seed_mask` 表示可靠的强缺陷核心；
- `grow_mask` 表示可能属于缺陷的弱响应外围。

随后再次排除产品黑边。该过程类似双阈值滞后分割：弱响应只有与强缺陷核心连通才会被接受。

### 5.10 连通域分析和候选评分

算法对 `grow_mask` 执行 4 邻域连通域分析。

每个连通域必须与 `seed_mask` 有至少一个重叠像素，否则直接丢弃。剩余连通域按照以下公式评分：

```text
score =
    mean_response
  + 8 × seed_fraction
  + 3 × fill
  + 2 × log(1 + area)
  - 2 × center_distance
```

各项含义：

- `mean_response`：连通域的平均缺陷响应；
- `seed_fraction`：强种子像素占连通域面积的比例；
- `fill`：连通域面积与其外接矩形面积之比，越紧凑越高；
- `area`：适度偏好面积较大的有效区域；
- `center_distance`：连通域距离 YOLO 框中心越远，扣分越多。

算法最终只选择得分最高的一个连通域。因此，一个 YOLO 框中即使存在多个缺陷区域，当前也只会输出其中评分最高的一个，不会拆分成多个结果。

### 5.11 写回细化结果

选中最佳连通域后更新：

```text
x/y      = 连通域外接框在全图中的坐标
w/h      = 连通域最小轴对齐外接框尺寸
area     = 连通域非零像素数
mask     = 外接框内部的连通域二值 Mask
contrast = 使用新 Mask 重新计算的对比度
```

以下字段保持不变：

- `name`：类别不变；
- `score`：仍是 YOLO 置信度。

如果新对比度计算失败，只保留原对比度；几何细化结果仍然有效。

## 6. 对比度重新计算

细化后调用：

```cpp
CalculateContrastRatioByMode(...)
```

具体算法由 `[post] contrast_mode` 决定：

| 值 | 模式 | 说明 |
|---:|---|---|
| 0 | `LocalRing` | 使用缺陷 Mask 周围的局部环形背景 |
| 1 | `AdaptiveLocalPlane` | 使用自适应局部背景平面，当前默认 |
| 2 | `GlobalFovMedian` | 使用排除黑边后的全 FOV 中位数背景 |

当前配置：

```ini
[post]
contrast_mode=1
```

自适应局部平面模式会：

1. 根据 Mask 面积计算等效半径；
2. 在 Mask 周围生成背景环；
3. 排除产品外部黑边；
4. 使用背景环像素鲁棒拟合背景平面；
5. 对每个缺陷像素计算：

```text
ratio(x, y) = 缺陷灰度 / 该位置预测背景灰度
```

6. 对 ratio 按缺陷极性进行分位数截尾平均。

通常：

- 暗缺陷：`contrast < 1`；
- 亮缺陷：`contrast > 1`；
- 越接近 1：缺陷与背景越接近。

`contrast=0` 是合法的极暗结果；`contrast=NaN` 表示计算失败。

## 7. 细化失败与回退机制

调用细化前，主流程会复制原始 YOLO 结果：

```cpp
DetectionResult detail = d;
```

细化函数只在全部步骤成功后才写回 `detail`。如果中途任一步失败，函数返回 `false`，调用方继续使用复制出来的原始数据，包括：

- 原始 YOLO 框；
- 原始 YOLO Mask；
- 原始面积；
- 原始对比度；
- 原始类别；
- 原始置信度。

因此细化是增强步骤，不是检测结果成立的必要条件。

## 8. 细化后的过滤规则

细化结束后依次判断：

```text
Enable
confidence_threshold
contrast_threshold_1 / contrast_threshold_2
contrast_direction
confidence_threshold_1 / min_width_1 / min_height_1 / min_area_1
confidence_threshold_2 / min_width_2 / min_height_2 / min_area_2
```

其中置信度仍然是 YOLO 置信度；对比度、宽高和面积可能已经是细化后的值。

`contrast_direction=less_equal` 用于暗缺陷：

```text
contrast <= threshold_1              → 直接保留
threshold_1 < contrast <= threshold_2 → 检查第 1 组附加置信度、宽、高、面积
contrast > threshold_2               → 检查第 2 组附加置信度、宽、高、面积
```

`contrast_direction=greater_equal` 用于亮缺陷：

```text
contrast >= threshold_2              → 直接保留
threshold_1 <= contrast < threshold_2 → 检查第 1 组附加置信度、宽、高、面积
contrast < threshold_1               → 检查第 2 组附加置信度、宽、高、面积
```

当两个新对比度阈值均为 `0` 时关闭分段过滤，两组附加置信度和几何阈值随之被忽略，但最前面的统一 `confidence_threshold` 始终生效。旧配置项 `contrast_threshold`、`min_width`、`min_height`、`min_area` 仍然兼容；仅在新分段阈值未启用时走旧过滤逻辑。

细化模式与类别的对比度极性应保持一致。例如给暗缺陷类别配置 `bright`，细化算法会寻找亮区域，但后续仍可能按暗缺陷方向过滤，造成不符合预期的结果。

## 9. 当前实际行为总结

当前配置下：

- `Stain`：框不超过 `30×30` 时尝试暗缺陷细化；
- 大于 `30×30` 的 `Stain`：保留 YOLO 原结果；
- `BrightStripes`、`Bright_clusters`、`LineArtifacts`：支持配置为 `bright`，但当前未开启；
- 其他类别：当前不细化；
- 细化失败：不会删除缺陷，自动回退到原 YOLO 结果；
- 一个 YOLO 框：最多选择一个细化连通域；
- 类别和置信度：始终保持 YOLO 输出不变。

如需开启亮缺陷细化，可以配置：

```ini
[BrightStripes_config]
traditional_measure_mode=bright
```

建议先用代表性样本验证 30 像素尺寸限制、背景渐变、产品黑边和多个亮点共框等情况，再决定是否在生产配置中启用。

## 10. 相关源码

- `internal/result_builder.cpp`：细化调用位置及细化后的过滤顺序；
- `internal/dark_defect_refiner_adaptive.cpp`：当前实际使用的暗/亮缺陷自适应细化算法；
- `internal/dark_defect_refiner.h`：细化函数声明；
- `internal/image_utils_adaptive.cpp`：自适应局部平面对比度；
- `internal/image_utils_global_fov.cpp`：对比度模式分发和全 FOV 对比度；
- `internal/image_utils.cpp`：局部环形对比度及灰度归一化；
- `config/config.cpp`：`traditional_measure_mode` 配置解析；
- `config/config.h`：细化模式枚举和类别过滤配置结构；
- `config.ini`：当前运行参数。

`internal/dark_defect_refiner.cpp` 中还保留了旧函数 `RefineDarkDefectGeometryOnly()`，但当前主处理流程没有调用它。当前实际运行的是 `dark_defect_refiner_adaptive.cpp` 中的自适应版本。

## 11. 类别极性与计算对比度冲突时的处理

### 11.1 什么是极性冲突

通常情况下：

- 暗缺陷期望 `contrast < 1`；
- 亮缺陷期望 `contrast > 1`。

如果出现以下结果，可以认为类别极性和对比度表现存在冲突：

```text
暗缺陷类别：contrast > 1
亮缺陷类别：contrast < 1
```

例如：

```text
类别：Stain
类别极性：Dark
计算结果：contrast=1.12
```

或者：

```text
类别：BrightStripes
类别极性：Bright
计算结果：contrast=0.76
```

### 11.2 当前代码不会自动纠正冲突

`Dark` 和 `Bright` 用于指导像素统计、缺陷响应和对比度计算，但不会强制最终结果必须小于或大于 1。

当前对比度的有效性判断只要求：

```cpp
std::isfinite(contrast_ratio) && contrast_ratio >= 0.0f
```

因此，暗缺陷计算出 `1.12` 仍然是一个有效数值；亮缺陷计算出 `0.76` 也仍然是有效数值。

发生冲突时，当前代码不会：

- 自动修改缺陷类别；
- 自动映射为 `Abnormal`；
- 自动将对比度转换为倒数；
- 仅因为跨过 1 就立即删除结果。

类别始终保持 YOLO 输出的类别，最终是否保留主要由分段对比度及其对应的宽高阈值决定。

### 11.3 对比度阈值的实际过滤规则

暗缺陷示例：

```ini
[Stain_config]
contrast_threshold_1=0.8
contrast_threshold_2=0.9
contrast_direction=less_equal
confidence_threshold_1=0.5
min_width_1=5
min_height_1=5
min_area_1=20
confidence_threshold_2=0.7
min_width_2=10
min_height_2=10
min_area_2=50
```

此时 `contrast <= 0.8` 直接保留；`0.8 < contrast <= 0.9` 要求附加置信度不小于 `0.5`、宽高均不小于 `5` 且面积不小于 `20`；`contrast > 0.9` 要求附加置信度不小于 `0.7`、宽高均不小于 `10` 且面积不小于 `50`。

亮缺陷使用镜像方向，例如：

```ini
[BrightStripes_config]
contrast_threshold_1=1.1
contrast_threshold_2=1.2
contrast_direction=greater_equal
confidence_threshold_1=0.5
min_width_1=5
min_height_1=5
min_area_1=20
confidence_threshold_2=0.7
min_width_2=10
min_height_2=10
min_area_2=50
```

此时 `contrast >= 1.2` 直接保留，`1.1 <= contrast < 1.2` 检查第 1 组附加置信度和宽高面积，`contrast < 1.1` 检查第 2 组附加置信度和宽高面积。

### 11.4 两个分段阈值均为 0 时不检查冲突

因此：

```ini
contrast_threshold_1=0
contrast_threshold_2=0
```

表示完全关闭对比度过滤。

在这种配置下，即使出现：

```text
Stain（Dark）得到 contrast=1.20
BrightStripes（Bright）得到 contrast=0.70
```

只要置信度、宽高和面积等其他条件通过，结果仍然会被保留。

还需要注意：如果对比度计算失败并得到 `NaN`，当两个分段阈值均为 `0` 时会直接通过分段对比度判断；启用任一分段阈值后，`NaN` 会被过滤。

### 11.5 细化过程中的极性冲突

暗缺陷细化使用：

```text
response = max(
    拟合背景平面 - 原图,
    高斯平滑图 - 原图,
    0
)
```

亮缺陷细化使用：

```text
response = max(
    原图 - 拟合背景平面,
    原图 - 高斯平滑图,
    0
)
```

如果真实缺陷是亮的，却配置为：

```ini
traditional_measure_mode=dark
```

那么亮区域在暗缺陷响应中通常不明显，最大响应可能达不到：

```text
seed_threshold = max(2.0, 2.5 × noise_sigma)
```

此时细化失败并自动回退到原始 YOLO 框、Mask、面积和对比度，不会删除检测结果。

但如果框中存在其他暗噪声或暗结构，也可能错误地选中这些区域并细化成功。因此，细化模式应当与缺陷真实极性一致。

### 11.6 三种极性设置彼此独立

当前实现中有三处与极性有关的设置：

1. `internal/yolo_categories.h` 中的 `ContrastPolarity`：决定初始对比度计算使用暗、亮还是自动极性；
2. `traditional_measure_mode`：决定细化时寻找暗区域还是亮区域；
3. `contrast_direction`：决定分段过滤采用暗缺陷方向还是亮缺陷方向。旧版单阈值配置仍使用类别定义中的 `keep_contrast_below_threshold`。

这三项目前不会自动进行一致性校验。

如果错误配置为：

```text
类别极性：Dark
细化模式：bright
过滤方向：less_equal 或 greater_equal
```

处理过程仍然会正常执行：

1. 按亮缺陷寻找细化区域；
2. 按亮缺陷重新计算对比度；
3. 最后按配置的分段方向执行宽高过滤。

这可能导致细化成功但随后被错误方向的阈值过滤，或者保留不符合业务语义的结果。

### 11.7 当前行为总结

```text
类别极性与对比度冲突
  ↓
不修改 YOLO 类别
  ↓
细化模式不匹配时，细化可能失败并回退原结果
  ↓
对比度仍作为有效数值输出
  ↓
contrast_threshold_1 或 contrast_threshold_2 > 0
  → 根据 contrast_direction 进入直接保留区或两组宽高过滤区

contrast_threshold_1 = 0 且 contrast_threshold_2 = 0
  → 不检查极性冲突，直接通过对比度过滤
```

### 11.8 可选的严格极性校验

如果业务要求类别和对比度极性必须一致，可以在细化完成后、阈值过滤前增加显式校验：

```cpp
bool polarity_matches = true;

if (category->contrast_polarity == ContrastPolarity::Dark) {
    polarity_matches = detail.contrast < 1.0f;
} else if (category->contrast_polarity == ContrastPolarity::Bright) {
    polarity_matches = detail.contrast > 1.0f;
}

if (!polarity_matches) {
    continue;
}
```

实际应用中不建议直接以 `1.0` 为硬边界，因为背景噪声可能使结果在 1 附近波动。可以设置容差，例如：

```text
暗缺陷：contrast < 0.98
亮缺陷：contrast > 1.02
```

分段配置本身不会无条件删除弱极性区的结果，而是让该区结果接受第 2 组宽高面积检查。如果业务要求不考虑几何尺寸、只按极性硬删除，仍应增加上述显式极性校验；不要通过设置不合理的超大阈值来模拟。
