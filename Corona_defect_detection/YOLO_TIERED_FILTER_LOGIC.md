# YOLO 分段过滤逻辑

本文说明 YOLO 检测结果的类别过滤顺序，以及两级对比度阈值和三组附加置信度、宽、高、面积参数的实际行为。

## 1. 总体处理顺序

每个 YOLO 检测结果依次经过：

```text
YOLO 检测结果
  → 传统算法细化（按类别配置，可关闭）
  → 类别 Enable
  → 统一 confidence_threshold
  → 分段对比度及对应组的附加置信度、宽、高、面积
  → AA / IQT 等附加过滤
  → 最终结果
```

当 `traditional_measure_mode=dark` 或 `bright` 且细化成功时，分段过滤使用细化后重新计算的宽、高、面积和对比度。类别和 YOLO 置信度不会被细化修改。

## 2. 基础过滤

```text
Enable=False
    → 删除该类别的检测结果

score < confidence_threshold
    → 删除检测结果

否则
    → 进入分段过滤
```

统一 `confidence_threshold` 始终先执行。各分段的附加置信度只会进一步提高要求，不会替代或降低统一阈值。

## 3. 启用或关闭分段过滤

满足以下任一条件时启用分段过滤：

```text
contrast_threshold_1 > 0
或
contrast_threshold_2 > 0
```

关闭整套分段过滤：

```ini
contrast_threshold_1=0
contrast_threshold_2=0
```

关闭后，第 0、1、2 组新增参数全部被忽略，统一 `confidence_threshold` 仍然生效。

旧版 `contrast_threshold`、`min_width`、`min_height`、`min_area` 仍兼容；只有未启用新分段过滤时才使用它们。要同时关闭新旧过滤，应将这些旧字段删除或设为 `0`。

## 4. 三组附加参数

每个对比度区间分别对应一组参数：

| 分组 | 对比度含义 | 参数 |
|---|---|---|
| 第 0 组 | 强极性区，原来的“直接保留区” | `confidence_threshold_0`、`min_width_0`、`min_height_0`、`min_area_0` |
| 第 1 组 | 中间区 | `confidence_threshold_1`、`min_width_1`、`min_height_1`、`min_area_1` |
| 第 2 组 | 弱极性区 | `confidence_threshold_2`、`min_width_2`、`min_height_2`、`min_area_2` |

每组内部必须同时满足：

```text
score  >= confidence_threshold_N
width  >= min_width_N
height >= min_height_N
area   >= min_area_N
```

其中 `N` 为 `0`、`1` 或 `2`。单个参数设为 `0` 表示不检查该项。

为了兼容旧配置，第 0 组参数缺省值均为 `0`。因此旧配置不添加第 0 组参数时，强极性区仍等价于原先的“直接保留”；只有显式配置第 0 组参数后，才会增加对应限制。

## 5. 暗缺陷方向

配置：

```ini
contrast_direction=less_equal
```

区间规则：

```text
contrast <= contrast_threshold_1
    → 检查第 0 组

contrast_threshold_1 < contrast <= contrast_threshold_2
    → 检查第 1 组

contrast > contrast_threshold_2
    → 检查第 2 组
```

示例：

```ini
contrast_threshold_1=0.80
contrast_threshold_2=0.90
contrast_direction=less_equal

confidence_threshold_0=0.40
min_width_0=2
min_height_0=3
min_area_0=6

confidence_threshold_1=0.50
min_width_1=5
min_height_1=6
min_area_1=20

confidence_threshold_2=0.70
min_width_2=10
min_height_2=12
min_area_2=50
```

| 对比度 | 实际要求 |
|---:|---|
| `contrast <= 0.70` | 置信度 ≥ 0.25、宽 ≥ 0、高 ≥ 0、面积 ≥ 0 |
| `0.70 < contrast <= 0.90` | 置信度 ≥ 0.25、宽 ≥ 5、高 ≥ 5、面积 ≥ 0 |
| `contrast > 0.90` | 置信度 ≥ 0.45、宽 ≥ 5、高 ≥ 5、面积 ≥ 0 |



## 6. 亮缺陷方向

配置：

```ini
contrast_direction=greater_equal
```

区间规则与暗缺陷镜像：

```text
contrast >= contrast_threshold_2
    → 检查第 0 组

contrast_threshold_1 <= contrast < contrast_threshold_2
    → 检查第 1 组

contrast < contrast_threshold_1
    → 检查第 2 组
```

例如阈值为 `1.10` 和 `1.20` 时，`contrast >= 1.20` 使用第 0 组，`1.10 <= contrast < 1.20` 使用第 1 组，`contrast < 1.10` 使用第 2 组。

## 7. 置信度的实际下限

由于统一置信度先执行，各区间最终生效的置信度下限为：

```text
第 0 组：max(confidence_threshold, confidence_threshold_0)
第 1 组：max(confidence_threshold, confidence_threshold_1)
第 2 组：max(confidence_threshold, confidence_threshold_2)
```

## 8. 配置约束

代码强制要求：

```text
contrast_threshold_2 >= contrast_threshold_1
所有附加置信度、宽、高、面积阈值 >= 0
```

三组附加阈值之间没有强制大小关系。通常弱极性区需要更严格的要求，可按业务需要配置为第 2 组 ≥ 第 1 组 ≥ 第 0 组，但这只是建议，并非加载条件。

启用分段过滤后，如果对比度为 `NaN` 或其他无效值，该检测结果会被过滤。
