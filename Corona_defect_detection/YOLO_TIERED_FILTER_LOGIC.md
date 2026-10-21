# YOLO 分段过滤逻辑

本文说明当前 YOLO 检测结果的类别过滤顺序，以及两级对比度和两组宽、高、面积参数的实际行为。

## 1. 总体处理顺序

每个 YOLO 检测结果依次经过：

```text
YOLO 检测结果
  ↓
传统算法细化（按类别配置，可关闭）
  ↓
类别 Enable
  ↓
confidence_threshold
  ↓
分段对比度及对应组的宽、高、面积
  ↓
AA / IQT 等附加过滤
  ↓
最终结果
```

当 `traditional_measure_mode=dark` 或 `bright` 且细化成功时，后续过滤使用细化后重新计算的宽、高、面积和对比度。类别和 YOLO 置信度不会被细化修改。

## 2. 基础过滤

```text
Enable=False
    → 删除该类别的检测结果

score < confidence_threshold
    → 删除检测结果

否则
    → 进入分段过滤
```

`Enable=False` 表示完全禁用该类别，不等同于只关闭分段过滤。

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

关闭后，`min_width_1/2`、`min_height_1/2` 和 `min_area_1/2` 全部被忽略。

如果配置文件中还保留旧版字段，则代码会回到旧版过滤逻辑。要完全关闭新旧过滤，旧字段也应删除或设为 `0`：

```ini
contrast_threshold=0
min_width=0
min_height=0
min_area=0
```

## 4. 暗缺陷方向

配置：

```ini
contrast_direction=less_equal
```

处理规则：

```text
contrast <= contrast_threshold_1
    → 直接保留，不检查宽、高、面积

contrast_threshold_1 < contrast <= contrast_threshold_2
    → 检查第 1 组宽、高、面积

contrast > contrast_threshold_2
    → 检查第 2 组宽、高、面积
```

第 1 组必须同时满足：

```text
width  >= min_width_1
height >= min_height_1
area   >= min_area_1
```

第 2 组必须同时满足：

```text
width  >= min_width_2
height >= min_height_2
area   >= min_area_2
```

例如：

```ini
contrast_threshold_1=0.80
contrast_threshold_2=0.90
contrast_direction=less_equal

min_width_1=5
min_height_1=6
min_area_1=20

min_width_2=10
min_height_2=12
min_area_2=50
```

| 对比度 | 处理方式 |
|---:|---|
| `contrast <= 0.80` | 直接保留 |
| `0.80 < contrast <= 0.90` | 宽 ≥ 5、高 ≥ 6、面积 ≥ 20 |
| `contrast > 0.90` | 宽 ≥ 10、高 ≥ 12、面积 ≥ 50 |

## 5. 亮缺陷方向

配置：

```ini
contrast_direction=greater_equal
```

处理规则：

```text
contrast >= contrast_threshold_2
    → 直接保留，不检查宽、高、面积

contrast_threshold_1 <= contrast < contrast_threshold_2
    → 检查第 1 组宽、高、面积

contrast < contrast_threshold_1
    → 检查第 2 组宽、高、面积
```

## 6. 两组几何阈值之间的关系

当前代码只要求所有宽、高、面积阈值不小于 `0`，没有强制两组之间的大小关系：

```text
min_width_1  与 min_width_2 之间没有强制大小关系
min_height_1 与 min_height_2 之间没有强制大小关系
min_area_1   与 min_area_2   之间没有强制大小关系
```

因此，下面两种配置在程序层面都合法：

```text
第 2 组 > 第 1 组
第 2 组 < 第 1 组
```

通常弱极性区需要更严格的尺寸要求，因此业务配置一般建议：

```text
min_width_2  >= min_width_1
min_height_2 >= min_height_1
min_area_2   >= min_area_1
```

这只是推荐关系，目前不是代码的强制校验条件。每一组内部的宽、高、面积是“并且”关系，三个条件全部通过才保留。

单个参数设为 `0` 表示不检查该项。例如：

```ini
min_width_1=5
min_height_1=0
min_area_1=20
```

表示第 1 组只检查宽度和面积，不限制高度。

## 7. 强制配置约束

当前唯一的组间顺序约束是：

```text
contrast_threshold_2 >= contrast_threshold_1
```

如果不满足，配置加载失败。启用分段过滤后，如果计算出的对比度是 `NaN` 或其他无效值，该检测结果会被过滤。

