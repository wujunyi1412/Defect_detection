# AA 大黑斑自动过滤原理

## 1. 目的

部分图像中会出现一种稳定的成像黑斑：中心较黑，灰度从中心向外逐渐恢复，但它在图像中的位置会发生较大变化，形状和大小也会有轻微差异。

原有 `position_filter_enabled` 规则只根据检测框的位置和尺寸过滤。当黑斑发生位置漂移时，固定坐标范围容易漏过滤；如果盲目扩大坐标范围，又可能误过滤真实缺陷。

因此新增独立的 `black_spot_filter_enabled` 规则。它不依赖固定位置，而是采用：

1. 类别、框宽高和 mask 面积快速预筛选。
2. 灰度外观算法确认黑斑特征。
3. 两个阶段都通过时才删除该检测结果。

原有 AA 中心带规则和位置规则保持不变，新规则可独立启停。

## 2. 总体流程

```text
YOLO 后处理及缺陷几何细化
        ↓
获得最终 DetectionResult
        ↓
类别是否属于 black_spot_categories？ ── 否 ─→ 保留
        ↓ 是
框宽、高和 mask 面积是否在配置范围内？ ── 否 ─→ 保留
        ↓ 是
从 gray_yolo 截取检测框灰度 ROI
        ↓
平滑背景纹理并寻找暗中心
        ↓
检查暗中心位置、中心暗度和径向渐亮规律
        ↓
符合黑斑外观？ ── 否 ─→ 保留为正常缺陷结果
        ↓ 是
过滤该检测结果
```

尺寸和面积只用于选出需要进一步计算的候选框，不能单独触发过滤。这样可以避免把所有 `Stain` 都送入灰度算法，也可以避免仅因为真实缺陷较大就将其删除。

## 3. 第一阶段：几何与 mask 面积预筛选

预筛选条件为：

```text
类别匹配
AND width_min  <= detail.w    <= width_max
AND height_min <= detail.h    <= height_max
AND area_min   <= detail.area <= area_max
```

其中 `detail.area` 是前面 YOLO 分割或缺陷细化流程得到的最终 mask 像素面积，不是 `width × height` 的检测框面积。

采用 mask 面积的原因：

- 检测框可能包含较多背景，框面积不能准确表示黑斑本体大小。
- 同样大小的检测框可能对应大块黑斑、细线或多个零散区域。
- mask 面积更能体现缺陷实际覆盖范围。
- 使用最终细化后的面积，可以让面积、宽度和高度处于当前输出结果的同一坐标尺度。

只有同时满足类别、宽度、高度和 mask 面积范围，才会执行第二阶段的灰度分析。

## 4. 第二阶段：黑斑灰度外观确认

### 4.1 截取并平滑 ROI

根据检测框从 `gray_yolo` 中截取灰度 ROI。坐标会裁剪到图像有效范围，过小或无效的 ROI 不参与过滤。

ROI 会先经过高斯平滑。平滑核尺寸随 ROI 尺寸变化，主要用于减弱图像表面的细密纹理和随机噪声，同时保留大尺度黑斑的渐变结构。

### 4.2 寻找暗中心

在平滑后的 ROI 中寻找最低灰度位置，将其作为候选黑斑中心。

然后计算该点相对检测框中心的归一化偏移：

```text
dx = (dark_x - box_center_x) / box_half_width
dy = (dark_y - box_center_y) / box_half_height
offset = sqrt(dx² + dy²)
```

当 `offset` 大于 `black_spot_max_center_offset_ratio` 时，不认为它是目标黑斑。该约束不是全图位置限制，只用于防止检测框边缘的偶然暗点被当成扩散中心。

### 4.3 计算椭圆径向灰度

以暗中心为原点，将有效区域按归一化椭圆半径划分为 8 个同心环，并分别计算每个环的平均灰度：

```text
暗中心 → 内环 → 中间环 → 外环
  暗       较暗      较亮      接近背景
```

使用椭圆归一化而不是固定圆形，是为了允许黑斑存在一定的长宽差异。

### 4.4 中心暗度条件

算法取最内侧两个环的平均值作为中心灰度，取最外侧两个环的平均值作为外围灰度：

```text
center_darkness = outer_mean - center_mean
```

只有满足：

```text
center_darkness >= black_spot_min_center_darkness
```

才说明中心相对外围足够暗。

### 4.5 径向渐亮条件

依次比较相邻圆环的平均灰度，统计由内向外保持变亮或仅有轻微波动的次数：

```text
monotonic_ratio = 符合渐亮趋势的相邻环数量 / 全部相邻环数量
```

判断相邻环时允许 `black_spot_monotonic_tolerance` 范围内的灰度下降，以适应背景纹理和轻微形状变化。

只有满足：

```text
monotonic_ratio >= black_spot_min_monotonic_ratio
```

才认为该 ROI 具有“中心黑、向外扩散并逐渐恢复”的规律。

## 5. 最终过滤条件

黑斑规则最终采用严格的 AND 关系：

```text
配置开关已打开
AND 类别匹配
AND 框宽度在范围内
AND 框高度在范围内
AND 最终 mask 面积在范围内
AND 暗中心位置合理
AND 中心相对外围足够暗
AND 径向灰度总体向外变亮
```

任意条件不满足都会保留该检测结果。图像为空、ROI 无效或灰度特征计算失败时也会保留，而不是过滤，以降低真实缺陷被误删的风险。

## 6. 配置说明

配置位于 `config.ini` 的 `[AA]` 段：

```ini
black_spot_filter_enabled=0
black_spot_debug_log_enabled=0
black_spot_categories=Stain

black_spot_width_min=145
black_spot_width_max=240
black_spot_height_min=145
black_spot_height_max=260
black_spot_mask_area_min=3000
black_spot_mask_area_max=50000

black_spot_min_center_darkness=12.0
black_spot_min_monotonic_ratio=0.71
black_spot_monotonic_tolerance=2.0
black_spot_max_center_offset_ratio=0.45
```

| 配置项 | 含义 |
|---|---|
| `black_spot_filter_enabled` | 新规则独立开关，`1` 开启、`0` 关闭 |
| `black_spot_debug_log_enabled` | 将预筛选结果、8 环灰度和最终判定原因写入现有日志 |
| `black_spot_categories` | 应用规则的 YOLO 类别，支持英文逗号分隔 |
| `black_spot_width_min/max` | 候选检测框宽度范围，包含边界 |
| `black_spot_height_min/max` | 候选检测框高度范围，包含边界 |
| `black_spot_mask_area_min/max` | 最终 mask 像素面积范围，包含边界 |
| `black_spot_min_center_darkness` | 外围灰度均值减中心灰度均值的最小值，当前图像为 8-bit 灰度 |
| `black_spot_min_monotonic_ratio` | 相邻径向环符合向外渐亮趋势的最低比例，取值范围 `[0, 1]` |
| `black_spot_monotonic_tolerance` | 相邻环比较时允许的灰度波动，值越大越宽松 |
| `black_spot_max_center_offset_ratio` | 暗中心相对框中心允许的最大归一化偏移，取值范围 `[0, 1]` |

默认配置中 `black_spot_filter_enabled=0`，因此加入新版本后不会自动改变现有生产结果。

### 调试日志

开启以下配置后，规则会把中间计算值写入项目现有日志，不会绘制到结果图片：

```ini
black_spot_debug_log_enabled=1
```

日志使用 `Info` 级别，内容包括检测框、mask 面积、几何预筛选结果、暗中心、8 个环的灰度均值、中心暗度、渐亮次数、单调比例和最终原因。例如：

```text
[AA_BLACK_SPOT] name=Stain x=454.220 y=304.300 w=170.520 h=211.390 mask_area=12580 geometry_pass=1
[AA_BLACK_SPOT] name=Stain x=454.220 y=304.300 w=170.520 h=211.390 mask_area=12580 appearance_pass=1 filtered=1 reason=passed dark_center=(86,102) center_offset_ratio=0.120 ring_means=[72.3,79.5,91.2,105.6,121.4,135.8,146.2,152.7] center_mean=75.9 outer_mean=149.5 center_darkness=73.6 darkness_threshold=12 rising_steps=7/7 monotonic_ratio=1 monotonic_threshold=0.71
```

几何条件或外观条件不通过时，`reason` 会说明原因，例如 `width_out_of_range`、`mask_area_out_of_range`、`center_offset_too_large`、`center_darkness_too_low` 或 `monotonic_ratio_too_low`。不属于 `black_spot_categories` 的类别不会输出该调试日志。

## 7. 与原有 AA 规则的关系

AA 后过滤包含三条彼此独立的规则：

1. 原中心带规则：中心 Y 范围加宽高比或最大宽度。
2. 原位置规则：左上角 X/Y 加框宽高范围。
3. 新黑斑规则：尺寸和 mask 面积预筛选加灰度外观确认。

三条规则使用 OR 关系，任意一个已启用的规则命中都会删除对应 YOLO 结果。关闭新规则不会影响原有两条规则。

## 8. 建议调参顺序

建议先收集包含目标黑斑和真实 `Stain` 的样本，并记录最终检测框宽高与 mask 面积。

1. 先调整 `width`、`height` 和 `mask_area` 范围，让绝大多数小型真实缺陷不进入灰度计算。
2. 调整 `min_center_darkness`，区分明显黑斑与普通亮度波动。
3. 调整 `min_monotonic_ratio`，排除轮廓不规则、局部突变明显的真实污点。
4. 仅在黑斑中心经常偏离检测框中心时，适当放宽 `max_center_offset_ratio`。
5. 仅在背景纹理造成相邻环轻微反向变化时，适当增大 `monotonic_tolerance`。
6. 使用包含真实大尺寸 `Stain` 的数据做回归测试，确认没有误过滤后再开启生产配置。

参数影响方向：

- 提高 `min_center_darkness`：过滤更严格，误过滤减少，但可能漏掉较浅黑斑。
- 提高 `min_monotonic_ratio`：过滤更严格，更强调平滑扩散规律。
- 提高 `monotonic_tolerance`：趋势判断更宽松，抗纹理能力增强，但误过滤风险可能增加。
- 提高 `max_center_offset_ratio`：允许暗中心离框中心更远，召回增加，但边缘暗物体更容易进入判断。

## 9. 对应实现位置

- 配置示例：`Corona_defect_detection/config.ini`
- 配置结构与默认值：`config/config.h`
- 配置读取和合法性校验：`config/config.cpp`
- 黑斑几何预筛选及灰度算法：`Corona_defect_detection/internal/aa_yolo_filter.cpp`
- 灰度图传入 AA 过滤器：`Corona_defect_detection/internal/result_builder.cpp`
- 单元与配置解析测试：`tests/smoke_tests.cpp`

