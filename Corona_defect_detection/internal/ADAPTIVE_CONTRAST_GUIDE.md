# 自适应前景/背景对比度计算：从零开始理解

本文专门解释：

```text
Corona_defect_detection/internal/image_utils_adaptive.cpp
```

中的函数：

```cpp
float CalculateContrastRatioAdaptive(
    const cv::Mat& gray_img,
    const cv::Mat& binary_mask,
    int x,
    int y,
    int w,
    int h,
    ContrastPolarity polarity);
```

目标是让没有传统图像处理基础的人也能理解：

1. 对比度到底在算什么。
2. 前景和背景从哪里来。
3. 为什么不能简单使用两个平均值相除。
4. 为什么需要背景环、平面拟合、MAD 和百分位。
5. 极小缺陷、亮度渐变、产品黑边怎样处理。
6. 该函数在 YOLO、PatchCore 和几何细化流程中的位置。

---

## 1. 一句话说明这个函数

它要计算的是：

```text
缺陷前景灰度 / 缺陷所在位置本来应该有的正常背景灰度
```

它不是简单计算：

```text
整个检测框平均灰度 / 整张图片平均灰度
```

而是：

```text
围绕缺陷 mask 找正常背景
        ↓
排除产品外部黑边
        ↓
拟合局部背景亮度变化
        ↓
在每个前景像素位置预测正常背景
        ↓
逐像素计算前景/背景比值
        ↓
用稳健百分位得到最终 contrast
```

---

## 2. 什么是灰度

对于常见的 8 位灰度图：

```text
0   = 黑色
255 = 白色
```

例如：

```text
正常产品背景灰度 = 200
黑污灰度         = 100
```

黑污相对背景的比值：

```text
100 / 200 = 0.50
```

如果是亮缺陷：

```text
正常背景灰度 = 150
亮缺陷灰度   = 210

210 / 150 = 1.40
```

因此：

```text
暗缺陷：contrast < 1
接近背景：contrast ≈ 1
亮缺陷：contrast > 1
```

---

## 3. 为什么不是“差值”

也可以计算：

```text
背景灰度 - 前景灰度
```

但差值对整体亮度比较敏感。

场景 A：

```text
背景=200
缺陷=160
差值=40
比值=0.80
```

场景 B：

```text
背景=100
缺陷=60
差值=40
比值=0.60
```

两个场景差值相同，但缺陷相对于背景的变暗程度不同。

比值更容易表达：

```text
缺陷保留了正常背景亮度的多少比例
```

---

## 4. 输入参数分别代表什么

函数接口：

```cpp
CalculateContrastRatioAdaptive(
    gray_img,
    binary_mask,
    x, y, w, h,
    polarity);
```

### 4.1 gray_img

单通道灰度图。

代码要求：

```text
gray_img 不能为空
gray_img.channels() 必须等于 1
```

YOLO 流程中，它通常是归一化后的 8 位灰度图。

函数内部会将当前 context 转成：

```cpp
CV_32F
```

这样可以进行平面拟合和浮点除法。

### 4.2 binary_mask

二值缺陷 mask：

```text
0   = 非缺陷
非0 = 缺陷
```

函数最终会统一阈值化成：

```text
0 或 255
```

### 4.3 x、y、w、h

检测框：

```text
x、y = 左上角
w、h = 宽和高
```

mask 只在这个检测框对应的位置作为前景使用。

### 4.4 polarity

缺陷明暗类型：

```cpp
ContrastPolarity::Dark
ContrastPolarity::Bright
ContrastPolarity::Auto
```

- `Dark`：按暗缺陷统计。
- `Bright`：按亮缺陷统计。
- `Auto`：根据比值中位位置自动判断。

---

## 5. 总体执行流程

```text
检查图像、mask 和框
        ↓
将 mask 对齐到检测框
        ↓
统计前景 mask 面积
        ↓
根据面积计算背景环半径
        ↓
扩展检测框得到 context
        ↓
在 context 中放置前景 mask
        ↓
生成内膨胀和外膨胀
        ↓
相减得到背景环
        ↓
识别并排除外部黑边
        ↓
收集有效背景样本
        ↓
第一次拟合背景平面
        ↓
用 MAD 找背景离群点
        ↓
使用正常样本第二次拟合
        ↓
逐个前景像素计算灰度比
        ↓
根据 Dark/Bright/Auto 稳健聚合
        ↓
返回最终 contrast
```

---

## 6. 第一步：校正检测框

辅助函数：

```cpp
ClampRect(...)
```

会检查：

```text
图像宽高是否有效
w、h 是否大于 0
框是否与图像存在有效交集
```

还会把坐标限制到图像内部。

例如：

```text
原框：x=-3, y=10, w=20, h=15
```

校正后：

```text
x=0
右边界仍限制在图像内部
```

如果校正后框为空，函数返回：

```text
0.0f
```

这里的 `0` 表示无法可靠计算，不一定表示缺陷像素真的是黑色零值。

---

## 7. 第二步：将 mask 对齐到检测框

辅助函数：

```cpp
BuildDetectionMask(...)
```

支持三种 mask 尺寸。

### 7.1 mask 与整张图一样大

```text
binary_mask.size() == gray_img.size()
```

函数会按照检测框裁出对应部分：

```cpp
binary_mask(detection)
```

### 7.2 mask 与检测框一样大

```text
binary_mask.size() == detection.size()
```

直接使用该 mask。

这类输入常见于几何细化完成后，因为最终连通域 mask 已经裁成了细化框大小。

### 7.3 mask 是其他尺寸

函数使用最近邻插值：

```cpp
cv::INTER_NEAREST
```

缩放到检测框大小。

为什么不用线性插值？

线性插值可能产生：

```text
0、37、128、219、255
```

最近邻更容易保持二值区域形状：

```text
0 或 255
```

最后函数仍会执行二值阈值化，所有非零值都变成 `255`。

---

## 8. 第三步：计算前景面积

```cpp
foreground_area =
    cv::countNonZero(detection_mask);
```

这个面积是 mask 中缺陷像素数量。

例如：

```text
0 1 1 0
1 1 1 0
0 1 0 0
```

前景面积：

```text
6
```

如果面积为 `0`，说明 mask 没有缺陷前景，函数直接返回 `0`。

---

## 9. 第四步：由面积计算等效半径

不同大小的缺陷不应该使用完全相同的背景环。

代码先假设缺陷面积相当于一个圆：

```text
area = π × radius²
```

反推半径：

```text
equivalent_radius =
    sqrt(foreground_area / π)
```

它不代表真实缺陷必须是圆形，只是用面积换算一个合理的尺度。

### 示例

面积为 `1`：

```text
equivalent_radius=sqrt(1/π)≈0.56
```

面积为 `100`：

```text
equivalent_radius=sqrt(100/π)≈5.64
```

面积为 `400`：

```text
equivalent_radius=sqrt(400/π)≈11.28
```

---

## 10. 第五步：计算背景环内外半径

内半径：

```text
inner_radius =
    clamp(round(equivalent_radius×0.20), 2, 8)
```

外半径：

```text
outer_radius =
    clamp(
        round(equivalent_radius×0.80),
        inner_radius+4,
        32
    )
```

含义：

- 内半径留出缺陷边缘缓冲区。
- 外半径决定取多远的正常背景。
- 小缺陷至少留 `2px` 内间隔。
- 背景环至少比内半径宽 `4px`。
- 最大外半径限制为 `32px`，防止取到太远区域。

### 小黑点示例

面积为 `1`：

```text
equivalent_radius≈0.56
inner_radius=2
outer_radius=6
```

所以 `1px` 黑点不会使用整个 YOLO 框大小决定背景，而是在缺陷附近约 `2～6px` 的范围取背景。

### 较大缺陷示例

面积为 `400`：

```text
equivalent_radius≈11.28
inner_radius≈2
outer_radius≈9
```

背景范围会随缺陷面积适当增大。

---

## 11. 第六步：扩展检测框得到 context

扩展距离：

```text
outer_radius + 2
```

结构可以想成：

```text
+--------------------------------+
|             context            |
|                                |
|       外背景环                 |
|      +-------------+           |
|      |  缓冲区     |           |
|      |   +-----+   |           |
|      |   |mask |   |           |
|      |   +-----+   |           |
|      +-------------+           |
|                                |
+--------------------------------+
```

扩展 context 的原因是：背景环必须能够走到 YOLO 框外。

如果只在检测框内部找背景，而 mask 又接近填满检测框，就没有足够的正常背景可用。

context 仍会被限制在图像范围内，不会越界。

---

## 12. 第七步：在 context 中放置前景 mask

函数创建一张与 context 同样大小的全零 mask：

```cpp
foreground_mask
```

再将检测框内的前景 mask 放到 context 的正确坐标位置。

这样后面的膨胀操作可以自然扩展到检测框外，但前景本身仍然来自原检测框。

---

## 13. 第八步：怎样生成背景环

对前景 mask 做两次椭圆膨胀：

```text
inner_dilated = 前景向外膨胀 inner_radius
outer_dilated = 前景向外膨胀 outer_radius
```

然后：

```text
background_ring =
    outer_dilated AND NOT inner_dilated
```

可以理解为用大区域减去小区域：

```text
OOOOOOOOOOOOO
OOOIIIIIIIOOO
OOOIIFFF I OOO
OOOIIIIIIIOOO
OOOOOOOOOOOOO
```

其中：

```text
F = 前景缺陷
I = 内缓冲区，不作为背景
O = 背景环
```

内缓冲区用于避免：

- 缺陷边缘模糊。
- mask 边界不准确。
- 缺陷光晕或拖尾。
- 插值造成的过渡像素。

---

## 14. 为什么不能直接使用背景环

缺陷靠近产品边缘时，背景环可能同时包含：

```text
正常产品表面：灰度200
产品外部黑边：灰度0
黑到亮过渡：灰度20～150
```

如果将这些像素全部用于背景拟合，预测背景可能严重偏暗。

例如本来应该计算：

```text
100 / 200 = 0.50
```

如果错误背景被估计为 `80`：

```text
100 / 80 = 1.25
```

一个黑污反而会被算成亮缺陷。

所以必须先排除产品外部黑区。

---

## 15. 第九步：识别与 context 外边连通的黑区

辅助函数：

```cpp
FindOuterDarkRegion(...)
```

### 15.1 收集 context 灰度

只收集有限值：

```text
不是 NaN
不是正负无穷
```

### 15.2 计算两个灰度参考值

```text
low_reference =
    第10百分位灰度

material_reference =
    第80百分位灰度
```

第 `80` 百分位用于近似较亮的正常产品材料。

第 `10` 百分位用于观察较暗部分的大致水平。

### 15.3 计算黑色候选阈值

```text
dark_threshold =
    min(
        0.35 × material_reference,
        low_reference
        + 0.20 × (material_reference-low_reference)
    )
```

假设：

```text
low_reference=0
material_reference=200
```

两个候选阈值：

```text
0.35×200=70
0+0.20×(200-0)=40
```

最终：

```text
dark_threshold=min(70,40)=40
```

灰度小于等于 `40` 的像素成为黑色候选。

### 15.4 为什么还要判断连通性

框内真实黑污也可能低于黑色阈值。

因此不能简单删除所有暗像素，只删除：

```text
与 context 四条外边相连的暗连通域
```

通常：

- 产品外部黑背景会接触 context 外边。
- 产品内部孤立黑污不会接触 context 外边。

### 15.5 覆盖黑边过渡带

找到外部黑区后，再膨胀：

```text
transition_guard =
    clamp(min(w,h)/12,1,3)
```

也就是向内扩展 `1～3px`，覆盖黑到亮的边缘过渡。

最后执行：

```cpp
background_ring.setTo(0, outer_dark_region);
```

这些黑边像素不再属于有效背景环。

---

## 16. 第十步：收集背景样本

每个背景样本包含：

```cpp
struct BackgroundSample {
    double x;
    double y;
    double value;
};
```

也就是：

```text
样本坐标 x
样本坐标 y
该位置灰度 value
```

为什么要保存坐标？

因为后面不是只计算一个背景平均值，而是拟合：

```text
background(x,y)=a*x+b*y+c
```

如果背景环中的像素值不是有限数字，就不会进入背景样本。

---

## 17. 为什么使用背景平面

真实产品表面可能存在亮度渐变：

```text
左边160 → 中间180 → 右边200
```

若只使用背景平均值 `180`：

- 左侧正常背景会被认为偏暗。
- 右侧正常背景会被认为偏亮。
- 相同程度的缺陷在不同位置得到不同 contrast。

平面模型：

```text
background(x,y)=a*x+b*y+c
```

可以表示：

- 水平方向缓慢变化。
- 垂直方向缓慢变化。
- 整体基础亮度。

---

## 18. 第十一步：第一次拟合背景平面

辅助函数：

```cpp
FitPlane(...)
```

每个样本满足：

```text
value ≈ a*x+b*y+c
```

算法通过最小二乘寻找 `a、b、c`，让所有样本的预测误差平方和尽量小。

代码构建法方程后使用：

```cpp
cv::solve(..., cv::DECOMP_SVD)
```

SVD 对接近退化的数值情况通常比直接求逆稳定。

最少需要 `3` 个样本才能求三个未知数，但后续稳健拟合要求至少 `12` 个背景样本。

---

## 19. 为什么第一次拟合还不够

背景环里仍可能残留：

- 另一个缺陷。
- 高亮反光。
- 孤立噪声点。
- 没有完全排除的边缘像素。

普通最小二乘容易被这些异常点带偏。

所以第一次拟合只是为了估计大致背景，再使用 MAD 找正常样本。

---

## 20. 第十二步：计算平面残差

对每个背景样本：

```text
predicted =
    a*x+b*y+c

residual =
    actual_value-predicted
```

例如：

```text
实际灰度=203
平面预测=200
residual=3
```

另一个异常黑点：

```text
实际灰度=80
平面预测=200
residual=-120
```

---

## 21. 第十三步：使用 MAD 估计正常残差

先计算残差中位数：

```text
residual_median =
    median(residuals)
```

再计算：

```text
deviation_i =
    |residual_i-residual_median|
```

MAD：

```text
MAD=median(deviations)
```

转换成类似标准差的尺度：

```text
sigma =
    max(0.5,1.4826×MAD)
```

这里设置：

```text
sigma>=0.5
```

用于避免完全平滑图像得到零容差。

---

## 22. 用数字手算背景离群点

假设残差：

```text
-2 -1 0 1 2 -1 0 1 -30
```

排序后：

```text
-30 -2 -1 -1 0 0 1 1 2
```

残差中位数：

```text
0
```

绝对偏差：

```text
2 1 0 1 2 1 0 1 30
```

排序：

```text
0 0 1 1 1 1 2 2 30
```

所以：

```text
MAD=1
sigma=max(0.5,1.4826×1)
     =1.4826
```

正常样本限制：

```text
inlier_limit=3×sigma
            ≈4.4478
```

正常残差 `-2～2` 都保留。

异常残差 `-30` 被删除。

---

## 23. 第十四步：使用正常样本第二次拟合

背景样本保留条件：

```text
|residual-residual_median|
    <=3×sigma
```

如果正常样本至少有 `12` 个，算法再次拟合背景平面。

第二次拟合不再容易被异常黑点或亮点带偏。

这就是：

```text
稳健背景平面
```

如果最初背景样本总数少于 `12`，或者第一次平面拟合失败，整个函数返回 `0`。

---

## 24. 第十五步：防止分母接近零

对比度需要做除法：

```text
foreground/background
```

如果背景接近 `0`，结果会非常大或不稳定。

代码设置：

```text
denominator_floor =
    max(
        1.0,
        |background_median|×0.10
    )
```

例如背景中位数为 `200`：

```text
denominator_floor=max(1,20)=20
```

某位置预测背景低于 `20` 时，不计算该前景像素的比值。

这可以避免产品外部黑色区域成为分母。

---

## 25. 第十六步：逐前景像素计算比值

对于 mask 中每个前景像素：

```text
predicted_background(x,y)=a*x+b*y+c

ratio(x,y)=
    foreground_gray(x,y)
    / predicted_background(x,y)
```

例如：

```text
像素A：
前景灰度=100
预测背景=200
ratio=0.50

像素B：
前景灰度=105
预测背景=210
ratio=0.50
```

两个像素原始灰度不同，但相对于各自背景的变暗比例相同。

只保留：

```text
ratio 是有限值
0 <= ratio <= 10
```

超出范围的异常计算结果会被忽略。

---

## 26. 为什么不能直接平均所有 ratio

YOLO mask 可能比真实缺陷稍大。

例如暗缺陷的前景比值：

```text
0.50 0.52 0.55 0.58 0.95 0.98 1.00
```

后面几个接近 `1` 的像素可能是 mask 包含的正常背景。

直接平均会把结果向 `1` 拉高，使缺陷看起来不够暗。

另一方面，单个极端坏点可能得到：

```text
0.02
```

只取最小值又会让结果过于敏感。

因此代码采用百分位区间平均。

---

## 27. 第十七步：Auto 怎样判断明暗

所有 ratio 会先排序。

`Auto` 查看排序中间位置：

```text
中间 ratio <=1
→ 按 Dark 处理

中间 ratio >1
→ 按 Bright 处理
```

当前项目中：

- `Stain`、`DarkClusters` 明确传入 `Dark`。
- `BrightStripes` 明确传入 `Bright`。
- PatchCore 某些调用使用默认 `Auto`。

---

## 28. 少于 5 个前景像素怎样处理

如果有效 ratio 少于 `5` 个，百分位统计样本太少。

代码直接返回：

```text
排序后中间位置的 ratio
```

例如 `1px` 缺陷：

```text
ratios=[0.50]
返回0.50
```

`2×2` 缺陷可能得到：

```text
ratios=[0.48,0.50,0.51,0.53]
```

排序后取索引：

```text
size/2=4/2=2
返回ratios[2]=0.51
```

注意：当前实现对偶数个小样本取偏上的中间位置，不是中间两个数的平均值。

---

## 29. 暗缺陷怎样聚合

当 ratio 数量不少于 `5`，暗缺陷使用：

```text
第10百分位 ～ 第40百分位
```

然后对这个区间中的 ratio 求平均值。

示例：

```text
排序后：
0.40 0.45 0.50 0.55 0.60
0.80 0.95 0.98 1.00 1.02
```

算法关注较低但不只取最极端的一部分。

作用：

- 避免 mask 中正常背景将结果拉向 `1`。
- 避免只由最黑的单个坏点决定结果。
- 更接近真实暗缺陷核心的代表比值。

---

## 30. 亮缺陷怎样聚合

亮缺陷使用：

```text
第60百分位 ～ 第90百分位
```

然后求区间平均。

它关注较亮的前景核心，同时避开最极端的少量高亮噪声。

---

## 31. 一个完整的暗缺陷计算示例

假设：

```text
检测框=20×19
YOLO mask面积=60
正常背景约为200
```

### 31.1 计算背景尺度

```text
equivalent_radius=
    sqrt(60/π)
    ≈4.37

inner_radius=
    clamp(round(4.37×0.20),2,8)
    =2

outer_radius=
    clamp(round(4.37×0.80),6,32)
    =6
```

### 31.2 生成背景环

前景外 `2px` 以内不作为背景，约 `2～6px` 的区域作为候选背景环。

### 31.3 排除黑边

假设 context 包含灰度为 `0` 的产品外部黑区，这些区域与 context 上边相连，因此从背景环删除。

### 31.4 稳健平面拟合

有效背景拟合得到：

```text
background(x,y)=0.4x+0.1y+190
```

### 31.5 逐像素比值

最终得到前景 ratios：

```text
0.58 0.60 0.61 0.62 0.64
0.65 0.90 0.95 0.98 1.00
```

### 31.6 稳健聚合

这是暗缺陷，选择第 `10～40` 百分位区间并平均，结果可能约为：

```text
contrast≈0.61
```

而不是直接平均全部数据得到更接近 `0.75～0.80` 的结果。

---

## 32. 黑边旁 1px 缺陷示例

```text
产品外部黑边=0
正常产品背景=200
1px缺陷=100
```

处理：

```text
1. mask面积=1
2. inner_radius=2
3. outer_radius=6
4. 与 context 外边连通的黑区被删除
5. 使用产品表面拟合背景≈200
6. ratio=100/200=0.50
7. 只有1个ratio，直接返回0.50
```

黑边不会使分母变成接近 `0`。

---

## 33. 亮度渐变示例

假设背景：

```text
background(x)=100+0.7x
```

缺陷在每个位置都变为正常背景的 `70%`：

```text
foreground(x)=0.70×background(x)
```

平面拟合可以在每个前景位置预测不同背景：

```text
左侧：70/100=0.70
中间：105/150=0.70
右侧：140/200=0.70
```

最终 contrast 仍然稳定在：

```text
0.70
```

---

## 34. 计算失败和真实 0 怎样区分

现在约定：

```text
有限数值（包括0.0f） = 计算成功
NaN                    = 计算失败
```

`0.0f` 是一个合法结果。例如前景像素确实为纯黑、预测背景大于零时，
前景除以背景就可能等于 `0.0f`。不能再用 `contrast == 0` 判断失败。

下面情况会返回 `NaN`。

### 输入无效时返回 NaN

```text
gray_img为空
binary_mask为空
gray_img不是单通道
```

### 检测框无效时返回 NaN

```text
w<=0
h<=0
框裁剪后为空
```

### 前景无效时返回 NaN

```text
mask没有非零像素
```

### 背景无效时返回 NaN

```text
黑边排除后背景样本少于12个
第一次平面拟合失败
```

### 比值无效时返回 NaN

```text
预测背景过低
前景像素不是有限值
ratio不在0～10范围
最终没有有效ratio
```

在 C++ 中统一这样判断：

```cpp
if (IsContrastRatioValid(contrast)) {
    // 计算成功，contrast可以等于0
} else {
    // 计算失败，contrast是NaN
}
```

在 C# 中这样判断：

```csharp
if (float.IsNaN(contrast)) {
    // 计算失败
}
```

---

## 35. 真实 0 和计算失败对分类有什么影响

YOLO 后处理中有逻辑：

```text
contrast计算成功，并且
Stain 且 contrast <= dark_clusters_threshold
→ DarkClusters
```

当前默认：

```ini
dark_clusters_threshold=0.8
```

所以：

- `contrast=0.65`：可能变为 `DarkClusters`。
- `contrast=0`：是合法的极暗结果，也会变为 `DarkClusters`。
- `contrast=0.90`：通常保留为 `Stain`。
- `contrast=NaN`：表示失败，不参与 `Stain/DarkClusters` 重分类。

当 `contrast_threshold=0` 时表示关闭对比度过滤，`NaN` 不会仅因该阈值被过滤。
当 `contrast_threshold>0` 时，`NaN` 不能通过对比度过滤。

---

## 36. 在 YOLO 流程中怎样调用

位置：

```text
Corona_defect_detection/internal/yolo_postprocess.cpp
```

输入：

```text
gray_yolo
YOLO输出mask
YOLO检测框
由类别确定的Dark/Bright/Auto
```

这里是第一次计算对比度。

结果用于：

- `detail.contrast`。
- 初次判断 `Stain/DarkClusters`。
- 后续类别过滤。

---

## 37. 在 PatchCore 流程中怎样调用

位置：

```text
Corona_defect_detection/internal/patchcore_postprocess.cpp
```

PatchCore 的异常 mask 会被缩放回原图，再传入自适应对比度函数。

默认极性通常是：

```cpp
ContrastPolarity::Auto
```

函数会根据前景比值中间位置判断异常区域偏暗还是偏亮。

---

## 38. 在小黑点细化后怎样调用

位置：

```text
Corona_defect_detection/internal/dark_defect_refiner_adaptive.cpp
```

小框细化成功后已经获得：

```text
final_mask
```

该 mask 是传统算法最终选出的精确连通域。

细化函数使用：

```text
精确final_mask
细化后的x/y/w/h
Dark极性
```

再次计算 contrast。

如果新 contrast：

```text
是有限的非负数（包括0）
```

则覆盖 YOLO 阶段的旧值。

如果新计算失败返回 `NaN`，则保留旧 contrast，避免因为局部背景不足将可靠的旧值覆盖成无效值。

---

## 39. 为什么对比度最多计算两次

第一次：

```text
YOLO mask
→ 得到初始contrast
→ 初步判断Stain/DarkClusters
```

第二次仅在小框细化成功后：

```text
最终传统连通域mask
→ 得到更准确contrast
→ 重新判断Stain/DarkClusters
→ 使用最终contrast_threshold过滤
```

因此最终输出可能是：

```text
x、y、w、h、area、contrast
全部来自同一个最终连通域
```

---

## 40. 新旧方法的主要差异

### 旧方法的限制

旧方法先将灰度图裁到检测框，再在框内构造背景环。

可能出现：

- mask 接近填满框时，没有框外正常背景。
- 背景环被裁断。
- bbox 尺寸 mask 按全图坐标裁剪时存在尺寸风险。
- 用单个背景代表值，不能补偿亮度渐变。
- 黑边占比较大时可能污染背景。

### 新方法

```text
背景环允许走到框外
支持全图mask和bbox mask
背景范围按mask面积变化
排除与context外边连通的黑区
拟合局部背景平面
用MAD删除离群背景样本
逐像素计算前景/预测背景
用百分位区间稳健聚合
```

---

## 41. 参数速查表

| 参数 | 当前值 | 作用 |
|---|---:|---|
| 等效半径 | `sqrt(area/π)` | 将 mask 面积转换为空间尺度 |
| 内半径比例 | `0.20` | 缺陷与背景环之间的缓冲 |
| 内半径范围 | `2～8px` | 防止过近或过远 |
| 外半径比例 | `0.80` | 背景环外侧范围 |
| 外半径范围 | `inner+4～32px` | 保证环宽并限制最远距离 |
| 黑区亮材料参考 | 第 `80` 百分位 | 估计产品较亮材料 |
| 黑区低灰参考 | 第 `10` 百分位 | 估计暗区水平 |
| 黑边过渡保护 | `1～3px` | 排除黑到亮过渡 |
| 最少背景样本 | `12` | 稳健平面拟合要求 |
| MAD换算 | `1.4826×MAD` | 得到稳健 sigma |
| sigma下限 | `0.5` | 防止零容差 |
| 背景离群阈值 | `3×sigma` | 删除异常背景样本 |
| 分母下限 | `max(1,0.1×背景中位数)` | 防止除以接近零 |
| ratio有效范围 | `0～10` | 排除异常计算 |
| 暗缺陷聚合 | 第 `10～40` 百分位 | 关注暗核心 |
| 亮缺陷聚合 | 第 `60～90` 百分位 | 关注亮核心 |
| 小样本策略 | `<5` 个取中间值 | 适配1px和2×2缺陷 |

---

## 42. 怎样调试对比度异常

遇到结果不符合预期时，建议依次保存或打印：

```text
1. 原始gray_img
2. detection框
3. 对齐后的detection_mask
4. foreground_area
5. inner_radius、outer_radius
6. context
7. background_ring
8. outer_dark_region
9. 黑边排除后的background_ring
10. background_samples数量
11. 第一次背景平面a、b、c
12. residual_median、MAD、sigma
13. inlier背景样本数量
14. 第二次背景平面a、b、c
15. 每个前景像素的predicted_background
16. foreground_ratios
17. 最终百分位区间
18. 最终contrast
```

这样可以判断问题属于：

- mask 错位。
- 背景环太小或太大。
- 黑边没有排除。
- 有效背景不足。
- 平面拟合被异常点影响。
- 前景 mask 包含太多正常像素。
- 百分位聚合不适合当前缺陷。

---

## 43. 常见问题

### 43.1 为什么 mask 越大，背景环也会变大

较大缺陷的边缘影响范围通常更大，需要离缺陷更远一些取正常背景。

但外半径限制为 `32px`，不会无限扩大。

### 43.2 为什么不是直接使用框外四条边

框可能不以缺陷为中心，或者缺陷 mask 只占框的一小部分。

围绕 mask 构造背景环，比围绕矩形框更符合真实缺陷形状。

### 43.3 为什么背景平面只使用一次线性模型

当前目标是补偿缓慢、近似线性的照明变化。更复杂的曲面可能拟合纹理或缺陷本身，稳定性更难控制。

### 43.4 为什么暗缺陷不是取最小 ratio

最小值可能只是一个坏点、死点或随机噪声。第 `10～40` 百分位平均更加稳定。

### 43.5 为什么亮缺陷不是取最大 ratio

最大值可能是反光或饱和像素。第 `60～90` 百分位平均可以减少极端值影响。

### 43.6 为什么细化后的 contrast 可能比 YOLO contrast 更低

YOLO mask 可能包含正常背景，使旧比值接近 `1`。细化后的 mask 更贴近黑点，所以新比值可能明显下降。

### 43.7 为什么细化后类别会变化

最终 contrast 会重新与：

```ini
dark_clusters_threshold
```

比较，所以 `Stain` 可能变成 `DarkClusters`，反之也可能重新变为 `Stain`。

---

## 44. 当前方法的限制

### 44.1 必须有正常产品背景

如果缺陷周围全部是产品外部黑区，没有任何正常材料，就无法定义“缺陷前景/正常背景”。

此时返回 `0` 比制造错误结果更安全。

### 44.2 背景模型只能描述平面渐变

强烈弯曲照明、周期性阴影或复杂反光不能被一个平面完全描述。

### 44.3 结果依赖 mask

如果 mask 完全错位或覆盖错误对象，再好的背景算法也无法得到正确前景比值。

### 44.4 极小缺陷可能是传感器噪声

算法能计算 `1px` 的对比度，但“能计算”不等于“业务上一定是真缺陷”。还需要结合：

- YOLO 置信度。
- 最小面积。
- 重复拍摄稳定性。
- 产品规格。

### 44.5 黑色产品上的黑缺陷难以测量

如果正常背景本身接近零，前景/背景比值的分母不稳定。当前函数会跳过过低背景预测，可能返回 `0`。

---

## 45. 建议怎样重新确定阈值

新旧函数虽然都叫“前景/背景比值”，但背景选取和稳健聚合不同，数值分布会变化。

准备已标注数据，分别统计：

```text
正常误检样本contrast
Stain样本contrast
DarkClusters样本contrast
BrightStripes样本contrast
靠黑边样本contrast
1～4px小点contrast
低对比度样本contrast
```

观察分布后再设置：

```ini
dark_clusters_threshold
contrast_threshold
```

不要只用一两张图片确定阈值。

---

## 46. 伪代码速查

```text
function CalculateContrastRatioAdaptive(
    gray, mask, box, polarity):

    if 输入无效:
        return NaN

    box = 限制到图像范围
    foreground = 将mask对齐到box

    if foreground面积为0:
        return NaN

    equivalent_radius = sqrt(area/π)
    inner_radius = 动态内半径
    outer_radius = 动态外半径

    context = 向box外扩展
    将foreground放入context坐标

    inner = 膨胀foreground(inner_radius)
    outer = 膨胀foreground(outer_radius)
    background_ring = outer - inner

    outer_dark = 找与context边缘连通的黑区
    background_ring -= outer_dark

    samples = 收集背景环中的(x,y,gray)

    if samples少于12:
        return NaN

    plane1 = 第一次拟合背景
    residuals = 实际背景-plane1预测
    sigma = max(0.5,1.4826×MAD)
    inliers = 残差在3×sigma内的样本
    plane2 = 使用inliers重新拟合

    ratios = []
    for 每个前景像素:
        predicted = plane2(x,y)
        if predicted足够大:
            ratios加入 gray(x,y)/predicted

    if ratios为空:
        return NaN

    if ratios数量<5:
        return 排序中间值

    if polarity是Dark:
        return 第10～40百分位区间平均

    if polarity是Bright:
        return 第60～90百分位区间平均

    if polarity是Auto:
        根据ratio中间位置决定Dark或Bright
```

---

## 47. 最容易记住的版本

```text
mask 告诉算法哪些像素是缺陷前景。

算法在 mask 周围建立背景环，
但先删除与局部外边连通的产品黑边。

正常背景不是一个固定数字，
而是一个随 x、y 缓慢变化的平面。

MAD 用来删除不符合背景平面的异常样本，
然后重新拟合更可靠的背景。

每个前景像素都除以自己位置的预测背景，
最后用稳健百分位区间得到代表性 contrast。

暗缺陷通常小于 1，
亮缺陷通常大于 1，
真实纯黑前景可以返回 0，
无法可靠计算时返回 NaN。
```
