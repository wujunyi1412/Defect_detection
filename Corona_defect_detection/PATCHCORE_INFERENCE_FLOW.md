# PatchCore 图片推理流程与特征评分说明

本文根据当前 C++ 实现说明 PatchCore 在推理图片时的完整流程，重点介绍多层特征的提取、合并、自适应平均池化及异常得分计算方法。

核心实现位于 [`patchcore_inference.cpp`](patchcore_inference.cpp)。

## 1. 整体流程

```text
原图
  → 转 8-bit/BGR、屏蔽边缘
  → 等比例缩放并补黑到 224×224
  → ImageNet 标准化
  → WideResNet50-2 backbone
  → 提取 layer2、layer3 特征
  → 每个位置展开 3×3 patch
  → 各层 patch 特征压缩到 1024 维
  → 空间对齐到 28×28
  → 合并两个层的特征并压缩到 1024 维
  → 使用 Faiss 与正常样本特征库做最近邻匹配
  → 得到 28×28 patch 异常分数图
  → 最大 patch 分数作为整图分数
  → 阈值化、映射回原图、计算面积和缺陷框
```

## 2. 图片预处理

在进入 PatchCore 前，公共预处理会：

- 将输入转换成 `uint8`。
- 转换为 BGR 三通道。
- 检测安全边缘区域。
- 将边缘区域设置为黑色，避免边框被识别为异常。

对应实现位于 [`internal/image_utils.cpp`](internal/image_utils.cpp)。

然后 PatchCore 自己再执行以下操作：

1. BGR 转 RGB。
2. 保持宽高比缩放，最长边缩放到 `224`。
3. 短边两侧补黑，形成 `224×224` 图像。
4. 使用 ImageNet 参数进行标准化：

```text
normalized = (pixel / 255 - mean) / std

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

5. 数据排列从 `HWC` 转换成 `NCHW`，最终输入形状为：

```text
1 × 3 × 224 × 224
```

## 3. Backbone 特征提取

模型元数据指定的 backbone 为：

```json
{
  "backbone.name": "wide_resnet50_2",
  "layers_to_extract_from": [
    "layer2",
    "layer3"
  ]
}
```

ONNX 推理一次会同时输出 WideResNet50-2 的 `layer2` 和 `layer3`。

对于常见的 `224×224` 输入，这两层通常对应：

```text
layer2: 1 × 512  × 28 × 28
layer3: 1 × 1024 × 14 × 14
```

代码没有把这些输出尺寸写死，而是在运行时读取 ONNX 输出的 `C/H/W`。

## 4. Patch 特征展开

元数据中的 patch 参数为：

```json
{
  "patchsize": 3,
  "patchstride": 1
}
```

程序对特征图上的每一个位置提取周围 `3×3` 的局部窗口。边缘位置使用零填充，因此特征图的空间尺寸保持不变：

```text
layer2 → 28×28 个 patch
layer3 → 14×14 个 patch
```

每个 patch 首先被展开成一维向量：

```text
patch 向量长度 = 通道数 × 3 × 3
```

因此通常为：

```text
layer2: 512  × 3 × 3 = 4608 维
layer3: 1024 × 3 × 3 = 9216 维
```

展开顺序为：

```text
通道0的3×3 → 通道1的3×3 → ... → 通道C-1的3×3
```

## 5. 每层特征压缩

元数据要求每一层先压缩到：

```json
{
  "pretrain_embed_dimension": 1024
}
```

但是 layer2 和 layer3 使用的方法不同。

### 5.1 layer2：自适应平均池化

配置为：

```json
{
  "mode": "adaptive",
  "kernel": null
}
```

程序把 `4608` 维向量自适应压缩成 `1024` 维：

```text
4608 → AdaptiveAvgPool1d → 1024
```

### 5.2 layer3：固定平均池化

配置为：

```json
{
  "mode": "avgpool",
  "kernel": 9
}
```

layer3 的 patch 向量是 `9216` 维，并且：

```text
9216 / 1024 = 9
```

所以程序每连续 9 个数求一次平均：

```text
9216 → AvgPool(kernel=9) → 1024
```

由于每个通道的 `3×3` 特征正好包含 9 个数，这一步实际上相当于对每个通道的 `3×3` 空间区域求平均。

因此 layer3 最终仍保留 1024 个通道维度，只是每个通道变成对应 `3×3` 邻域的平均响应。

## 6. 自适应平均池化

自适应平均池化和普通平均池化的主要区别是：

- 普通平均池化提前指定 kernel 大小。
- 自适应平均池化提前指定输出长度，由程序自动计算每个输出元素对应的输入区间。

当前实现使用以下公式计算池化区间：

```text
start(i) = floor(i × input_length / output_length)
end(i)   = ceil((i + 1) × input_length / output_length)

output[i] = mean(input[start(i) : end(i)])
```

例如，把长度 10 压缩到长度 4：

```text
输出0：平均 input[0:3]
输出1：平均 input[2:5]
输出2：平均 input[5:8]
输出3：平均 input[7:10]
```

当输入长度不能被输出长度整除时，相邻池化区间可能出现少量重叠。这与常见深度学习框架中的自适应平均池化逻辑一致。

代码使用前缀和优化区间求和：

```text
interval_sum = prefix[end] - prefix[start]
```

这样不需要为每个输出维度重新遍历整个输入区间。

在当前模型中，layer2 执行：

```text
4608 → 1024
```

平均每个输出维度覆盖的输入长度为：

```text
4608 / 1024 = 4.5
```

所以每个输出通常会平均约 5～6 个输入元素。

需要注意的是，这里池化的是展平后的 `C×3×3` 一维向量。因此部分自适应区间可能跨过两个通道的边界，并不是严格地对每个通道单独进行空间平均。

## 7. 多层特征的空间对齐

两层特征的空间尺寸不同：

```text
layer2: 28×28
layer3: 14×14
```

元数据指定参考 patch 网格为：

```json
{
  "ref_patch_shape": [28, 28]
}
```

因此代码使用双线性插值，将 layer3 的 patch 特征从：

```text
14×14×1024
```

插值为：

```text
28×28×1024
```

插值只发生在空间维度，1024 维特征深度保持不变。

最终两层都变成：

```text
layer2: 28×28×1024
layer3: 28×28×1024
```

## 8. 多层特征合并

在每一个 `28×28` 空间位置上，代码先拼接两个层的特征：

```text
[layer2 的 1024 维, layer3 的 1024 维]
                    ↓ concat
                   2048 维
```

聚合配置为：

```json
{
  "agg_pool": {
    "mode": "avgpool",
    "kernel": 2
  },
  "target_embed_dimension": 1024
}
```

程序对拼接后的 2048 维向量每连续两个数求平均：

```text
2048 → AvgPool(kernel=2) → 1024
```

最终得到：

```text
28 × 28 = 784 个 patch
每个 patch 是 1024 维特征

embeddings.shape = 784 × 1024
```

### 8.1 合并方式的注意事项

它不是直接计算：

```text
(layer2[i] + layer3[i]) / 2
```

代码的实际拼接顺序是：

```text
layer2[0...1023], layer3[0...1023]
```

然后对连续两个元素求平均。因此实际计算更接近：

```text
输出[0]   = mean(layer2[0], layer2[1])
输出[1]   = mean(layer2[2], layer2[3])
...
输出[511] = mean(layer2[1022], layer2[1023])

输出[512]  = mean(layer3[0], layer3[1])
...
输出[1023] = mean(layer3[1022], layer3[1023])
```

也就是说，它相当于：

```text
layer2 1024维 → 压缩成512维 ┐
                              ├→ 拼接成1024维
layer3 1024维 → 压缩成512维 ┘
```

而不是将两个层的对应维度直接求平均。

## 9. Faiss 特征匹配

特征合并后，共有 784 个待检测 patch 特征：

```text
784 × 1024
```

Faiss 索引中保存的是训练阶段从正常图片中提取的正常 patch 特征库。

当前索引的实际信息为：

```text
类型：IndexFlatL2
特征维度：1024
正常特征数量：4704
```

推理时调用：

```cpp
index->search(
    n,
    embeddings.data(),
    num_nn_,
    distances.data(),
    labels.data());
```

元数据配置为：

```json
{
  "anomaly_scorer_num_nn": 1
}
```

因此每个待检测 patch 只寻找一个最近的正常 patch。

## 10. Patch 异常得分

`IndexFlatL2` 返回平方 L2 距离：

```text
distance(x, m) = Σ (x[d] - m[d])²
```

这里没有进行开平方，因此严格来说是欧氏距离的平方。

每个 patch 的异常分数是最近 `K` 个正常特征距离的平均值：

```text
patch_score(i) =
    (distance(i, neighbor1) + ... + distance(i, neighborK)) / K
```

当前 `K=1`，所以：

```text
patch_score(i) = 当前 patch 与最近正常特征的平方 L2 距离
```

其含义为：

- 距离小：当前 patch 很像正常特征库中的 patch。
- 距离大：当前 patch 找不到相似的正常模式，更可能是异常。

## 11. 整图异常得分

784 个 patch 分数会被排列成：

```text
28×28 异常分数图
```

整张图片的 PatchCore 分数直接取所有 patch 分数的最大值：

```text
image_score = max(patch_scores)
```

因此只要有一个局部 patch 与正常特征库差异很大，整图分数就会升高。

当前实现没有使用：

- Sigmoid。
- Softmax。
- 距离归一化。
- 原始 PatchCore 部分实现中的加权重评分。
- 多个高分 patch 的平均值。

虽然类中定义了 `Sigmoid()`，但当前推理流程并没有调用它。

## 12. 缺陷后处理

后处理实现位于 [`internal/patchcore_postprocess.cpp`](internal/patchcore_postprocess.cpp)。

主要步骤为：

1. 使用 `area_threshold` 对 `28×28` patch 分数图进行二值化。
2. 使用最近邻插值放大到 `224×224`。
3. 去掉图片预处理时添加的黑色 padding。
4. 将异常 mask 映射回原图尺寸。
5. 统计异常面积比例、包围框和对比度。
6. 根据整图分数和异常面积比例判断是否存在缺陷。

最终判定条件为：

```text
has_defect =
    image_score >= score_threshold
    &&
    area_ratio >= mask_area_threshold
```

所以最终结果不只取决于最高 patch 分数，还要求异常区域面积达到指定比例。

## 13. 核心逻辑总结

### 13.1 特征合并

```text
layer2 patch → 自适应池化到1024维 ┐
                                     ├→ 空间对齐 → 拼接2048维 → 每2个平均 → 1024维
layer3 patch → 3×3通道内平均到1024维 ┘
```

### 13.2 异常评分

```text
局部特征
  → 与正常特征库做 1-NN 平方 L2 匹配
  → 每个 patch 的最近邻距离作为异常分数
  → 784 个 patch 中的最大距离作为整图分数
```

