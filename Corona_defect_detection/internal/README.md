# internal/ 模块说明

本目录包含 Corona 缺陷检测的**内部后处理逻辑**，负责将 YOLO 和 PatchCore 两个模型的推理结果整合为最终输出。

---

## 文件职责概览

```
inference_dll.cpp (主控)
    │
    ├── image_utils        ← 图像预处理 & 统计工具函数
    ├── yolo_postprocess   ← YOLO 检测结果 → YoloDerived
    ├── aa_yolo_filter    ← YOLO 最终结果的 AA 区域/形状过滤
    ├── iqt_yolo_filter   ← YOLO 最终结果的 IQT 独立区域/形状过滤
    ├── patchcore_postprocess ← PatchCore 异常图 → PatchCoreDerived
    │
    ├── dark_defect_refiner ← 暗缺陷几何精修 (传统CV)
    │
    └── result_builder     ← 融合 + 过滤 → InferenceResult
```

---

## 各文件详细说明

### 1. image_utils — 图像工具函数

| 函数 | 作用 |
|---|---|
| `ProcessTIF32ForPatchcore` | TIF32 图像转 BGR，并遮盖边界无效区域 |
| `ProcessForYolo` | TIF32 图像转 BGR（YOLO 用，无边界遮盖） |
| `ConvertGrayToU8Normalized` | 灰度图 min-max 线性拉伸到 0-255（CV_8U 直接返回） |
| `Median` | 中位数（用 nth_element，不完整排序） |
| `Percentile` | 百分位数（线性插值） |
| `CalculateContrastRatio` | 缺陷区域对比度：`defect_mean / bg_mean`，bg 通过形态学膨胀环获取 |

### 2. yolo_postprocess — YOLO 后处理

**输入**：`YOLO::Detection` 列表 + 灰度图  
**输出**：`YoloDerived`（含 score、是否检测到缺陷、DetectionResult 列表）

核心逻辑 `AnalyzeYolo()`：
1. 遍历 YOLO 检测框，按 class_id 映射 8 类：Glue_overflow、Decolorization、Stain、Stripes、BrightStripes、Bright_clusters、Line_artifacts、LineArtifacts
2. 从分割 mask 计算缺陷面积
3. 调用 `CalculateContrastRatio` 计算对比度
4. 0/1/2/6 类使用暗缺陷极性，4/5/7 类使用亮缺陷极性，3 类使用自动极性

### 3. patchcore_postprocess — PatchCore 后处理

**输入**：`PatchCoreResult`（异常分数图） + 图像尺寸 + 阈值  
**输出**：`PatchCoreDerived`

核心逻辑 `AnalyzePatchCore()`：
1. **二值化**：用 `area_threshold` 生成 cropped binary mask
2. **缺陷判定**：`score ≥ score_threshold` 且 `area_ratio ≥ mask_area_threshold` → `has_defect = true`
3. **对比度**：将 mask 缩放到原图尺寸，调用 `CalculateContrastRatio`
4. **十字线检测**（`Detect3x3GridComponents`）：检测异常区域是否形成 3×3 网格（十字线特征）：
   - 对连通域按面积取 Top-9
   - 用 K-Means 1D 分别对 x、y 坐标聚类为 3 组
   - 验证 9 个分量是否正好填入 3×3 网格（行列各 3 个，间距均匀）

### 4. dark_defect_refiner — 暗缺陷几何精修

**输入**：灰度图 + YOLO 检测框（`DetectionResult`）  
**输出**：`bool`（是否成功精修），直接修改传入的 `detail`

核心逻辑 `RefineDarkDefectGeometryOnly()`：
1. **ROI 裁剪**：根据 YOLO 框裁剪 ROI，并 clamp 到图像边界
2. **小 ROI 保护**：宽高 < 4 的 ROI 直接用原框
3. **背景估计**：
   - 小框（≤30×30）：`medianBlur(3)` 估计背景
   - 大框：`GaussianBlur` 估计背景，并计算边界区域均值作为参考
4. **暗响应**：`dark_response = background - original`，亮响应 = 暗缺陷
5. **双阈值分割**（seed/grow）：
   - seed_mask：严格阈值找种子点（灰度低 + 暗响应高）
   - grow_mask：宽松阈值扩展区域
6. **形态学处理**：开运算去噪 + 闭运算填孔
7. **连通域评分**：综合考虑响应均值、灰度反转值、填充率、种子重叠度，选出最佳连通域
8. **安全回退**：若精修框过小（< 原框 1/6）或变化极小（> 98%），放弃精修

### 5. result_builder — 结果融合 & 过滤

提供两个层级的输出组合：

#### `ComposeOutput`（简单模式，无过滤）
- PatchCore 有十字线 → 输出 9 个 Abnormal 分量
- 仅有 YOLO → 直接输出 YOLO 结果
- 仅有 PatchCore（无十字线）→ 输出单个 Abnormal
- 两者都有 → 合并输出

#### `ComposeOutputWithDefectFilter`（完整模式，带类别过滤）
1. **YOLO 过滤**：通过“类别名 → `CategoryFilterConfig`”映射逐类别过滤；`use_traditional_measure=1` 的类别调用传统图像算法精修，当前仅 Stain 开启
2. **PatchCore 过滤**：按 `AbnormalFilterConfig` 过滤，十字线模式逐分量过滤。`contrast_threshold > 0` 时，暗类（0/1/2/6）保留 `contrast < threshold`，其余 YOLO 类和 PatchCore Abnormal 保留 `contrast >= threshold`；配置为 `0` 时关闭对比度过滤
3. **AA 后过滤**：标准 YOLO 过滤后，`aa_yolo_filter` 包含两条独立规则：“中心 Y + 最小宽高比”和“左上角 X/Y + 宽/高范围”。两条规则分别由 `[AA]` 开关控制，任一启用规则命中就移除该 YOLO 结果
4. **PatchCore/YOLO IoU 过滤**：对每条已通过过滤的 PatchCore `Abnormal`，计算它与全部最终 YOLO 框的 IoU；任一 IoU 严格大于 `post.patchcore_yolo_iou_threshold` 时移除该 PatchCore 明细。没有 YOLO 时始终保留，阈值设为 `1` 可关闭此过滤
5. **IQT 后过滤**：在 PatchCore/YOLO IoU 过滤之后，按独立的 `[IQT]` 配置执行“中心 Y +（最小宽高比或最大宽度）”规则；`max_width=0` 时关闭宽度条件，参数不与 `[AA]` 共用
6. **OK/NG 判定**：任一有效检测 → `"NG"`，否则 `"OK"`

---

## 数据流总结

```
图像输入
  ├─→ ProcessForYolo() → YOLO 推理 → AnalyzeYolo() → YoloDerived
  │                                              ↓
  │                              RefineDarkDefectGeometryOnly() (可选)
  │                                              ↓
  └─→ ProcessTIF32ForPatchcore() → PatchCore 推理 → AnalyzePatchCore() → PatchCoreDerived
                                                                          ↓
                                              ComposeOutput / ComposeOutputWithDefectFilter
                                                                          ↓
                                                                   InferenceResult
```
