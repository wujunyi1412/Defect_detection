# C++ DLL 调用示例

`use_examples` 展示如何在 C++ 中初始化一次 `InspectionEngine`，并处理单个文件或批量处理文件夹。

支持的输入：

- PNG：`.png`
- TIFF：`.tif`、`.tiff`
- BIN：无文件头的连续 `float32` 灰度数据，元素数量必须等于 `width * height`

每次运行会在输出目录生成：

- `detect_images/`：带检测框、掩膜和结果文字的 PNG 图片
- `summary.csv`：每个文件的类别数量、推理耗时、保存耗时、总耗时和错误信息

## 构建

```powershell
cmake -S Use_DLL -B Use_DLL/build
cmake --build Use_DLL/build --config Release
```

构建后，配置、模型及运行时 DLL 会复制到：

```text
Use_DLL/build/bin/Release
```

## 单文件

PNG：

```powershell
Use_DLL\build\bin\Release\use_examples.exe `
  --mode single `
  --input E:\images\test.png `
  --input-type png `
  --config Use_DLL\build\bin\Release\config.ini
```

TIFF：

```powershell
Use_DLL\build\bin\Release\use_examples.exe `
  --mode single `
  --input E:\images\test.tiff `
  --input-type tiff `
  --config Use_DLL\build\bin\Release\config.ini
```

BIN：

```powershell
Use_DLL\build\bin\Release\use_examples.exe `
  --mode single `
  --input E:\images\test.bin `
  --input-type bin `
  --width 1000 `
  --height 800 `
  --config Use_DLL\build\bin\Release\config.ini
```

## 批量文件夹

只处理一种格式：

```powershell
Use_DLL\build\bin\Release\use_examples.exe `
  --mode batch `
  --input E:\images `
  --input-type tiff `
  --output-dir E:\output `
  --recurse `
  --config Use_DLL\build\bin\Release\config.ini
```

同时查找 PNG、TIFF 和 BIN：

```powershell
Use_DLL\build\bin\Release\use_examples.exe `
  --mode batch `
  --input E:\images `
  --input-type auto `
  --width 1000 `
  --height 800 `
  --output-dir E:\output `
  --recurse `
  --config Use_DLL\build\bin\Release\config.ini
```

兼容 C# 示例的 `--input-dir` 写法；它会自动启用批量模式：

```powershell
Use_DLL\build\bin\Release\use_examples.exe `
  --input-dir E:\images `
  --input-type bin `
  --width 1000 `
  --height 800
```

使用 `--repeat N` 重复处理，使用 `--no-save-images` 关闭检测图片保存。
