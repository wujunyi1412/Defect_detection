# Corona 瑕疵检测工作站

Windows x64 桌面程序，C# WPF/XAML 界面通过 P/Invoke 调用
`Corona_defect_detection.dll` 的 C API。

## 构建

项目以 `CMakeLists.txt` 为唯一构建入口。可以使用 Visual Studio 2022
“打开本地文件夹”，选择 `vs2022-x64` 预设；也可以在项目根目录运行：

```powershell
cmake --preset vs2022-x64
cmake --build --preset vs2022-release
```

输出程序位于：

`build\bin\Release\CoronaDetection.exe`

默认生成 Windows x64 框架依赖版本，目标电脑需要安装 .NET 9 Desktop Runtime。
若需自包含版本，联网配置 CMake 时加入 `-DCORONA_SELF_CONTAINED=ON`。

## 使用提示

- 修改配置后，开始检测会自动保存配置并重启模型。
- 结果目录包含标注 PNG、`summary.csv` 和 `details.csv`。
- 批量任务可只扫描一级目录或递归扫描，并可选择是否保留原目录层级。
- 模型和原生 DLL 位于 `Runtime`，构建时自动复制到 exe 同目录。
- 当前原生 C API 使用 `char*` 路径；程序会对含非 ASCII 字符的输入图片自动使用临时英文路径。

## 原生接口自检

构建后可用一张测试图片验证 DLL、模型和 C# 结构体封送：

```powershell
.\build\bin\Release\CoronaDetection.exe --self-test D:\test.png
```
