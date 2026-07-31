@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
set "APP_PROJECT=%PROJECT_ROOT%\windows_corona_detection\CoronaDetection.App\CoronaDetection.App.csproj"
set "PACKAGE_ROOT=%PROJECT_ROOT%\windows_corona_detection\package"
set "APP_VERSION="
for /f "tokens=2,3 delims=<>" %%A in ('findstr /C:"<Version>" "%APP_PROJECT%"') do set "APP_VERSION=%%B"
if not defined APP_VERSION set "APP_VERSION=1.0.0"
set "PACKAGE_DIR=%PACKAGE_ROOT%\CoronaDetection_%APP_VERSION%_win-x64"

pushd "%PROJECT_ROOT%" || goto :error

call "%PROJECT_ROOT%\build_windows_app.bat"
if errorlevel 1 goto :error

echo.
echo [1/3] Preparing portable package directory...
if exist "%PACKAGE_DIR%" cmake -E remove_directory "%PACKAGE_DIR%"
if errorlevel 1 goto :error
cmake -E make_directory "%PACKAGE_DIR%"
if errorlevel 1 goto :error

echo [2/3] Publishing self-contained .NET 9 WPF application...
dotnet publish "%APP_PROJECT%" ^
    --configuration Release ^
    --runtime win-x64 ^
    --self-contained true ^
    --output "%PACKAGE_DIR%" ^
    --nologo ^
    -p:PublishSingleFile=false ^
    -p:PublishReadyToRun=false ^
    -p:DebugType=None ^
    -p:DebugSymbols=false
if errorlevel 1 goto :error

echo [3/3] Adding required Visual C++ runtime libraries...
for %%F in (
    msvcp140.dll
    msvcp140_1.dll
    vcruntime140.dll
    vcruntime140_1.dll
    concrt140.dll
    vcomp140.dll
) do (
    if exist "%SystemRoot%\System32\%%F" (
        copy /Y "%SystemRoot%\System32\%%F" "%PACKAGE_DIR%\%%F" >nul
    ) else (
        echo [WARNING] Visual C++ runtime library was not found: %%F
    )
)

if exist "%PACKAGE_DIR%\CoronaDetection.pdb" del /Q "%PACKAGE_DIR%\CoronaDetection.pdb"
if exist "%PACKAGE_DIR%\inspection.log" del /Q "%PACKAGE_DIR%\inspection.log"
for %%F in (
    createdump.exe
    Microsoft.DiaSymReader.Native.amd64.dll
    mscordaccore.dll
    mscordbi.dll
) do (
    if exist "%PACKAGE_DIR%\%%F" del /Q "%PACKAGE_DIR%\%%F"
)
for %%F in ("%PACKAGE_DIR%\mscordaccore_*.dll") do (
    if exist "%%~fF" del /Q "%%~fF"
)

if not exist "%PACKAGE_DIR%\CoronaDetection.exe" goto :error
if not exist "%PACKAGE_DIR%\halcon.dll" goto :error
if not exist "%PACKAGE_DIR%\onnx_model\yolov8_seg_0720.onnx" goto :error
if not exist "%PACKAGE_DIR%\onnx_model\patchcore_backbone.onnx" goto :error
if not exist "%PACKAGE_DIR%\onnx_model\nnscorer_search_index.faiss" goto :error
if not exist "%PACKAGE_DIR%\onnx_model\model_metadata.json" goto :error

echo.
echo [SUCCESS] Portable package completed.
echo [OUTPUT]  %PACKAGE_DIR%
echo [NOTE]    A valid HALCON runtime license is still required.
popd
exit /b 0

:error
echo.
echo [FAILED] Portable package was not created.
popd
exit /b 1
