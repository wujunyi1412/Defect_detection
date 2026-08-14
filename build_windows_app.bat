@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
set "NATIVE_BUILD=%PROJECT_ROOT%\build"
set "APP_SOURCE=%PROJECT_ROOT%\windows_corona_detection"
set "APP_BUILD=%APP_SOURCE%\build"
set "RUNTIME_DIR=%APP_SOURCE%\Runtime"
set "CONFIGURATION=Release"

pushd "%PROJECT_ROOT%" || goto :error

where cmake >nul 2>nul
if errorlevel 1 (
    echo [ERROR] CMake was not found in PATH.
    goto :error
)

where dotnet >nul 2>nul
if errorlevel 1 (
    echo [ERROR] dotnet was not found in PATH.
    goto :error
)

tasklist /FI "IMAGENAME eq CoronaDetection.exe" 2>nul | find /I "CoronaDetection.exe" >nul
if not errorlevel 1 (
    echo [ERROR] CoronaDetection.exe is running.
    echo         Close the application before building.
    goto :error
)

echo [1/6] Configuring native C++ project...
if exist "%NATIVE_BUILD%\CMakeCache.txt" (
    cmake -S "%PROJECT_ROOT%" -B "%NATIVE_BUILD%"
) else (
    cmake -S "%PROJECT_ROOT%" -B "%NATIVE_BUILD%" -G "Visual Studio 17 2022" -A x64
)
if errorlevel 1 goto :error

echo [2/6] Building native inference and evaluation DLLs...
cmake --build "%NATIVE_BUILD%" --config "%CONFIGURATION%" --target Corona_defect_detection evaluation_metrics evaluation_metrics_tests
if errorlevel 1 goto :error

set "NATIVE_DLL=%NATIVE_BUILD%\bin\%CONFIGURATION%\Corona_defect_detection.dll"
if not exist "%NATIVE_DLL%" (
    echo [ERROR] Native DLL was not generated:
    echo         %NATIVE_DLL%
    goto :error
)

echo [3/6] Running evaluation metrics tests...
ctest --test-dir "%NATIVE_BUILD%" -C "%CONFIGURATION%" -R evaluation_metrics_tests --output-on-failure
if errorlevel 1 goto :error

echo [4/6] Updating native runtime DLLs...
for %%F in (
    Corona_defect_detection.dll
    evaluation_metrics.dll
    onnxruntime.dll
    halcon.dll
    halconcpp.dll
    faiss.dll
    openblas.dll
    opencv_core453.dll
    opencv_imgproc453.dll
    opencv_imgcodecs453.dll
) do (
    if not exist "%NATIVE_BUILD%\bin\%CONFIGURATION%\%%F" (
        echo [ERROR] Required native DLL was not generated: %%F
        goto :error
    )
    copy /Y "%NATIVE_BUILD%\bin\%CONFIGURATION%\%%F" "%RUNTIME_DIR%\%%F" >nul
    if errorlevel 1 goto :error
)

echo [5/6] Configuring the WPF CMake project...
if exist "%APP_BUILD%\CMakeCache.txt" (
    cmake -S "%APP_SOURCE%" -B "%APP_BUILD%"
) else (
    cmake -S "%APP_SOURCE%" -B "%APP_BUILD%" -G "Visual Studio 17 2022" -A x64
)
if errorlevel 1 goto :error

echo [6/6] Publishing CoronaDetection.exe...
cmake --build "%APP_BUILD%" --config "%CONFIGURATION%" --target CoronaDetection
if errorlevel 1 goto :error

set "APP_EXE=%APP_BUILD%\bin\%CONFIGURATION%\CoronaDetection.exe"
if not exist "%APP_EXE%" (
    echo [ERROR] WPF executable was not generated:
    echo         %APP_EXE%
    goto :error
)

echo.
echo [SUCCESS] Build completed.
echo [OUTPUT]  %APP_EXE%
popd
exit /b 0

:error
echo.
echo [FAILED] Build did not complete.
popd
exit /b 1
