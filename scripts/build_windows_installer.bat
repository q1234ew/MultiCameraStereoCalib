@echo off
REM 1) PyInstaller 生成分发目录 dist\MultiCamera
REM 2) Inno Setup 6 生成安装程序到 installer_output\（若未安装 Inno，则仅生成 ZIP 便携包）

setlocal
cd /d "%~dp0.."

call scripts\build_windows.bat %*
if errorlevel 1 exit /b 1

if not exist "installer_output" mkdir installer_output

set ISCC=
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"

if not "%ISCC%"=="" (
    echo.
    echo 使用 Inno Setup 生成安装包...
    "%ISCC%" packaging\windows\MultiCamera.iss
    if errorlevel 1 exit /b 1
    echo.
    echo 安装包: installer_output\MultiCamera_Setup_0.1.0_x64.exe
    goto :eof
)

echo.
echo [未检测到 Inno Setup 6] 已跳过 .exe 安装包。
echo 请安装：https://jrsoftware.org/isdl.php  后重新运行本脚本。
echo 正在生成 ZIP 便携包作为替代...
powershell -NoProfile -Command ^
  "Compress-Archive -Path (Join-Path '%CD%' 'dist\MultiCamera\*') -DestinationPath (Join-Path '%CD%' 'installer_output\MultiCamera_0.1.0_portable_x64.zip') -Force"
if errorlevel 1 (
    echo ZIP 打包失败，请手动压缩 dist\MultiCamera 文件夹。
    exit /b 2
)
echo.
echo 便携 ZIP: installer_output\MultiCamera_0.1.0_portable_x64.zip
