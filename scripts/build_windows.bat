@echo off
REM 在 Windows 上于仓库根目录生成分发包 dist\MultiCamera\
REM 需要已安装 Python 3.10+（建议 64 位），且网络可访问 pip。
REM
REM 默认生成轻量包：不包含 Open3D 点云运行库，体积更小、打包更快。
REM 需要点云完整包时运行：
REM   scripts\build_windows.bat --pointcloud

cd /d "%~dp0.."

set "WITH_POINTCLOUD=0"
if /I "%~1"=="--pointcloud" set "WITH_POINTCLOUD=1"
if /I "%~1"=="/pointcloud" set "WITH_POINTCLOUD=1"

if not exist ".venv\Scripts\activate.bat" (
    python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install -U pip
if "%WITH_POINTCLOUD%"=="1" (
    echo 构建点云完整包，将安装并打包 Open3D...
    pip install -e ".[build-win,pointcloud]"
    set "MULTICAMERA_WITH_OPEN3D=1"
) else (
    echo 构建轻量包，不打包 Open3D；点云功能在此包中不可用。
    pip install -e ".[build-win]"
    set "MULTICAMERA_WITH_OPEN3D="
)

pyinstaller --noconfirm MultiCamera.spec
if errorlevel 1 exit /b 1

echo.
echo 输出目录: dist\MultiCamera\
echo 将整目录复制到目标机器，运行 MultiCamera.exe；sessions 会写在同目录下的 sessions 文件夹。
if "%WITH_POINTCLOUD%"=="1" (
    echo 当前包: 点云完整包（包含 Open3D）
) else (
    echo 当前包: 轻量包（不包含 Open3D，标定/导出可用）
)
echo 可选：在 exe 同级放置 assets\logo.png 作为窗口图标。
echo 需要 Windows 安装包时，请运行 scripts\build_windows_installer.bat（需额外安装 Inno Setup 6）。
