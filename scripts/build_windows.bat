@echo off
REM 在 Windows 上于仓库根目录生成分发包 dist\MultiCamera\
REM 需要已安装 Python 3.10+（建议 64 位），且网络可访问 pip。

cd /d "%~dp0.."

if not exist ".venv\Scripts\activate.bat" (
    python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install -U pip
pip install -e ".[build-win]"

pyinstaller --noconfirm MultiCamera.spec
if errorlevel 1 exit /b 1

echo.
echo 输出目录: dist\MultiCamera\
echo 将整目录复制到目标机器，运行 MultiCamera.exe；sessions 会写在同目录下的 sessions 文件夹。
echo 可选：在 exe 同级放置 assets\logo.png 作为窗口图标。
echo 需要 Windows 安装包时，请运行 scripts\build_windows_installer.bat（需额外安装 Inno Setup 6）。
