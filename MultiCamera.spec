# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 配置：在 Windows 上生成 dist/MultiCamera/ 目录（含 MultiCamera.exe）。

用法（在仓库根目录、已安装本项目与构建依赖时）：
    pyinstaller --noconfirm MultiCamera.spec

建议在 Windows 64 位环境、Python 3.10–3.12 下执行；体积较大（含 Open3D / Qt / OpenCV）。
"""
import pathlib

from PyInstaller.building.build_main import Analysis, COLLECT, EXE, PYZ
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs, collect_submodules

# PyInstaller sets SPECPATH to the directory containing this .spec file.
ROOT = pathlib.Path(SPECPATH).resolve()
SRC = ROOT / "src"
ENTRY = SRC / "multicamera" / "__main__.py"

datas: list = []
binaries: list = []
hiddenimports: list = collect_submodules("multicamera")

_assets = ROOT / "assets"
if _assets.is_dir():
    datas.append((str(_assets), "assets"))

for pkg in ("PySide6", "open3d"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

try:
    datas += collect_data_files("cv2")
except Exception:
    pass
try:
    binaries += collect_dynamic_libs("cv2")
except Exception:
    pass
try:
    datas += collect_data_files("scipy")
except Exception:
    pass

block_cipher = None

a = Analysis(
    [str(ENTRY)],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "torch",
        "open3d.ml.torch",
        "open3d.ml.tf",
        "PySide6.scripts.deploy_lib",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MultiCamera",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MultiCamera",
)
