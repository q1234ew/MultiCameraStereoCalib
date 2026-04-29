"""运行时路径（开发与 PyInstaller 单文件/单目录打包）。"""

from __future__ import annotations

import sys
from pathlib import Path


def _frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _meipass() -> Path | None:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return None


def exe_dir() -> Path:
    """可执行文件所在目录（打包后为包含 .exe 的目录）。"""
    if _frozen():
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def sessions_dir() -> Path:
    """标定会话目录：打包后默认在 exe 同级的 sessions，便于便携部署。"""
    d = exe_dir() / "sessions"
    if _frozen():
        d.mkdir(parents=True, exist_ok=True)
    return d


def logo_png_path() -> Path | None:
    """应用图标；优先 exe 旁 assets，其次打包内嵌资源。"""
    candidates: list[Path] = []
    if _frozen():
        candidates.append(exe_dir() / "assets" / "logo.png")
        mp = _meipass()
        if mp is not None:
            candidates.append(mp / "assets" / "logo.png")
    else:
        here = Path(__file__).resolve().parent
        repo = here.parent.parent
        candidates.append(repo / "assets" / "logo.png")
        candidates.append(here / "assets" / "logo.png")
    for p in candidates:
        if p.is_file():
            return p
    return None
