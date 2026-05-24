"""Background workers for image import and board detection."""

from __future__ import annotations

import glob
import os
import re
import traceback
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal

from ...board.detector import CharucoDetector
from ...calibration.planar import PlanarPatternDetector


class StereoImageImportWorker(QThread):
    """Load stereo image pairs and detect ChArUco corners off the UI thread."""

    finished_ok = Signal(object)
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, folder: str, board, parent=None):
        super().__init__(parent)
        self.folder = folder
        self.board = board

    def run(self):
        try:
            pairs = _discover_stereo_pairs(self.folder)
            if not pairs:
                self.failed.emit(
                    "未找到离线左右图。支持 *_left/*_right、left_*/right_*、*_L/*_R、"
                    "left/right 子目录；可直接选择父目录。"
                )
                return

            self.log.emit(f"发现 {len(pairs)} 对候选离线图片")
            detector = CharucoDetector(self.board)
            imported = []
            skipped = 0
            for idx, lp, rp in pairs:
                img_l = cv2.imread(str(lp))
                img_r = cv2.imread(str(rp))
                if img_l is None or img_r is None:
                    skipped += 1
                    continue
                det_l = detector.detect(img_l)
                det_r = detector.detect(img_r)
                imported.append((idx, img_l, img_r, det_l, det_r))

            if not imported:
                self.failed.emit("未能加载任何有效图片对，请检查图片格式或文件是否损坏")
                return
            if skipped:
                self.log.emit(f"跳过 {skipped} 对无法读取的图片")
            self.finished_ok.emit(imported)
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc(limit=3)}")


class AuxImageImportWorker(QThread):
    """Load RGB_L/AUX image pairs and detect planar board points off the UI thread."""

    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, folder: str, board_config, parent=None):
        super().__init__(parent)
        self.folder = folder
        self.board_config = board_config

    def run(self):
        try:
            rgb_files = _indexed_files(self.folder, "rgb_left")
            aux_files = _indexed_files(self.folder, "aux")
            keys = sorted(set(rgb_files) & set(aux_files))
            if not keys:
                self.failed.emit("未找到 rgb_left_XXXX.* 与 aux_XXXX.* 配对图像")
                return

            detector = PlanarPatternDetector(self.board_config)
            imported = []
            for key in keys:
                rgb = cv2.imread(rgb_files[key])
                aux = cv2.imread(aux_files[key])
                if rgb is None or aux is None:
                    continue
                rgb_det = detector.detect(rgb)
                aux_det = detector.detect(aux)
                imported.append((key, rgb, aux, rgb_det, aux_det))

            if not imported:
                self.failed.emit("未能加载任何有效多模态图片对")
                return
            self.finished_ok.emit(imported)
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc(limit=3)}")


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _discover_stereo_pairs(folder: str) -> list[tuple[str, Path, Path]]:
    root = Path(folder)
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
    by_norm = {_norm_path(p): p for p in files}
    pairs: list[tuple[str, Path, Path]] = []
    seen: set[tuple[str, str]] = set()

    for left in files:
        right = _right_path_for_left(left, by_norm)
        if right is None:
            continue
        key = (_norm_path(left), _norm_path(right))
        if key in seen:
            continue
        seen.add(key)
        idx = _pair_index(root, left)
        pairs.append((idx, left, right))
    pairs.sort(key=lambda item: _natural_key(item[0]))
    return pairs


def _right_path_for_left(left: Path, by_norm: dict[str, Path]) -> Path | None:
    parent = left.parent
    stem = left.stem
    suffix = left.suffix
    lower_stem = stem.lower()
    candidates: list[Path] = []

    replacements = (
        ("_left", "_right"),
        ("-left", "-right"),
        (".left", ".right"),
        ("_l", "_r"),
        ("-l", "-r"),
        (".l", ".r"),
    )
    for ltoken, rtoken in replacements:
        if lower_stem.endswith(ltoken):
            candidates.append(parent / f"{stem[: -len(ltoken)]}{rtoken}{suffix}")

    prefixes = (
        ("left_", "right_"),
        ("left-", "right-"),
        ("l_", "r_"),
        ("l-", "r-"),
    )
    for ltoken, rtoken in prefixes:
        if lower_stem.startswith(ltoken):
            candidates.append(parent / f"{rtoken}{stem[len(ltoken):]}{suffix}")

    parts = list(left.parts)
    for idx, part in enumerate(parts[:-1]):
        lowered = part.lower()
        if lowered in {"left", "lefts", "l", "cam_l", "camera_l"}:
            for replacement in ("right", "rights", "r", "cam_r", "camera_r"):
                cand_parts = parts.copy()
                cand_parts[idx] = replacement
                candidates.append(Path(*cand_parts))

    for candidate in candidates:
        found = by_norm.get(_norm_path(candidate))
        if found is not None:
            return found
    return None


def _pair_index(root: Path, left: Path) -> str:
    try:
        rel = left.relative_to(root)
    except ValueError:
        rel = left.name
    stem = Path(rel).stem
    for token in ("_left", "-left", ".left", "_L", "-L", ".L", "_l", "-l", ".l"):
        if stem.endswith(token):
            return stem[: -len(token)]
    for token in ("left_", "left-", "L_", "L-", "l_", "l-"):
        if stem.startswith(token):
            return stem[len(token):]
    return str(Path(rel).with_suffix(""))


def _norm_path(path: Path) -> str:
    return str(path).replace("\\", "/").lower()


def _natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _indexed_files(folder: str, prefix: str) -> dict[str, str]:
    files: dict[str, str] = {}
    for ext in ("png", "jpg", "jpeg", "bmp", "tif", "tiff"):
        for path in glob.glob(os.path.join(folder, f"{prefix}_*.{ext}")):
            stem = os.path.splitext(os.path.basename(path))[0]
            files[stem[len(prefix) + 1 :]] = path
    return files
