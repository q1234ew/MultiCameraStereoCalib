"""Background workers for image import and board detection."""

from __future__ import annotations

import glob
import os
import traceback

import cv2
from PySide6.QtCore import QThread, Signal

from ...board.detector import CharucoDetector
from ...calibration.planar import PlanarPatternDetector


class StereoImageImportWorker(QThread):
    """Load stereo image pairs and detect ChArUco corners off the UI thread."""

    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, folder: str, board, parent=None):
        super().__init__(parent)
        self.folder = folder
        self.board = board

    def run(self):
        try:
            lefts = []
            for ext in ("png", "jpg", "jpeg", "bmp", "tif", "tiff"):
                lefts.extend(glob.glob(os.path.join(self.folder, f"*_left.{ext}")))
            lefts = sorted(lefts)
            if not lefts:
                self.failed.emit("未找到 *_left.* 文件")
                return

            detector = CharucoDetector(self.board)
            imported = []
            for lp in lefts:
                rp = _right_path_for_left(lp)
                if rp is None:
                    continue
                img_l = cv2.imread(lp)
                img_r = cv2.imread(rp)
                if img_l is None or img_r is None:
                    continue
                det_l = detector.detect(img_l)
                det_r = detector.detect(img_r)
                idx = os.path.basename(lp).split("_")[0]
                imported.append((idx, img_l, img_r, det_l, det_r))

            if not imported:
                self.failed.emit("未能加载任何有效图片对")
                return
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


def _right_path_for_left(left_path: str) -> str | None:
    root, ext = os.path.splitext(left_path)
    if not root.endswith("_left"):
        return None
    candidate = root[: -len("_left")] + "_right" + ext
    return candidate if os.path.isfile(candidate) else None


def _indexed_files(folder: str, prefix: str) -> dict[str, str]:
    files: dict[str, str] = {}
    for ext in ("png", "jpg", "jpeg", "bmp", "tif", "tiff"):
        for path in glob.glob(os.path.join(folder, f"{prefix}_*.{ext}")):
            stem = os.path.splitext(os.path.basename(path))[0]
            files[stem[len(prefix) + 1 :]] = path
    return files
