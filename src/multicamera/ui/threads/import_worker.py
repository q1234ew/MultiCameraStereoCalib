"""Background workers for image import and board detection."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import traceback
from copy import deepcopy
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from ...board.detector import CharucoDetector
from ...calibration.assessment import DataAssessor
from ...calibration.aux_capture_quality import assess_aux_frame, quality_reason_text
from ...calibration.planar import PatternBoardConfig, PlanarPatternDetector
from ...calibration.offline_aux import offline_pair_review


class StereoImageImportWorker(QThread):
    """Load stereo image pairs and detect ChArUco corners off the UI thread."""

    finished_ok = Signal(object)
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, folder: str, board, parent=None):
        super().__init__(parent)
        self.folder = folder
        self.board = board
        self.frame_qualities = {}
        self.source_paths = {}

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
            paired_files = {path for _, left, right in pairs for path in (left, right)}
            unused = [path for path in Path(self.folder).rglob('*')
                      if path.is_file() and path.suffix.lower() in _IMAGE_EXTS and path not in paired_files]
            if unused:
                self.log.emit(f"有 {len(unused)} 张图片未参与配对，请检查左右命名或缺失文件")
            detector = CharucoDetector(self.board)
            assessor = DataAssessor()
            self.frame_qualities = {}
            self.source_paths = {}
            imported = []
            skipped = 0
            for idx, lp, rp in pairs:
                review_path = lp.with_name(lp.stem.replace('left_', 'quality_', 1) + '.json')
                if lp.name.startswith('left_') and review_path.exists():
                    review = json.loads(review_path.read_text(encoding='utf-8'))
                    if review.get('board') is not None and review['board'] != self.board.to_dict():
                        skipped += 1
                        self.log.emit(f'[{idx}] 历史 RGB 标定板与当前会话不一致，未纳入；原图保留')
                        continue
                    if review.get('state') != '可采纳':
                        skipped += 1
                        self.log.emit(f'[{idx}] {review.get("state", "待复核")}：未纳入导入标定数据；原图保留')
                        continue
                img_l = cv2.imdecode(np.frombuffer(lp.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                img_r = cv2.imdecode(np.frombuffer(rp.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                if img_l is None or img_r is None:
                    skipped += 1
                    self.log.emit(f'[{idx}] 图片无法读取：{lp.name} / {rp.name}，请检查文件损坏')
                    continue
                det_l = detector.detect(img_l)
                det_r = detector.detect(img_r)
                self.frame_qualities[str(idx)] = (
                    assessor.assess_frame(img_l, det_l.num_corners, len(imported)),
                    assessor.assess_frame(img_r, det_r.num_corners, len(imported)),
                )
                imported.append((idx, img_l, img_r, det_l, det_r))
                self.source_paths[str(idx)] = (str(lp.resolve()), str(rp.resolve()))

            if not imported:
                self.failed.emit("未能加载任何有效图片对，请检查图片格式或文件是否损坏")
                return
            if skipped:
                self.log.emit(f"跳过 {skipped} 对未通过采图验收或无法读取的图片")
            self.finished_ok.emit(imported)
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc(limit=3)}")


class AuxImageImportWorker(QThread):
    """Load RGB_L/AUX image pairs and detect planar board points off the UI thread."""

    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, folder: str, board_config, parent=None, *, modality='ir',
                 offline_recheck=False, rgb_intrinsics=None):
        super().__init__(parent)
        self.folder = folder
        self.board_config = board_config
        self.modality = modality
        self.frame_reviews = {}
        self.offline_recheck = offline_recheck
        self.offline_reviews = {}
        self.import_context = None
        self.read_errors = []
        self.rgb_intrinsics = deepcopy(rgb_intrinsics)
        self.geometry_messages = []

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
                snapshots = []
                if self.offline_recheck:
                    images = []
                    for source in (rgb_files[key], aux_files[key]):
                        path = Path(source).resolve()
                        payload = path.read_bytes()
                        images.append(cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR))
                        snapshots.append(dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest()))
                    rgb, aux = images
                else:
                    rgb = cv2.imread(rgb_files[key])
                    aux = cv2.imread(aux_files[key])
                if rgb is None or aux is None:
                    self.read_errors.append(f'[{key}] 图片无法读取：{rgb_files[key]} / {aux_files[key]}')
                    continue
                rgb_det = (detector.detect_offline(rgb) if self.offline_recheck
                           else detector.detect(rgb))
                if self.offline_recheck:
                    rgb_det = detector.recheck_rgb_geometry(rgb, rgb_det, self.rgb_intrinsics)
                    check = rgb_det.detection_info.get('geometry_recheck')
                    if check:
                        status = {'recovered': '原尺寸SB重检恢复', 'unresolved': '重检未恢复',
                                  'skipped': '未执行重检'}[check['status']]
                        self.geometry_messages.append(
                            f'[{key}] RGB原图几何检查：{status}；'
                            f'原检测RMS={check.get("original_geometry_rms_px")} px，'
                            f'原尺寸SB RMS={check.get("sb_geometry_rms_px")} px；'
                            + ('重检已恢复，仍须逐对确认' if check['status'] == 'recovered' else
                               check.get('reason', '未恢复，请复核或补拍此图像')))
                aux_det = (detector.detect_offline(aux, modality=self.modality)
                           if self.offline_recheck else detector.detect(aux))
                qualities = [assess_aux_frame(image, det, self.board_config,
                    modality='rgb' if i == 0 or self.modality == 'acoustic_rgb' else 'ir')
                    for i, (image, det) in enumerate(((rgb, rgb_det), (aux, aux_det)))]
                reasons = [f'{name}：' + quality_reason_text(reason)
                           for name, quality in zip(('RGB', 'IR/辅助'), qualities)
                           for reason in quality.reasons]
                review_path = Path(rgb_files[key]).with_name(f'quality_{key}.json')
                prior = None
                board_reasons = []
                if review_path.exists():
                    prior = json.loads(review_path.read_text(encoding='utf-8'))
                    if prior.get('board') is not None:
                        try:
                            recorded_board = PatternBoardConfig.from_dict(prior['board'])
                            if recorded_board != self.board_config:
                                board_reasons.append('历史标定板类型、检测点数或物理间距与当前配置不一致')
                        except (TypeError, ValueError, KeyError):
                            board_reasons.append('历史标定板记录无效，不能确定物理尺度')
                    if prior.get('state') != '可采纳':
                        reasons += ['原采图验收未通过，不能仅凭重新检测自动纳入']
                        reasons += prior.get('reasons', [])
                reasons += board_reasons
                if self.offline_recheck:
                    self.offline_reviews[key] = offline_pair_review(key, (rgb, aux),
                        (rgb_det, aux_det), self.board_config, modality=self.modality,
                        paths=(rgb_files[key], aux_files[key]), capture_review=prior)
                    if self.offline_reviews[key]['sources'] != snapshots:
                        raise ValueError(f'{key}: 导入期间原图已变化，请等待文件保存完成后重新导入')
                    if board_reasons:
                        self.offline_reviews[key]['image_accepted'] = False
                        self.offline_reviews[key]['image_reasons'] += board_reasons
                self.frame_reviews[key] = dict(schema_version=1,
                    detected=bool(rgb_det.valid and aux_det.valid),
                    state='待复核' if reasons else '可采纳', reasons=reasons,
                    metrics=[q.metrics for q in qualities], temporal_stability='not_verified_offline',
                    physical_corner_order='requires_manual_confirmation')
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
    if len(by_norm) != len(files):
        raise ValueError("存在仅大小写不同的图片路径，无法可靠区分左右配对，请先整理目录")
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
    counts = Counter(index for index, _, _ in pairs)
    pairs = [(left.relative_to(root).as_posix() if counts[index] > 1 or not index else index, left, right)
             for index, left, right in pairs]
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
            key = stem[len(prefix) + 1:]
            if key in files:
                raise ValueError(f'配对编号重复：{prefix}_{key}')
            files[key] = path
    return files
