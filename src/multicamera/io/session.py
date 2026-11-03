"""Calibration session management: save/load images, detections, and results."""

from __future__ import annotations

import json
import logging
import shutil
import hashlib
import uuid
from contextlib import contextmanager
from copy import deepcopy
from threading import RLock
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..board.charuco_board import CharucoBoard
from ..calibration.models import CameraModel, MultiCameraRig
from ..perf import perf_timer
from ..streaming.stream_manager import AuxiliaryCameraConfig, StereoPairConfig
from .image_io import write_png

logger = logging.getLogger(__name__)


@contextmanager
def _staging_directory(**kwargs):
    """Keep cleanup errors from masking a completed commit or its original failure."""
    staging = TemporaryDirectory(**kwargs)
    try:
        yield staging.name
    finally:
        try:
            staging.cleanup()
        except Exception:
            logger.exception("Failed to clean staging directory: %s", staging.name)


class CalibrationSession:
    """Persistent storage for a single calibration run."""

    METADATA_FILE = "session.json"
    IMAGES_DIR = "images"
    MULTIMODAL_DIR = "multimodal"
    RESULTS_DIR = "results"

    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self._metadata: Dict = {}
        self._io_lock = RLock()
        self._ensure_dirs()

    def _ensure_dirs(self):
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / self.IMAGES_DIR).mkdir(exist_ok=True)
        (self.session_dir / self.MULTIMODAL_DIR).mkdir(exist_ok=True)
        (self.session_dir / self.RESULTS_DIR).mkdir(exist_ok=True)

    # ── Factory ───────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        base_dir: Path,
        board: CharucoBoard,
        pairs: List[StereoPairConfig],
        auxiliary: AuxiliaryCameraConfig | None = None,
        name: str = "",
        camera_model: str = "pinhole",
    ) -> CalibrationSession:
        camera_model = CameraModel(camera_model).value
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"{name}_{timestamp}" if name else timestamp
        base_dir.mkdir(parents=True, exist_ok=True)
        session_dir = base_dir / session_name
        suffix = 0
        while True:
            try:
                session_dir.mkdir()
                break
            except FileExistsError:
                suffix += 1
                session_dir = base_dir / f"{session_name}_{suffix:02d}"
        session_name = session_dir.name
        session = cls(session_dir)

        session._metadata = {
            "name": session_name,
            "created": datetime.now().isoformat(),
            "board": board.to_dict(),
            "camera_model": camera_model,
            "pairs": [
                {
                    "name": p.name,
                    "left_id": p.left.camera_id,
                    "left_url": p.left.url,
                    "right_id": p.right.camera_id,
                    "right_url": p.right.url,
                    "stream_type": p.left.stream_type,
                }
                for p in pairs
            ],
            "auxiliary_camera": (
                auxiliary.to_dict()
                if auxiliary is not None
                else None
            ),
            "frame_count": 0,
            "multimodal_frame_count": 0,
            "multimodal_rgb_left_id": next((p.left.camera_id for p in pairs
                                              if p.left.stream_type == 'rgb'), None),
        }
        session._creation_metadata = deepcopy(session._metadata)
        try:
            session._save_metadata()
        except Exception:
            session.discard_unpublished()
            raise
        return session

    def is_pristine(self) -> bool:
        """Only metadata and empty folders; counters alone cannot establish emptiness."""
        if self.frame_count or self._metadata.get('multimodal_frame_count', 0):
            return False
        if self.session_dir.is_symlink() or not self.session_dir.is_dir():
            return False
        metadata = self.session_dir / self.METADATA_FILE
        return not any(path.is_symlink() or (path.is_file() and path != metadata)
                       for path in self.session_dir.rglob('*'))

    def discard_unpublished(self) -> bool:
        """Remove only this object's newly created, unused directory, never loaded history."""
        original = getattr(self, '_creation_metadata', None)
        if original is None or self._metadata != original or self.session_dir.is_symlink():
            return False
        metadata = self.session_dir / self.METADATA_FILE
        folders = [self.session_dir / name for name in
                   (self.IMAGES_DIR, self.MULTIMODAL_DIR, self.RESULTS_DIR)]
        with self._io_lock:
            try:
                if metadata.exists() and json.loads(metadata.read_text(encoding='utf-8')) != original:
                    return False
                allowed = set(folders + [metadata])
                if (any(path not in allowed or path.is_symlink() for path in self.session_dir.iterdir())
                        or any(folder.exists() and any(folder.iterdir()) for folder in folders)):
                    return False
                for folder in folders:
                    if folder.exists():
                        folder.rmdir()
                metadata.unlink(missing_ok=True)
                self.session_dir.rmdir()
                self._creation_metadata = None
                return True
            except (OSError, ValueError):
                # A late writer wins: keep its files and restore session metadata if needed.
                if self.session_dir.exists():
                    try:
                        self._ensure_dirs()
                        if not metadata.exists():
                            self._atomic_json(metadata, original)
                    except OSError:
                        logger.exception('Could not restore unpublished session metadata')
                logger.warning('Could not discard unused session: %s', self.session_dir, exc_info=True)
                return False

    @classmethod
    def load(cls, session_dir: Path) -> CalibrationSession:
        session_dir = Path(session_dir)
        meta_path = session_dir / cls.METADATA_FILE
        if not meta_path.is_file():
            raise ValueError("所选目录不是标定会话：缺少 session.json")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("session.json 必须包含会话对象")
        for counter in ("frame_count", "multimodal_frame_count"):
            value = metadata.get(counter, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"会话计数 {counter} 无效")
        session = cls(session_dir)
        session._metadata = metadata
        return session

    # ── Image storage ─────────────────────────────────────────

    def record_configuration(self, board: CharucoBoard, pairs: List[StereoPairConfig],
                             auxiliary: AuxiliaryCameraConfig | None, camera_model: str = "pinhole",
                             *, continuation: dict | None = None):
        """Record configuration revisions without deleting historical images/results."""
        keys = ("board", "pairs", "auxiliary_camera", "camera_model")
        previous = {key: self._metadata.get(key) for key in keys}
        current = {
            "board": board.to_dict(),
            "pairs": [{"name": p.name, "left_id": p.left.camera_id, "left_url": p.left.url,
                       "right_id": p.right.camera_id, "right_url": p.right.url,
                       "stream_type": p.left.stream_type} for p in pairs],
            "auxiliary_camera": auxiliary.to_dict() if auxiliary else None,
            "camera_model": camera_model,
        }
        if previous == current and continuation is None:
            return
        revision = {
            "changed_at": datetime.now().isoformat(), "previous": previous,
            "frame_count": self.frame_count,
            "multimodal_frame_count": self._metadata.get("multimodal_frame_count", 0),
        }
        updated = deepcopy(self._metadata)
        if not updated.get('multimodal_rgb_left_id'):
            updated['multimodal_rgb_left_id'] = next((p.get('left_id') for p in self._metadata.get('pairs', [])
                if p.get('stream_type', 'rgb') == 'rgb'), None)
        if previous != current:
            updated["configuration_history"] = [*updated.get("configuration_history", []), revision]
        updated.update(current)
        if continuation is not None:
            updated['rgb_continuation_history'] = [*updated.get('rgb_continuation_history', []),
                                                    deepcopy(continuation)]
        path = self.session_dir / self.METADATA_FILE
        with _staging_directory(dir=self.session_dir) as staging:
            staged = Path(staging) / path.name
            staged.write_text(json.dumps(updated, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            staged.replace(path)
        self._metadata = updated

    def save_frame_pair(
        self,
        pair_name: str,
        left: np.ndarray,
        right: np.ndarray,
        frame_idx: Optional[int] = None,
    ) -> int:
        """Save a stereo frame pair and return the frame index."""
        if frame_idx is None:
            frame_idx = self._metadata.get("frame_count", 0)

        pair_dir = self.session_dir / self.IMAGES_DIR / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)

        with perf_timer("save frame pair", threshold_ms=100.0):
            self._save_image_pair(pair_dir, "left", "right", left, right, frame_idx, "frame_count")
        return frame_idx

    def save_capture_review(self, pair_name: str, frame_idx: int, review: dict) -> None:
        """Persist admission separately from successful image-save counters."""
        folder = self.session_dir / self.IMAGES_DIR / pair_name
        folder.mkdir(parents=True, exist_ok=True)
        self._atomic_json(folder / f'quality_{frame_idx:04d}.json', review)

    def delete_saved_pair_frame(self, pair_name: str, frame_idx: int) -> bool:
        """删除某组指定索引的左右 PNG；并依据磁盘重新计算 ``frame_count``。"""
        pair_dir = self.session_dir / self.IMAGES_DIR / pair_name
        lf = pair_dir / f"left_{frame_idx:04d}.png"
        rf = pair_dir / f"right_{frame_idx:04d}.png"
        removed = False
        if lf.exists():
            lf.unlink()
            removed = True
        if rf.exists():
            rf.unlink()
            removed = True
        # Retain admission evidence if either image cannot be removed (e.g. Windows locks).
        (pair_dir / f'quality_{frame_idx:04d}.json').unlink(missing_ok=True)
        if removed:
            self._recompute_frame_count_from_disk()
            self._save_metadata()
        return removed

    def _recompute_frame_count_from_disk(self) -> None:
        mx = -1
        root = self.session_dir / self.IMAGES_DIR
        if root.exists():
            for pd in root.iterdir():
                if not pd.is_dir():
                    continue
                for fp in pd.glob("left_*.png"):
                    if not fp.with_name(fp.name.replace("left_", "right_", 1)).exists():
                        continue
                    try:
                        idx = int(fp.stem.replace("left_", ""))
                        mx = max(mx, idx)
                    except ValueError:
                        continue
        self._metadata["frame_count"] = mx + 1 if mx >= 0 else 0

    def resolved_images_base(self) -> Path:
        """当前会话写入图像的根目录（绝对路径）。"""
        return (self.session_dir / self.IMAGES_DIR).resolve()

    def load_frame_pairs(
        self, pair_name: str
    ) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """Load all saved frame pairs for a stereo pair."""
        pair_dir = self.session_dir / self.IMAGES_DIR / pair_name
        if not pair_dir.exists():
            return []

        frames = []
        left_files = sorted(pair_dir.glob("left_*.png"))
        for lf in left_files:
            idx_str = lf.stem.replace("left_", "")
            idx = int(idx_str)
            rf = pair_dir / f"right_{idx_str}.png"
            if rf.exists():
                left = cv2.imdecode(np.frombuffer(lf.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                right = cv2.imdecode(np.frombuffer(rf.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                if left is not None and right is not None:
                    frames.append((idx, left, right))
        return frames

    def save_multimodal_pair(
        self,
        rgb_left: np.ndarray,
        aux: np.ndarray,
        frame_idx: Optional[int] = None,
    ) -> int:
        """Save a paired RGB_L/AUX observation under the session multimodal folder."""
        mm_dir = self.session_dir / self.MULTIMODAL_DIR
        mm_dir.mkdir(parents=True, exist_ok=True)
        with self._io_lock:
            if frame_idx is None:
                frame_idx = self.multimodal_resume_state()['next_index']
            self._save_image_pair(mm_dir, "rgb_left", "aux", rgb_left, aux, frame_idx,
                                  "multimodal_frame_count", pending_review=True)
        return frame_idx

    def multimodal_resume_state(self):
        """Count saved PNG pairs, reserving even orphan image/review indices."""
        folder = self.session_dir / self.MULTIMODAL_DIR
        indices = {}
        for prefix, suffix in (('rgb_left_', '.png'), ('aux_', '.png'), ('quality_', '.json')):
            indices[prefix] = {int(path.stem[len(prefix):])
                for path in folder.glob(f'{prefix}*{suffix}')
                if path.stem[len(prefix):].isascii() and path.stem[len(prefix):].isdecimal()}
        reserved = set().union(*indices.values())
        return dict(saved_pairs=len(indices['rgb_left_'] & indices['aux_']),
            next_index=max(self._metadata.get('multimodal_frame_count', 0),
                           max(reserved, default=-1) + 1))

    def save_multimodal_review(self, frame_idx: int, review: dict) -> None:
        """Keep raw admission evidence next to the successfully saved pair."""
        self._atomic_json(self.session_dir / self.MULTIMODAL_DIR / f'quality_{frame_idx:04d}.json',
                          review)

    def _save_image_pair(self, directory, first_name, second_name, first, second, index, counter,
                         *, pending_review=False):
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("图片索引必须为非负整数")
        targets = [directory / f"{name}_{index:04d}.png" for name in (first_name, second_name)]
        if pending_review:
            targets.append(directory / f'quality_{index:04d}.json')
        with self._io_lock:
            if any(path.exists() for path in targets):
                raise FileExistsError(f"采图目标已存在，未覆盖历史图片：{targets[0]}")
            previous = self._metadata.copy()
            committed = []
            try:
                with _staging_directory(prefix=".capture-", dir=directory) as staging:
                    staged = [Path(staging) / target.name for target in targets]
                    for path, image in zip(staged, (first, second)):
                        write_png(path, image)
                    if pending_review:
                        self._atomic_json(staged[-1], dict(schema_version=1, state='待复核',
                            reasons=['原图已落盘，验收尚未提交；请勿自动纳入标定'],
                            physical_corner_order='requires_manual_confirmation'))
                    for source, target in zip(staged, targets):
                        source.replace(target)
                        committed.append(target)
                    self._metadata[counter] = max(self._metadata.get(counter, 0), index + 1)
                    self._save_metadata()
            except Exception:
                self._metadata = previous
                for target in committed:
                    try:
                        target.unlink(missing_ok=True)
                    except OSError:
                        logger.exception("Failed to remove uncommitted capture: %s", target)
                raise

    def record_multimodal_result(
        self,
        aux_camera_id: str,
        modality: str,
        board_type: str,
        valid_pairs: int,
        rms_error: float,
        board_config: dict | None = None,
    ):
        self._metadata["multimodal_result"] = {
            "aux_camera_id": aux_camera_id,
            "modality": modality,
            "board_type": board_type,
            "valid_pairs": valid_pairs,
            "rms_error": rms_error,
        }
        if board_config is not None:
            self._metadata["multimodal_result"]["board_config"] = dict(board_config)
        self._save_metadata()

    @property
    def frame_count(self) -> int:
        return self._metadata.get("frame_count", 0)

    @property
    def next_stereo_frame_index(self) -> int:
        indices = []
        for path in (self.session_dir / self.IMAGES_DIR).rglob('*'):
            if path.stem.startswith(('left_', 'right_', 'quality_')):
                key = path.stem.rsplit('_', 1)[-1]
                if key.isascii() and key.isdecimal():
                    indices.append(int(key))
        return max(self.frame_count, max(indices, default=-1) + 1)

    # ── Calibration results ───────────────────────────────────

    def save_rig(self, rig: MultiCameraRig, directory: Path | None = None):
        path = (directory if directory is not None else self.session_dir / self.RESULTS_DIR) / "rig.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with _staging_directory(dir=path.parent) as staging:
            tmp = Path(staging) / path.name
            tmp.write_text(
                json.dumps(rig.to_dict(), indent=2, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            tmp.replace(path)
        return path

    def save_calibration_results(self, rig: MultiCameraRig) -> tuple[list[Path], list[str]]:
        """Save an isolated generation, then atomically publish its latest pointer."""
        from .export import (
            KalibrDistortionError,
            _validate_rig,
            export_rig_kalibr_yaml,
            export_rig_opencv_standard_yaml,
            export_rig_opencv_yaml,
        )

        _validate_rig(rig)

        generation_id = uuid.uuid4().hex
        root = self.session_dir / self.RESULTS_DIR
        results = root / f"calib_{datetime.now():%Y%m%d_%H%M%S}_{generation_id}"
        results.mkdir()
        saved = [self.save_rig(rig, directory=results)]
        errors: list[str] = []
        exports = [
            ("rig_opencv.yml", export_rig_opencv_yaml, rig, {}),
            ("rig_kalibr.yaml", export_rig_kalibr_yaml, rig, {}),
        ]
        for index, (name, pair) in enumerate(rig.pairs.items(), start=1):
            filename = (
                "rig_opencv_standard.yml" if len(rig.pairs) == 1
                else f"rig_pair_{index:02d}_opencv_standard.yml"
            )
            exports.append((filename, export_rig_opencv_standard_yaml, rig, {"pair_name": name}))
            if len(rig.pairs) > 1:
                pair_rig = MultiCameraRig(pairs={name: pair}, reference_camera=pair.left_id)
                exports.append((
                    f"rig_pair_{index:02d}_kalibr.yaml", export_rig_kalibr_yaml, pair_rig, {},
                ))
        for filename, exporter, export_rig, kwargs in exports:
            try:
                # Publish only successful formats inside this fresh generation.
                with _staging_directory(dir=results) as staging:
                    tmp = Path(staging) / filename
                    exporter(export_rig, tmp, **kwargs)
                    if not tmp.is_file() or not tmp.stat().st_size:
                        raise OSError("Export did not produce a non-empty file")
                    path = results / filename
                    tmp.replace(path)
                saved.append(path)
            except Exception as exc:
                logger.exception("Automatic calibration export failed: file=%s", filename)
                reason = ('高阶畸变不能无损转换为 Kalibr radtan，请使用 JSON/OpenCV；详细参数见日志'
                          if isinstance(exc, KalibrDistortionError) else str(exc))
                errors.append(f"{filename}: {reason}")
        manifest_path = results / "calibration_manifest.json"
        try:
            manifest = {
                "schema_version": 1, "generation_id": generation_id,
                "created": datetime.now().isoformat(), "reference_camera": rig.reference_camera,
                "pair_names": list(rig.pairs), "errors": list(errors),
                "files": [{"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for path in saved],
                "note": "This directory contains only this generation; failed formats are absent.",
            }
            self._atomic_json(manifest_path, manifest)
            saved.append(manifest_path)
        except Exception as exc:
            logger.exception("Save calibration manifest failed")
            errors.append(f"calibration_manifest.json: {exc}")
        try:
            self._atomic_json(root / 'latest.json', {
                'schema_version': 1, 'generation_id': generation_id, 'directory': results.name,
            })
        except Exception as exc:
            logger.exception("Publish latest calibration pointer failed")
            errors.append(f"latest.json: {exc}（本批次已保存，最新批次索引未更新）")
        return saved, errors

    @property
    def result_directory(self) -> Path:
        """Resolve the latest batch, falling back only when no batch pointer exists."""
        root = self.session_dir / self.RESULTS_DIR
        pointer = root / 'latest.json'
        if not pointer.exists():
            return root
        data = json.loads(pointer.read_text(encoding='utf-8'))
        name = data.get('directory') if isinstance(data, dict) else None
        if (not isinstance(name, str) or not name.startswith('calib_') or
                '/' in name or '\\' in name or (root / name).resolve().parent != root.resolve()):
            raise ValueError('最新标定批次索引无效，拒绝回退到历史结果')
        directory = root / name
        if not (directory / 'rig.json').is_file():
            raise ValueError('最新标定批次缺少 rig.json，拒绝回退到历史结果')
        return directory

    def load_rig(self) -> Optional[MultiCameraRig]:
        path = self.result_path
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return MultiCameraRig.from_dict(data)

    @property
    def result_path(self) -> Path:
        path = self.result_directory / 'rig.json'
        if path.exists() or (self.session_dir / self.RESULTS_DIR / 'latest.json').exists():
            return path
        # Legacy manual complete-JSON export; never bypass a damaged latest batch pointer.
        legacy = self.session_dir / 'calibration_all_json.json'
        return legacy if legacy.is_file() else path

    # ── Metadata ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._metadata.get("name", self.session_dir.name)

    @property
    def metadata(self) -> Dict:
        return dict(self._metadata)

    def _save_metadata(self):
        path = self.session_dir / self.METADATA_FILE
        self._atomic_json(path, self._metadata)

    @staticmethod
    def _atomic_json(path, data):
        with _staging_directory(prefix=".metadata-", dir=path.parent) as staging:
            staged = Path(staging) / path.name
            staged.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            staged.replace(path)

    def delete(self):
        if self.session_dir.exists():
            shutil.rmtree(self.session_dir)


class SessionManager:
    """Discovers and manages calibration sessions in a base directory."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_sessions(self) -> List[CalibrationSession]:
        sessions = []
        for d in sorted(self.base_dir.iterdir()):
            if d.is_dir() and (d / CalibrationSession.METADATA_FILE).exists():
                sessions.append(CalibrationSession.load(d))
        return sessions

    def create_session(
        self,
        board: CharucoBoard,
        pairs: List[StereoPairConfig],
        auxiliary: AuxiliaryCameraConfig | None = None,
        name: str = "",
        camera_model: str = "pinhole",
    ) -> CalibrationSession:
        return CalibrationSession.create(self.base_dir, board, pairs, auxiliary, name, camera_model)

    def delete_session(self, session_name: str):
        session_dir = self.base_dir / session_name
        if session_dir.exists():
            shutil.rmtree(session_dir)
