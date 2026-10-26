"""Calibration session management: save/load images, detections, and results."""

from __future__ import annotations

import json
import logging
import shutil
import hashlib
import uuid
from contextlib import contextmanager
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
        }
        session._save_metadata()
        return session

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
                             auxiliary: AuxiliaryCameraConfig | None, camera_model: str = "pinhole"):
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
        if previous == current:
            return
        revision = {
            "changed_at": datetime.now().isoformat(), "previous": previous,
            "frame_count": self.frame_count,
            "multimodal_frame_count": self._metadata.get("multimodal_frame_count", 0),
        }
        self._metadata["configuration_history"] = [*self._metadata.get("configuration_history", []), revision]
        self._metadata.update(current)
        path = self.session_dir / self.METADATA_FILE
        with _staging_directory(dir=self.session_dir) as staging:
            staged = Path(staging) / path.name
            staged.write_text(json.dumps(self._metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            staged.replace(path)

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
                left = cv2.imread(str(lf))
                right = cv2.imread(str(rf))
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
        if frame_idx is None:
            frame_idx = self._metadata.get("multimodal_frame_count", 0)

        mm_dir = self.session_dir / self.MULTIMODAL_DIR
        mm_dir.mkdir(parents=True, exist_ok=True)
        self._save_image_pair(mm_dir, "rgb_left", "aux", rgb_left, aux, frame_idx, "multimodal_frame_count")
        return frame_idx

    def _save_image_pair(self, directory, first_name, second_name, first, second, index, counter):
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("图片索引必须为非负整数")
        targets = [directory / f"{name}_{index:04d}.png" for name in (first_name, second_name)]
        with self._io_lock:
            if any(path.exists() for path in targets):
                raise FileExistsError(f"采图目标已存在，未覆盖历史图片：{targets[0]}")
            previous = self._metadata.copy()
            committed = []
            try:
                with _staging_directory(prefix=".capture-", dir=directory) as staging:
                    staged = [Path(staging) / target.name for target in targets]
                    for path, image in zip(staged, (first, second)):
                        if not cv2.imwrite(str(path), image):
                            raise OSError(f"无法保存图片：{path.name}")
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

    # ── Calibration results ───────────────────────────────────

    def save_rig(self, rig: MultiCameraRig):
        path = self.session_dir / self.RESULTS_DIR / "rig.json"
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
        """Persist the rig first, then export formats independently of each other."""
        from .export import (
            KalibrDistortionError,
            export_rig_kalibr_yaml,
            export_rig_opencv_standard_yaml,
            export_rig_opencv_yaml,
        )

        saved = [self.save_rig(rig)]
        errors: list[str] = []
        results = self.session_dir / self.RESULTS_DIR
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
                # Replace a previous result only after its new export has completed.
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
                "schema_version": 1, "generation_id": uuid.uuid4().hex,
                "created": datetime.now().isoformat(), "reference_camera": rig.reference_camera,
                "pair_names": list(rig.pairs), "errors": list(errors),
                "files": [{"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for path in saved],
                "note": "Only listed files with matching hashes belong to this generation; other files may be historical.",
            }
            self._atomic_json(manifest_path, manifest)
            saved.append(manifest_path)
        except Exception as exc:
            logger.exception("Save calibration manifest failed")
            errors.append(f"calibration_manifest.json: {exc}")
        return saved, errors

    def load_rig(self) -> Optional[MultiCameraRig]:
        path = self.session_dir / self.RESULTS_DIR / "rig.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return MultiCameraRig.from_dict(data)

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
