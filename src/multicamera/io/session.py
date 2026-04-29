"""Calibration session management: save/load images, detections, and results."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..board.charuco_board import CharucoBoard
from ..calibration.models import MultiCameraRig, StereoPairCalibration
from ..streaming.stream_manager import StereoPairConfig


class CalibrationSession:
    """Persistent storage for a single calibration run."""

    METADATA_FILE = "session.json"
    IMAGES_DIR = "images"
    RESULTS_DIR = "results"

    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self._metadata: Dict = {}
        self._ensure_dirs()

    def _ensure_dirs(self):
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / self.IMAGES_DIR).mkdir(exist_ok=True)
        (self.session_dir / self.RESULTS_DIR).mkdir(exist_ok=True)

    # ── Factory ───────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        base_dir: Path,
        board: CharucoBoard,
        pairs: List[StereoPairConfig],
        name: str = "",
    ) -> CalibrationSession:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"{name}_{timestamp}" if name else timestamp
        session_dir = base_dir / session_name
        session = cls(session_dir)

        session._metadata = {
            "name": session_name,
            "created": datetime.now().isoformat(),
            "board": board.to_dict(),
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
            "frame_count": 0,
        }
        session._save_metadata()
        return session

    @classmethod
    def load(cls, session_dir: Path) -> CalibrationSession:
        session = cls(session_dir)
        meta_path = session_dir / cls.METADATA_FILE
        if meta_path.exists():
            session._metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        return session

    # ── Image storage ─────────────────────────────────────────

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

        cv2.imwrite(str(pair_dir / f"left_{frame_idx:04d}.png"), left)
        cv2.imwrite(str(pair_dir / f"right_{frame_idx:04d}.png"), right)

        self._metadata["frame_count"] = max(
            self._metadata.get("frame_count", 0), frame_idx + 1
        )
        self._save_metadata()
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

    @property
    def frame_count(self) -> int:
        return self._metadata.get("frame_count", 0)

    # ── Calibration results ───────────────────────────────────

    def save_rig(self, rig: MultiCameraRig):
        path = self.session_dir / self.RESULTS_DIR / "rig.json"
        path.write_text(
            json.dumps(rig.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

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
        path.write_text(
            json.dumps(self._metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

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
        name: str = "",
    ) -> CalibrationSession:
        return CalibrationSession.create(self.base_dir, board, pairs, name)

    def delete_session(self, session_name: str):
        session_dir = self.base_dir / session_name
        if session_dir.exists():
            shutil.rmtree(session_dir)
