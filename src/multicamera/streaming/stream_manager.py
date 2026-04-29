"""Manages multiple MJPEG streams and provides soft-synchronised frame bundles."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal

from .mjpeg_grabber import MJPEGGrabber


@dataclass
class CameraConfig:
    camera_id: str
    url: str
    role: str = ""  # e.g. "left" or "right"
    group: str = ""  # stereo pair group name
    stream_type: str = "rgb"  # e.g. "rgb" or "ir"


@dataclass
class FrameEntry:
    frame: np.ndarray
    timestamp: float


@dataclass
class StereoPairConfig:
    name: str
    left: CameraConfig
    right: CameraConfig


class StreamManager(QObject):
    """Lifecycle manager for all camera streams with soft-sync capability."""

    frame_received = Signal(str, np.ndarray, float)  # camera_id, frame, ts
    sync_pair_ready = Signal(str, np.ndarray, np.ndarray, float)  # group, left, right, ts
    camera_connected = Signal(str)
    camera_disconnected = Signal(str, str)
    fps_updated = Signal(str, float)

    # 双目帧时间戳允许偏差；过小易导致「采集一帧」拿不到同步对、进度与磁盘保存都不触发
    DEFAULT_SYNC_TOLERANCE = 0.35  # 350 ms（嵌入式 MJPEG 左右帧到达 UI 常有抖动）

    def __init__(self, sync_tolerance: float = DEFAULT_SYNC_TOLERANCE, parent=None):
        super().__init__(parent)
        self._sync_tolerance = sync_tolerance
        self._grabbers: Dict[str, MJPEGGrabber] = {}
        self._latest_frames: Dict[str, FrameEntry] = {}
        self._stereo_pairs: Dict[str, StereoPairConfig] = {}
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────

    def add_stereo_pair(self, pair: StereoPairConfig):
        self._stereo_pairs[pair.name] = pair
        self._add_camera(pair.left)
        self._add_camera(pair.right)

    def remove_stereo_pair(self, name: str):
        pair = self._stereo_pairs.pop(name, None)
        if pair:
            self._remove_camera(pair.left.camera_id)
            self._remove_camera(pair.right.camera_id)

    def start_all(self):
        # Ensure grabber objects exist for every configured camera, then start.
        # After stop_all(), MJPEGGrabber QThreads are finished — they must be recreated
        # (Qt does not reliably allow restarting the same QThread instance).
        for pair in self._stereo_pairs.values():
            self._add_camera(pair.left)
            self._add_camera(pair.right)
        for g in self._grabbers.values():
            if not g.isRunning():
                g.start()

    def stop_all(self):
        for g in list(self._grabbers.values()):
            g.stop()
        self._grabbers.clear()
        self._latest_frames.clear()

    def get_latest_frame(self, camera_id: str) -> Optional[Tuple[np.ndarray, float]]:
        with self._lock:
            entry = self._latest_frames.get(camera_id)
            if entry:
                return entry.frame, entry.timestamp
        return None

    def get_sync_pair(self, group_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return the latest soft-synchronised frame pair, or None."""
        pair = self._stereo_pairs.get(group_name)
        if not pair:
            return None

        with self._lock:
            left_entry = self._latest_frames.get(pair.left.camera_id)
            right_entry = self._latest_frames.get(pair.right.camera_id)

        if left_entry is None or right_entry is None:
            return None
        if abs(left_entry.timestamp - right_entry.timestamp) > self._sync_tolerance:
            return None
        return left_entry.frame, right_entry.frame

    def get_latest_pair_relaxed(self, group_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """左右流各有最新帧即返回（忽略时间戳），用于严格同步失败时的采集回退。"""
        pair = self._stereo_pairs.get(group_name)
        if not pair:
            return None
        with self._lock:
            left_entry = self._latest_frames.get(pair.left.camera_id)
            right_entry = self._latest_frames.get(pair.right.camera_id)
        if left_entry is None or right_entry is None:
            return None
        return left_entry.frame, right_entry.frame

    @property
    def sync_tolerance_seconds(self) -> float:
        return self._sync_tolerance

    @property
    def stereo_pairs(self) -> Dict[str, StereoPairConfig]:
        return dict(self._stereo_pairs)

    @property
    def camera_ids(self) -> List[str]:
        return list(self._grabbers.keys())

    # ── Internal ──────────────────────────────────────────────

    def _add_camera(self, cfg: CameraConfig):
        existing = self._grabbers.get(cfg.camera_id)
        if existing is not None:
            if existing.isRunning() and getattr(existing, "url", "") == cfg.url:
                return
            existing.stop()
            existing.wait(3000)
            self._grabbers.pop(cfg.camera_id, None)

        grabber = MJPEGGrabber(cfg.camera_id, cfg.url, parent=self)
        grabber.frame_ready.connect(self._on_frame)
        grabber.connection_established.connect(self.camera_connected)
        grabber.connection_lost.connect(self.camera_disconnected)
        grabber.fps_updated.connect(self.fps_updated)
        self._grabbers[cfg.camera_id] = grabber

    def _remove_camera(self, camera_id: str):
        grabber = self._grabbers.pop(camera_id, None)
        if grabber:
            grabber.stop()
        with self._lock:
            self._latest_frames.pop(camera_id, None)

    def _on_frame(self, camera_id: str, frame: np.ndarray, ts: float):
        with self._lock:
            self._latest_frames[camera_id] = FrameEntry(frame, ts)

        self.frame_received.emit(camera_id, frame, ts)
        self._try_emit_sync_pairs(camera_id, ts)

    def _try_emit_sync_pairs(self, camera_id: str, ts: float):
        for name, pair in self._stereo_pairs.items():
            if camera_id not in (pair.left.camera_id, pair.right.camera_id):
                continue

            with self._lock:
                left_entry = self._latest_frames.get(pair.left.camera_id)
                right_entry = self._latest_frames.get(pair.right.camera_id)

            if left_entry is None or right_entry is None:
                continue
            if abs(left_entry.timestamp - right_entry.timestamp) <= self._sync_tolerance:
                self.sync_pair_ready.emit(
                    name, left_entry.frame, right_entry.frame, ts
                )
