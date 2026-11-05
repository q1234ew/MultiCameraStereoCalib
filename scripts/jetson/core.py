"""Bounded CPU primitives shared by replay and a future raw transport adapter."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def load_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("listen") != "127.0.0.1":
        raise ValueError("Expected schema 1 and loopback listener")
    if type(config.get("port")) is not int or not 1024 <= config["port"] <= 65535:
        raise ValueError("Invalid service port")
    for name in ("max_skew_ms", "max_age_ms"):
        if (type(config.get(name)) not in (int, float)
                or not math.isfinite(config[name]) or not 0 < config[name] <= 1000):
            raise ValueError(f"Invalid {name}")
    if config["max_age_ms"] < config["max_skew_ms"]:
        raise ValueError("Age bound must cover skew bound")
    devices = config.get("devices")
    if not isinstance(devices, list) or not 1 <= len(devices) <= 3:
        raise ValueError("Expected one to three devices")
    ids, hosts, macs = set(), set(), set()
    for device in devices:
        ident, host, mac = device["id"], device["host"], device["expected_mac"]
        if not re.fullmatch(r"device_[1-3]", ident) or ident in ids:
            raise ValueError("Invalid or duplicate device id")
        if not ipaddress.ip_address(host).is_private or host in hosts:
            raise ValueError("Expected distinct private camera IPs")
        if not re.fullmatch(r"(?:[0-9a-f]{2}:){5}[0-9a-f]{2}", mac) or mac in macs:
            raise ValueError("Invalid or duplicate MAC")
        ids.add(ident)
        hosts.add(host)
        macs.add(mac)
        if type(device.get("physical_binding_verified")) is not bool:
            raise ValueError("Explicit physical binding status required")
        if not re.fullmatch(r"calib_\d{8}_\d{6}", device["source_session"]):
            raise ValueError("Missing source session binding")
        if device.get("translation_unit") != "mm":
            raise ValueError("Candidate translations must be millimeters")
        if not re.fullmatch(r"[0-9a-f]{64}", device["calibration_sha256"]):
            raise ValueError("Calibration SHA256 required")
        if not Path(device["calibration_path"]).is_absolute():
            raise ValueError("Absolute calibration path required")
        if set(device["raw_ports"]) != {"left", "right", "ir"}:
            raise ValueError("Three raw roles required")
        ports = list(device["raw_ports"].values())
        if len(set(ports)) != 3 or any(type(p) is not int or p not in (8080, 8081, 8082)
                                       for p in ports):
            raise ValueError("Raw endpoints must be distinct original camera ports")
        if set(device["resolution"]) != {"left", "right", "ir"}:
            raise ValueError("Resolution binding required for every camera")
        for size in device["resolution"].values():
            if (not isinstance(size, list) or len(size) != 2
                    or any(type(v) is not int or not 0 < v <= 4096 for v in size)):
                raise ValueError("Invalid bound resolution")
    return config


@dataclass(frozen=True)
class Frame:
    device_id: str
    camera: str
    timestamp_ns: int
    image: np.ndarray


class Pairer:
    """Two latest-frame slots per device; timestamps share one monotonic clock."""

    def __init__(self, device: dict, skew_ms: float, age_ms: float):
        self.device = device
        self.skew_ns = int(skew_ms * 1_000_000)
        self.age_ns = int(age_ms * 1_000_000)
        self.pending: dict[str, Frame] = {}
        self.last_timestamp: dict[str, int] = {}
        self.dropped = 0

    def push(self, frame: Frame, now_ns: int) -> tuple[Frame, Frame] | None:
        if frame.device_id != self.device["id"] or frame.camera not in ("left", "right"):
            raise ValueError("Frame belongs to another device or camera role")
        if list(frame.image.shape[1::-1]) != self.device["resolution"][frame.camera]:
            raise ValueError("Raw frame resolution mismatch; resizing forbidden")
        previous = self.last_timestamp.get(frame.camera, -1)
        if (frame.timestamp_ns <= previous or frame.timestamp_ns > now_ns
                or now_ns - frame.timestamp_ns > self.age_ns):
            self.dropped += 1
            return None
        self.last_timestamp[frame.camera] = frame.timestamp_ns
        for camera, old in list(self.pending.items()):
            if now_ns - old.timestamp_ns > self.age_ns:
                del self.pending[camera]
                self.dropped += 1
        if frame.camera in self.pending:
            self.dropped += 1
        self.pending[frame.camera] = frame
        if len(self.pending) != 2:
            return None
        left, right = self.pending["left"], self.pending["right"]
        if abs(left.timestamp_ns - right.timestamp_ns) > self.skew_ns:
            del self.pending[min(self.pending, key=lambda k: self.pending[k].timestamp_ns)]
            self.dropped += 1
            return None
        self.pending.clear()
        return left, right


class Rectifier:
    """One immutable calibration and two cached maps per device, CPU only."""

    def __init__(self, device: dict, path: Path):
        raw = path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if self.sha256 != device["calibration_sha256"]:
            raise ValueError("Calibration hash mismatch")
        fs = cv2.FileStorage(raw.decode("utf-8"), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
        try:
            expected = {"device_id": device["id"], "translation_unit": "mm",
                        "coord_frame": "C_L", "stereo_transform_direction": "left_to_right",
                        "ir_transform_direction": "ir_to_left"}
            if any(fs.getNode(k).string() != v for k, v in expected.items()):
                raise ValueError("Calibration identity, unit or transform direction mismatch")
            self.maps = {}
            self.projection = None
            for camera in ("left", "right", "ir"):
                intr = fs.getNode("camera_intrinsics").getNode(camera)
                size = [int(intr.getNode(k).real()) for k in ("width", "height")]
                if size != device["resolution"][camera]:
                    raise ValueError("Calibration resolution mismatch")
                model = intr.getNode("model").string()
                if model not in ("pinhole", "fisheye"):
                    raise ValueError("Unsupported camera model")
                if camera == "ir":
                    continue
                index = "1" if camera == "left" else "2"
                rect = fs.getNode("stereo_rectify")
                k, d = intr.getNode("K").mat(), intr.getNode("D").mat()
                r, p = rect.getNode("R" + index).mat(), rect.getNode("P" + index).mat()
                if any(m is None or not np.isfinite(m).all() for m in (k, d, r, p)):
                    raise ValueError("Missing or non-finite calibration matrices")
                builder = (cv2.fisheye.initUndistortRectifyMap if model == "fisheye"
                           else cv2.initUndistortRectifyMap)
                self.maps[camera] = builder(k, d, r, p[:, :3], tuple(size), cv2.CV_32FC1)
                if camera == "left":
                    self.projection = p[:, :3].copy()
        finally:
            fs.release()
        self.device = device

    def rectify(self, frame: Frame) -> np.ndarray:
        if frame.device_id != self.device["id"] or frame.camera not in self.maps:
            raise ValueError("Rectifier device/role mismatch")
        if list(frame.image.shape[1::-1]) != self.device["resolution"][frame.camera]:
            raise ValueError("Rectifier resolution mismatch")
        return cv2.remap(frame.image, *self.maps[frame.camera], cv2.INTER_LINEAR)


def depth_points(depth: np.ndarray, projection: np.ndarray, *, unit: str,
                 expected_size: tuple[int, int], stride: int = 4) -> np.ndarray:
    """Backproject actual registered rectified-left axial Z; output meters in C_L_rect."""
    if depth.ndim != 2 or depth.shape[::-1] != expected_size:
        raise ValueError("Depth must match rectified-left resolution")
    if unit not in ("mm", "m") or type(stride) is not int or stride < 1:
        raise ValueError("Explicit depth unit and positive stride required")
    if (projection.shape != (3, 3) or not np.isfinite(projection).all()
            or projection[0, 0] <= 0 or projection[1, 1] <= 0):
        raise ValueError("Invalid rectified projection")
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[::stride, ::stride].astype(np.float64) * (0.001 if unit == "mm" else 1.)
    valid = np.isfinite(z) & (z > 0)
    rays = np.linalg.solve(projection, np.stack((u, v, np.ones_like(u))).reshape(3, -1))
    return (rays * z.reshape(1, -1)).T[valid.ravel()]
