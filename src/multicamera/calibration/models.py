"""Data models for calibration parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class CameraModel(str, Enum):
    PINHOLE = "pinhole"
    FISHEYE = "fisheye"


@dataclass
class CameraIntrinsics:
    """Single camera intrinsic calibration result."""

    camera_matrix: np.ndarray  # 3x3
    dist_coeffs: np.ndarray  # pinhole: 1x5/1x8, fisheye: 1x4
    image_size: Tuple[int, int]  # (width, height)
    model: CameraModel = CameraModel.PINHOLE
    rms_error: float = 0.0
    per_frame_errors: Optional[List[float]] = None
    rvecs: Optional[List[np.ndarray]] = None
    tvecs: Optional[List[np.ndarray]] = None

    def to_dict(self) -> dict:
        d = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "image_size": list(self.image_size),
            "model": self.model.value,
            "rms_error": self.rms_error,
        }
        if self.per_frame_errors is not None:
            d["per_frame_errors"] = self.per_frame_errors
        return d

    @classmethod
    def from_dict(cls, d: dict) -> CameraIntrinsics:
        return cls(
            camera_matrix=np.array(d["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(d["dist_coeffs"], dtype=np.float64),
            image_size=tuple(d["image_size"]),
            model=CameraModel(d.get("model", "pinhole")),
            rms_error=d.get("rms_error", 0.0),
            per_frame_errors=d.get("per_frame_errors"),
        )


@dataclass
class StereoCalibResult:
    """Stereo pair calibration result."""

    R: np.ndarray  # 3x3 rotation
    T: np.ndarray  # 3x1 translation
    E: np.ndarray  # 3x3 essential matrix
    F: np.ndarray  # 3x3 fundamental matrix
    rms_error: float = 0.0
    model: CameraModel = CameraModel.PINHOLE

    R1: Optional[np.ndarray] = None
    R2: Optional[np.ndarray] = None
    P1: Optional[np.ndarray] = None
    P2: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None
    map1_left: Optional[np.ndarray] = None
    map2_left: Optional[np.ndarray] = None
    map1_right: Optional[np.ndarray] = None
    map2_right: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        d = {
            "R": self.R.tolist(),
            "T": self.T.tolist(),
            "E": self.E.tolist(),
            "F": self.F.tolist(),
            "rms_error": self.rms_error,
            "model": self.model.value,
        }
        if self.Q is not None:
            d["Q"] = self.Q.tolist()
        if self.R1 is not None:
            d["R1"] = self.R1.tolist()
            d["R2"] = self.R2.tolist()
            d["P1"] = self.P1.tolist()
            d["P2"] = self.P2.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> StereoCalibResult:
        result = cls(
            R=np.array(d["R"], dtype=np.float64),
            T=np.array(d["T"], dtype=np.float64),
            E=np.array(d["E"], dtype=np.float64),
            F=np.array(d["F"], dtype=np.float64),
            rms_error=d.get("rms_error", 0.0),
            model=CameraModel(d.get("model", "pinhole")),
        )
        if "Q" in d:
            result.Q = np.array(d["Q"], dtype=np.float64)
        if "R1" in d:
            result.R1 = np.array(d["R1"], dtype=np.float64)
            result.R2 = np.array(d["R2"], dtype=np.float64)
            result.P1 = np.array(d["P1"], dtype=np.float64)
            result.P2 = np.array(d["P2"], dtype=np.float64)
        return result


@dataclass
class CameraExtrinsics:
    """Extrinsic parameters of a camera relative to the world frame."""

    R: np.ndarray  # 3x3 rotation
    T: np.ndarray  # 3x1 translation

    @property
    def transformation_matrix(self) -> np.ndarray:
        """4x4 homogeneous transformation [R|T; 0 1]."""
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = self.R
        mat[:3, 3] = self.T.flatten()
        return mat

    def to_dict(self) -> dict:
        return {"R": self.R.tolist(), "T": self.T.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> CameraExtrinsics:
        return cls(
            R=np.array(d["R"], dtype=np.float64),
            T=np.array(d["T"], dtype=np.float64),
        )


@dataclass
class StereoPairCalibration:
    """Complete calibration data for one stereo pair."""

    pair_name: str
    left_id: str
    right_id: str
    left_intrinsics: CameraIntrinsics
    right_intrinsics: CameraIntrinsics
    stereo: StereoCalibResult

    def to_dict(self) -> dict:
        return {
            "pair_name": self.pair_name,
            "left_id": self.left_id,
            "right_id": self.right_id,
            "left_intrinsics": self.left_intrinsics.to_dict(),
            "right_intrinsics": self.right_intrinsics.to_dict(),
            "stereo": self.stereo.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> StereoPairCalibration:
        return cls(
            pair_name=d["pair_name"],
            left_id=d["left_id"],
            right_id=d["right_id"],
            left_intrinsics=CameraIntrinsics.from_dict(d["left_intrinsics"]),
            right_intrinsics=CameraIntrinsics.from_dict(d["right_intrinsics"]),
            stereo=StereoCalibResult.from_dict(d["stereo"]),
        )


@dataclass
class MultiCameraRig:
    """Full multi-camera rig calibration: all pairs + global extrinsics."""

    pairs: Dict[str, StereoPairCalibration] = field(default_factory=dict)
    extrinsics: Dict[str, CameraExtrinsics] = field(default_factory=dict)
    reference_camera: str = ""

    def get_pair(self, name: str) -> Optional[StereoPairCalibration]:
        return self.pairs.get(name)

    def get_extrinsics(self, camera_id: str) -> Optional[CameraExtrinsics]:
        return self.extrinsics.get(camera_id)

    def to_dict(self) -> dict:
        return {
            "reference_camera": self.reference_camera,
            "pairs": {k: v.to_dict() for k, v in self.pairs.items()},
            "extrinsics": {k: v.to_dict() for k, v in self.extrinsics.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> MultiCameraRig:
        return cls(
            reference_camera=d.get("reference_camera", ""),
            pairs={
                k: StereoPairCalibration.from_dict(v)
                for k, v in d.get("pairs", {}).items()
            },
            extrinsics={
                k: CameraExtrinsics.from_dict(v)
                for k, v in d.get("extrinsics", {}).items()
            },
        )
