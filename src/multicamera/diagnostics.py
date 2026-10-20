"""OpenCV runtime evidence without importing the GUI or optional point cloud code."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

OPENCV_DISTRIBUTIONS = (
    "opencv-contrib-python", "opencv-python",
    "opencv-contrib-python-headless", "opencv-python-headless",
)
REQUIRED_APIS = (
    "calibrateCamera", "stereoCalibrate", "stereoRectify", "projectPoints",
    "solvePnP", "FileStorage", "initUndistortRectifyMap",
    "aruco.CharucoBoard", "aruco.CharucoDetector", "aruco.DetectorParameters",
    "aruco.CharucoParameters", "aruco.getPredefinedDictionary",
    "aruco.drawDetectedCornersCharuco", "fisheye.calibrate", "fisheye.stereoCalibrate",
)
LEGACY_API = "aruco.calibrateCameraCharuco"


def opencv_runtime_info(module=None, *, hash_native: bool = False) -> dict:
    if module is None:
        import cv2 as module

    def available(name):
        obj = module
        for part in name.split("."):
            obj = getattr(obj, part, None)
        return callable(obj)

    distributions = {}
    for name in OPENCV_DISTRIBUTIONS:
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    native = getattr(module, "_native", module)
    native_path = Path(native.__file__).resolve()
    info = {
        "version": module.__version__, "module_path": str(Path(module.__file__).resolve()),
        "native_path": str(native_path), "distributions": distributions,
        "apis": {name: available(name) for name in (*REQUIRED_APIS, LEGACY_API)},
    }
    if hash_native:
        with native_path.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            info["native_sha256"] = digest.hexdigest()
        info["build_information"] = module.getBuildInformation()
    return info
