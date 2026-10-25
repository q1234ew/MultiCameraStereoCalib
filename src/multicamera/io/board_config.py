"""Read independent auxiliary-board specifications, never camera K/D or extrinsics."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from ..calibration.planar import PatternBoardConfig


def load_pattern_board_config(path: str | Path) -> PatternBoardConfig:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("Board JSON must contain an object")
        if isinstance(data.get("auxiliary_camera"), dict):
            data = data["auxiliary_camera"]
        elif "aux_cameras" in data:
            cameras = data["aux_cameras"]
            if not isinstance(cameras, dict) or len(cameras) != 1:
                raise ValueError("Select a board-only file when the rig has zero or multiple AUX cameras")
            data = next(iter(cameras.values()))
            if not isinstance(data, dict):
                raise ValueError("Invalid auxiliary camera entry")
        elif isinstance(data.get("multimodal_result"), dict):
            data = data["multimodal_result"]
        data = data.get("board_config", data)
    else:
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            if not fs.isOpened():
                raise ValueError("Cannot open OpenCV YAML board file")
            root = fs.getNode("board_config")
            if root.empty():
                root = fs.root()
            data = {}
            for key in ("cols", "rows", "board_cols", "board_rows", "square_size",
                        "square_mm", "pattern_type", "asymmetric"):
                node = root.getNode(key)
                if node.empty():
                    continue
                value = node.string() if node.isString() else node.real()
                if key == "asymmetric":
                    if value not in (0, 1, "true", "false"):
                        raise ValueError("asymmetric must be 0/1 or true/false")
                    value = value in (1, "true")
                data[key] = value
        finally:
            fs.release()
    return PatternBoardConfig.from_dict(data)
