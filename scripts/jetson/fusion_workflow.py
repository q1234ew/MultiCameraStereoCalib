"""Migration adapter: metric rectified RGB-D -> Owen point-cloud/TSDF pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .core import load_config
from .fusion_preview import load_cloud_preview
from .fusion_geometry import (
    aggregate_poses,
    check_anchor_depth,
    detect_tags,
    pose_from_tag,
    rectify_for_tags,
    validate_active,
)

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor/owen_fusion"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def calibrate(config_path, bundle_root, out, ids, side_mm=80.0, tag_id=-1, samples=3):
    config = load_config(config_path)
    devices = [d for d in config["devices"] if d["id"] in ids]
    if len(devices) != len(ids) or not 2 <= len(devices) <= 3 or not 3 <= samples <= 8:
        raise ValueError("请选择2–3台相机，采样帧数3–8")
    if not 10 <= side_mm <= 1000 or not -1 <= tag_id <= 586:
        raise ValueError("AprilTag 尺寸或编号无效")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    observations = {d["id"]: [] for d in devices}
    visible = {d["id"]: set() for d in devices}
    for sample in range(samples):
        for device in devices:
            best = {}
            path = Path(bundle_root) / device["id"] / "calibration_mm.yml"
            for role in ("left", "right"):
                try:
                    url = f"http://{device['host']}:{device['raw_ports'][role]}/snapshot"
                    with opener.open(url, timeout=4) as response:
                        raw = response.read(4 * 1024 * 1024 + 1)
                    if len(raw) > 4 * 1024 * 1024:
                        raise ValueError("Image exceeds limit")
                    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("Camera image cannot be decoded")
                    for zoom in (1, 2):
                        rect, k, right_from_left = rectify_for_tags(device, path, role, image, zoom)
                        tags = detect_tags(rect)
                        visible[device["id"]].update(tags)
                        for ident, corners in tags.items():
                            try:
                                world_camera, error = pose_from_tag(corners, k, side_mm * 0.001)
                            except ValueError:
                                continue
                            if role == "right":
                                world_camera = world_camera @ right_from_left
                            if error > 2:
                                continue
                            if ident not in best or error < best[ident]["error"]:
                                best[ident] = {"T": world_camera, "error": error, "role": role}
                                annotated = rect.copy()
                                cv2.aruco.drawDetectedMarkers(
                                    annotated,
                                    [corners.astype(np.float32).reshape(1, 4, 2)],
                                    np.array([[ident]], np.int32),
                                )
                                cv2.imwrite(
                                    str(out / f"{device['id']}_{sample}_tag{ident}.jpg"), annotated
                                )
                except (OSError, ValueError, cv2.error) as exc:
                    print(device["id"], role, str(exc), flush=True)
            observations[device["id"]].append(best)
        time.sleep(0.1)
    common = set.intersection(
        *(set().union(*(set(x) for x in obs)) for obs in observations.values())
    )
    if tag_id >= 0:
        common &= {tag_id}
    result = {
        "source": "foundation_stereo_rectified",
        "pose_solver": "IPPE_two_branch_LM_ambiguity_checked",
        "translation_unit": "m",
        "tag_family": "36h11",
        "tag_side_mm": side_mm,
        "anchor_tag_id": None,
        "quality_passed": False,
        "measurement_accuracy_verified": False,
        "created_at_unix_ms": round(time.time() * 1000),
        "cameras": {},
        "visible_ids": {k: sorted(v) for k, v in visible.items()},
    }
    if not common:
        result["error"] = "没有两台设备共同识别的同一 AprilTag；请把实测尺寸的码朝向各相机并保持静止"
        write_json(out / "calibration_report.json", result)
        return result
    def anchor_score(ident):
        groups = [[x[ident] for x in obs if ident in x] for obs in observations.values()]
        # Prefer coverage across every camera, then the clearest unambiguous marker.
        return (-min(len(group) for group in groups),
                float(np.median([x["error"] for group in groups for x in group])))
    chosen = min(common, key=anchor_score)
    result["anchor_tag_id"] = chosen
    result["world_frame"] = f"AprilTag36h11:{chosen}:top_left_m"
    passed = True
    for device in devices:
        obs = [x[chosen] for x in observations[device["id"]] if chosen in x]
        if not obs:
            passed = False
            continue
        transform, stats, good = aggregate_poses([x["T"] for x in obs], [x["error"] for x in obs])
        result["cameras"][device["id"]] = {
            "T_world_camera": transform.tolist(),
            "T_camera_world": np.linalg.inv(transform).tolist(),
            "calibration_sha256": device["calibration_sha256"],
            "point_frame": device["id"] + ":C_L_rect",
            "quality_passed": good,
            "statistics": stats,
            "observation_roles": [x["role"] for x in obs],
        }
        passed &= good
    result["quality_passed"] = bool(passed and len(result["cameras"]) == len(devices))
    if not result["quality_passed"]:
        result["error"] = "标定样本不足或位姿稳定性未通过门槛，保留候选结果但不启用融合"
    write_json(out / "calibration_report.json", result)
    return result


def run_fusion(captures, extrinsics, out, frames=3, min_depth=0.2, max_depth=10,
               build_mesh=True, tolerance_profile='strict'):
    """Run the migrated algorithms on registered RGB-D, without any camera SDK dependency."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    reference = json.loads(Path(extrinsics).read_text())
    alignment = check_anchor_depth(captures, reference, tolerance_profile)
    write_json(out / "alignment_report.json", alignment)
    if not alignment["passed"]:
        raise ValueError("标记外参与深度的尺度或方向不一致，暂停融合；请核对黑色码尺寸、标定与深度")
    common = [
        "--captures",
        *[str(Path(p).resolve()) for p in captures],
        "--extrinsics",
        str(Path(extrinsics).resolve()),
        "--frame-count",
        str(frames),
        "--min-depth-m",
        str(min_depth),
        "--min-valid-frame-ratio",
        "0.5",
        "--max-depth-mad-m",
        "0.05",
        "--edge-threshold-m",
        "0.08",
    ]
    jobs = [
        (
            "pointcloud",
            VENDOR / "make_multicamera_pointcloud.py",
            [
                "--max-depth-m",
                str(max_depth),
                "--voxel-m",
                "0.005",
                "--html-points",
                "12000",
                "--stride",
                "1",
            ],
        ),
        (
            "tsdf",
            VENDOR / "make_tsdf_mesh.py",
            [
                "--depth-trunc-m",
                str(max_depth),
                "--voxel-m",
                "0.01",
                "--sdf-trunc-m",
                "0.04",
                "--min-cluster-triangles",
                "50",
            ],
        ),
    ]
    if not build_mesh:
        jobs = jobs[:1]
    for name, script, extra in jobs:
        cmd = [sys.executable, str(script), *common, "--out", str(out / name), *extra]
        env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
        with (out / f"{name}.log").open("w") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=300)
        if proc.returncode:
            raise ValueError(f"{name} 融合失败，请检查任务日志")
    preview = load_cloud_preview(out / "pointcloud", reference["cameras"])
    result = {
        "alignment_report": alignment,
        "tolerance_profile": tolerance_profile,
        "strict_alignment_passed": alignment.get("strict_passed", alignment["passed"]),
        "preview_only": tolerance_profile == "preview",
        **preview,
        "point_unit": "m",
        "world_frame": reference.get(
            "world_frame", reference.get("reference", {}).get("world_frame", "supplied_world_frame")
        ),
        "measurement_accuracy_verified": False,
        "tsdf": json.loads((out / "tsdf/summary.json").read_text()) if build_mesh else None,
        "build_mesh": build_mesh,
        "pointcloud": json.loads((out / "pointcloud/summary.json").read_text()),
    }
    write_json(out / "fusion_result.json", result)
    return result


def validate_capture_calibration(calibration, config, ids):
    devices = [d for d in config["devices"] if d["id"] in ids]
    if len(devices) != len(ids) or not 2 <= len(devices) <= 3:
        raise ValueError("请选择2–3台设备")
    validate_active(calibration, devices)
    return devices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--config", type=Path, default=Path("packaging/jetson/devices.json"))
    cal.add_argument("--bundle-root", type=Path, default=Path("bundles"))
    cal.add_argument("--devices", nargs="+", required=True)
    cal.add_argument("--tag-size-mm", type=float, default=80)
    cal.add_argument("--tag-id", type=int, default=-1)
    cal.add_argument("--samples", type=int, default=3)
    cal.add_argument("--out", type=Path, required=True)
    fuse = sub.add_parser("fuse")
    fuse.add_argument("--captures", nargs="+", type=Path, required=True)
    fuse.add_argument("--extrinsics", type=Path, required=True)
    fuse.add_argument("--frames", type=int, default=3)
    fuse.add_argument("--max-depth-m", type=float, default=10)
    fuse.add_argument("--out", type=Path, required=True)
    fuse.add_argument("--pointcloud-only", action="store_true")
    fuse.add_argument("--tolerance-profile", choices=["strict", "preview"], default="strict")
    args = parser.parse_args()
    if args.action == "calibrate":
        result = calibrate(
            args.config,
            args.bundle_root,
            args.out,
            args.devices,
            args.tag_size_mm,
            args.tag_id,
            args.samples,
        )
        print(
            json.dumps(
                {
                    "quality_passed": result["quality_passed"],
                    "visible_ids": result["visible_ids"],
                    "error": result.get("error"),
                }
            )
        )
        return 0 if result["quality_passed"] else 2
    result = run_fusion(
        args.captures, args.extrinsics, args.out, args.frames, max_depth=args.max_depth_m,
        build_mesh=not args.pointcloud_only, tolerance_profile=args.tolerance_profile
    )
    print(json.dumps({"point_count": result["point_count"], "world_frame": result["world_frame"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
