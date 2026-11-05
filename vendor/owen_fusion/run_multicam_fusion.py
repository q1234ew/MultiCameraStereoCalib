#!/usr/bin/env python3
"""Run fixed multi-Orbbec capture, point-cloud fusion, and TSDF fusion."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent


def resolve_path(value: str | Path, base: Path = REPO_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(cmd: list[str], cwd: Path, log_path: Path) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nLog: {log_path}\n{proc.stdout}")
    return proc


def list_devices(capture_tool: Path, log_path: Path) -> dict[str, str]:
    proc = run_command([str(capture_tool), "--list-devices"], cwd=REPO_ROOT, log_path=log_path)
    devices = {}
    for line in proc.stdout.splitlines():
        match = re.search(r"name=(.*?) serial=([^ ]+) firmware=([^ ]+)", line)
        if match:
            name, serial, firmware = match.groups()
            devices[serial] = f"{name} {firmware}"
    return devices


def capture_camera(capture_tool: Path, serial: str, raw_dir: Path, settings: dict[str, object], log_path: Path) -> dict[str, object]:
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(capture_tool),
        "--serial",
        serial,
        "--out-dir",
        str(raw_dir),
        "--frames",
        str(int(settings["frames"])),
        "--drop-frames",
        str(int(settings["drop_frames"])),
        "--timeout-ms",
        str(int(settings["timeout_ms"])),
        "--imu-rate-hz",
        str(int(settings["imu_rate_hz"])),
    ]
    run_command(cmd, cwd=REPO_ROOT, log_path=log_path)
    color_count = len(list((raw_dir / "color").glob("*.png")))
    depth_count = len(list((raw_dir / "depth").glob("*.png")))
    camera_json = raw_dir / "camera.json"
    if not camera_json.exists():
        raise RuntimeError(f"Capture did not write camera.json for {serial}: {raw_dir}")
    return {"serial": serial, "raw_dir": str(raw_dir), "color_frames": color_count, "depth_frames": depth_count}


def capture_all_cameras(
    capture_tool: Path,
    cameras: list[dict[str, object]],
    raw_root: Path,
    capture_settings: dict[str, object],
    logs_dir: Path,
) -> tuple[list[dict[str, object]], str]:
    errors = []
    results = []
    with ThreadPoolExecutor(max_workers=len(cameras)) as executor:
        futures = {
            executor.submit(
                capture_camera,
                capture_tool,
                str(camera["serial"]),
                raw_root / str(camera["serial"]),
                capture_settings,
                logs_dir / f"capture_{camera['serial']}.log",
            ): camera
            for camera in cameras
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 - preserve all capture failures for fallback logging.
                errors.append(exc)

    if not errors:
        return results, "parallel"

    fallback_log = logs_dir / "capture_parallel_fallback.log"
    fallback_log.write_text(
        "\n\n".join(f"Parallel capture failed, will retry sequentially:\n{error}" for error in errors),
        encoding="utf-8",
    )
    for camera in cameras:
        raw_dir = raw_root / str(camera["serial"])
        if raw_dir.exists():
            shutil.rmtree(raw_dir)

    results = []
    for camera in cameras:
        serial = str(camera["serial"])
        results.append(
            capture_camera(
                capture_tool,
                serial,
                raw_root / serial,
                capture_settings,
                logs_dir / f"capture_{serial}_sequential.log",
            )
        )
    return results, "sequential_fallback"


def validate_capture(raw_dir: Path, expected_frames: int) -> dict[str, object]:
    camera = load_json(raw_dir / "camera.json")
    color_files = sorted((raw_dir / "color").glob("*.png"))
    depth_files = sorted((raw_dir / "depth").glob("*.png"))
    if len(color_files) != expected_frames or len(depth_files) != expected_frames:
        raise RuntimeError(f"Expected {expected_frames} RGB-D frames in {raw_dir}, got color={len(color_files)} depth={len(depth_files)}")
    color = camera["color"]
    depth = camera["depth"]
    if int(color["width"]) != 1280 or int(color["height"]) != 800:
        raise RuntimeError(f"Unexpected color resolution in {raw_dir}: {color['width']}x{color['height']}")
    if int(depth["width"]) != 1280 or int(depth["height"]) != 800:
        raise RuntimeError(f"Unexpected depth resolution in {raw_dir}: {depth['width']}x{depth['height']}")
    return {
        "serial": camera["device"]["serial_number"],
        "color_resolution": [int(color["width"]), int(color["height"])],
        "depth_resolution": [int(depth["width"]), int(depth["height"])],
        "frames": expected_frames,
    }


def make_output_dir(config: dict[str, object], output: str | None) -> Path:
    if output:
        out = resolve_path(output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = resolve_path(str(config["output_root"])) / timestamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=THIS_DIR / "deployment.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-capture", action="store_true", help="Reuse raw/ directories already present in the output directory.")
    parser.add_argument("--frames", type=int, default=None, help="Override deployment capture.frames.")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    config = load_json(config_path)
    capture_settings = dict(config["capture"])
    if args.frames is not None:
        capture_settings["frames"] = args.frames
    fusion = config["fusion"]
    calibration_eval = config.get("calibration_evaluation", {"enabled": False})
    thresholds = config["quality_thresholds"]
    cameras = list(config["cameras"])

    out_dir = make_output_dir(config, args.output)
    logs_dir = out_dir / "logs"
    raw_root = out_dir / "raw"
    pointcloud_dir = out_dir / "pointcloud_fine_5mm"
    tsdf_dir = out_dir / "tsdf_1cm"
    calibration_dir = out_dir / "calibration_eval"
    raw_root.mkdir(parents=True, exist_ok=True)

    capture_tool = resolve_path(str(config["capture_tool"]))
    extrinsics = resolve_path(str(config["extrinsics"]))
    landmark_boards = resolve_path(str(config.get("landmark_boards", ""))) if config.get("landmark_boards") else None
    board_extrinsics = resolve_path(str(config.get("board_extrinsics", ""))) if config.get("board_extrinsics") else None
    if not capture_tool.exists():
        raise FileNotFoundError(capture_tool)
    if not extrinsics.exists():
        raise FileNotFoundError(extrinsics)

    available_devices = list_devices(capture_tool, logs_dir / "list_devices.log")
    missing = [camera["serial"] for camera in cameras if camera["serial"] not in available_devices]
    if missing and not args.skip_capture:
        raise RuntimeError(f"Missing required Orbbec device(s): {missing}. Available: {sorted(available_devices)}")

    capture_results = []
    capture_mode = "skip_capture"
    if args.skip_capture:
        for camera in cameras:
            serial = camera["serial"]
            capture_results.append({"serial": serial, "raw_dir": str(raw_root / serial), "color_frames": None, "depth_frames": None})
    else:
        capture_results, capture_mode = capture_all_cameras(capture_tool, cameras, raw_root, capture_settings, logs_dir)

    raw_dirs = [raw_root / str(camera["serial"]) for camera in cameras]
    capture_validation = [validate_capture(raw_dir, int(capture_settings["frames"])) for raw_dir in raw_dirs]

    python_exe = sys.executable
    pointcloud_script = THIS_DIR / "make_multicamera_pointcloud.py"
    tsdf_script = THIS_DIR / "make_tsdf_mesh.py"

    pointcloud_cmd = [
        python_exe,
        str(pointcloud_script),
        "--captures",
        *[str(p) for p in raw_dirs],
        "--extrinsics",
        str(extrinsics),
        "--out",
        str(pointcloud_dir),
        "--frame-start",
        "0",
        "--frame-count",
        str(int(capture_settings["frames"])),
        "--stride",
        str(int(fusion["pointcloud_stride"])),
        "--min-depth-m",
        str(float(fusion["min_depth_m"])),
        "--max-depth-m",
        str(float(fusion["max_depth_m"])),
        "--voxel-m",
        str(float(fusion["pointcloud_voxel_m"])),
        "--html-points",
        str(int(fusion["pointcloud_html_points"])),
        "--min-valid-frame-ratio",
        str(float(fusion["min_valid_frame_ratio"])),
        "--max-depth-mad-m",
        str(float(fusion["max_depth_mad_m"])),
        "--edge-threshold-m",
        str(float(fusion["edge_threshold_m"])),
    ]
    run_command(pointcloud_cmd, cwd=REPO_ROOT, log_path=logs_dir / "pointcloud_fusion.log")

    tsdf_cmd = [
        python_exe,
        str(tsdf_script),
        "--captures",
        *[str(p) for p in raw_dirs],
        "--extrinsics",
        str(extrinsics),
        "--out",
        str(tsdf_dir),
        "--frame-start",
        "0",
        "--frame-count",
        str(int(capture_settings["frames"])),
        "--voxel-m",
        str(float(fusion["tsdf_voxel_m"])),
        "--min-depth-m",
        str(float(fusion["min_depth_m"])),
        "--sdf-trunc-m",
        str(float(fusion["tsdf_sdf_trunc_m"])),
        "--depth-trunc-m",
        str(float(fusion["max_depth_m"])),
        "--min-cluster-triangles",
        str(int(fusion["tsdf_min_cluster_triangles"])),
        "--min-valid-frame-ratio",
        str(float(fusion["min_valid_frame_ratio"])),
        "--max-depth-mad-m",
        str(float(fusion["max_depth_mad_m"])),
        "--edge-threshold-m",
        str(float(fusion["edge_threshold_m"])),
    ]
    run_command(tsdf_cmd, cwd=REPO_ROOT, log_path=logs_dir / "tsdf_fusion.log")

    pointcloud_summary = load_json(pointcloud_dir / "summary.json")
    tsdf_summary = load_json(tsdf_dir / "summary.json")
    calibration_summary = None
    if calibration_eval.get("enabled", False):
        if landmark_boards is None or board_extrinsics is None:
            raise RuntimeError("calibration_evaluation is enabled but landmark_boards/board_extrinsics are missing from deployment.json")
        if not landmark_boards.exists():
            raise FileNotFoundError(landmark_boards)
        if not board_extrinsics.exists():
            raise FileNotFoundError(board_extrinsics)
        calibration_script = THIS_DIR / "evaluate_calibration_quality.py"
        calibration_cmd = [
            python_exe,
            str(calibration_script),
            "--captures",
            *[str(p) for p in raw_dirs],
            "--boards",
            str(landmark_boards),
            "--board-extrinsics",
            str(board_extrinsics),
            "--camera-extrinsics",
            str(extrinsics),
            "--out",
            str(calibration_dir),
            "--frame-start",
            "0",
            "--frame-count",
            str(int(capture_settings["frames"])),
            "--frame-step",
            str(int(calibration_eval["frame_step"])),
            "--min-tags",
            str(int(calibration_eval["min_tags"])),
            "--max-reproj-px",
            str(float(calibration_eval["max_reprojection_error_px"])),
        ]
        run_command(calibration_cmd, cwd=REPO_ROOT, log_path=logs_dir / "calibration_eval.log")
        calibration_summary = load_json(calibration_dir / "summary.json")

    quality = {
        "passed": True,
        "checks": [],
    }
    for camera in pointcloud_summary["cameras"]:
        serial = camera["serial"]
        valid_ratio = float(camera["depth_summary"]["median_valid_depth_ratio"])
        filtered_ratio = float(camera["filter_summary"]["final_valid_ratio"])
        points = int(camera["input_valid_points"])
        ratio_ok = valid_ratio >= float(thresholds["min_depth_valid_ratio"])
        filtered_ratio_ok = filtered_ratio >= float(thresholds["min_filtered_depth_valid_ratio"])
        points_ok = points >= int(thresholds["min_projected_points_per_camera"])
        quality["checks"].append({"serial": serial, "metric": "median_valid_depth_ratio", "value": valid_ratio, "passed": ratio_ok})
        quality["checks"].append({"serial": serial, "metric": "filtered_depth_valid_ratio", "value": filtered_ratio, "passed": filtered_ratio_ok})
        quality["checks"].append({"serial": serial, "metric": "input_valid_points", "value": points, "passed": points_ok})
        quality["passed"] = bool(quality["passed"] and ratio_ok and filtered_ratio_ok and points_ok)
    mesh_ok = int(tsdf_summary["clean_stats"]["triangles_after_clean"]) > 0
    quality["checks"].append({"metric": "tsdf_clean_triangles", "value": int(tsdf_summary["clean_stats"]["triangles_after_clean"]), "passed": mesh_ok})
    quality["passed"] = bool(quality["passed"] and mesh_ok)
    if calibration_summary is not None:
        if calibration_summary["status"] == "ok":
            metrics = calibration_summary["aggregate_metrics"]
            reproj = metrics["reprojection_error_px_median"]
            trans = metrics["translation_error_m_p90"]
            rot = metrics["rotation_error_deg_p90"]
            reproj_ok = reproj is not None and reproj <= float(thresholds["max_calibration_reprojection_median_px"])
            trans_ok = trans is not None and trans <= float(thresholds["max_calibration_translation_p90_m"])
            rot_ok = rot is not None and rot <= float(thresholds["max_calibration_rotation_p90_deg"])
            quality["checks"].append({"metric": "calibration_reprojection_error_px_median", "value": reproj, "passed": reproj_ok})
            quality["checks"].append({"metric": "calibration_translation_error_m_p90", "value": trans, "passed": trans_ok})
            quality["checks"].append({"metric": "calibration_rotation_error_deg_p90", "value": rot, "passed": rot_ok})
            quality["passed"] = bool(quality["passed"] and reproj_ok and trans_ok and rot_ok)
        else:
            quality["checks"].append({"metric": "calibration_eval", "value": calibration_summary["status"], "passed": True, "note": "No visible landmarks; fusion result is not failed."})

    summary = {
        "config": str(config_path),
        "output": str(out_dir),
        "available_devices": available_devices,
        "capture": capture_results,
        "capture_mode": capture_mode,
        "capture_validation": capture_validation,
        "extrinsics": str(extrinsics),
        "pointcloud": pointcloud_summary,
        "tsdf": tsdf_summary,
        "calibration_eval": calibration_summary,
        "quality": quality,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_dir), "quality_passed": quality["passed"], "summary": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
