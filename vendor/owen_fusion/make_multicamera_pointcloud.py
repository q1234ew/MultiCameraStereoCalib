#!/usr/bin/env python3
"""Fuse multiple Orbbec RGB-D captures into one world-frame RGB point cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw


def load_camera(capture_dir: Path) -> dict[str, object]:
    path = capture_dir / "camera.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_extrinsics(path: Path) -> dict[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    poses = {}
    for serial, payload in data["cameras"].items():
        t = payload.get("T_world_camera")
        if t is not None:
            poses[serial] = np.asarray(t, dtype=np.float64)
    if not poses:
        raise ValueError(f"No camera poses found in {path}")
    return poses


def frame_name(index: int) -> str:
    return f"{index:06d}.png"


def resolve_frames(capture_dir: Path, frame: str | None, frame_start: int, frame_count: int) -> list[str]:
    if frame_count <= 1:
        return [frame or frame_name(frame_start)]
    frames = [frame_name(i) for i in range(frame_start, frame_start + frame_count)]
    missing = [name for name in frames if not (capture_dir / "depth" / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing depth frames in {capture_dir}: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    return frames


def depth_edge_keep_mask(depth: np.ndarray, valid: np.ndarray, threshold_m: float) -> np.ndarray:
    if threshold_m <= 0:
        return valid.copy()
    keep = valid.copy()
    horizontal = valid[:, 1:] & valid[:, :-1] & (np.abs(depth[:, 1:] - depth[:, :-1]) > threshold_m)
    vertical = valid[1:, :] & valid[:-1, :] & (np.abs(depth[1:, :] - depth[:-1, :]) > threshold_m)
    right = keep[:, 1:]
    left = keep[:, :-1]
    right[horizontal] = False
    left[horizontal] = False
    keep[:, 1:] = right
    keep[:, :-1] = left
    bottom = keep[1:, :]
    top = keep[:-1, :]
    bottom[vertical] = False
    top[vertical] = False
    keep[1:, :] = bottom
    keep[:-1, :] = top
    return keep


def load_depth_median(capture_dir: Path, frames: list[str], depth_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    depths = []
    valid_counts = []
    for name in frames:
        depth_path = capture_dir / "depth" / name
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(depth_path)
        depth_m = depth_raw.astype(np.float32) * depth_scale
        valid_counts.append(int(np.count_nonzero(depth_raw)))
        depths.append(depth_m)
    if len(depths) == 1:
        depth = depths[0]
        support_ratio = (depth > 0).astype(np.float32)
        mad = np.zeros_like(depth, dtype=np.float32)
    else:
        stack = np.stack(depths, axis=0)
        valid_stack = stack > 0
        support_ratio = valid_stack.mean(axis=0).astype(np.float32)
        stack[~valid_stack] = np.nan
        depth = np.nanmedian(stack, axis=0).astype(np.float32)
        mad = np.nanmedian(np.abs(stack - depth[None, :, :]), axis=0).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        mad[~np.isfinite(mad)] = np.inf
    total = int(depth.size)
    summary = {
        "frames": frames,
        "frame_count": len(frames),
        "per_frame_valid_depth_ratio_min": float(min(valid_counts) / total) if total else 0.0,
        "per_frame_valid_depth_ratio_max": float(max(valid_counts) / total) if total else 0.0,
        "median_valid_depth_ratio": float(np.count_nonzero(depth > 0) / total) if total else 0.0,
        "support_ratio_median": float(np.median(support_ratio)) if total else 0.0,
        "depth_mad_median_m": float(np.median(mad[np.isfinite(mad)])) if np.any(np.isfinite(mad)) else None,
        "depth_mad_p90_m": float(np.percentile(mad[np.isfinite(mad)], 90)) if np.any(np.isfinite(mad)) else None,
    }
    return depth, support_ratio, mad, summary


def make_depth_quality_mask(
    depth: np.ndarray,
    support_ratio: np.ndarray,
    mad: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    min_valid_frame_ratio: float,
    max_depth_mad_m: float,
    edge_threshold_m: float,
) -> tuple[np.ndarray, dict[str, object]]:
    base = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    support_ok = support_ratio >= min_valid_frame_ratio
    mad_ok = mad <= max_depth_mad_m if max_depth_mad_m > 0 else np.ones_like(base, dtype=bool)
    before_edge = base & support_ok & mad_ok
    after_edge = depth_edge_keep_mask(depth, before_edge, edge_threshold_m)
    total = int(depth.size)
    range_valid_pixels = int(np.count_nonzero(base))
    support_valid_pixels = int(np.count_nonzero(base & support_ok))
    confidence_valid_pixels = int(np.count_nonzero(base & support_ok & mad_ok))
    final_valid_pixels = int(np.count_nonzero(after_edge))
    stats = {
        "pixels_total": total,
        "range_valid_pixels": range_valid_pixels,
        "support_valid_pixels": support_valid_pixels,
        "mad_valid_pixels": confidence_valid_pixels,
        "temporal_support_filtered_pixels": range_valid_pixels - support_valid_pixels,
        "temporal_mad_filtered_pixels": support_valid_pixels - confidence_valid_pixels,
        "confidence_filtered_pixels": range_valid_pixels - confidence_valid_pixels,
        "edge_filtered_pixels": confidence_valid_pixels - final_valid_pixels,
        "final_valid_pixels": final_valid_pixels,
        "final_valid_ratio": float(final_valid_pixels / total) if total else 0.0,
        "min_valid_frame_ratio": min_valid_frame_ratio,
        "max_depth_mad_m": max_depth_mad_m,
        "edge_threshold_m": edge_threshold_m,
    }
    return after_edge, stats


def deproject_depth(
    depth_m: np.ndarray,
    quality_mask: np.ndarray,
    intrinsic: dict[str, float],
    distortion: dict[str, float],
    stride: int,
    min_depth_m: float,
    max_depth_m: float,
    use_distortion: bool,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth_m.shape
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    z = depth_m[ys, xs].astype(np.float32)
    valid = quality_mask[ys, xs] & np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)

    pix = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=-1).reshape(-1, 1, 2)
    if use_distortion:
        k = np.asarray(
            [
                [float(intrinsic["fx"]), 0.0, float(intrinsic["cx"])],
                [0.0, float(intrinsic["fy"]), float(intrinsic["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist = np.asarray(
            [
                float(distortion.get("k1", 0.0)),
                float(distortion.get("k2", 0.0)),
                float(distortion.get("p1", 0.0)),
                float(distortion.get("p2", 0.0)),
                float(distortion.get("k3", 0.0)),
                float(distortion.get("k4", 0.0)),
                float(distortion.get("k5", 0.0)),
                float(distortion.get("k6", 0.0)),
            ],
            dtype=np.float64,
        )
        rays = cv2.undistortPoints(pix, k, dist).reshape(ys.shape[0], ys.shape[1], 2).astype(np.float32)
        x_norm = rays[..., 0]
        y_norm = rays[..., 1]
    else:
        x_norm = (xs.astype(np.float32) - float(intrinsic["cx"])) / float(intrinsic["fx"])
        y_norm = (ys.astype(np.float32) - float(intrinsic["cy"])) / float(intrinsic["fy"])

    points = np.stack([x_norm * z, y_norm * z, z], axis=-1)
    return points.reshape(-1, 3), valid.reshape(-1)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    ones = np.ones((len(points), 1), dtype=np.float64)
    hom = np.concatenate([points.astype(np.float64), ones], axis=1)
    return (transform @ hom.T).T[:, :3].astype(np.float32)


def voxel_downsample(points: np.ndarray, colors: np.ndarray, labels: np.ndarray, voxel_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if voxel_m <= 0 or len(points) == 0:
        return points, colors, labels
    keys = np.floor(points / voxel_m).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    unique_idx.sort()
    return points[unique_idx], colors[unique_idx], labels[unique_idx]


def write_ply_binary(path: Path, points: np.ndarray, colors: np.ndarray, labels: np.ndarray) -> None:
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("camera_id", "<i4"),
        ]
    )
    vertices = np.empty(len(points), dtype=vertex_dtype)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]
    vertices["camera_id"] = labels

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property int camera_id",
            "end_header",
            "",
        ]
    ).encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        vertices.tofile(f)


def make_html(path: Path, points: np.ndarray, colors: np.ndarray, labels: np.ndarray, serials: list[str], max_points: int) -> None:
    if len(points) == 0:
        return
    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        pts = points[idx]
        cols = colors[idx]
        labs = labels[idx]
    else:
        pts, cols, labs = points, colors, labels

    fig = go.Figure()
    for camera_id, serial in enumerate(serials):
        mask = labs == camera_id
        if not np.any(mask):
            continue
        c = cols[mask]
        color_strings = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in c]
        fig.add_trace(
            go.Scatter3d(
                x=pts[mask, 0],
                y=pts[mask, 1],
                z=pts[mask, 2],
                mode="markers",
                marker=dict(size=1.2, color=color_strings, opacity=0.95),
                name=serial,
            )
        )
    fig.update_layout(
        title="Multi-Orbbec World RGB Point Cloud",
        scene=dict(aspectmode="data", xaxis_title="world x (m)", yaxis_title="world y (m)", zaxis_title="world z (m)"),
        margin=dict(l=0, r=0, t=36, b=0),
    )
    fig.write_html(path, include_plotlyjs=True)


def make_preview_png(path: Path, points: np.ndarray, colors: np.ndarray, axes: tuple[int, int], title: str, max_points: int = 160000) -> None:
    if len(points) == 0:
        return
    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        pts = points[idx]
        cols = colors[idx]
    else:
        pts, cols = points, colors

    width, height = 1200, 900
    margin = 40
    xy = pts[:, list(axes)]
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    keep = np.all((xy >= lo) & (xy <= hi), axis=1)
    xy = xy[keep]
    cols = cols[keep]
    px = ((xy[:, 0] - lo[0]) / span[0] * (width - 2 * margin) + margin).astype(np.int32)
    py = (height - margin - (xy[:, 1] - lo[1]) / span[1] * (height - 2 * margin)).astype(np.int32)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for x, y, color in zip(px, py, cols):
        draw.point((int(x), int(y)), fill=tuple(int(c) for c in color))
    draw.rectangle((margin, margin, width - margin, height - margin), outline=(80, 80, 80), width=1)
    draw.text((margin, 12), title, fill=(0, 0, 0))
    draw.text((margin, height - 28), f"axis {axes[0]} range {lo[0]:.2f}..{hi[0]:.2f} m, axis {axes[1]} range {lo[1]:.2f}..{hi[1]:.2f} m", fill=(0, 0, 0))
    image.save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures", nargs="+", required=True, type=Path)
    parser.add_argument("--extrinsics", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frame", default=None, help="Single frame name for backwards-compatible one-frame fusion.")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=1)
    parser.add_argument("--color-frame", default=None, help="RGB frame used for coloring; defaults to middle selected frame.")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--min-depth-m", type=float, default=0.3)
    parser.add_argument("--max-depth-m", type=float, default=6.0)
    parser.add_argument("--voxel-m", type=float, default=0.01)
    parser.add_argument("--html-points", type=int, default=90000)
    parser.add_argument("--min-valid-frame-ratio", type=float, default=0.5)
    parser.add_argument("--max-depth-mad-m", type=float, default=0.05)
    parser.add_argument("--edge-threshold-m", type=float, default=0.08)
    parser.add_argument("--ignore-distortion", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    poses = load_extrinsics(args.extrinsics)

    all_points = []
    all_colors = []
    all_labels = []
    serials = []
    cameras_summary = []

    for camera_id, capture_dir in enumerate(args.captures):
        camera = load_camera(capture_dir)
        serial = camera["device"]["serial_number"]
        if serial not in poses:
            print(f"Skipping {serial}: no T_world_camera in extrinsics")
            continue
        serials.append(serial)
        selected_frames = resolve_frames(capture_dir, args.frame, args.frame_start, args.frame_count)
        color_frame = args.color_frame or selected_frames[len(selected_frames) // 2]
        color_path = capture_dir / "color" / color_frame
        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color is None:
            raise FileNotFoundError(color_path)

        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        depth_scale = float(camera.get("depth_scale_m_per_unit", 0.001))
        depth_m, support_ratio, mad, depth_summary = load_depth_median(capture_dir, selected_frames, depth_scale)
        if depth_m.shape[:2] != color_rgb.shape[:2]:
            raise ValueError(f"Depth/color size mismatch for {capture_dir}: {depth_m.shape} vs {color_rgb.shape}")
        quality_mask, filter_summary = make_depth_quality_mask(
            depth_m,
            support_ratio,
            mad,
            args.min_depth_m,
            args.max_depth_m,
            args.min_valid_frame_ratio,
            args.max_depth_mad_m,
            args.edge_threshold_m,
        )

        sampled_points, valid = deproject_depth(
            depth_m,
            quality_mask,
            camera["color"]["intrinsic"],
            camera["color"]["distortion"],
            args.stride,
            args.min_depth_m,
            args.max_depth_m,
            not args.ignore_distortion,
        )
        sampled_colors = color_rgb[0 : color_rgb.shape[0] : args.stride, 0 : color_rgb.shape[1] : args.stride].reshape(-1, 3)
        sampled_points = sampled_points[valid]
        sampled_colors = sampled_colors[valid].astype(np.uint8)
        world_points = transform_points(sampled_points, poses[serial])
        labels = np.full(len(world_points), len(serials) - 1, dtype=np.int32)

        all_points.append(world_points)
        all_colors.append(sampled_colors)
        all_labels.append(labels)
        cameras_summary.append(
            {
                "serial": serial,
                "capture_dir": str(capture_dir),
                "color": str(color_path),
                "depth_frames": [str(capture_dir / "depth" / name) for name in selected_frames],
                "depth_fusion": "single_frame" if len(selected_frames) == 1 else "temporal_median",
                "depth_summary": depth_summary,
                "filter_summary": filter_summary,
                "input_valid_points": int(len(sampled_points)),
                "input_valid_ratio_after_range_filter": float(len(sampled_points) / depth_m.size),
                "T_world_camera": poses[serial].tolist(),
            }
        )

    if not all_points:
        raise RuntimeError("No points generated.")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    before_voxel = len(points)
    points, colors, labels = voxel_downsample(points, colors, labels, args.voxel_m)

    ply_path = args.out / "multi_orbbec_world_rgb_pointcloud.ply"
    html_path = args.out / "multi_orbbec_world_rgb_pointcloud.html"
    preview_xy_path = args.out / "preview_world_xy.png"
    preview_xz_path = args.out / "preview_world_xz.png"
    summary_path = args.out / "summary.json"
    write_ply_binary(ply_path, points, colors, labels)
    make_html(html_path, points, colors, labels, serials, args.html_points)
    make_preview_png(preview_xy_path, points, colors, (0, 1), "Multi-Orbbec point cloud preview: world XY")
    make_preview_png(preview_xz_path, points, colors, (0, 2), "Multi-Orbbec point cloud preview: world XZ")

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    summary = {
        "captures": [str(p) for p in args.captures],
        "extrinsics": str(args.extrinsics),
        "frame": args.frame,
        "frame_start": args.frame_start,
        "frame_count": args.frame_count,
        "color_frame": args.color_frame,
        "stride": args.stride,
        "min_depth_m": args.min_depth_m,
        "max_depth_m": args.max_depth_m,
        "voxel_m": args.voxel_m,
        "min_valid_frame_ratio": args.min_valid_frame_ratio,
        "max_depth_mad_m": args.max_depth_mad_m,
        "edge_threshold_m": args.edge_threshold_m,
        "points_before_voxel": int(before_voxel),
        "points_after_voxel": int(len(points)),
        "serials": serials,
        "bounds_world_m": {"min": mins.tolist(), "max": maxs.tolist()},
        "ply": str(ply_path),
        "html": str(html_path),
        "preview_world_xy": str(preview_xy_path),
        "preview_world_xz": str(preview_xz_path),
        "cameras": cameras_summary,
        "coordinate_frame": "world frame from AprilTag board extrinsics; camera coordinates are Orbbec/OpenCV x-right y-down z-forward.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
