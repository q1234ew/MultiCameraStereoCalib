#!/usr/bin/env python3
"""Fuse Orbbec RGB-D captures into a TSDF mesh using Open3D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def load_camera(capture_dir: Path) -> dict[str, object]:
    return json.loads((capture_dir / "camera.json").read_text(encoding="utf-8"))


def invert_t(t: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ t[:3, 3]
    return out


def load_world_camera_poses(path: Path) -> dict[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    poses: dict[str, np.ndarray] = {}
    for serial, payload in data["cameras"].items():
        pose = payload.get("T_world_camera")
        if pose is not None:
            poses[serial] = np.asarray(pose, dtype=np.float64)
    if not poses:
        raise ValueError(f"No T_world_camera poses found in {path}")
    return poses


def frame_name(index: int) -> str:
    return f"{index:06d}.png"


def resolve_frames(capture_dir: Path, frame: str | None, frame_start: int, frame_count: int) -> list[str]:
    if frame_count <= 1:
        return [frame or frame_name(frame_start)]
    frames = [frame_name(i) for i in range(frame_start, frame_start + frame_count)]
    missing = [name for name in frames if not (capture_dir / "depth" / name).exists() or not (capture_dir / "color" / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing RGB-D frames in {capture_dir}: {missing[:5]}{'...' if len(missing) > 5 else ''}")
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


def make_depth_quality_mask(
    capture_dir: Path,
    frames: list[str],
    depth_scale_m_per_unit: float,
    min_depth_m: float,
    max_depth_m: float,
    min_valid_frame_ratio: float,
    max_depth_mad_m: float,
    edge_threshold_m: float,
) -> tuple[np.ndarray, dict[str, object]]:
    depths = []
    valid_counts = []
    for frame in frames:
        depth_path = capture_dir / "depth" / frame
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(depth_path)
        depths.append(depth_raw.astype(np.float32) * depth_scale_m_per_unit)
        valid_counts.append(int(np.count_nonzero(depth_raw)))
    stack = np.stack(depths, axis=0)
    valid_stack = stack > 0
    support_ratio = valid_stack.mean(axis=0).astype(np.float32)
    stack_nan = stack.copy()
    stack_nan[~valid_stack] = np.nan
    median = np.nanmedian(stack_nan, axis=0).astype(np.float32)
    mad = np.nanmedian(np.abs(stack_nan - median[None, :, :]), axis=0).astype(np.float32)
    median[~np.isfinite(median)] = 0.0
    mad[~np.isfinite(mad)] = np.inf

    base = np.isfinite(median) & (median >= min_depth_m) & (median <= max_depth_m)
    support_ok = support_ratio >= min_valid_frame_ratio
    mad_ok = mad <= max_depth_mad_m if max_depth_mad_m > 0 else np.ones_like(base, dtype=bool)
    before_edge = base & support_ok & mad_ok
    after_edge = depth_edge_keep_mask(median, before_edge, edge_threshold_m)
    total = int(median.size)
    range_valid_pixels = int(np.count_nonzero(base))
    support_valid_pixels = int(np.count_nonzero(base & support_ok))
    confidence_valid_pixels = int(np.count_nonzero(base & support_ok & mad_ok))
    final_valid_pixels = int(np.count_nonzero(after_edge))
    stats = {
        "frames": frames,
        "frame_count": len(frames),
        "per_frame_valid_depth_ratio_min": float(min(valid_counts) / total) if total else 0.0,
        "per_frame_valid_depth_ratio_max": float(max(valid_counts) / total) if total else 0.0,
        "median_valid_depth_ratio": float(np.count_nonzero(median > 0) / total) if total else 0.0,
        "support_ratio_median": float(np.median(support_ratio)) if total else 0.0,
        "depth_mad_median_m": float(np.median(mad[np.isfinite(mad)])) if np.any(np.isfinite(mad)) else None,
        "depth_mad_p90_m": float(np.percentile(mad[np.isfinite(mad)], 90)) if np.any(np.isfinite(mad)) else None,
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


def make_intrinsic(camera: dict[str, object]) -> o3d.camera.PinholeCameraIntrinsic:
    color = camera["color"]
    intrinsic = color["intrinsic"]
    return o3d.camera.PinholeCameraIntrinsic(
        int(color["width"]),
        int(color["height"]),
        float(intrinsic["fx"]),
        float(intrinsic["fy"]),
        float(intrinsic["cx"]),
        float(intrinsic["cy"]),
    )


def read_rgbd(
    capture_dir: Path,
    frame: str,
    depth_scale: float,
    depth_scale_m_per_unit: float,
    min_depth_m: float,
    depth_trunc: float,
    quality_mask: np.ndarray | None,
) -> o3d.geometry.RGBDImage:
    color_path = capture_dir / "color" / frame
    depth_path = capture_dir / "depth" / frame
    color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if color_bgr is None:
        raise FileNotFoundError(color_path)
    if depth is None:
        raise FileNotFoundError(depth_path)
    depth_m = depth.astype(np.float32) * depth_scale_m_per_unit
    valid = (depth_m >= min_depth_m) & (depth_m <= depth_trunc)
    if quality_mask is not None:
        valid &= quality_mask
    depth = depth.copy()
    depth[~valid] = 0

    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    color_o3d = o3d.geometry.Image(color_rgb)
    depth_o3d = o3d.geometry.Image(depth)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )


def clean_mesh(mesh: o3d.geometry.TriangleMesh, min_cluster_triangles: int) -> tuple[o3d.geometry.TriangleMesh, dict[str, int]]:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()

    before_vertices = len(mesh.vertices)
    before_triangles = len(mesh.triangles)
    if min_cluster_triangles > 0 and before_triangles > 0:
        triangle_clusters, cluster_n_triangles, _cluster_area = mesh.cluster_connected_triangles()
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        triangle_clusters = np.asarray(triangle_clusters)
        remove_mask = cluster_n_triangles[triangle_clusters] < min_cluster_triangles
        mesh.remove_triangles_by_mask(remove_mask)
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()
    return mesh, {
        "vertices_before_clean_clusters": int(before_vertices),
        "triangles_before_clean_clusters": int(before_triangles),
        "vertices_after_clean": int(len(mesh.vertices)),
        "triangles_after_clean": int(len(mesh.triangles)),
    }


def make_preview(mesh: o3d.geometry.TriangleMesh, out_path: Path) -> None:
    # Headless-friendly preview: render vertices projected onto world XY.
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    if len(vertices) == 0:
        return
    if len(colors) != len(vertices):
        colors = np.full((len(vertices), 3), 0.55)
    if len(vertices) > 220000:
        idx = np.linspace(0, len(vertices) - 1, 220000).astype(np.int64)
        vertices = vertices[idx]
        colors = colors[idx]

    from PIL import Image, ImageDraw

    width, height = 1200, 900
    margin = 40
    xy = vertices[:, :2]
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    keep = np.all((xy >= lo) & (xy <= hi), axis=1)
    xy = xy[keep]
    colors = colors[keep]
    px = ((xy[:, 0] - lo[0]) / span[0] * (width - 2 * margin) + margin).astype(np.int32)
    py = (height - margin - (xy[:, 1] - lo[1]) / span[1] * (height - 2 * margin)).astype(np.int32)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for x, y, color in zip(px, py, colors):
        draw.point((int(x), int(y)), fill=tuple(int(c * 255) for c in color))
    draw.rectangle((margin, margin, width - margin, height - margin), outline=(80, 80, 80), width=1)
    draw.text((margin, 12), "TSDF mesh preview: world XY vertex projection", fill=(0, 0, 0))
    draw.text(
        (margin, height - 28),
        f"x {lo[0]:.2f}..{hi[0]:.2f} m, y {lo[1]:.2f}..{hi[1]:.2f} m",
        fill=(0, 0, 0),
    )
    image.save(out_path)


def make_mesh_html(mesh: o3d.geometry.TriangleMesh, out_path: Path, max_triangles: int = 120000) -> None:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    if len(vertices) == 0 or len(triangles) == 0:
        return
    if len(colors) != len(vertices):
        colors = np.full((len(vertices), 3), 0.65)
    if len(triangles) > max_triangles:
        tri_idx = np.linspace(0, len(triangles) - 1, max_triangles).astype(np.int64)
        triangles = triangles[tri_idx]

    vertexcolor = [f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})" for r, g, b in np.clip(colors, 0, 1)]
    fig = o3d_mesh_to_plotly(vertices, triangles, vertexcolor)
    fig.write_html(out_path, include_plotlyjs=True)


def o3d_mesh_to_plotly(vertices: np.ndarray, triangles: np.ndarray, vertexcolor: list[str]) -> "go.Figure":
    import plotly.graph_objects as go

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                vertexcolor=vertexcolor,
                opacity=1.0,
                flatshading=False,
                name="TSDF mesh",
            )
        ]
    )
    fig.update_layout(
        title="TSDF Mesh",
        scene=dict(aspectmode="data", xaxis_title="world x (m)", yaxis_title="world y (m)", zaxis_title="world z (m)"),
        margin=dict(l=0, r=0, t=36, b=0),
    )
    return fig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures", nargs="+", required=True, type=Path)
    parser.add_argument("--extrinsics", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frame", default=None, help="Single frame name for backwards-compatible one-frame fusion.")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=1)
    parser.add_argument("--voxel-m", type=float, default=0.02)
    parser.add_argument("--min-depth-m", type=float, default=0.3)
    parser.add_argument("--sdf-trunc-m", type=float, default=0.08)
    parser.add_argument("--depth-trunc-m", type=float, default=4.0)
    parser.add_argument("--min-cluster-triangles", type=int, default=120)
    parser.add_argument("--min-valid-frame-ratio", type=float, default=0.5)
    parser.add_argument("--max-depth-mad-m", type=float, default=0.05)
    parser.add_argument("--edge-threshold-m", type=float, default=0.08)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    poses = load_world_camera_poses(args.extrinsics)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_m,
        sdf_trunc=args.sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    integrated = []
    for capture_dir in args.captures:
        camera = load_camera(capture_dir)
        serial = camera["device"]["serial_number"]
        if serial not in poses:
            print(f"Skipping {serial}: no pose in extrinsics")
            continue
        depth_scale_m_per_unit = float(camera.get("depth_scale_m_per_unit", 0.001))
        depth_scale = 1.0 / depth_scale_m_per_unit
        selected_frames = resolve_frames(capture_dir, args.frame, args.frame_start, args.frame_count)
        quality_mask, quality_summary = make_depth_quality_mask(
            capture_dir,
            selected_frames,
            depth_scale_m_per_unit,
            args.min_depth_m,
            args.depth_trunc_m,
            args.min_valid_frame_ratio,
            args.max_depth_mad_m,
            args.edge_threshold_m,
        )
        intrinsic = make_intrinsic(camera)
        extrinsic_world_to_camera = invert_t(poses[serial])
        for frame in selected_frames:
            rgbd = read_rgbd(capture_dir, frame, depth_scale, depth_scale_m_per_unit, args.min_depth_m, args.depth_trunc_m, quality_mask)
            volume.integrate(rgbd, intrinsic, extrinsic_world_to_camera)
        integrated.append(
            {
                "serial": serial,
                "capture_dir": str(capture_dir),
                "frames": selected_frames,
                "frame_count": len(selected_frames),
                "depth_scale": depth_scale,
                "depth_quality": quality_summary,
                "T_world_camera": poses[serial].tolist(),
                "T_camera_world_used_by_open3d": extrinsic_world_to_camera.tolist(),
            }
        )

    mesh_raw = volume.extract_triangle_mesh()
    raw_vertices = int(len(mesh_raw.vertices))
    raw_triangles = int(len(mesh_raw.triangles))
    raw_mesh_path = args.out / "tsdf_mesh_raw.ply"
    o3d.io.write_triangle_mesh(str(raw_mesh_path), mesh_raw, write_ascii=False, compressed=False)

    mesh, clean_stats = clean_mesh(mesh_raw, args.min_cluster_triangles)
    mesh_path = args.out / "tsdf_mesh_clean.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False, compressed=False)

    pcd = volume.extract_point_cloud()
    pcd_path = args.out / "tsdf_pointcloud.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False, compressed=False)

    preview_path = args.out / "preview_tsdf_world_xy.png"
    html_path = args.out / "tsdf_mesh_clean.html"
    make_preview(mesh, preview_path)
    make_mesh_html(mesh, html_path)

    bounds = None
    if len(mesh.vertices) > 0:
        vertices = np.asarray(mesh.vertices)
        bounds = {"min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()}

    summary = {
        "captures": [str(p) for p in args.captures],
        "extrinsics": str(args.extrinsics),
        "frame": args.frame,
        "frame_start": args.frame_start,
        "frame_count": args.frame_count,
        "voxel_m": args.voxel_m,
        "min_depth_m": args.min_depth_m,
        "sdf_trunc_m": args.sdf_trunc_m,
        "depth_trunc_m": args.depth_trunc_m,
        "min_valid_frame_ratio": args.min_valid_frame_ratio,
        "max_depth_mad_m": args.max_depth_mad_m,
        "edge_threshold_m": args.edge_threshold_m,
        "integrated_cameras": integrated,
        "raw_mesh": str(raw_mesh_path),
        "clean_mesh": str(mesh_path),
        "pointcloud": str(pcd_path),
        "preview_world_xy": str(preview_path),
        "html": str(html_path),
        "raw_vertices": raw_vertices,
        "raw_triangles": raw_triangles,
        "clean_stats": clean_stats,
        "bounds_world_m": bounds,
        "note": "Open3D TSDF integration uses world-to-camera extrinsics; input camera poses are inverted from T_world_camera.",
    }
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
