"""Frozen stereo models: common-canvas alignment, valid angular coverage and stretch."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path

import cv2
import mrcal
import numpy as np

import compare_mrcal as common
import compare_mrcal_spline as spline
import diagnose_mrcal_spline as diagnosis
import test_mrcal_external as external
from assess_mrcal_triangulation import statistics

NAMES = ("opencv5", "spline12x8_fov120", "spline16x10_fov140")
PROJECTIONS = {"pinhole": "LENSMODEL_PINHOLE", "latlon": "LENSMODEL_LATLON"}


def inside(pixels, size, bilinear=False):
    pixels = np.asarray(pixels)
    maximum = np.asarray(size) - 1
    return (np.isfinite(pixels).all(axis=-1) & (pixels >= 0).all(axis=-1)
            & ((pixels < maximum).all(axis=-1) if bilinear else (pixels <= maximum).all(axis=-1)))


def rotation_to_rectified(source, rectified):
    return mrcal.compose_Rt(rectified.Rt_cam_ref(), source.Rt_ref_cam())[:3]


def transform_pixels(pixels, source, rectified):
    """Rotate observation rays about the camera center; no scene-depth assumption."""
    pixels = np.asarray(pixels, dtype=float)
    rays = mrcal.unproject(pixels, *source.intrinsics(), normalize=True)
    valid = diagnosis.valid_inverse(rays, mrcal.project(rays, *source.intrinsics()), pixels)
    rays_rect = mrcal.rotate_point_R(rotation_to_rectified(source, rectified), rays)
    result = mrcal.project(rays_rect, *rectified.intrinsics())
    valid &= np.isfinite(result).all(axis=-1) & (rays_rect[..., 2] > 0)
    return result, valid


def canonical_pair(models):
    return mrcal.rectified_system(models, az_fov_deg=100., el_fov_deg=65., az0_deg=0., el0_deg=0.)


def reference_fov(models, size):
    """Training-only OPENCV5 border, symmetric outward whole-degree angular extent."""
    width, height = size
    x, y = np.linspace(0, width - 1, 129), np.linspace(0, height - 1, 73)
    pixels = np.concatenate((np.stack((x, x * 0), -1), np.stack((x, x * 0 + height - 1), -1),
                             np.stack((y * 0, y), -1), np.stack((y * 0 + width - 1, y), -1)))
    angles = []
    for source, rect in zip(models, canonical_pair(models)):
        rays = mrcal.unproject(pixels, *source.intrinsics(), normalize=True)
        if not diagnosis.valid_inverse(rays, mrcal.project(rays, *source.intrinsics()), pixels).all():
            raise ValueError("Reference border contains invalid inverse")
        rays = mrcal.rotate_point_R(rotation_to_rectified(source, rect), rays)
        angles.append(mrcal.project(rays, "LENSMODEL_LATLON", np.array([1., 1., 0., 0.])))
    angles = np.degrees(np.concatenate(angles))
    return (2 * np.ceil(np.max(np.abs(angles), axis=0))).astype(int)


def rectified_pair(models, lensmodel, size, fov):
    """Identical output intrinsics for all candidates; each uses its own calibrated pose."""
    extent = (np.asarray(size) - 1) / 2
    half_angles = np.radians(fov) / 2
    focal = extent / (np.tan(half_angles) if lensmodel == "LENSMODEL_PINHOLE" else half_angles)
    intrinsics = np.r_[focal, extent]
    pair = tuple(mrcal.cameramodel(intrinsics=(lensmodel, intrinsics.copy()),
                                  imagersize=np.asarray(size, np.int32),
                                  extrinsics_rt_fromref=rect.rt_cam_ref().copy())
                 for rect in canonical_pair(models))
    # Do not let file-role reversal yield a negative-baseline rectification accidentally.
    relative = mrcal.compose_Rt(pair[1].Rt_cam_ref(), pair[0].Rt_ref_cam())
    np.testing.assert_allclose(relative[:3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(relative[3, 1:], 0, atol=1e-12)
    if relative[3, 0] >= 0:
        raise ValueError("Expected camera1 center on positive rectified X")
    return pair


def solid_angle_weights(rectified):
    """Midpoint differential solid-angle quadrature per output pixel, steradians."""
    width, height = rectified.imagersize()
    x, y = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    _, gradients, _ = mrcal.unproject(np.stack((x, y), -1), *rectified.intrinsics(),
                                     normalize=True, get_gradients=True)
    return np.linalg.norm(np.cross(gradients[..., 0], gradients[..., 1]), axis=-1)


def largest_rectangle(mask):
    """Largest all-valid axis-aligned output ROI, O(HW) histogram stack."""
    heights = np.zeros(mask.shape[1], int)
    best = (0, 0, 0, 0)
    area = 0
    for row, values in enumerate(mask):
        heights = (heights + 1) * values
        stack = []
        for col in range(mask.shape[1] + 1):
            height = heights[col] if col < len(heights) else 0
            start = col
            while stack and stack[-1][1] > height:
                index, previous_height = stack.pop()
                candidate = (col - index) * previous_height
                if candidate > area:
                    area = candidate
                    best = (index, row - previous_height + 1, col - index, previous_height)
                start = index
            if not stack or stack[-1][1] < height:
                stack.append((start, height))
    return {"xywh": [int(x) for x in best], "pixels": int(area), "fraction": float(area / mask.size)}


def coverage_metrics(maps, source_size, weights):
    masks = [inside(mapping, source_size, bilinear=True) for mapping in maps]
    common_mask = masks[0] & masks[1]
    cameras = [{"valid_pixels": int(mask.sum()), "valid_fraction": float(mask.mean()),
                "solid_angle_sr": float(weights[mask].sum())} for mask in masks]
    return {"cameras": cameras, "common_valid_pixels": int(common_mask.sum()),
            "common_valid_fraction": float(common_mask.mean()),
            "common_solid_angle_sr": float(weights[common_mask].sum()),
            "canvas_solid_angle_sr": float(weights.sum()),
            "largest_common_valid_rectangle": largest_rectangle(common_mask)}, masks, common_mask


def local_stretch(pixels, source, rectified, step=.25):
    """Singular values of source-pixel -> output-pixel Jacobian, central differences."""
    jacobian = np.empty((len(pixels), 2, 2))
    valid = np.ones(len(pixels), bool)
    for axis in (0, 1):
        offset = np.zeros(2)
        offset[axis] = step
        plus, good_plus = transform_pixels(pixels + offset, source, rectified)
        minus, good_minus = transform_pixels(pixels - offset, source, rectified)
        jacobian[..., axis] = (plus - minus) / (2 * step)
        valid &= good_plus & good_minus
    finite = np.isfinite(jacobian).all(axis=(-1, -2))
    singular = np.full((len(pixels), 2), np.nan)
    singular[finite] = np.linalg.svd(jacobian[finite], compute_uv=False)
    determinant = np.linalg.det(jacobian)
    valid &= finite & (singular[:, 1] > 1e-12) & (determinant > 0)
    return {"max_magnification": singular[:, 0], "min_magnification": singular[:, 1],
            "anisotropy": singular[:, 0] / singular[:, 1], "area_magnification": determinant}, valid


def angular_rows(pixels, rectified):
    lensmodel, intrinsics = rectified.intrinsics()
    angle = (pixels[..., 1] - intrinsics[3]) / intrinsics[1]
    return np.arctan(angle) if lensmodel == "LENSMODEL_PINHOLE" else angle


def map_roundtrip(maps, models, rectified, size):
    indices = np.array([(int(x), int(y)) for y in np.linspace(0, size[1] - 1, 37)
                        for x in np.linspace(0, size[0] - 1, 65)])
    result = []
    for camera in (0, 1):
        source_pixels = maps[camera][indices[:, 1], indices[:, 0]]
        usable = inside(source_pixels, size, bilinear=True)
        transformed, valid = transform_pixels(source_pixels[usable], models[camera], rectified[camera])
        error = np.linalg.norm(transformed - indices[usable], axis=-1)
        passed = bool(valid.all() and (error < .01).all())
        result.append({"camera": camera, "checked_valid_map_points": int(usable.sum()),
                       "failed_inverse_points": int((~valid).sum()), "max_error_px": float(error.max()),
                       "tolerance_px": .01, "passed": passed})
    return result


def preview(session, record, models, rectified, maps, masks, name, projection):
    tiles = []
    for camera, side in enumerate(("left", "right")):
        image = cv2.imread(str(session / record["paths"][camera]))
        if image is None:
            raise ValueError("Missing preview image")
        warped = cv2.remap(image, maps[camera], None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        warped[~masks[camera]] = (30, 0, 50)
        pixels, valid = transform_pixels(np.array(record[side]), models[camera], rectified[camera])
        for point in pixels[valid & inside(pixels, models[camera].imagersize())]:
            cv2.circle(warped, tuple(np.round(point).astype(int)), 3, (0, 220, 255), 1, cv2.LINE_AA)
        for y in range(0, warped.shape[0], 60):
            cv2.line(warped, (0, y), (warped.shape[1] - 1, y), (0, 180, 0), 1)
        cv2.putText(warped, f"{name} {projection} {record['capture_id']} file cam{camera}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(warped, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))
    return np.concatenate(tiles, axis=1)


def summarize(report):
    fov = report["protocol"]["fov_deg"]
    lines = ["# test1 双目校正评估", "", f"输出统一为 1280×720、中心视角 {fov[0]}°×{fov[1]}°；冻结相机，原固定留出 6 对。", ""]
    for projection, results in report["projections"].items():
        lines += [f"## {projection}", "", "| 模型 | 全点垂直 RMS px | 边缘 RMS px | 共同有效画幅 % | 共同立体角 sr | 边缘最大倍率 P95 | 边缘横纵变形 P95 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name, result in results["models"].items():
            s = result["stretch"]["cameras"]
            magnification = max(c["regions"]["edge"]["max_magnification"]["p95_abs"] for c in s)
            anisotropy = max(c["regions"]["edge"]["anisotropy"]["p95_abs"] for c in s)
            lines.append(f"| {name} | {result['alignment']['regions']['all']['vertical_px']['rms']:.4f} | "
                         f"{result['alignment']['regions']['edge']['vertical_px']['rms']:.4f} | "
                         f"{result['coverage']['common_valid_fraction'] * 100:.2f} | "
                         f"{result['coverage']['common_solid_angle_sr']:.4f} | {magnification:.3f} | {anisotropy:.3f} |")
    lines += ["", "边缘由原图外侧 20% 定义；三模型拉伸统计使用共同可见的同一批源像素查询。",
              "倍率为输出/源图像素的局部最大奇异值；横纵变形为最大/最小奇异值。表内取两目各自 P95 的较大值。",
              "共同有效区域只表示同一输出射线方向有输入像素支持，不保证近处场景无视差越界或遮挡。",
              "PINHOLE 与 LATLON 的像素角分辨率不同，跨投影还应查看角度对齐误差和立体角。",
              "不新增标定、删点或独立最终测试；上沿仍没有验证角点。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        baseline = json.loads((args.baseline / "report.json").read_text())
        data = common.collect_observations(args.session.resolve(), "rgb_pair_1", args.baseline / "observations.json")
        if data["dataset_hash"] != baseline["dataset_hash"]:
            raise ValueError("Dataset hash mismatch")
        split = baseline["protocol"]["split"]
        records = [r for r in data["records"] if r["capture_id"] in split["validation"]]
        if len(records) != 6 or [r["capture_id"] for r in records] != split["validation"]:
            raise ValueError("Expected unchanged six-capture holdout")
        if not all(r["usable"] for r in records) or set(split["train"]) & set(split["validation"]):
            raise ValueError("Invalid holdout split")
        size = tuple(data["image_size"])
        if size != (1280, 720):
            raise ValueError("Expected test1 1280x720")
        inputs, models, fingerprints, source_hashes = {}, {}, {}, {}
        for name in NAMES:
            paths = [args.baseline / f"{name}_{c}.cameramodel" for c in (0, 1)]
            source_hashes[name] = [common.digest(p) for p in paths]
            inputs[name] = diagnosis.load_inputs(paths[0])
            if len(inputs[name]["rt_ref_frame"]) != len(split["train"]):
                raise ValueError("Model training frame count mismatch")
            fingerprints[name] = external.model_fingerprint(inputs[name])
            models[name] = tuple(mrcal.cameramodel(optimization_inputs=copy.deepcopy(inputs[name]), icam_intrinsics=c) for c in (0, 1))
        fov = reference_fov(models["opencv5"], size)
        protocol = {"split": split, "output_size": size, "fov_deg": fov.tolist(),
                    "fov_source": "training-only OPENCV5 dense border in canonical LATLON angles; symmetric outward whole-degree bound",
                    "projection_models": PROJECTIONS, "identical_output_intrinsics_within_projection": True,
                    "rectified_orientation": "per-model calibrated baseline and mean forward, az0=el0=0",
                    "new_camera_or_board_pose_fitting": False, "file_roles": "preserved without swapping",
                    "alignment": "all original holdout correspondences, also all-model common-visible subset",
                    "edge_definition": "outer 20% of either original image for correspondences; each original image for stretch",
                    "valid_map": "finite source coordinates and complete bilinear footprint inside input",
                    "effective_fov": "both per-camera and common output support; midpoint solid-angle quadrature",
                    "stretch": "source->rectified Jacobian central difference 0.25 source pixel, 65x37 grid; common visible queries across models",
                    "source_report_sha256": common.digest(args.baseline / "report.json")}
        common.write_json(args.output / "protocol.json", protocol)
        report = {"status": "completed", "dataset_hash": data["dataset_hash"], "protocol": protocol,
                  "script_sha256": common.digest(__file__), "source_model_sha256": source_hashes,
                  "helper_sha256": {m.__file__: common.digest(m.__file__) for m in (common, spline, diagnosis, external)},
                  "environment": {n: importlib.metadata.version(n) for n in ("mrcal", "numpy", "scipy", "opencv-python-headless")},
                  "projections": {}}
        grid = np.array([(x, y) for y in np.linspace(0, size[1] - 1, 37) for x in np.linspace(0, size[0] - 1, 65)])
        raw_left = np.concatenate([r["left"] for r in records])
        raw_right = np.concatenate([r["right"] for r in records])
        regions = {key: spline.region_masks(raw_left, size)[key] | spline.region_masks(raw_right, size)[key]
                   for key in ("edge", "corners", "top", "bottom", "left", "right")}
        regions.update(all=np.ones(len(raw_left), bool), center=~regions["edge"])
        for projection, lensmodel in PROJECTIONS.items():
            print("rectify", projection, flush=True)
            results, query_data, visible_data, preview_tiles = {}, {}, [], {}
            weights = None
            for name in NAMES:
                print(projection, name, flush=True)
                rectified = rectified_pair(models[name], lensmodel, size, fov)
                for c, model in enumerate(rectified):
                    model.write(str(args.output / f"{projection}_{name}_{c}.cameramodel"))
                maps = mrcal.rectification_maps(models[name], rectified)
                if weights is None:
                    weights = solid_angle_weights(rectified[0])
                coverage, masks, common_mask = coverage_metrics(maps, size, weights)
                roundtrip = map_roundtrip(maps, models[name], rectified, size)
                if not all(c["passed"] for c in roundtrip):
                    raise ValueError("Rectification map round-trip failed")
                xy, valid, region_stretch, grid_checks = [], [], [], []
                for c, raw in enumerate((raw_left, raw_right)):
                    transformed, good = transform_pixels(raw, models[name][c], rectified[c])
                    xy.append(transformed)
                    valid.append(good)
                    quantities, good_grid = local_stretch(grid, models[name][c], rectified[c])
                    mapped_grid, inverse_good = transform_pixels(grid, models[name][c], rectified[c])
                    visible_grid = good_grid & inverse_good & inside(mapped_grid, size)
                    grid_checks.append({"camera": c, "queries": len(grid),
                                        "invalid_inverse_queries": int((~inverse_good).sum()),
                                        "invalid_or_nonpositive_jacobian_queries": int((~good_grid).sum()),
                                        "valid_queries_clipped_by_canvas": int((good_grid & inverse_good & ~inside(mapped_grid, size)).sum())})
                    center_quantities, center_good = local_stretch(np.array([[(size[0] - 1) / 2, (size[1] - 1) / 2]]), models[name][c], rectified[c])
                    if not center_good.all():
                        raise ValueError("Invalid center stretch")
                    quantities["max_magnification_relative_center"] = quantities["max_magnification"] / center_quantities["max_magnification"][0]
                    quantities["area_magnification_relative_center"] = quantities["area_magnification"] / center_quantities["area_magnification"][0]
                    region_stretch.append((quantities, visible_grid, center_quantities))
                if not np.logical_and(*valid).all():
                    raise ValueError("Invalid holdout corner transformation")
                delta = xy[0][:, 1] - xy[1][:, 1]
                angular_delta = (angular_rows(xy[0], rectified[0]) - angular_rows(xy[1], rectified[1])) * 1000
                visible = inside(xy[0], size) & inside(xy[1], size)
                visible_data.append(visible)
                frames, offset = [], 0
                for record in records:
                    count = len(record["ids"])
                    frames.append({"capture_id": record["capture_id"], "points": count,
                                   "visible_pairs": int(visible[offset:offset + count].sum()),
                                   "vertical_px": statistics(delta[offset:offset + count]),
                                   "angular_row_mrad": statistics(angular_delta[offset:offset + count])})
                    offset += count
                alignment = {"points": len(delta), "finite_transforms": int(np.logical_and(*valid).sum()),
                             "visible_pairs": int(visible.sum()), "clipped_pairs": int((~visible).sum()),
                             "regions": {key: {"vertical_px": statistics(delta[mask]),
                                 "angular_row_mrad": statistics(angular_delta[mask])} for key, mask in regions.items()},
                             "frames": frames}
                query_data[name] = (region_stretch, delta, angular_delta)
                results[name] = {"alignment": alignment, "coverage": coverage, "map_roundtrip": roundtrip,
                                 "output_intrinsics": rectified[0].intrinsics()[1].tolist(),
                                 "stretch_query_checks": grid_checks}
                np.savez_compressed(args.output / f"{projection}_{name}_maps.npz", map0=maps[0], map1=maps[1],
                                    valid0=masks[0], valid1=masks[1], common_valid=common_mask,
                                    holdout_rectified0=xy[0], holdout_rectified1=xy[1], holdout_visible=visible)
                for record in (records[0], records[-1]):
                    preview_tiles.setdefault(record["capture_id"], []).append(preview(args.session, record, models[name], rectified, maps, masks, name, projection))
            common_visible = np.logical_and.reduce(visible_data)
            stretch_queries = [np.logical_and.reduce([query_data[name][0][c][1] for name in NAMES]) for c in (0, 1)]
            for name in NAMES:
                data_query, delta, angular = query_data[name]
                results[name]["alignment"]["all_model_common_visible"] = {
                    "points": int(common_visible.sum()), "vertical_px": statistics(delta[common_visible]),
                    "angular_row_mrad": statistics(angular[common_visible])}
                cameras = []
                for c, (quantities, visible_grid, center) in enumerate(data_query):
                    cameras.append({"camera": c, "grid_queries": len(grid), "valid_visible_queries": int(visible_grid.sum()),
                        "all_model_common_visible_queries": int(stretch_queries[c].sum()),
                        "center": {k: float(v[0]) for k, v in center.items()},
                        "regions": {region: {quantity: statistics(values[mask & stretch_queries[c]])
                            for quantity, values in quantities.items()} for region, mask in spline.region_masks(grid, size).items()}})
                    np.savez_compressed(args.output / f"{projection}_{name}_stretch_cam{c}.npz", pixels=grid,
                                        visible=visible_grid, common_visible=stretch_queries[c], **quantities)
                results[name]["stretch"] = {"cameras": cameras}
            for capture, tiles in preview_tiles.items():
                cv2.imwrite(str(args.output / f"preview_{projection}_{capture}.png"), np.concatenate(tiles, axis=0))
            report["projections"][projection] = {"models": results}
            common.write_json(args.output / "report.partial.json", report)
        for name in NAMES:
            if external.model_fingerprint(inputs[name]) != fingerprints[name]:
                raise ValueError("Changed frozen camera parameters")
            if source_hashes[name] != [common.digest(args.baseline / f"{name}_{c}.cameramodel") for c in (0, 1)]:
                raise ValueError("Changed source model files")
        common.write_json(args.output / "report.json", report)
        (args.output / "summary.md").write_text(summarize(report), encoding="utf-8")
        print("Completed", args.output, flush=True)
    except Exception as exc:
        common.write_json(args.output / "failure.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
