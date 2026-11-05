"""Propagate offline detector perturbations through full stereo BA, retaining board poses.

Conditional linearization only: this does not certify physical calibration accuracy.
Relative clipped weights require sandwich covariance, not inverse weighted normal matrix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.linalg import qr, solve_triangular

import compare_mrcal as common
import verify_calibration_information as ba
import verify_discocal_transfer as study


def influence_matrix(design, information):
    """Global-parameter rows of full BA influence; eliminate no nuisance by fixing it."""
    whitener = ba.sqrt_information(information)
    weighted = (whitener @ design.reshape(-1, 2, design.shape[1])).reshape(design.shape)
    scale = 1 / np.linalg.norm(weighted, axis=0)
    q, r = qr(weighted * scale, mode="economic")
    singular = np.linalg.svd(r, compute_uv=False)
    if not np.isfinite(singular).all() or singular[-1] <= singular[0] * 1e-10:
        raise ValueError("Full BA Jacobian rank deficient; covariance unavailable")
    # Unscale parameters, then map ORIGINAL pixel errors through the whitener.
    influence = (scale[:, None] * solve_triangular(r, q.T))[:24]
    influence = np.einsum("pni,nij->pnj", influence.reshape(24, -1, 2), whitener).reshape(24, -1)
    return influence, float(singular[0] / singular[-1])


def sandwich_covariance(influence, covariance):
    blocks = influence.reshape(influence.shape[0], -1, 2)
    return np.einsum("pni,nij,qnj->pq", blocks, covariance, blocks)


def assess(state, records, objects, precision, archive, size):
    jacobians, covariance, information, draws = [], [], [], []
    for frame, record in enumerate(records):
        obj, left, right = common.points(record, objects)
        identity = [np.tile(np.eye(2), (len(obj), 1, 1))] * 2
        _, a, b, _ = ba.frame_system(state[:24], state[24 + frame * 6:30 + frame * 6], obj, (left, right), identity)
        jac = np.zeros((len(a), len(state)))
        jac[:, :24], jac[:, 24 + frame * 6:30 + frame * 6] = a, b
        jacobians.append(jac)
        for camera in (0, 1):
            samples = archive[f"frame{frame}_cam{camera}_draws"]
            if not np.isfinite(samples).all():
                return {"status": "unavailable", "reason": "Incomplete perturbation detections; no imputation for joint covariance"}
            draws.append(samples)
            covariance.extend(archive[f"frame{frame}_cam{camera}_covariance"])
            information.extend(precision[frame][camera])
    influence, condition = influence_matrix(np.concatenate(jacobians), np.array(information))
    diagonal = sandwich_covariance(influence, np.array(covariance))
    # Block covariance across all corners WITHIN each image. No inversion of a
    # rank-deficient sample covariance, no assumed correlations between images.
    correlated = np.zeros((24, 24))
    check_delta = np.zeros((24, 32))
    begin = 0
    for samples in draws:
        columns = samples.shape[1] * 2
        operator = influence[:, begin:begin + columns]
        train = samples[:32].reshape(32, -1)
        check = samples[32:].reshape(32, -1)
        delta = operator @ (train - train.mean(axis=0)).T
        # Full covariance is applied as sample products, never materialized.
        cov_blocks = np.einsum("sni,snj->nij", samples[:32] - samples[:32].mean(axis=0), samples[:32] - samples[:32].mean(axis=0)) / 31
        isotropic = np.trace(cov_blocks, axis1=-2, axis2=-1)[:, None, None] / 2 * np.eye(2)
        correlated += .9 * (delta @ delta.T / 31) + sandwich_covariance(operator, .1 * isotropic + 1e-4 * np.eye(2))
        check_delta += operator @ (check - check.mean(axis=0)).T
        begin += columns
    empirical = check_delta @ check_delta.T / 31
    model = ba.unpack_model(state[:24])
    map_entries, summaries = [], []
    grid = np.array([[x, y] for y in np.linspace(.05, .95, 11) for x in np.linspace(.05, .95, 17)]) * size
    for camera in (0, 1):
        ray_xy = cv2.undistortPointsIter(grid[:, None], model["K"][camera], model["D"][camera], None, None,
                                      (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)).reshape(-1, 2)
        rays = np.c_[ray_xy, np.ones(len(ray_xy))]
        pixels, derivative = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), model["K"][camera], model["D"][camera])
        valid = np.isfinite(rays).all(axis=1) & (np.linalg.norm(pixels.reshape(-1, 2) - grid, axis=1) <= 1e-5)
        if not valid.any():
            raise ValueError("No valid map rays")
        j = np.zeros((len(grid), 2, 24))
        j[:, :, camera * 9:camera * 9 + 9] = derivative[:, 6:15].reshape(-1, 2, 9)
        entry = {"camera": camera, "grid_points": len(grid), "valid_points": int(valid.sum())}
        edge = np.any((grid / size < .2) | (grid / size > .8), axis=1)
        trace_maps = {}
        for label, cov in (("block_diagonal", diagonal), ("within_image_correlated", correlated), ("independent_check", empirical)):
            pixel_cov = j @ cov @ j.swapaxes(-1, -2)
            trace = np.trace(pixel_cov, axis1=-2, axis2=-1)
            if np.min(trace[valid]) < -1e-10:
                raise ValueError("Negative projected covariance")
            sigma = np.sqrt(np.maximum(trace, 0.) / 2)
            trace_maps[label] = trace
            entry[label] = {"rms_sigma_px": float(np.sqrt(np.mean(trace[valid]) / 2)),
                            "edge_rms_sigma_px": float(np.sqrt(np.mean(trace[valid & edge]) / 2)),
                            "sigma_p95_px": float(np.percentile(sigma[valid], 95))}
            map_entries.append({"camera": camera, "method": label, "grid_pixels": grid.tolist(),
                                "valid": valid.tolist(), "sigma_px": [float(x) if ok else None for x, ok in zip(sigma, valid)]})
        entry["check_variance_over_block_diagonal"] = float(trace_maps["independent_check"][valid].sum() / trace_maps["block_diagonal"][valid].sum())
        entry["check_variance_over_correlated"] = float(trace_maps["independent_check"][valid].sum() / trace_maps["within_image_correlated"][valid].sum())
        summaries.append(entry)
    return {"status": "completed", "scaled_jacobian_condition": condition,
            "camera_summaries": summaries, "maps": map_entries,
            "limits": "32 independent check draws; linearized conditional sensitivity, no nonlinear refit or physical accuracy coverage"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.experiment / "report.json").read_text())
    data = json.loads(args.observations.read_text())
    train, _, split = common.split_records(data["records"], .25, 2)
    if split != report["protocol"]["split"] or data["dataset_hash"] != report["protocol"]["dataset_hash"]:
        raise ValueError("Dataset or split mismatch")
    archive = np.load(args.experiment / "training_perturbations.npz")
    scale = report["weights"]["normalization_scale"]
    precision = []
    for frame in range(len(train)):
        sides = []
        for camera in (0, 1):
            _, cov, good, _ = study.covariance_statistics(archive[f"frame{frame}_cam{camera}_draws"][:32])
            sides.append(ba.normalize_information(np.linalg.inv(cov), scale, ~good)[0])
        precision.append(sides)
    result = {"script_sha256": common.digest(__file__), "dataset_hash": data["dataset_hash"], "models": {}}
    for mode in ("uniform", "anisotropic"):
        weights = precision if mode == "anisotropic" else [[np.tile(np.eye(2), (len(m), 1, 1)) for m in frame] for frame in precision]
        state = np.load(args.experiment / f"{mode}_state.npz")["state"]
        result["models"][mode] = assess(state, train, np.array(data["object_points_m"]), weights, archive, tuple(data["image_size"]))
    common.write_json(args.experiment / "uncertainty.json", result)


if __name__ == "__main__":
    main()
