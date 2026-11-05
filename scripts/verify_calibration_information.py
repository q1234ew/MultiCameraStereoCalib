"""Offline OPENCV5 weighted BA and stereo next-view information prototype."""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.optimize import least_squares

import compare_mrcal as common


def intrinsic_matrix(parameters):
    fx, fy, cx, cy = parameters[:4]
    return np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])


def pack_model(model):
    intrinsics = [np.r_[k[0, 0], k[1, 1], k[0, 2], k[1, 2], d]
                  for k, d in zip(model["K"], model["D"])]
    return np.concatenate((*intrinsics, cv2.Rodrigues(model["R"])[0].ravel(), model["T"]))


def unpack_model(parameters):
    return {"K": [intrinsic_matrix(parameters[c * 9:c * 9 + 9]) for c in (0, 1)],
            "D": [parameters[c * 9 + 4:c * 9 + 9].copy() for c in (0, 1)],
            "R": cv2.Rodrigues(parameters[18:21])[0], "T": parameters[21:24].copy()}


def sqrt_information(information):
    """Symmetric L such that L.T L = information; keep xy correlations."""
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    if not np.isfinite(information).all() or (eigenvalues <= 0).any():
        raise ValueError("Corner information must be finite positive definite")
    return (eigenvectors * np.sqrt(eigenvalues)[..., None, :]) @ np.swapaxes(eigenvectors, -1, -2)


def patch_information(gray, pixels):
    """Image-gradient proxy, not a calibrated covariance of ChArUco interpolation."""
    gray = np.asarray(gray, np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8.)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8.)
    gaussian = cv2.getGaussianKernel(9, 2.)
    weight = gaussian @ gaussian.T
    matrices, fallback = [], []
    for xy in np.asarray(pixels):
        bad = (not np.isfinite(xy).all() or (xy < 5).any()
               or (xy > np.array(gray.shape[::-1]) - 6).any())
        if bad:
            matrices.append(np.eye(2))
        else:
            dx = cv2.getRectSubPix(gx, (9, 9), tuple(xy.astype(float)))
            dy = cv2.getRectSubPix(gy, (9, 9), tuple(xy.astype(float)))
            matrices.append([[np.sum(weight * dx * dx), np.sum(weight * dx * dy)],
                             [np.sum(weight * dx * dy), np.sum(weight * dy * dy)]])
        fallback.append(bool(bad))
    return np.array(matrices), np.array(fallback)


def normalize_information(raw, scale, fallback=None):
    """Training-only common scale and predeclared [0.25,4] eigenvalue bounds."""
    eigenvalues, eigenvectors = np.linalg.eigh(raw / scale)
    clipped = np.clip(eigenvalues, .25, 4.)
    matrices = (eigenvectors * clipped[..., None, :]) @ eigenvectors.swapaxes(-1, -2)
    if fallback is not None:
        matrices[fallback] = np.eye(2)
    return matrices, int(np.count_nonzero(np.any(eigenvalues != clipped, axis=-1)))


def training_information(session, records):
    raw, bad = [], []
    for record in records:
        sides, fallback = [], []
        for camera, side in enumerate(("left", "right")):
            image = cv2.imread(str(session / record["paths"][camera]), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Missing training image")
            matrix, invalid = patch_information(image.astype(np.float32) / 255., record[side])
            sides.append(matrix)
            fallback.append(invalid)
        raw.append(sides)
        bad.append(fallback)
    strengths = np.concatenate([np.trace(m, axis1=-2, axis2=-1)[~b] / 2
                                for frame, failures in zip(raw, bad) for m, b in zip(frame, failures)])
    scale = float(np.median(strengths))
    if scale <= 1e-12:
        raise ValueError("Training corner gradients are degenerate")
    precision, clipped = [], 0
    for frame, failures in zip(raw, bad):
        normalized = [normalize_information(m, scale, b) for m, b in zip(frame, failures)]
        precision.append([m[0] for m in normalized])
        clipped += sum(m[1] for m in normalized)
    eigenvalues = np.concatenate([np.linalg.eigvalsh(m) for frame in precision for m in frame])
    return precision, {"training_scale": scale, "image_observations": len(eigenvalues),
                       "fallback_to_identity": int(sum(b.sum() for frame in bad for b in frame)),
                       "bounded_corner_observations": clipped,
                       "information_eigenvalue_min_median_max": [float(eigenvalues.min()),
                           float(np.median(eigenvalues)), float(eigenvalues.max())],
                       "anisotropy_information_p95": float(np.percentile(eigenvalues[:, 1] / eigenvalues[:, 0], 95))}


def frame_system(parameters, pose, objects, observations, whiteners):
    """Analytic chain-rule Jacobian: 24 rig/intrinsic parameters, 6 board-pose parameters."""
    composition = cv2.composeRT(pose[:3], pose[3:], parameters[18:21], parameters[21:24])
    residuals, globals_, boards, projected = [], [], [], []
    for camera in (0, 1):
        intr = parameters[camera * 9:camera * 9 + 9]
        rv, tv = (pose[:3], pose[3:]) if camera == 0 else composition[:2]
        pixels, jac = cv2.projectPoints(objects, rv, tv, intrinsic_matrix(intr), intr[4:])
        pixels, jac = pixels.reshape(-1, 2), jac[:, :15].reshape(-1, 2, 15)
        a, b = np.zeros((len(objects), 2, 24)), np.zeros((len(objects), 2, 6))
        a[..., camera * 9:camera * 9 + 9] = jac[..., 6:15]
        if camera == 0:
            b[:] = jac[..., :6]
        else:
            # composeRT: board is transform1; rig is transform2.
            b[..., :3] = jac[..., :3] @ composition[2] + jac[..., 3:6] @ composition[6]
            b[..., 3:] = jac[..., :3] @ composition[3] + jac[..., 3:6] @ composition[7]
            a[..., 18:21] = jac[..., :3] @ composition[4] + jac[..., 3:6] @ composition[8]
            a[..., 21:24] = jac[..., :3] @ composition[5] + jac[..., 3:6] @ composition[9]
        w = whiteners[camera]
        residuals.append(np.einsum("nij,nj->ni", w, pixels - observations[camera]).ravel())
        globals_.append((w @ a).reshape(-1, 24))
        boards.append((w @ b).reshape(-1, 6))
        projected.append(pixels)
    return np.concatenate(residuals), np.concatenate(globals_), np.concatenate(boards), projected


def fit(records, objects, initial, information):
    global_seed = pack_model(initial)
    poses = [common.pose_from_left(*common.points(r, objects)[:2], initial) for r in records]
    seed = np.r_[global_seed, np.array(poses).ravel()]
    whiteners = [[sqrt_information(m) for m in frame] for frame in information]
    cache = {}

    def evaluate(state):
        if "x" not in cache or not np.array_equal(state, cache["x"]):
            residuals, jacobians = [], []
            for frame, record in enumerate(records):
                obj, left, right = common.points(record, objects)
                begin = 24 + frame * 6
                residual, a, b, _ = frame_system(state[:24], state[begin:begin + 6], obj,
                                               (left, right), whiteners[frame])
                jac = np.zeros((len(residual), len(state)))
                jac[:, :24], jac[:, begin:begin + 6] = a, b
                residuals.append(residual)
                jacobians.append(jac)
            cache.update(x=state.copy(), residual=np.concatenate(residuals), jacobian=np.concatenate(jacobians))
        return cache

    start = time.perf_counter()
    answer = least_squares(lambda x: evaluate(x)["residual"], seed,
                           jac=lambda x: evaluate(x)["jacobian"], method="lm", x_scale="jac",
                           max_nfev=300, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    if not answer.success or not np.isfinite(answer.x).all():
        raise ValueError(f"Weighted BA failed: {answer.message}")
    model = unpack_model(answer.x[:24])
    if any(min(k[0, 0], k[1, 1]) <= 0 for k in model["K"]) or np.linalg.norm(model["T"]) <= 0:
        raise ValueError("Invalid fitted camera parameters")
    return answer.x, model, {"seconds": time.perf_counter() - start, "nfev": answer.nfev,
                             "cost": float(answer.cost), "optimality": float(answer.optimality),
                             "stop_reason": answer.message}


def marginal_information(a, b):
    """Schur elimination via orthogonal projection, avoiding an explicit (B.T B)^-1."""
    u, singular, _ = np.linalg.svd(b, full_matrices=False)
    if singular[-1] <= singular[0] * 1e-10:
        raise ValueError("Board pose is rank deficient")
    reduced = a - u @ (u.T @ a)
    return reduced.T @ reduced


def information_trace(information):
    eigenvalues, eigenvectors = np.linalg.eigh((information + information.T) / 2)
    if eigenvalues[0] <= eigenvalues[-1] * 1e-12:
        raise ValueError("Rig information is rank deficient; no pseudo-inverse confidence")
    covariance = (eigenvectors / eigenvalues) @ eigenvectors.T
    return float(np.trace(covariance[:18, :18])), float(eigenvalues[-1] / eigenvalues[0])


def linearized_noise_control(state, records, objects, information, size, samples=300):
    """Known-noise control of the full linearized BA, not a real-data covariance claim."""
    jacobians, precisions = [], []
    for i, (record, precision) in enumerate(zip(records, information)):
        obj, left, right = common.points(record, objects)
        identity = [np.tile(np.eye(2), (len(obj), 1, 1))] * 2
        _, a, b, _ = frame_system(state[:24], state[24 + 6 * i:30 + 6 * i], obj, (left, right), identity)
        jac = np.zeros((len(a), len(state)))
        jac[:, :24], jac[:, 24 + 6 * i:30 + 6 * i] = a, b
        jacobians.append(jac)
        precisions.extend(precision)
    jacobian = np.concatenate(jacobians)
    w = sqrt_information(np.concatenate(precisions))
    inverse_w = np.linalg.inv(w)
    weighted_jacobian = (w @ jacobian.reshape(-1, 2, len(state))).reshape(-1, len(state))
    scale = np.r_[np.tile([size[0], size[1], size[0], size[1], .1, .1, .01, .01, .1], 2),
                  [.1] * 3, [np.linalg.norm(state[21:24])] * 3,
                  np.tile([.3, .3, .3, 1., 1., 1.], len(records))]
    rng = np.random.default_rng(20261029)
    noise = inverse_w @ (rng.normal(size=(len(w), 2, samples)) * .2)
    traces, empirical = {}, {}
    for label, design, residual_noise in (
            ("uniform", jacobian, noise.reshape(-1, samples)),
            ("known_anisotropic", weighted_jacobian, (w @ noise).reshape(-1, samples))):
        q, r = qr(design * scale, mode="economic")
        if np.min(np.abs(np.diag(r))) < np.max(np.abs(np.diag(r))) * 1e-10:
            raise ValueError("Linear noise control rank deficient")
        solution = solve_triangular(r, q.T @ residual_noise)
        empirical[label] = float(np.mean(np.sum(solution[:18]**2, axis=0)))
        influence = solve_triangular(r, q.T)[:18]
        # Map independent unit image noise through C^(1/2), retaining xy correlation.
        if label == "uniform":
            influence = np.einsum("pnk,nkj->pnj", influence.reshape(18, -1, 2), inverse_w).reshape(18, -1)
        traces[label] = float(np.sum(influence**2) * .2**2)
    return {"trials": samples, "seed": 20261029,
            "noise_model": "known covariance 0.2^2 * inverse(training precision), synthetic Gaussian",
            "predicted_normalized_intrinsic_mse": traces, "empirical_normalized_intrinsic_mse": empirical,
            "predicted_weighted_over_uniform": traces["known_anisotropic"] / traces["uniform"],
            "empirical_weighted_over_uniform": empirical["known_anisotropic"] / empirical["uniform"],
            "limits": "fixed full-BA linearization, not nonlinear recalibration Monte Carlo or measured detector noise"}


@lru_cache(maxsize=1)
def corner_table():
    """Opening-angle lookup for a synthetic crossing with assumed 1px Gaussian blur."""
    xy = np.stack(np.meshgrid(np.arange(-15, 16), np.arange(-15, 16)), axis=-1)
    matrices = []
    for angle in range(15, 166):
        half = np.radians(angle / 2)
        normal0, normal1 = np.array([-np.sin(half), np.cos(half)]), np.array([np.sin(half), np.cos(half)])
        image = (((xy @ normal0) * (xy @ normal1)) > 0).astype(np.float32)
        image = cv2.GaussianBlur(image, (9, 9), 1.)
        matrices.append(patch_information(image, np.array([[15., 15.]]))[0][0])
    return np.array(matrices)


def predicted_information(parameters, pose, objects):
    """Projected grid directions -> synthetic opening-angle information, not measured future quality."""
    base = frame_system(parameters, pose, objects, [np.zeros((len(objects), 2))] * 2,
                        [np.tile(np.eye(2), (len(objects), 1, 1))] * 2)[3]
    neighbors = []
    for offset in ([.04, 0., 0.], [0., .04, 0.]):
        neighbors.append(frame_system(parameters, pose, objects + offset,
                         [np.zeros((len(objects), 2))] * 2,
                         [np.tile(np.eye(2), (len(objects), 1, 1))] * 2)[3])
    precisions, spacing = [], []
    table = corner_table()
    reference = float(np.trace(table[90 - 15]) / 2)
    for camera in (0, 1):
        u, v = neighbors[0][camera] - base[camera], neighbors[1][camera] - base[camera]
        spacing.append(np.minimum(np.linalg.norm(u, axis=-1), np.linalg.norm(v, axis=-1)))
        u, v = u / np.linalg.norm(u, axis=-1)[:, None], v / np.linalg.norm(v, axis=-1)[:, None]
        angle = np.degrees(np.arccos(np.clip(np.sum(u * v, axis=-1), -1, 1)))
        bisector = u + v
        beta = np.arctan2(bisector[:, 1], bisector[:, 0])
        r = np.stack((np.cos(beta), -np.sin(beta), np.sin(beta), np.cos(beta)), -1).reshape(-1, 2, 2)
        predicted = r @ table[np.clip(np.round(angle).astype(int), 15, 165) - 15] @ r.swapaxes(-1, -2)
        precisions.append(normalize_information(predicted, reference)[0])
    return precisions, base, spacing


def rank_next_views(state, records, objects, information, size, spacing):
    """Finite candidate search, not the paper's continuous global optimization."""
    if not np.isclose(spacing, .04):
        raise ValueError("Prototype currently requires 40mm board spacing")
    parameters = state[:24]
    baseline = np.linalg.norm(parameters[21:24])
    scale = np.r_[np.tile([size[0], size[1], size[0], size[1], .1, .1, .01, .01, .1], 2),
                  [.1] * 3, [baseline] * 3]
    information0 = np.zeros((24, 24))
    for i, (record, precision) in enumerate(zip(records, information)):
        obj, left, right = common.points(record, objects)
        _, a, b, _ = frame_system(parameters, state[24 + i * 6:30 + i * 6], obj, (left, right),
                                  [sqrt_information(m) for m in precision])
        information0 += marginal_information(a * scale, b)
    before, condition = information_trace(information0)
    model = unpack_model(parameters)
    center = objects.mean(axis=0)
    outer = np.array([[-spacing, -spacing, 0], [objects[:, 0].max() + spacing, -spacing, 0],
                      [objects[:, 0].max() + spacing, objects[:, 1].max() + spacing, 0],
                      [-spacing, objects[:, 1].max() + spacing, 0]])
    rows, rejected, candidates = [], {}, 0
    rotations = {"front": [0., 0., 0.], "pitch_up": [-25., 0., 0.],
                 "pitch_down": [25., 0., 0.], "yaw_left": [0., -25., 0.], "yaw_right": [0., 25., 0.]}
    for row, y in enumerate((.2, .5, .8)):
        for col, x in enumerate((.2, .5, .8)):
            normalized = cv2.undistortPointsIter(
                np.array([[[x * size[0], y * size[1]]]]), model["K"][0], model["D"][0], None, None,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-12)).ravel()
            for depth in (.5, .75, 1.):
                for label, rotation in rotations.items():
                    candidates += 1
                    rv = np.radians(rotation)
                    r = cv2.Rodrigues(rv)[0]
                    pose = np.r_[rv, np.r_[normalized * depth, depth] - r @ center]
                    xyz = objects @ r.T + pose[3:]
                    other = xyz @ model["R"].T + model["T"]
                    if (xyz[:, 2] <= 0).any() or (other[:, 2] <= 0).any():
                        rejected["behind_camera"] = rejected.get("behind_camera", 0) + 1
                        continue
                    projected_outer = frame_system(parameters, pose, outer, [np.zeros((4, 2))] * 2,
                                                     [np.tile(np.eye(2), (4, 1, 1))] * 2)[3]
                    if any(((p / size < .04) | (p / size > .96)).any() for p in projected_outer):
                        rejected["board_border_or_stereo_visibility"] = rejected.get("board_border_or_stereo_visibility", 0) + 1
                        continue
                    precision, predicted, spacings = predicted_information(parameters, pose, objects)
                    if min(float(s.min()) for s in spacings) < 8:
                        rejected["corner_spacing"] = rejected.get("corner_spacing", 0) + 1
                        continue
                    _, a, b, _ = frame_system(parameters, pose, objects, predicted,
                                               [sqrt_information(m) for m in precision])
                    score, _ = information_trace(information0 + marginal_information(a * scale, b))
                    rows.append({"zone_row_col": [row, col], "orientation": label, "center_depth_m": depth,
                                 "pose_board_to_camera0": pose.tolist(), "expected_normalized_intrinsic_trace": score,
                                 "expected_trace_reduction_percent": float((before - score) / before * 100),
                                 "min_corner_spacing_px": min(float(s.min()) for s in spacings)})
    rows.sort(key=lambda r: r["expected_normalized_intrinsic_trace"])
    return {"candidate_count": candidates, "feasible_candidates": len(rows), "rejected": rejected,
            "current_normalized_intrinsic_trace": before, "scaled_information_condition": condition,
            "parameter_scale": scale.tolist(), "top_candidates": rows[:10],
            "median_feasible_trace_reduction_percent": float(np.median([r["expected_trace_reduction_percent"] for r in rows])) if rows else None,
            "future_corner_model": "synthetic crossing, assumed sigma_blur=1px; no measured future quality",
            "objective": "trace of normalized marginal two-camera intrinsic covariance, rig covariance marginalized too",
            "limits": "finite 3x3/depth/25deg candidate grid, no continuous global optimum; approximate conditional linear information"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = common.collect_observations(args.session.resolve(), "rgb_pair_1", args.baseline / "observations.json")
    train, holdout, split = common.split_records(data["records"], .25, 2)
    historical = json.loads((args.baseline / "report.json").read_text())
    if split != historical.get("split", historical.get("protocol", {}).get("split")):
        raise ValueError("Historical split changed")
    common.write_json(args.output / "protocol.json", {"split": split,
        "weight_bounds": [.25, 4.], "gradient_window": [9, 9], "gradient_window_sigma": 2.,
        "weight_normalization": "training-only median half trace", "fit_modes": ["uniform", "scalar", "anisotropic"],
        "new_application_changes": False, "holdout_usage": "unweighted evaluation only; no weight estimation/tuning",
        "planning": "training-only anisotropic solve; 3x3 positions, depths .5/.75/1m, front and four 25deg tilts"})
    objects, size = np.array(data["object_points_m"]), tuple(data["image_size"])
    initial = common.solve_opencv(train, objects, size)["opencv_joint"]
    precision, weight_report = training_information(args.session, train)
    report = {"status": "completed", "dataset_hash": data["dataset_hash"], "split": split,
              "script_sha256": common.digest(__file__), "helper_sha256": common.digest(common.__file__),
              "opencv": cv2.__version__, "weights": weight_report, "models": {}}
    state_for_planning = uniform_state = None
    for name in ("uniform", "scalar", "anisotropic"):
        if name == "uniform":
            weights = [[np.tile(np.eye(2), (len(m), 1, 1)) for m in frame] for frame in precision]
        elif name == "scalar":
            weights = [[np.trace(m, axis1=-2, axis2=-1)[:, None, None] / 2 * np.eye(2) for m in frame] for frame in precision]
        else:
            weights = precision
        print("weighted BA", name, flush=True)
        state, model, solver = fit(train, objects, initial, weights)
        result = {"solver": solver, "model": common.serializable(model),
                  "training": common.evaluate(train, objects, size, model),
                  "validation": common.evaluate(holdout, objects, size, model)}
        report["models"][name] = result
        np.savez_compressed(args.output / f"{name}_state.npz", state=state)
        print(name, result["validation"]["cross_camera_transfer"], flush=True)
        if name == "anisotropic":
            state_for_planning = state
        if name == "uniform":
            uniform_state = state
        common.write_json(args.output / "report.partial.json", report)
    print("rank next stereo views", flush=True)
    report["planning"] = rank_next_views(state_for_planning, train, objects, precision, size, data["board"]["square_length"])
    print("linearized known-noise control", flush=True)
    report["linearized_noise_control"] = linearized_noise_control(uniform_state, train, objects, precision, size)
    np.savez_compressed(args.output / "training_precision.npz", **{f"frame{i}_cam{c}": m for i, frame in enumerate(precision) for c, m in enumerate(frame)})
    common.write_json(args.output / "report.json", report)
    print("Completed", args.output, flush=True)


if __name__ == "__main__":
    main()
