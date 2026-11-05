"""Native calibcam OPENCV5 solver comparison on frozen stereo observations.

Run in a separate Python 3.10 calibcam environment. No re-detection, rejection,
application mutation or holdout tuning. Native projection/Jacobian are not ported.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from unittest.mock import patch

import cv2
import numpy as np
from scipy.optimize import least_squares

import compare_mrcal as common
import verify_calibration_information as ba
import verify_opencv5_convergence as application

SOURCE_COMMIT = "ca2a49d61252995708730f7d9aebca024a3dfceb"


def backend():
    import jax
    jax.config.update("jax_enable_x64", True)
    from calibcam import camfunctions, calibrator_opts, optimization
    return jax, camfunctions, calibrator_opts, optimization


def native_input(records, objects, initial):
    """World = file-left camera; board poses and all lengths remain in metres."""
    corners = np.full((2, len(records), len(objects), 2), np.nan, dtype=np.float64)
    poses = []
    for frame, record in enumerate(records):
        obj, left, right = common.points(record, objects)
        poses.append(common.pose_from_left(obj, left, initial))
        for camera, values in enumerate((left, right)):
            corners[camera, frame, record["ids"]] = values
    poses = np.array(poses)
    calibs = []
    for camera in (0, 1):
        calibs.append({"A": initial["K"][camera].copy(), "k": initial["D"][camera].copy(), "xi": 0.,
                       "rvec_cam": np.zeros(3) if camera == 0 else cv2.Rodrigues(initial["R"])[0].ravel(),
                       "tvec_cam": np.zeros(3) if camera == 0 else initial["T"].copy(),
                       "rvecs": poses[:, :3].copy(), "tvecs": poses[:, 3:].copy(),
                       "projection_model": "perspective"})
    return corners, calibs


def native_options():
    _, _, options, _ = backend()
    opts = options.get_default_opts(2, do_fill=True)
    opts.update(debug=False, projection_models="perspective", coord_cam=0)
    for free in opts["free_vars"]:
        free["k"] = np.ones(5, dtype=int)  # Native pinhole default frees k1,k2 only.
        free["xi"] = False
    opts["optimization"].update(ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=500, verbose=0)
    return opts


def common_model(calibs):
    rotations = [cv2.Rodrigues(np.asarray(c["rvec_cam"]))[0] for c in calibs]
    r = rotations[1] @ rotations[0].T
    t = np.asarray(calibs[1]["tvec_cam"]) - r @ np.asarray(calibs[0]["tvec_cam"])
    return {"K": [np.array(c["A"]).copy() for c in calibs],
            "D": [np.array(c["k"]).copy() for c in calibs], "R": r, "T": t}


def projection_audit(parameters, args):
    _, _, _, optimization = backend()
    calibs, rotations, translations = optimization.unravel_to_calibs(parameters, args)
    model = common_model(calibs)
    r0 = cv2.Rodrigues(calibs[0]["rvec_cam"])[0]
    residual = optimization.obj_fcn_wrapper(parameters, args).reshape(args["corners"].shape)
    expected = np.zeros_like(residual)
    for frame, (rv, tv) in enumerate(zip(rotations, translations)):
        pose = np.r_[cv2.Rodrigues(r0 @ cv2.Rodrigues(rv)[0])[0].ravel(), r0 @ tv + calibs[0]["tvec_cam"]]
        for camera in (0, 1):
            observed = args["corners"][camera, frame]
            finite = np.isfinite(observed).all(axis=-1)
            expected[camera, frame, finite] = observed[finite] - common.project_camera(args["board_coords_3d_0"][finite], pose, model, camera)
    maximum = float(np.max(np.abs(residual - expected)))
    if maximum > 1e-7:
        raise ValueError(f"Native projection differs from OpenCV: {maximum}")
    return {"max_absolute_residual_difference_px": maximum,
            "observed_camera_points": int(np.isfinite(args["corners"]).all(axis=-1).sum()),
            "missing_camera_point_slots": int(np.isnan(args["corners"]).all(axis=-1).sum())}


def jacobian_audit(parameters, args):
    _, _, _, optimization = backend()
    jacobian = optimization.obj_fcn_jacobian_wrapper(parameters, args).toarray()
    # Camera variables plus a fixed spread of nuisance pose variables.
    columns = np.unique(np.r_[np.arange(min(24, len(parameters))), np.linspace(24, len(parameters) - 1, 12).astype(int)])
    maximum_absolute, maximum_relative = 0., 0.
    for column in columns:
        step = 1e-6 * max(1., abs(parameters[column]))
        delta = np.zeros_like(parameters)
        delta[column] = step
        numerical = (optimization.obj_fcn_wrapper(parameters + delta, args) - optimization.obj_fcn_wrapper(parameters - delta, args)) / (2 * step)
        difference = np.abs(numerical - jacobian[:, column])
        maximum_absolute = max(maximum_absolute, float(difference.max()))
        maximum_relative = max(maximum_relative, float(np.max(difference / (1 + np.abs(numerical)))))
    optimization.obj_fcn_wrapper(parameters, args)
    if not np.isfinite(jacobian).all() or maximum_relative > 2e-5:
        raise ValueError(f"Native Jacobian failed independent finite differences: {maximum_relative}")
    return {"checked_columns": columns.tolist(), "max_absolute_difference": maximum_absolute,
            "max_scaled_difference": maximum_relative}


def solve_native(records, objects, initial, mode):
    _, functions, _, optimization = backend()
    corners, calibs = native_input(records, objects, initial)
    before = corners.copy()
    opts = native_options()
    args, parameters = functions.make_optim_input(objects.copy(), deepcopy(calibs), corners, opts)
    audits = {"projection": projection_audit(parameters, args), "jacobian": jacobian_audit(parameters, args)}
    if len(parameters) != 24 + 6 * len(records):
        raise ValueError("Free parameter count differs from OPENCV5 joint BA")
    start = time.perf_counter()
    if mode == "sparse":
        fit_calibs, rotations, translations, answer, args = functions.optimize_calib_parameters(corners, deepcopy(calibs), objects.copy(), opts=opts, verbose=0)
    elif mode == "dense_control":
        # Same untouched native objective and automatic derivatives; change only
        # numerical linear algebra to isolate sparse LSMR convergence effects.
        answer = least_squares(optimization.obj_fcn_wrapper, parameters,
                               jac=lambda x, a: optimization.obj_fcn_jacobian_wrapper(x, a).toarray(),
                               args=(args,), method="lm", x_scale="jac", max_nfev=500,
                               ftol=1e-10, xtol=1e-10, gtol=1e-10)
        fit_calibs, rotations, translations = optimization.unravel_to_calibs(answer.x, args)
    else:
        raise ValueError("Unknown native solve mode")
    seconds = time.perf_counter() - start
    if not np.array_equal(corners, before, equal_nan=True):
        raise ValueError("Native solver changed observations or mask")
    model = common_model(fit_calibs)
    if not np.isfinite(answer.x).all() or not all(np.isfinite(v).all() for v in (*model["K"], *model["D"], model["R"], model["T"])):
        raise ValueError("Native solve produced invalid parameters")
    if any(min(k[0, 0], k[1, 1]) <= 0 for k in model["K"]) or np.linalg.norm(model["T"]) <= 0:
        raise ValueError("Native solve produced invalid focal length/baseline")
    # Decode exact fitted board poses for independent objective comparison.
    r0 = cv2.Rodrigues(fit_calibs[0]["rvec_cam"])[0]
    poses = [np.r_[cv2.Rodrigues(r0 @ cv2.Rodrigues(rv)[0])[0].ravel(), r0 @ tv + fit_calibs[0]["tvec_cam"]]
             for rv, tv in zip(rotations, translations)]
    state = np.r_[ba.pack_model(model), np.array(poses).ravel()]
    actual = []
    for pose, record in zip(poses, records):
        obj, left, right = common.points(record, objects)
        actual.extend(common.stereo_residual(pose, obj, left, right, model))
    actual_sse = float(np.sum(np.array(actual) ** 2))
    if not np.isclose(actual_sse, np.sum(answer.fun ** 2), rtol=1e-8, atol=1e-8):
        raise ValueError("Native and independent training objective differ")
    return state, model, {"success": bool(answer.success), "status": int(answer.status),
                          "nfev": answer.nfev, "njev": answer.njev, "optimality": float(answer.optimality),
                          "sse_px2": actual_sse, "seconds": seconds, "stop_reason": str(answer.message),
                          "backend": mode, "audits": audits}


def solve_application(data, train, reference=False):
    left_cal, right_cal, stereo = application.calibrators(data, train)
    left, right = left_cal.calibrate(), right_cal.calibrate()
    original = cv2.stereoCalibrate
    calls = []

    def solve(*args, **kwargs):
        if reference:
            kwargs["criteria"] = (3, 10000, 1e-10)
        calls.append({"criteria": kwargs.get("criteria"), "flags": kwargs.get("flags")})
        return original(*args, **kwargs)

    start = time.perf_counter()
    with patch.object(cv2, "stereoCalibrate", solve):
        result = stereo.calibrate(left, right, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    model = {"K": [left.camera_matrix.copy(), right.camera_matrix.copy()],
             "D": [left.dist_coeffs.ravel().copy(), right.dist_coeffs.ravel().copy()],
             "R": result.R.copy(), "T": result.T.ravel().copy()}
    return model, {"native_rms_px": float(result.rms_error), "solve_and_maps_seconds": time.perf_counter() - start,
                   "calls": calls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    data = common.collect_observations(args.session.resolve(), "rgb_pair_1", args.baseline / "observations.json")
    train, holdout, split = common.split_records(data["records"], .25, 2)
    history = json.loads((args.baseline / "report.json").read_text())
    if split != history.get("split", history.get("protocol", {}).get("split")):
        raise ValueError("Historical split changed")
    # Use EXACT application board coordinates in every solve, including float32
    # representation, to prevent geometry changes from masquerading as gains.
    objects = common.CharucoBoard.from_dict(data["board"]).object_points.astype(np.float64)
    size = tuple(data["image_size"])
    protocol = {"dataset_hash": data["dataset_hash"], "split": split,
                "source_commit": SOURCE_COMMIT, "script_sha256": common.digest(__file__),
                "common_sha256": common.digest(common.__file__), "ba_sha256": common.digest(ba.__file__),
                "geometry": "exact application float32 board coordinates promoted to float64; metres",
                "observations": "same original common IDs, no redetection/reweighting/rejection",
                "model": "OPENCV5 fx fy cx cy k1 k2 p1 p2 k3 per camera; xi=0, skew=0",
                "gauge": "file camera0 fixed; 24+6*N free parameters",
                "native_termination": {"max_nfev": 500, "ftol": 1e-10, "xtol": 1e-10, "gtol": 1e-10},
                "native_sparse_solver": "unmodified optimize_calib_parameters; TRF/LSMR x_scale=jac linear loss",
                "dense_control": "unmodified native residual/AD Jacobian; dense LM x_scale=jac",
                "holdout": "frozen raw pixels; nuisance board poses only; no tuning or solution selection",
                "python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__,
                "packages": {p: importlib.metadata.version(p) for p in ("bbo-calibcam", "bbo-calibcamlib", "jax", "jaxlib", "scipy")},
                "application_sources": {name: common.digest(Path("src/multicamera/calibration") / name)
                                        for name in ("intrinsic.py", "stereo.py", "solver_settings.py")}}
    common.write_json(args.output / "protocol.json", protocol)
    print("loading native backend", flush=True)
    jax, _, _, optimization = backend()
    protocol["jax_float64"] = bool(jax.config.read("jax_enable_x64"))
    protocol["native_source_hashes"] = {name: common.digest(Path(optimization.__file__).parent / name)
                                       for name in ("camfunctions.py", "optimization.py", "calibrator_opts.py", "camfunctions_ag.py", "helper_ag.py", "repro/optfunctions.py", "repro/optfunctions_ag.py")}
    common.write_json(args.output / "protocol.json", protocol)
    report = {"protocol": protocol, "models": {}}

    def record(label, model, solver, state=None):
        report["models"][label] = {"solver": solver, "model": common.serializable(model),
                                   "baseline_mm": float(np.linalg.norm(model["T"]) * 1000),
                                   "training": common.evaluate(train, objects, size, model),
                                   "validation": common.evaluate(holdout, objects, size, model)}
        if state is not None:
            np.savez_compressed(args.output / f"{label}_state.npz", state=state)
        common.write_json(args.output / "report.partial.json", report)
        print(label, report["models"][label]["validation"]["cross_camera_transfer"], flush=True)

    app_model, solver = solve_application(data, train)
    record("application_joint", app_model, solver)
    reference, solver = solve_application(data, train, reference=True)
    record("opencv_long_reference", reference, solver)
    opencv = common.solve_opencv(train, objects, size)
    fixed = opencv["opencv_fixed"]
    identity = [[np.tile(np.eye(2), (len(r["ids"]), 1, 1))] * 2 for r in train]
    state, model, solver = ba.fit(train, objects, app_model, identity)
    record("analytic_double_control", model, solver, state)
    for label, initial, mode in (("calibcam_sparse_from_fixed", fixed, "sparse"),
                                  ("calibcam_sparse_from_application", app_model, "sparse"),
                                  ("calibcam_dense_control_from_fixed", fixed, "dense_control")):
        print("native solve", label, flush=True)
        state, model, solver = solve_native(train, objects, initial, mode)
        record(label, model, solver, state)
    report["status"] = "completed"
    common.write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
