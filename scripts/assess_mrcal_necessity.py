"""Fixed-model offline comparison with explicitly validated initialization rays.

Missing virtual seed rays never remove real calibration observations. This entry
point records the incomplete OPENCV5 seed domain instead of feeding NaNs to SVD.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
from pathlib import Path
import sys

import cv2
import mrcal
import numpy as np
import scipy

import assess_mrcal_capture as capture
import compare_mrcal as common
import compare_mrcal_session as session
import compare_mrcal_spline as spline
import diagnose_mrcal_spline as diagnosis


def inverse_summary(inputs, camera, pixels):
    rays = mrcal.unproject(pixels, inputs["lensmodel"], inputs["intrinsics"][camera],
                           normalize=True)
    finite_nonzero = (np.isfinite(rays).all(axis=-1)
                      & (np.linalg.norm(rays, axis=-1) > 1e-12))
    valid = np.zeros(len(pixels), dtype=bool)
    valid[finite_nonzero] = diagnosis.valid_inverse(
        rays[finite_nonzero], mrcal.project(rays[finite_nonzero], inputs["lensmodel"],
                                          inputs["intrinsics"][camera]), pixels[finite_nonzero])
    return {"queries": len(pixels), "valid": int(valid.sum()),
            "invalid_ray_count": int((~finite_nonzero).sum()),
            "invalid_roundtrip_count": int((finite_nonzero & ~valid).sum()),
            "invalid_pixels": pixels[~valid].tolist()}


def audit_frozen_models(baseline):
    """Audit each model without assuming the reference OPENCV5 inverse is valid."""
    import json

    report = json.loads((baseline / "report.json").read_text())
    data = json.loads((baseline / "observations.json").read_text())
    split = report["protocol"]["split"]
    subsets = {name: [r for r in data["records"] if r["capture_id"] in split[name]]
               for name in ("train", "validation")}
    width, height = data["image_size"]
    pixels = np.array([(x, y) for y in np.linspace(0, height - 1, 37)
                       for x in np.linspace(0, width - 1, 65)])
    audit = {"dataset_hash": data["dataset_hash"], "split": split,
             "script_sha256": common.digest(__file__),
             "coverage": {name: capture.coverage(rs, (width, height))
                          for name, rs in subsets.items()}, "models": {}}
    for name, entry in report["models"].items():
        if entry["status"] != "completed":
            continue
        inputs = diagnosis.load_inputs(baseline / f"{name}_0.cameramodel")
        before = inputs["intrinsics"].tobytes(), inputs["rt_cam_ref"].tobytes()
        cameras = []
        for camera, side in enumerate(("left", "right")):
            queries = np.concatenate([r[side] for r in subsets["validation"]])
            cameras.append({"camera": camera, "full_image_grid": inverse_summary(inputs, camera, pixels),
                            "holdout_corners": inverse_summary(inputs, camera, queries)})
        audit["models"][name] = {"inverse": cameras,
                                  "uncertainty": capture.masked_uncertainty(inputs),
                                  "knot_support": spline.knot_support(inputs)}
        assert before == (inputs["intrinsics"].tobytes(), inputs["rt_cam_ref"].tobytes())
        common.write_json(baseline / "audit.partial.json", audit)
    common.write_json(baseline / "audit.json", audit)
    return audit


def seed_from_valid_rays(source, lensmodel):
    result = copy.deepcopy(source)
    result.update(lensmodel=lensmodel, do_optimize_intrinsics_core=False,
                  do_apply_regularization=True)
    seeds, diagnostics = [], []
    for camera in (0, 1):
        width, height = source["imagersizes"][camera]
        pixels = np.array([(x, y) for y in np.linspace(0, height - 1, 17)
                           for x in np.linspace(0, width - 1, 25)])
        rays = mrcal.unproject(pixels, source["lensmodel"], source["intrinsics"][camera],
                               normalize=True)
        valid = np.isfinite(rays).all(axis=-1) & (np.linalg.norm(rays, axis=-1) > 1e-12)
        positions = np.flatnonzero(valid)
        roundtrip = mrcal.project(rays[valid], source["lensmodel"], source["intrinsics"][camera])
        valid[positions] &= np.isfinite(roundtrip).all(axis=-1)
        valid[positions] &= np.linalg.norm(roundtrip - pixels[positions], axis=-1) <= 1e-5
        if np.count_nonzero(valid) < 100:
            raise ValueError("Insufficient valid virtual initialization rays")
        intrinsics = np.zeros(mrcal.lensmodel_num_params(lensmodel))
        intrinsics[:4] = source["intrinsics"][camera, :4]
        predicted, _, jacobian = mrcal.project(rays[valid], lensmodel, intrinsics,
                                              get_gradients=True)
        design = jacobian[..., 4:].reshape(-1, len(intrinsics) - 4)
        if not np.isfinite(design).all():
            raise ValueError("Nonfinite initialization design")
        intrinsics[4:], _, rank, _ = np.linalg.lstsq(
            design, (pixels[valid] - predicted).ravel(), rcond=1e-10)
        error = mrcal.project(rays[valid], lensmodel, intrinsics) - pixels[valid]
        diagnostics.append({"camera": camera, "queries": len(pixels),
                            "valid_virtual_rays": int(valid.sum()),
                            "invalid_virtual_pixels": pixels[~valid].tolist(),
                            "rank": int(rank), "coefficient_count": len(intrinsics) - 4,
                            "approximation": common.error_summary(error)})
        seeds.append(intrinsics)
    result["intrinsics"] = np.array(seeds)
    return result, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous runs; output must not exist")
    args.output.mkdir(parents=True)
    data = common.collect_observations(args.session.resolve(), "rgb_pair_1", args.observations)
    train, holdout, split = common.split_records(data["records"], .25, 2)
    protocol = {"split": split, "models": session.MODELS, "regularization": True,
                "outlier_rejection": False, "board_warp": False,
                "initialization": "training-only OPENCV5; validated finite inverse rays on fixed 25x17 grid",
                "invalid_virtual_rays": "reported and omitted from initialization only; real observations unchanged",
                "holdout_use": "board pose only; cameras frozen; no tuning or split changes",
                "context": "strict historical initializer failed before evaluation; this fallback is diagnostic"}
    common.write_json(args.output / "protocol.json", protocol)
    common.write_json(args.output / "observations.json", data)
    objects, size = np.array(data["object_points_m"]), tuple(data["image_size"])
    initial = common.solve_opencv(train, objects, size)["opencv_fixed"]
    base, stats = spline.solve(session.training_inputs(train, data, initial))
    report = {"status": "completed", "dataset_hash": data["dataset_hash"], "protocol": protocol,
              "train_captures": len(train), "validation_captures": len(holdout),
              "script_sha256": common.digest(__file__),
              "helper_sha256": {m.__file__: common.digest(m.__file__)
                                 for m in (common, session, spline, diagnosis, capture)},
              "environment": {"python": sys.version, "cv2": cv2.__version__,
                              "numpy": np.__version__, "scipy": scipy.__version__,
                              "mrcal": importlib.metadata.version("mrcal"),
                              "cv2_module": cv2.__file__, "mrcal_module": mrcal.__file__},
              "train_coverage": spline.coverage(train, size),
              "validation_coverage": spline.coverage(holdout, size), "models": {}}
    for name, lensmodel in session.MODELS.items():
        print("fit", name, flush=True)
        try:
            if name == "opencv5":
                fitted, details = base, {"solver": stats}
            else:
                candidate, init = seed_from_valid_rays(base, lensmodel)
                fitted, timing = spline.solve(candidate)
                details = {"initialization": init, "solver": timing}
            before = (fitted["intrinsics"].tobytes(), fitted["rt_cam_ref"].tobytes(),
                      fitted["observations_board"].tobytes())
            entry = {"status": "completed", "lensmodel": lensmodel, **details,
                     "train": spline.evaluate(fitted, train, objects, size, initial),
                     "validation": spline.evaluate(fitted, holdout, objects, size, initial),
                     "intrinsics": fitted["intrinsics"].tolist(),
                     "rt_cam_ref": fitted["rt_cam_ref"].tolist(),
                     "baseline_mm": float(np.linalg.norm(fitted["rt_cam_ref"][0, 3:]) * 1000)}
            assert before == (fitted["intrinsics"].tobytes(), fitted["rt_cam_ref"].tobytes(),
                              fitted["observations_board"].tobytes())
            for camera in (0, 1):
                mrcal.cameramodel(optimization_inputs=copy.deepcopy(fitted),
                                  icam_intrinsics=camera).write(
                    str(args.output / f"{name}_{camera}.cameramodel"))
            report["models"][name] = entry
            print(name, entry["validation"]["transfer"]["all"], flush=True)
        except Exception as exc:
            report["models"][name] = {"status": "failed", "error": str(exc)}
        common.write_json(args.output / "report.partial.json", report)
    if any(m["status"] != "completed" for m in report["models"].values()):
        report["status"] = "completed_with_model_failures"
    common.write_json(args.output / "report.json", report)
    audit_frozen_models(args.output)


if __name__ == "__main__":
    main()
