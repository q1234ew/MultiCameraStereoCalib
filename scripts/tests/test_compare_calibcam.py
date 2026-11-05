"""Projection, gauge, masking and convergence controls for the native comparison."""
from copy import deepcopy
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import compare_calibcam as study


def synthetic_system():
    objects = np.array([[x, y, 0.] for y in np.arange(6) * .04 for x in np.arange(9) * .04])
    model = {"K": [np.array([[620., 0., 640.], [0., 615., 360.], [0., 0., 1.]])] * 2,
             "D": [np.array([-.1, .02, .001, -.002, .005])] * 2,
             "R": cv2.Rodrigues(np.array([.03, -.02, .01]))[0], "T": np.array([-.08, .002, -.001])}
    records = []
    for frame in range(7):
        pose = np.array([.15 + frame * .04, -.3 + frame * .07, frame * .02, -.18 + .025 * frame, -.12, .7 + .035 * frame])
        ids = np.arange(len(objects))[np.arange(len(objects)) % 7 != frame]
        xy = [study.common.project_camera(objects[ids], pose, model, camera) for camera in (0, 1)]
        records.append({"capture_id": f"{frame:04}", "ids": ids.tolist(), "left": xy[0].tolist(), "right": xy[1].tolist()})
    return records, objects, model


def require_native():
    pytest.importorskip("jax")
    pytest.importorskip("calibcam")
    return study.backend()


def test_adapter_retains_original_ids_pixels_missing_slots_and_input():
    records, objects, model = synthetic_system()
    before = deepcopy(records)
    corners, calibs = study.native_input(records, objects, model)
    assert records == before
    assert corners.dtype == np.float64
    assert corners.shape == (2, len(records), len(objects), 2)
    for frame, record in enumerate(records):
        for camera, side in enumerate(("left", "right")):
            np.testing.assert_array_equal(corners[camera, frame, record["ids"]], record[side])
            missing = np.setdiff1d(np.arange(len(objects)), record["ids"])
            assert np.isnan(corners[camera, frame, missing]).all()
    np.testing.assert_array_equal(calibs[0]["rvec_cam"], np.zeros(3))
    np.testing.assert_array_equal(calibs[0]["tvec_cam"], np.zeros(3))
    assert all(c["xi"] == 0. for c in calibs)


def test_native_to_relative_camera_transform_handles_nonidentity_world_gauge():
    _, _, model = synthetic_system()
    r0 = cv2.Rodrigues(np.array([.2, -.1, .05]))[0]
    t0 = np.array([.1, -.2, .3])
    r1, t1 = model["R"] @ r0, model["R"] @ t0 + model["T"]
    calibs = [{"rvec_cam": cv2.Rodrigues(r)[0].ravel(), "tvec_cam": t,
               "A": model["K"][i], "k": model["D"][i]} for i, (r, t) in enumerate(((r0, t0), (r1, t1)))]
    converted = study.common_model(calibs)
    np.testing.assert_allclose(converted["R"], model["R"], atol=1e-14)
    np.testing.assert_allclose(converted["T"], model["T"], atol=1e-14)


def test_native_all_five_distortions_free_with_fixed_reference_and_matching_projection():
    _, functions, _, _ = require_native()
    records, objects, model = synthetic_system()
    corners, calibs = study.native_input(records, objects, model)
    before = corners.copy()
    opts = study.native_options()
    for free in opts["free_vars"]:
        np.testing.assert_array_equal(free["k"], np.ones(5))
        assert not free["xi"]
    args, parameters = functions.make_optim_input(objects, calibs, corners, opts)
    assert len(parameters) == 24 + 6 * len(records)
    assert not args["mask_opt"][:3].any()
    audit = study.projection_audit(parameters, args)
    assert audit["max_absolute_residual_difference_px"] < 1e-9
    assert audit["observed_camera_points"] == 2 * sum(len(r["ids"]) for r in records)
    np.testing.assert_array_equal(corners, before)


def test_native_sparse_automatic_jacobian_matches_independent_finite_differences():
    _, functions, _, _ = require_native()
    records, objects, model = synthetic_system()
    corners, calibs = study.native_input(records, objects, model)
    args, parameters = functions.make_optim_input(objects, calibs, corners, study.native_options())
    audit = study.jacobian_audit(parameters, args)
    assert audit["max_scaled_difference"] < 2e-5


def test_native_dense_control_recovers_known_synthetic_projection_without_dropping_points():
    require_native()
    records, objects, truth = synthetic_system()
    initial = deepcopy(truth)
    for k in initial["K"]:
        k[0, 0] *= 1.005
    before = deepcopy(records)
    _, model, result = study.solve_native(records, objects, initial, "dense_control")
    assert records == before
    assert result["success"]
    assert result["sse_px2"] < 1e-10
    np.testing.assert_allclose(model["T"], truth["T"], atol=1e-7)
