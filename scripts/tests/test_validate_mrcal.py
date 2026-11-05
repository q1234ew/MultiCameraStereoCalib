"""Check ground-truth injection, gauge compensation and uncertainty calibration."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("mrcal")
sys.path.insert(0, str(Path(__file__).parents[1]))
import validate_mrcal as validation


@pytest.fixture
def geometry():
    parameters = np.array([900., 910., 640., 360., -.1, .02, .001, -.002, .005])
    frames = np.array([[.4 * np.sin(i), .4 * np.cos(i), .12 * np.sin(i * .3),
                        -.28 + .22 * np.sin(i * .6), -.2 + .13 * np.cos(i * .7), .85 + .05 * (i % 8)]
                       for i in range(30)])
    observations = np.zeros((60, 6, 8, 3))
    observations[..., 2] = 1
    observations[:, 0, 0, 2] = -1
    inputs = {"intrinsics": np.array([parameters, parameters]),
              "rt_cam_ref": np.array([[.005, -.01, .003, -.12, .001, .002]]),
              "rt_ref_frame": frames, "observations_board": observations,
              "indices_frame_camintrinsics_camextrinsics": np.array(
                  [(i, camera, camera - 1) for i in range(30) for camera in range(2)], dtype=np.int32),
              "lensmodel": "LENSMODEL_OPENCV5", "imagersizes": np.array([[1280, 720]] * 2, dtype=np.int32),
              "calibration_object_spacing": .08,
              "do_optimize_intrinsics_core": True, "do_optimize_intrinsics_distortions": True,
              "do_optimize_extrinsics": True, "do_optimize_frames": True}
    rendered = validation.render_observations(inputs)
    xy = rendered["observations_board"][..., :2]
    outside = np.any((xy < 0) | (xy >= np.array([1280, 720])), axis=-1)
    rendered["observations_board"][..., 2][outside] = -1
    return rendered


def test_injection_is_reproducible_exact_and_never_modifies_missing_slots(geometry):
    original = copy.deepcopy(geometry["observations_board"])
    a, truth = validation.inject_outliers(geometry, .05, 20., np.random.default_rng(4))
    b, truth_b = validation.inject_outliers(geometry, .05, 20., np.random.default_rng(4))
    np.testing.assert_array_equal(a["observations_board"], b["observations_board"])
    np.testing.assert_array_equal(truth, truth_b)
    np.testing.assert_array_equal(geometry["observations_board"], original)
    assert not truth[:, 0, 0].any()
    delta = a["observations_board"][..., :2] - original[..., :2]
    np.testing.assert_allclose(np.linalg.norm(delta[truth], axis=-1), 20., atol=1e-10)
    np.testing.assert_array_equal(delta[~truth], 0)


def test_detection_metrics_do_not_count_missing_corners_as_rejected():
    valid = np.array([True, True, True, False])
    truth = np.array([True, False, False, False])
    measured = validation.detection_metrics(valid, truth, np.array([-1., -1., 1., -1.]))
    assert measured["true_positive"] == 1
    assert measured["false_positive"] == 1
    assert measured["recall"] == 1
    assert measured["false_positive_rate"] == .5


def test_mean_pcam_removes_reference_frame_gauge_changes(geometry):
    import mrcal
    pixels = np.array([[128., 72.], [640., 360.], [1152., 648.]])
    moved = validation.copy_inputs(geometry)
    # Equivalent description after changing reference coordinates. Camera0 remains fixed
    # in normal solves; this transformation-only check exercises camera1 compensation.
    shift = np.array([.03, -.02, .01, .1, -.05, .02])
    moved["rt_ref_frame"] = mrcal.compose_rt(shift, geometry["rt_ref_frame"])
    moved["rt_cam_ref"] = mrcal.compose_rt(geometry["rt_cam_ref"], mrcal.invert_rt(shift))
    for distance in (1., 3., None):
        q = validation.reproject_mean_pcam(geometry, moved, pixels, 1, distance)
        np.testing.assert_allclose(q, pixels, atol=1e-7)


def test_rejection_recovers_known_faults(geometry):
    noisy = validation.add_noise(geometry, .1, np.random.default_rng(10))
    bad, truth = validation.inject_outliers(noisy, .03, 15., np.random.default_rng(12))
    fitted, _ = validation.fit(bad, rejection=True)
    stats = validation.detection_metrics(geometry["observations_board"][..., 2] > 0,
                                         truth, fitted["observations_board"][..., 2])
    assert stats["recall"] == 1.
    assert stats["false_positive"] == 0


def test_uncertainty_matches_independent_resolves_and_catches_wrong_scale(geometry, tmp_path):
    result = validation.uncertainty_experiment(geometry, tmp_path, 60, seed=20261026)
    assert not result["failures"]
    assert result["checks_pass"]
    saved = np.load(tmp_path / "uncertainty_sigma_0.1.npz")
    wrong = validation.covariance_check(saved["projections"], saved["predicted_covariance"] * 9,
                                         saved["pixels"], .1, family_queries=162)
    assert not wrong["variance_checks_pass"]


def test_weak_coverage_extrapolation_does_not_get_a_false_pass(geometry, tmp_path):
    weak = validation.copy_inputs(geometry)
    weak["rt_ref_frame"] = np.array(
        [[.22 * np.sin(i), .3 * np.cos(i), .12 * np.sin(i * .3),
          -.15 + .03 * np.cos(i), -.1 + .02 * np.sin(i), .8 + .06 * i]
         for i in range(12)])
    weak["observations_board"] = np.zeros((24, 6, 8, 3))
    weak["observations_board"][..., 2] = 1
    weak["observations_board"][:, 0, 0, 2] = -1
    weak["indices_frame_camintrinsics_camextrinsics"] = np.array(
        [(i, c, c - 1) for i in range(12) for c in range(2)], dtype=np.int32)
    weak["calibration_object_spacing"] = .04
    weak = validation.render_observations(weak)
    report = validation.uncertainty_experiment(weak, tmp_path, 60, seed=20261026)
    # This small-board, narrow-coverage fixture has nonlinear extrapolation at edge queries.
    # It must remain a reported failure instead of broadening the acceptance interval.
    assert not report["checks_pass"]
    assert any(not level["variance_checks_pass"] for level in report["levels"])
