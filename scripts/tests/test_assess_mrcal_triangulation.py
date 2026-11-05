"""Geometric controls: plane fit cannot hide scale errors; correlations matter."""
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("mrcal")
sys.path.insert(0, str(Path(__file__).parents[1]))
import assess_mrcal_triangulation as audit


def test_summary_reports_actual_capture_and_point_counts():
    text = audit.summarize({"holdout_captures": 6, "holdout_points": 617, "models": {}})
    assert "6 对" in text
    assert "617 个三维角点" in text
    assert "756" not in text


def board():
    return np.array([[x * .04, y * .04, 0.] for y in range(3) for x in range(4)])


def test_plane_fit_does_not_establish_correct_scale():
    objects = board()
    points = objects * 1.1 + [0, 0, 1]
    assert np.max(np.abs(audit.plane_residuals(points))) < 1e-12
    _, _, _, nominal, error = audit.pair_geometry(points, objects)
    np.testing.assert_allclose(error / nominal, .1)


def test_distance_and_plane_are_invariant_to_rigid_pose():
    objects = board()
    rotation = audit.mrcal.R_from_r(np.array([.2, -.3, .7]))
    points = objects @ rotation.T + [.2, -.4, 1.2]
    assert np.max(np.abs(audit.pair_geometry(points, objects)[-1])) < 1e-12
    assert np.max(np.abs(audit.plane_residuals(points))) < 1e-12
    np.testing.assert_allclose(audit.rigid_board_residuals(points, objects), 0, atol=1e-12)


def test_common_translation_uncertainty_cancels_in_board_distance():
    # A 2mm shared translation changes each Z, but no inter-corner distances.
    points = board() + [0, 0, 1]
    i, j, direction, _, _ = audit.pair_geometry(points, board())
    covariance = np.tile(np.eye(3) * .002**2, (len(points), len(points), 1, 1)).transpose(0, 2, 1, 3)
    np.testing.assert_allclose(audit.pair_variance(covariance, i, j, direction), 0, atol=1e-20)
    assert (audit.diagonal_blocks(covariance)[:, 2, 2] > 0).all()


def test_invalid_covariance_is_rejected():
    covariance = np.eye(6).reshape(2, 3, 2, 3)
    covariance[0, 0, 0, 0] = -1
    with pytest.raises(ValueError, match="Invalid covariance"):
        audit.covariance_diagnostics(covariance)
    with pytest.raises(ValueError, match="Non-finite metric"):
        audit.statistics([0., np.nan])


@pytest.mark.parametrize("translation_x", [-.05, .05])
def test_triangulation_and_observation_uncertainty_with_either_file_role(translation_x):
    intrinsics = ("LENSMODEL_OPENCV5", np.array([900., 900., 640., 360., 0., 0., 0., 0., 0.]))
    rt = np.array([0., 0., 0., translation_x, 0., 0.])
    models = tuple(audit.mrcal.cameramodel(intrinsics=intrinsics, imagersize=np.array([1280, 720]),
                                         extrinsics_rt_fromref=np.zeros(6) if c == 0 else rt)
                   for c in (0, 1))
    points = board() + [-.1, -.1, 1.]
    q = np.stack((audit.mrcal.project(points, *intrinsics),
                  audit.mrcal.project(audit.mrcal.transform_point_rt(rt, points), *intrinsics)), axis=1)
    p, obs = audit.mrcal.triangulate(q, models, q_observation_stdev=.2)
    np.testing.assert_allclose(p, points, atol=1e-10)
    checks = audit.observation_sampling(q, models, obs, (1280, 720), 1000, 20261027)
    assert all(c["passed"] for c in checks)
    wrong = audit.observation_sampling(q, models, obs * 9, (1280, 720), 1000, 20261027)
    assert all(not c["passed"] for c in wrong)
