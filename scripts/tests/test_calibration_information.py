"""Independent controls for anisotropic whitening, pose elimination and BA derivatives."""
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import verify_calibration_information as information


def fixture_system():
    model = {"K": [np.array([[500., 0, 320.], [0, 510., 240.], [0, 0, 1.]])] * 2,
             "D": [np.array([.03, -.01, .001, -.002, .005])] * 2,
             "R": cv2.Rodrigues(np.array([.03, -.02, .01]))[0], "T": np.array([-.05, .002, -.001])}
    parameters = information.pack_model(model)
    pose = np.array([.15, -.2, .03, -.1, -.1, .8])
    objects = np.array([[x, y, 0.] for y in (0., .04, .08) for x in (0., .04, .08, .12)])
    w = np.tile(information.sqrt_information(np.array([[2., .3], [.3, .5]])), (len(objects), 1, 1))
    observations = [np.zeros((len(objects), 2))] * 2
    return parameters, pose, objects, observations, [w, w]


def test_analytic_stereo_chain_matches_finite_differences():
    parameters, pose, objects, observations, w = fixture_system()
    _, a, b, _ = information.frame_system(parameters, pose, objects, observations, w)
    state = np.r_[parameters, pose]
    numerical = []
    for i in range(len(state)):
        step = 1e-6 if i < 18 else 1e-7
        delta = np.zeros(len(state))
        delta[i] = step
        plus, minus = state + delta, state - delta
        rp = information.frame_system(plus[:24], plus[24:], objects, observations, w)[0]
        rm = information.frame_system(minus[:24], minus[24:], objects, observations, w)[0]
        numerical.append((rp - rm) / (2 * step))
    np.testing.assert_allclose(np.c_[a, b], np.array(numerical).T, rtol=3e-5, atol=2e-5)


def test_whitening_retains_xy_covariance_instead_of_scalar_weight():
    precision = np.array([[4., 1.2], [1.2, 1.]])
    residual = np.array([.3, -.7])
    w = information.sqrt_information(precision)
    np.testing.assert_allclose(w.T @ w, precision, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(w @ residual)**2, residual @ precision @ residual)
    assert not np.isclose(np.linalg.norm(w @ residual)**2, np.trace(precision) / 2 * (residual @ residual))
    with pytest.raises(ValueError, match="positive definite"):
        information.sqrt_information(np.array([[1., 0.], [0., 0.]]))


def test_pose_elimination_matches_full_covariance_and_removes_pose_only_information():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=(80, 5)), rng.normal(size=(80, 6))
    full = np.c_[a, b]
    expected = np.linalg.inv(full.T @ full)[:5, :5]
    reduced = information.marginal_information(a, b)
    np.testing.assert_allclose(np.linalg.inv(reduced), expected, rtol=1e-12, atol=1e-14)
    pose_only = b @ rng.normal(size=(6, 5))
    assert np.linalg.norm(information.marginal_information(pose_only, b)) < 1e-20
    with pytest.raises(ValueError, match="rank deficient"):
        information.marginal_information(a, np.zeros_like(b))


def test_information_gain_reduces_marginal_intrinsic_trace_and_rejects_rank_loss():
    rng = np.random.default_rng(8)
    j = rng.normal(size=(120, 24))
    information0 = j.T @ j
    before, _ = information.information_trace(information0)
    extra = rng.normal(size=(100, 24))
    after, _ = information.information_trace(information0 + extra.T @ extra)
    assert after < before
    with pytest.raises(ValueError, match="rank deficient"):
        information.information_trace(np.zeros((24, 24)))


def test_gradient_information_decreases_under_blur_and_bounds_preserve_all_points():
    y, x = np.indices((61, 61))
    sharp = ((x >= 30) ^ (y >= 30)).astype(np.float32)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 2.)
    sharp_c, _ = information.patch_information(sharp, [[30., 30.]])
    blur_c, _ = information.patch_information(blurred, [[30., 30.]])
    assert np.trace(blur_c[0]) < np.trace(sharp_c[0])
    raw, fallback = information.patch_information(sharp, [[30., 30.], [0., 0.]])
    normalized, _ = information.normalize_information(raw, float(np.trace(sharp_c[0]) / 2), fallback)
    assert len(normalized) == 2
    np.testing.assert_allclose(normalized[1], np.eye(2))
    assert fallback.tolist() == [False, True]
    assert (np.linalg.eigvalsh(normalized) >= .25).all()


def test_predicted_opening_angle_weights_are_spd_and_do_not_change_camera_state():
    parameters, pose, objects, _, _ = fixture_system()
    before = parameters.copy()
    precision, projected, spacing = information.predicted_information(parameters, pose, objects)
    for c in (0, 1):
        assert np.linalg.eigvalsh(precision[c]).min() >= .25 - 1e-12
        assert np.isfinite(projected[c]).all() and (spacing[c] > 0).all()
    np.testing.assert_array_equal(parameters, before)


def test_known_heteroscedastic_noise_control_favors_correct_weighting():
    parameters, _, _, _, _ = fixture_system()
    objects = np.array([[x, y, 0.] for y in np.linspace(0, .45, 4) for x in np.linspace(0, .6, 5)])
    poses, records, weights = [], [], []
    rng = np.random.default_rng(20)
    for index in range(8):
        pose = np.r_[rng.uniform(-.4, .4, 3), -.3, -.2, rng.uniform(.65, 1.)]
        poses.append(pose)
        _, _, _, pixels = information.frame_system(parameters, pose, objects,
            [np.zeros((len(objects), 2))] * 2, [np.tile(np.eye(2), (len(objects), 1, 1))] * 2)
        records.append({"capture_id": str(index), "ids": list(range(len(objects))),
                        "left": pixels[0].tolist(), "right": pixels[1].tolist()})
        matrices = np.array([np.diag([.3, 3.]) if i % 2 else np.diag([3., .3]) for i in range(len(objects))])
        weights.append([matrices, matrices])
    state = np.r_[parameters, np.array(poses).ravel()]
    result = information.linearized_noise_control(state, records, objects, weights, (640, 480), samples=300)
    assert result["predicted_weighted_over_uniform"] < .8
    assert result["empirical_weighted_over_uniform"] < .8
