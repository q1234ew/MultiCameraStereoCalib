"""Controls with known geometry/noise for the independent DiscoCal transfer study."""
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
import verify_discocal_transfer as study
import assess_discocal_uncertainty as propagation


def test_projected_region_centroid_matches_analytic_ellipse_not_projected_center():
    k = np.array([[800., 0., 640.], [0., 800., 360.], [0., 0., 1.]])
    r = cv2.Rodrigues(np.array([.7, -.4, 0.]))[0]
    center = np.array([.32, .12])
    radius = .04
    conic = np.eye(3)
    conic[:2, 2] = conic[2, :2] = -center
    conic[2, 2] = center @ center - radius ** 2
    homography = k @ np.c_[r[:, :2], [0., 0., .7]]
    inverse = np.linalg.inv(homography)
    projected_conic = inverse.T @ conic @ inverse
    expected = -np.linalg.solve(projected_conic[:2, :2], projected_conic[:2, 2])
    theta = np.arange(8192) * 2 * np.pi / 8192
    boundary = np.c_[center + radius * np.c_[np.cos(theta), np.sin(theta)], np.ones(len(theta))]
    homogeneous = boundary @ homography.T
    polygon = homogeneous[:, :2] / homogeneous[:, 2:]
    measured = study.polygon_centroid(polygon)
    np.testing.assert_allclose(measured, expected, atol=1e-6, rtol=0.)
    np.testing.assert_allclose(study.polygon_centroid(polygon[::-1]), expected, atol=1e-6, rtol=0.)
    center_h = homography @ np.r_[center, 1.]
    assert np.linalg.norm(expected - center_h[:2] / center_h[2]) > .1


def test_distortion_and_perspective_controls_converge_and_front_parallel_is_unbiased():
    result = study.circle_control()
    assert result["frontoparallel_undistorted"]["center_to_region_centroid_px"] < 1e-8
    for case in ("perspective_only", "radial_only", "opencv5_combined"):
        assert result[case]["center_to_region_centroid_px"] > .1
        assert result[case]["integration_refinement_delta_px"] < 1e-5


def test_covariance_prediction_on_independent_known_gaussian_draws():
    rng = np.random.default_rng(19)
    covariance = np.array([[.8, .35], [.35, .4]])
    train = rng.multivariate_normal([0., 0.], covariance, size=(500, 30))
    check = rng.multivariate_normal([0., 0.], covariance, size=(500, 30))
    result = study.covariance_check(train, check)
    assert 1.85 < result["mean_nees"] < 2.15
    assert .93 < result["coverage_95"] < .97
    assert .7 < result["frame_mean_variance_over_independent"] < 1.3


def test_shared_detector_motion_is_visible_in_cross_corner_diagnostic():
    rng = np.random.default_rng(20)
    shared = rng.normal(size=(300, 1, 2))
    train = shared + rng.normal(scale=.1, size=(300, 25, 2))
    check = rng.normal(size=(300, 1, 2)) + rng.normal(scale=.1, size=(300, 25, 2))
    result = study.covariance_check(train, check)
    assert result["frame_mean_variance_over_independent"] > 20


def test_missing_detections_flag_fallback_and_flat_covariance_remains_spd():
    draws = np.zeros((32, 3, 2))
    draws[:9, 1] = np.nan
    draws[:, 2] = np.nan
    mean, cov, usable, count = study.covariance_statistics(draws)
    assert usable.tolist() == [True, False, False]
    assert count.tolist() == [32, 23, 0]
    assert np.isfinite(mean).all()
    assert (np.linalg.eigvalsh(cov) >= 1e-4).all()
    raw = np.linalg.inv(cov)
    weights, _ = study.ba.normalize_information(raw, 1., ~usable)
    assert len(weights) == 3
    np.testing.assert_allclose(weights[1:], np.tile(np.eye(2), (2, 1, 1)))


def test_covariance_rotation_equivariance_preserves_off_diagonal_information():
    rng = np.random.default_rng(21)
    draws = rng.multivariate_normal([1., -2.], [[.3, .08], [.08, .03]], size=(50, 2))
    rotation = np.array([[.6, -.8], [.8, .6]])
    mean, cov, _, _ = study.covariance_statistics(draws)
    rotated_mean, rotated_cov, _, _ = study.covariance_statistics(draws @ rotation.T)
    np.testing.assert_allclose(rotated_mean, mean @ rotation.T, atol=1e-14)
    np.testing.assert_allclose(rotated_cov, rotation @ cov @ rotation.T, atol=1e-14)
    assert abs(cov[0, 0, 1]) > .01


def test_full_ba_influence_predicts_known_noise_with_nuisance_parameters():
    rng = np.random.default_rng(22)
    design = rng.normal(size=(160, 30)) * np.geomspace(1e-3, 1e3, 30)
    covariance = np.tile([[.4, .12], [.12, .1]], (80, 1, 1))
    precision = np.linalg.inv(covariance)
    influence, condition = propagation.influence_matrix(design, precision)
    assert np.isfinite(condition)
    # Independent direct weighted least-squares solutions of many noise draws.
    noise = np.linalg.cholesky(covariance) @ rng.normal(size=(80, 2, 4000))
    w = np.linalg.cholesky(precision).swapaxes(-1, -2)
    solution = np.linalg.lstsq((w @ design.reshape(80, 2, 30)).reshape(160, 30),
                              (w @ noise).reshape(160, 4000), rcond=None)[0][:24]
    np.testing.assert_allclose(influence @ noise.reshape(160, 4000), solution, rtol=1e-7, atol=1e-8)
    predicted = propagation.sandwich_covariance(influence, covariance)
    empirical = np.cov(solution)
    np.testing.assert_allclose(np.diag(empirical), np.diag(predicted), rtol=.08)


def test_uncertainty_map_control_and_missing_detection_refusal():
    rng = np.random.default_rng(23)
    model = {"K": [np.array([[800., 0., 640.], [0., 800., 360.], [0., 0., 1.]])] * 2,
             "D": [np.array([-.1, .01, .001, -.001, 0.])] * 2,
             "R": cv2.Rodrigues(np.array([.01, -.02, .005]))[0], "T": np.array([-.08, .001, .002])}
    objects = np.array([[x, y, 0.] for y in np.arange(6) * .04 for x in np.arange(9) * .04])
    records, poses, archive, information = [], [], {}, []
    for frame in range(8):
        pose = np.r_[rng.uniform(-.45, .45, 3), rng.uniform(-.2, -.05, 2), rng.uniform(.6, 1.)]
        poses.append(pose)
        xy = [study.common.project_camera(objects, pose, model, camera) for camera in (0, 1)]
        records.append({"ids": list(range(len(objects))), "left": xy[0].tolist(), "right": xy[1].tolist()})
        information.append([np.tile(np.eye(2), (len(objects), 1, 1))] * 2)
        for camera in (0, 1):
            samples = xy[camera] + rng.normal(scale=.03, size=(64, len(objects), 2))
            archive[f"frame{frame}_cam{camera}_draws"] = samples
            archive[f"frame{frame}_cam{camera}_covariance"] = study.covariance_statistics(samples[:32])[1]
    state = np.r_[study.ba.pack_model(model), np.array(poses).ravel()]
    result = propagation.assess(state, records, objects, information, archive, (1280, 720))
    assert result["status"] == "completed"
    assert len(result["maps"]) == 6
    for side in result["camera_summaries"]:
        assert side["valid_points"] == side["grid_points"]
        assert .5 < side["check_variance_over_correlated"] < 1.5
    archive["frame0_cam0_draws"][0, 0] = np.nan
    refused = propagation.assess(state, records, objects, information, archive, (1280, 720))
    assert refused["status"] == "unavailable"
