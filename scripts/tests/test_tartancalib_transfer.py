"""Known-image controls for the ChArUco adaptation of iterative subpixel refinement."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).parents[1]))
import verify_tartancalib_transfer as study
import diagnose_tartancalib_transfer as diagnostic


def synthetic_corner():
    y, x = np.indices((81, 91))
    center = np.array([43.2, 39.7])
    image = .5 + .45 * erf((x - center[0]) / 1.2) * erf((y - center[1]) / 1.2)
    return np.rint(image * 255).astype(np.uint8), center + .5


def test_refinement_converges_from_different_seeds_without_half_pixel_shift():
    gray, truth = synthetic_corner()
    original = truth + [.2, -.2]
    outputs = []
    for offset in ([.6, -.4], [-.6, .4], [0., 0.]):
        refined, reasons, _ = study.refine_pixels(gray, [original], [original + offset], [5])
        assert reasons == {"accepted": 1}
        np.testing.assert_allclose(refined[0], truth, atol=.03, rtol=0.)
        outputs.append(refined[0])
    assert np.max(np.ptp(outputs, axis=0)) < .002


def test_no_texture_border_invalid_seed_retain_original_instead_of_model():
    original = np.array([[30., 30.], [1., 1.], [20., 20.], [40., 40.]])
    seed = original.copy()
    seed[2] += [3., 0.]
    seed[3] = np.nan
    output, reasons, _ = study.refine_pixels(np.full((61, 61), 127, np.uint8), original, seed, [3] * 4)
    np.testing.assert_array_equal(output, original)
    assert reasons == {"border": 1, "invalid_seed": 2, "no_corner_texture": 1}


def test_window_adapts_to_scale_and_ignores_point_itself():
    pixels = np.array([[0., 0.], [20., 0.], [0., 20.]])
    np.testing.assert_allclose(study.nearest_spacing(pixels), 20.)
    np.testing.assert_array_equal(study.window_sizes(pixels, .1, 1), [2, 2, 2])
    np.testing.assert_array_equal(study.window_sizes(pixels * 2, .1, 1), [4, 4, 4])
    np.testing.assert_array_equal(study.window_sizes(pixels * 10, .1, 1), [8, 8, 8])
    with pytest.raises(ValueError, match="Degenerate"):
        study.nearest_spacing([[0., 0.], [0., 0.]])


def test_rendered_charuco_preserves_native_pixel_convention_and_ids():
    board = study.common.CharucoBoard(study.common.CharucoBoard.from_dict({
        "cols": 15, "rows": 10, "square_length": .04, "marker_length": .03,
        "dict_type": "DICT_4X4_1000"}).config)
    image = board.generate_image(pixel_size=40, margin=20)
    detected = study.common.CharucoDetector(board).detect(image)
    assert detected.valid
    ids = detected.charuco_ids.ravel()
    xy = detected.charuco_corners.reshape(-1, 2)
    expected = board.object_points[ids, :2].astype(float) * 1000 + 20
    refined, reasons, _ = study.refine_pixels(image, xy, xy, np.full(len(ids), 4))
    assert reasons == {"accepted": len(ids)}
    np.testing.assert_allclose(refined, expected, atol=.03, rtol=0.)
    record = {"capture_id": "synthetic", "ids": ids.tolist(), "left": xy.tolist(), "right": xy.tolist()}
    source = [record]
    output, diagnostic = study.refine_records(source, {("synthetic", c): image for c in (0, 1)},
                                               board.object_points, None, .1, 1, image_only=True)
    assert output[0]["ids"] == record["ids"]
    assert source == [record]
    assert diagnostic["original_observations_retained"] == 2 * len(ids)


def test_calibrated_detector_configuration_does_not_mutate_intrinsics():
    board = study.common.CharucoBoard.from_dict({"cols": 15, "rows": 10,
        "square_length": .04, "marker_length": .03, "dict_type": "DICT_4X4_1000"})
    k = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
    d = np.zeros(5)
    detector = study.calibrated_detector(board, k, d)
    parameters = detector.getCharucoParameters()
    np.testing.assert_array_equal(parameters.cameraMatrix, k)
    np.testing.assert_array_equal(parameters.distCoeffs.ravel(), d)
    parameters.cameraMatrix[0, 0] += 10.
    assert k[0, 0] == 500.


def test_native_redetection_retains_ids_and_raw_values_on_missing_or_large_moves(monkeypatch):
    class FakeDetector:
        def detectBoard(self, gray):
            return (np.array([[[10.2, 10.]], [[40., 40.]], [[50., 50.]]], np.float32),
                    np.array([[0], [2], [3]], np.int32), [], None)

    monkeypatch.setattr(study, "calibrated_detector", lambda *args: FakeDetector())
    points = [[10., 10.], [20., 20.], [30., 30.]]
    record = {"capture_id": "a", "ids": [0, 1, 2], "left": points, "right": points}
    model = {"K": [np.eye(3)] * 2, "D": [np.zeros(5)] * 2}
    images = {("a", camera): np.zeros((61, 61), np.uint8) for camera in (0, 1)}
    output, report = diagnostic.redetect([record], images, None, model)
    assert output[0]["ids"] == [0, 1, 2]
    assert record["left"] == points
    for side in ("left", "right"):
        np.testing.assert_allclose(output[0][side][0], [10.2, 10.], atol=1e-6)
        assert output[0][side][1:] == points[1:]
    assert report["original_observations_retained"] == 6
    assert report["new_common_ids_not_used"] == 1
    assert report["per_image"][0]["counts"] == {"accepted": 1, "missing": 1, "invalid_or_over_2px": 1}


def test_calibrated_redetection_on_known_rendered_board():
    board = study.common.CharucoBoard.from_dict({"cols": 15, "rows": 10,
        "square_length": .04, "marker_length": .03, "dict_type": "DICT_4X4_1000"})
    image = board.generate_image(pixel_size=40, margin=20)
    detected = study.common.CharucoDetector(board).detect(image)
    ids = detected.charuco_ids.ravel()
    xy = detected.charuco_corners.reshape(-1, 2)
    record = {"capture_id": "a", "ids": ids.tolist(), "left": xy.tolist(), "right": xy.tolist()}
    k = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
    model = {"K": [k] * 2, "D": [np.zeros(5)] * 2}
    output, report = diagnostic.redetect([record], {("a", c): image for c in (0, 1)}, board, model)
    expected = board.object_points[ids, :2].astype(float) * 1000 + 20
    np.testing.assert_allclose(output[0]["left"], expected, atol=.03, rtol=0.)
    assert report["new_common_ids_not_used"] == 0
    assert report["per_image"][0]["counts"]["accepted"] == len(ids)
