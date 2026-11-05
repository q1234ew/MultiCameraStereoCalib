"""Rectification controls: common canvas, epipolar geometry, masks and stretch."""
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

pytest.importorskip("mrcal")
sys.path.insert(0, str(Path(__file__).parents[1]))
import assess_mrcal_rectification as audit


def cameras(translation_x=-.05):
    intrinsics = ("LENSMODEL_OPENCV5", np.array([300., 300., 320., 180., .01, 0., 0., 0., 0.]))
    return tuple(audit.mrcal.cameramodel(intrinsics=intrinsics, imagersize=np.array([640, 360]),
                 extrinsics_rt_fromref=np.zeros(6) if c == 0 else np.array([.01, -.02, .005, translation_x, 0., 0.]))
                 for c in (0, 1))


@pytest.mark.parametrize("lensmodel", audit.PROJECTIONS.values())
@pytest.mark.parametrize("translation_x", [-.05, .05])
def test_perfect_scene_has_same_rectified_rows_for_either_file_role(lensmodel, translation_x):
    models = cameras(translation_x)
    rectified = audit.rectified_pair(models, lensmodel, (640, 360), np.array([100., 70.]))
    points = np.array([[x, y, z] for x in (-.3, .2) for y in (-.2, .1) for z in (.5, 2.)])
    output = []
    for source, rect in zip(models, rectified):
        pixels = audit.mrcal.project(audit.mrcal.transform_point_rt(source.rt_cam_ref(), points), *source.intrinsics())
        transformed, valid = audit.transform_pixels(pixels, source, rect)
        assert valid.all()
        output.append(transformed)
    np.testing.assert_allclose(output[0][:, 1], output[1][:, 1], atol=1e-8)
    np.testing.assert_array_equal(rectified[0].imagersize(), [640, 360])
    np.testing.assert_array_equal(rectified[0].intrinsics()[1], rectified[1].intrinsics()[1])


def test_pinhole_forward_transform_matches_opencv_on_distorted_pixels():
    models = cameras()
    rectified = audit.rectified_pair(models, "LENSMODEL_PINHOLE", (640, 360), np.array([100., 70.]))
    pixels = np.array([[1., 1.], [320., 180.], [630., 350.], [30., 280.]])
    for source, rect in zip(models, rectified):
        f = source.intrinsics()[1]
        k = np.array([[f[0], 0., f[2]], [0., f[1], f[3]], [0., 0., 1.]])
        f_out = rect.intrinsics()[1]
        k_out = np.array([[f_out[0], 0., f_out[2]], [0., f_out[1], f_out[3]], [0., 0., 1.]])
        expected = cv2.undistortPointsIter(pixels[:, None], k, f[4:],
                    audit.rotation_to_rectified(source, rect), k_out,
                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)).reshape(-1, 2)
        actual, valid = audit.transform_pixels(pixels, source, rect)
        assert valid.all()
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        # Native rectification maps contain float32; unprojection requires float64.
        from_map, map_valid = audit.transform_pixels(pixels.astype(np.float32), source, rect)
        assert map_valid.all()
        np.testing.assert_array_equal(from_map, actual)


def test_same_pinhole_camera_has_unit_stretch():
    source = audit.mrcal.cameramodel(intrinsics=("LENSMODEL_PINHOLE", np.array([300., 300., 320., 180.])),
                                     imagersize=np.array([640, 360]), extrinsics_rt_fromref=np.zeros(6))
    quantities, valid = audit.local_stretch(np.array([[1., 1.], [320., 180.], [630., 350.]]), source, source)
    assert valid.all()
    for values in quantities.values():
        np.testing.assert_allclose(values, 1., atol=1e-10)


def test_valid_map_rejects_nan_and_partial_bilinear_footprints():
    pixels = np.array([[0., 0.], [638.9, 358.9], [639., 100.], [-.1, 100.], [np.nan, 20.]])
    np.testing.assert_array_equal(audit.inside(pixels, (640, 360), bilinear=True), [True, True, False, False, False])
    np.testing.assert_array_equal(audit.inside(pixels, (640, 360)), [True, True, True, False, False])


def test_largest_valid_rectangle_avoids_hole():
    mask = np.ones((5, 6), bool)
    mask[2, 2] = False
    result = audit.largest_rectangle(mask)
    x, y, width, height = result["xywh"]
    assert mask[y:y + height, x:x + width].all()
    assert result["pixels"] == 15
    assert audit.largest_rectangle(np.zeros((3, 4), bool))["pixels"] == 0


def test_pinhole_solid_angle_matches_rectangle_formula():
    model = audit.mrcal.cameramodel(intrinsics=("LENSMODEL_PINHOLE", np.array([500., 500., 319.5, 179.5])),
                                   imagersize=np.array([640, 360]), extrinsics_rt_fromref=np.zeros(6))
    weights = audit.solid_angle_weights(model)
    # Boundaries are +/- half a pixel beyond the outer pixel centers.
    x, y = 320 / 500, 180 / 500
    exact = 4 * np.arctan(x * y / np.sqrt(1 + x*x + y*y))
    np.testing.assert_allclose(weights.sum(), exact, rtol=2e-6)
    assert np.isfinite(weights).all() and (weights > 0).all()
