"""Same-ID rectification diagnostics, including vertical stereo and invalid data."""

import copy
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from multicamera.calibration.models import CameraIntrinsics, CameraModel, StereoCalibResult
from multicamera.calibration.stereo_diagnostics import measure_rectification


def scene(model, axis, residuals=(0., 0., 0., 0.)):
    k = np.array([[300., 0., 320.], [0., 310., 240.], [0., 0., 1.]])
    d = np.zeros(4 if model == CameraModel.FISHEYE else 5)
    left = CameraIntrinsics(k.copy(), d.copy(), (640, 480), model)
    right = CameraIntrinsics(k.copy(), d.copy(), (640, 480), model)
    p1 = np.column_stack([k, np.zeros(3)])
    p2 = p1.copy()
    direction = 0 if axis == 'horizontal' else 1
    component = 1 - direction
    p2[direction, 3] = -.1 * k[direction, direction]
    stereo = StereoCalibResult(
        np.eye(3), np.array([[-.1], [0.], [0.]]), np.eye(3), np.eye(3),
        model=model, R1=np.eye(3), R2=np.eye(3), P1=p1, P2=p2,
    )
    ideal_left = np.array([[200., 150.], [250., 200.], [300., 250.], [350., 300.]])
    ideal_right = ideal_left.copy()
    ideal_right[:, direction] -= 30
    ideal_right[:, component] -= residuals

    def distort(points):
        if model == CameraModel.PINHOLE:
            return points.copy()
        normalized = (points - [k[0, 2], k[1, 2]]) / [k[0, 0], k[1, 1]]
        return cv2.fisheye.distortPoints(normalized.reshape(-1, 1, 2), k, d).reshape(-1, 2)

    cal = SimpleNamespace(
        model=model,
        _img_points_left=[distort(ideal_left[:1]), distort(ideal_left[1:])],
        _img_points_right=[distort(ideal_right[:1]), distort(ideal_right[1:])],
        frame_ids=['0002', '0018'],
        _corner_ids=[np.array([7]), np.array([2, 9, 23])],
    )
    return cal, left, right, stereo


@pytest.mark.parametrize('model', list(CameraModel))
@pytest.mark.parametrize('axis', ['horizontal', 'vertical'])
def test_known_aligned_geometry_and_source_id_gaps(model, axis):
    args = scene(model, axis)
    result = measure_rectification(*args)
    assert result['axis'] == axis
    assert result['coordinate'] == ('y' if axis == 'horizontal' else 'x')
    assert result['status'] == 'ok' and result['unit'] == 'px'
    assert result['point_count'] == 4 and result['frame_count'] == 2
    assert result['invalid_count'] == 0 and result['valid_point_count'] == 4
    assert result['rms_px'] < 1e-10
    assert [f['frame_id'] for f in result['per_frame']] == ['0002', '0018']
    assert [f['point_count'] for f in result['per_frame']] == [1, 3]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('model', list(CameraModel))
@pytest.mark.parametrize('axis', ['horizontal', 'vertical'])
def test_signed_perturbations_and_point_weighted_aggregate(model, axis):
    expected = np.array([1., -2., 3., -4.])
    result = measure_rectification(*scene(model, axis, expected))
    actual = [e for frame in result['per_frame'] for e in frame['signed_errors_px']]
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert result['rms_px'] == pytest.approx(np.sqrt(7.5))
    assert result['mean_abs_px'] == pytest.approx(2.5)
    assert result['median_abs_px'] == pytest.approx(2.5)
    assert result['p95_abs_px'] == pytest.approx(3.85)
    assert result['max_abs_px'] == pytest.approx(4.)
    assert result['per_frame'][0]['rms_px'] == pytest.approx(1.)
    assert result['per_frame'][1]['rms_px'] == pytest.approx(np.sqrt(29 / 3))


@pytest.mark.parametrize('model', list(CameraModel))
@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_observations_are_counted_not_silently_omitted(model, side, value):
    args = scene(model, 'horizontal', [1., 2., 3., 4.])
    getattr(args[0], f'_img_points_{side}')[1][0, 0] = value
    result = measure_rectification(*args)
    assert result['status'] == 'invalid'
    assert result['point_count'] == 4 and result['invalid_count'] == 1
    assert result['valid_point_count'] == 3
    assert result['per_frame'][0]['status'] == 'ok'
    frame = result['per_frame'][1]
    assert frame['status'] == 'invalid' and frame['invalid_count'] == 1
    assert frame['signed_errors_px'][0] is None
    for key in ('rms_px', 'mean_abs_px', 'median_abs_px', 'p95_abs_px', 'max_abs_px'):
        assert result[key] is None and frame[key] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('model', list(CameraModel))
def test_nonfinite_projected_point_and_overflow_are_invalid(model, monkeypatch):
    args = scene(model, 'vertical')
    namespace = cv2.fisheye if model == CameraModel.FISHEYE else cv2
    calls = []

    def project(points, *a, **kw):
        calls.append(1)
        out = np.zeros_like(points)
        if len(calls) == 1:
            out[0, 0, 1] = np.inf  # Invalid even though the measured component is x.
        elif len(calls) == 3:
            out[0, 0, 0] = np.finfo(np.float64).max
        elif len(calls) == 4:
            out[0, 0, 0] = -np.finfo(np.float64).max
        return out

    monkeypatch.setattr(namespace, 'undistortPoints', project)
    result = measure_rectification(*args)
    assert result['status'] == 'invalid' and result['invalid_count'] == 2
    assert result['per_frame'][0]['signed_errors_px'] == [None]
    assert result['per_frame'][1]['signed_errors_px'][0] is None
    json.dumps(result, allow_nan=False)


def test_huge_finite_outlier_is_not_clipped_or_converted_to_distance(monkeypatch):
    args = scene(CameraModel.PINHOLE, 'horizontal')
    calls = []

    def project(points, *a, **kw):
        calls.append(1)
        out = np.zeros_like(points)
        if len(calls) == 3:
            out[0, 0, 1] = 1e200
        return out

    monkeypatch.setattr(cv2, 'undistortPoints', project)
    result = measure_rectification(*args)
    assert result['status'] == 'ok'
    assert result['max_abs_px'] == 1e200
    assert result['rms_px'] == pytest.approx(5e199)
    assert result['point_count'] == 4 and result['invalid_count'] == 0
    assert result['unit'] == 'px'
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('field,shape', [('R1', (3, 3)), ('R2', (3, 3)),
                                       ('P1', (3, 4)), ('P2', (3, 4))])
@pytest.mark.parametrize('bad', ['missing', 'shape', 'nonfinite'])
def test_invalid_or_missing_rectification_rejected(field, shape, bad):
    args = scene(CameraModel.PINHOLE, 'horizontal')
    value = None if bad == 'missing' else np.ones((2, 2) if bad == 'shape' else shape)
    if bad == 'nonfinite':
        value.flat[0] = np.nan
    setattr(args[3], field, value)
    with pytest.raises(ValueError, match=field):
        measure_rectification(*args)


@pytest.mark.parametrize('index', [0, 1, 2, 3])
def test_model_mismatch_is_explicit(index):
    args = scene(CameraModel.PINHOLE, 'horizontal')
    args[index].model = CameraModel.FISHEYE
    with pytest.raises(ValueError, match='models must match'):
        measure_rectification(*args)


@pytest.mark.parametrize('bad', [np.zeros(8), np.zeros((2, 3)), np.zeros((2, 2, 2)),
                               np.zeros((0, 2))])
def test_bad_point_shape_rejected(bad):
    args = scene(CameraModel.PINHOLE, 'horizontal')
    args[0]._img_points_left[0] = bad
    with pytest.raises(ValueError):
        measure_rectification(*args)


@pytest.mark.parametrize('bad', ['point_count', 'frame_count', 'frame_ids', 'missing_id'])
def test_missing_correspondences_or_frame_ids_are_not_zipped_away(bad):
    args = scene(CameraModel.PINHOLE, 'horizontal')
    cal = args[0]
    if bad == 'point_count':
        cal._img_points_right[1] = cal._img_points_right[1][:2]
    elif bad == 'frame_count':
        cal._img_points_right.pop()
    elif bad == 'frame_ids':
        cal.frame_ids.pop()
    else:
        cal.frame_ids[0] = None
    with pytest.raises(ValueError):
        measure_rectification(*args)


@pytest.mark.parametrize('layout', ['flat', 'column', 'row'])
def test_supported_point_layouts(layout):
    args = scene(CameraModel.PINHOLE, 'vertical', [1., 2., 3., 4.])
    for name in ('_img_points_left', '_img_points_right'):
        points = getattr(args[0], name)
        for i, p in enumerate(points):
            points[i] = p.reshape((-1, 1, 2) if layout == 'column' else
                                  (1, -1, 2) if layout == 'row' else (-1, 2))
    assert measure_rectification(*args)['rms_px'] == pytest.approx(np.sqrt(7.5))


@pytest.mark.parametrize('model', list(CameraModel))
def test_inputs_unchanged_and_rotations_and_projections_forwarded(model, monkeypatch):
    args = scene(model, 'horizontal')
    args[3].R1 = cv2.Rodrigues(np.array([0., 0., .03]))[0]
    args[3].R2 = cv2.Rodrigues(np.array([-.02, 0., 0.]))[0]
    before = copy.deepcopy(args)
    namespace = cv2.fisheye if model == CameraModel.FISHEYE else cv2
    original = namespace.undistortPoints
    calls = []

    def project(points, k, d, *, R, P):
        calls.append((R.copy(), P.copy()))
        return original(points, k, d, R=R, P=P)

    monkeypatch.setattr(namespace, 'undistortPoints', project)
    measure_rectification(*args)
    for i, (r, p) in enumerate(calls):
        np.testing.assert_array_equal(r, before[3].R1 if i % 2 == 0 else before[3].R2)
        np.testing.assert_array_equal(p, before[3].P1 if i % 2 == 0 else before[3].P2)
    for actual, expected in zip(args, before):
        for key, old in vars(expected).items():
            new = getattr(actual, key)
            if isinstance(old, np.ndarray):
                np.testing.assert_array_equal(new, old)
            elif isinstance(old, list) and old and isinstance(old[0], np.ndarray):
                for a, b in zip(new, old):
                    np.testing.assert_array_equal(a, b)
            else:
                assert new == old


@pytest.mark.parametrize('field,value', [
    ('camera_matrix', np.eye(2)),
    ('camera_matrix', np.diag([0., 1., 1.])),
    ('dist_coeffs', np.zeros((2, 2))),
    ('dist_coeffs', np.zeros(6)),
    ('dist_coeffs', np.array([0., 0., 0., 0., np.nan])),
])
@pytest.mark.parametrize('side', [1, 2])
def test_bad_intrinsics_rejected(side, field, value):
    args = scene(CameraModel.PINHOLE, 'horizontal')
    setattr(args[side], field, value)
    with pytest.raises(ValueError):
        measure_rectification(*args)


def test_fisheye_rejects_pinhole_coefficient_count():
    args = scene(CameraModel.FISHEYE, 'vertical')
    args[1].dist_coeffs = np.zeros(5)
    with pytest.raises(ValueError, match='dist_coeffs'):
        measure_rectification(*args)


def test_axis_uses_dominant_projection_translation_not_stored_T():
    args = scene(CameraModel.PINHOLE, 'vertical', [1., 2., 3., 4.])
    args[3].P2[0, 3] = 1.
    result = measure_rectification(*args)
    assert result['axis'] == 'vertical' and result['coordinate'] == 'x'
    assert result['rms_px'] == pytest.approx(np.sqrt(7.5))


def test_zero_baseline_projection_cannot_determine_axis():
    args = scene(CameraModel.PINHOLE, 'horizontal')
    args[3].P2[:, 3] = 0
    with pytest.raises(ValueError, match='translation'):
        measure_rectification(*args)


def test_empty_dataset_does_not_report_perfect_alignment():
    args = scene(CameraModel.PINHOLE, 'horizontal')
    cal = args[0]
    cal._img_points_left.clear()
    cal._img_points_right.clear()
    cal.frame_ids.clear()
    result = measure_rectification(*args)
    assert result['status'] == 'invalid'
    assert result['point_count'] == result['frame_count'] == 0
    assert result['rms_px'] is None
    json.dumps(result, allow_nan=False)
