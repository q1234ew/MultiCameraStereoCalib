"""Invalid calibration results must never receive a usable quality verdict."""

from copy import deepcopy

import numpy as np
import pytest

from multicamera.calibration.assessment import CalibrationEvaluator, Grade
from multicamera.calibration.models import (
    AuxCameraCalibration, CameraIntrinsics, CameraModel, MultiCameraRig,
    StereoCalibResult, StereoPairCalibration,
)
from multicamera.calibration.quality_summary import summarize_quality


@pytest.fixture
def rig():
    intr = CameraIntrinsics(
        np.array([[900., 0., 640.], [0., 900., 360.], [0., 0., 1.]]),
        np.zeros((1, 5)), (1280, 720), rms_error=.2, per_frame_errors=[.2] * 12,
    )
    stereo = StereoCalibResult(np.eye(3), np.array([[-.1], [0.], [0.]]),
                              np.eye(3), np.eye(3), rms_error=.2)
    pair = StereoPairCalibration('rgb', 'L', 'R', intr, deepcopy(intr), stereo)
    return MultiCameraRig(pairs={'rgb': pair})


def assert_invalid(rig, reason):
    result = summarize_quality(rig)
    assert result.level == 'poor'
    assert result.title == '质量指标无效，需重新核对'
    assert any(reason in risk for risk in result.risks), result.risks


@pytest.mark.parametrize('error', [-1., float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('target', ['intrinsic', 'stereo'])
def test_invalid_rms_scores_zero_and_explains_failure(rig, target, error):
    pair = rig.pairs['rgb']
    evaluator = CalibrationEvaluator()
    if target == 'intrinsic':
        pair.left_intrinsics.rms_error = error
        report = evaluator.evaluate_intrinsic(pair.left_intrinsics)
    else:
        pair.stereo.rms_error = error
        report = evaluator.evaluate_stereo(pair)
    assert report.grade == Grade.POOR
    assert report.score == 0.
    assert any('无效' in message for message in report.suggestions)
    assert_invalid(rig, '误差指标无效')


@pytest.mark.parametrize('model,distortion', [
    (CameraModel.PINHOLE, None),
    (CameraModel.PINHOLE, np.array([])),
    (CameraModel.PINHOLE, np.array([np.nan] * 5)),
    (CameraModel.PINHOLE, np.array([np.inf] * 5)),
    (CameraModel.PINHOLE, np.zeros(6)),
    (CameraModel.PINHOLE, np.zeros((2, 2))),
    (CameraModel.PINHOLE, np.array(['0'] * 5)),
    (CameraModel.PINHOLE, np.zeros(5, dtype=complex)),
    (CameraModel.FISHEYE, np.zeros(5)),
])
def test_invalid_distortion(rig, model, distortion):
    intr = rig.pairs['rgb'].left_intrinsics
    intr.model, intr.dist_coeffs = model, distortion
    assert_invalid(rig, '畸变系数无效')


@pytest.mark.parametrize('rotation', [
    np.eye(4), np.ones((3, 1)), np.zeros((3, 3)),
    np.diag([1., 1., -1.]), np.diag([1., 1., 2.]),
    np.array([[1., .1, 0.], [0., 1., 0.], [0., 0., 1.]]),
    np.full((3, 3), np.nan), np.full((3, 3), np.inf),
])
def test_invalid_stereo_rotation(rig, rotation):
    rig.pairs['rgb'].stereo.R = rotation
    assert_invalid(rig, '双目旋转 R 无效')


@pytest.mark.parametrize('translation', [
    np.zeros((3, 1)), np.zeros(3), np.zeros((1, 3)),
    np.eye(3), np.ones((3, 1, 1)), np.ones(2),
    np.array([np.nan, 0., 0.]), np.array([np.inf, 0., 0.]),
])
def test_invalid_stereo_translation(rig, translation):
    rig.pairs['rgb'].stereo.T = translation
    assert_invalid(rig, '双目平移 T')


@pytest.mark.parametrize('transform', [
    np.eye(3), np.diag([2., 1., 1., 1.]), np.diag([1., 1., -1., 1.]),
    np.diag([1., 1., 1., 2.]), np.full((4, 4), np.nan), np.full((4, 4), np.inf),
    np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
              [0., 0., 1., 0.], [.1, 0., 0., 1.]]),
])
def test_auxiliary_transform_must_be_rigid(rig, transform):
    rig.aux_cameras['ir'] = AuxCameraCalibration(
        'ir', 'ir', deepcopy(rig.pairs['rgb'].left_intrinsics), transform,
        rms_error=.2, frame_count=12,
    )
    assert_invalid(rig, '4×4 刚体变换')


@pytest.mark.parametrize('model,count', [
    (CameraModel.PINHOLE, count) for count in (4, 5, 8, 12, 14)
] + [(CameraModel.FISHEYE, 4)])
@pytest.mark.parametrize('layout', ['flat', 'row', 'column'])
def test_valid_result_vector_layouts_remain_accepted(rig, model, count, layout):
    shape = {'flat': (count,), 'row': (1, count), 'column': (count, 1)}[layout]
    pair = rig.pairs['rgb']
    for intr in (pair.left_intrinsics, pair.right_intrinsics):
        intr.model = model
        intr.dist_coeffs = np.zeros(shape)
        intr.per_frame_errors = np.full(12, .2)
    pair.stereo.model = model
    pair.stereo.T = pair.stereo.T.reshape({'flat': (3,), 'row': (1, 3),
                                         'column': (3, 1)}[layout])
    pair.stereo.R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    assert summarize_quality(rig).level == 'good'


def test_valid_auxiliary_identity_only_requires_existing_manual_review(rig):
    rig.aux_cameras['ir'] = AuxCameraCalibration(
        'ir', 'ir', deepcopy(rig.pairs['rgb'].left_intrinsics), np.eye(4),
        rms_error=.2, frame_count=12,
    )
    result = summarize_quality(rig)
    assert result.level == 'fair'
    assert not any('无效' in risk for risk in result.risks)


@pytest.mark.parametrize('error', [-1., np.nan, np.inf, -np.inf])
def test_invalid_auxiliary_rms(rig, error):
    rig.aux_cameras['ir'] = AuxCameraCalibration(
        'ir', 'ir', deepcopy(rig.pairs['rgb'].left_intrinsics), np.eye(4),
        rms_error=error, frame_count=12,
    )
    assert_invalid(rig, '融合误差指标无效')


@pytest.mark.parametrize('errors', [[-.1], [np.nan], [np.inf], [[.2]]])
def test_invalid_per_frame_errors_have_specific_reason(rig, errors):
    rig.pairs['rgb'].left_intrinsics.per_frame_errors = errors
    assert_invalid(rig, '逐帧误差包含无效值')


@pytest.mark.parametrize('model', list(CameraModel))
@pytest.mark.parametrize('stereo', [False, True])
def test_existing_rms_thresholds_are_unchanged(model, stereo):
    evaluator = CalibrationEvaluator()
    thresholds = (evaluator.STEREO_THRESHOLDS if stereo else
                  evaluator.INTRINSIC_THRESHOLDS)[model]
    for rms, grade, score in zip(thresholds, (Grade.EXCELLENT, Grade.GOOD, Grade.FAIR),
                                 (90., 70., 40.)):
        assert evaluator._grade_rms(rms, thresholds) == grade
        assert evaluator._rms_to_score(rms, thresholds) == score
    assert evaluator._grade_rms(thresholds[-1] + .001, thresholds) == Grade.POOR
