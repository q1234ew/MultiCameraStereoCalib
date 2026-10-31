"""Physical-domain rejection must precede RMS-based acceptance or file writes."""

from copy import deepcopy
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from multicamera.calibration.health import check_intrinsics, check_stereo_health
from multicamera.calibration.models import (
    CameraIntrinsics, CameraModel, MultiCameraRig, StereoCalibResult, StereoPairCalibration,
)
from multicamera.calibration.quality_summary import summarize_quality
from multicamera.calibration.validated_selection import calibrate_validated
from multicamera.io.export import export_rig_json, export_rig_opencv_standard_yaml
from test_model_selection import calibrators, model_window as model_window


def intrinsics():
    return CameraIntrinsics(np.array([[800., 0, 640], [0, 800., 360], [0, 0, 1]]),
                            np.zeros(5), (1280, 720), rms_error=.2)


def rig_for(intr):
    stereo = StereoCalibResult(np.eye(3), np.array([[-.05], [0.], [0.]]),
                              np.eye(3), np.eye(3), rms_error=.1)
    stereo.R1, stereo.R2 = np.eye(3), np.eye(3)
    stereo.P1 = np.c_[intr.camera_matrix, np.zeros(3)]
    stereo.P2 = stereo.P1.copy()
    stereo.P2[0, 3] = -.05 * intr.camera_matrix[0, 0]
    fx, fy = intr.camera_matrix[0, 0], intr.camera_matrix[1, 1]
    cx, cy = intr.camera_matrix[:2, 2]
    stereo.Q = np.array([[1., 0., 0., -cx], [0., fx/fy, 0., -cy*fx/fy],
                         [0., 0., 0., fx], [0., 0., 20., 0.]])
    return MultiCameraRig(pairs={'pair': StereoPairCalibration('pair', 'L', 'R',
                                                              intr, deepcopy(intr), stereo)})


@pytest.mark.parametrize('model', list(CameraModel))
def test_healthy_lens_and_view_pass(model):
    intr = intrinsics()
    intr.model = model
    if model == CameraModel.FISHEYE:
        intr.dist_coeffs = np.zeros(4)
    assert check_intrinsics(intr)['status'] == 'ok'
    pair = rig_for(intr).pairs['pair']
    assert check_stereo_health(intr, pair.right_intrinsics, pair.stereo)['status'] == 'ok'


def test_low_rms_does_not_allow_folded_lens_to_export(tmp_path):
    intr = intrinsics()
    intr.camera_matrix[:2, :2] = [[463., 0.], [0., 463.]]
    intr.dist_coeffs = np.array([.058, -.034, .0002, .0001, -.0004])
    health = check_intrinsics(intr)
    assert health['status'] == 'failed'
    assert any('反算' in text or '非单调' in text for text in health['reasons'])
    rig = rig_for(intr)
    rig.pairs['pair'].calibration_info['health'] = {'status': 'ok'}  # Never trust stored claims.
    assert not summarize_quality(rig).valid
    for writer, suffix in [(export_rig_json, '.json'), (export_rig_opencv_standard_yaml, '.yml')]:
        target = tmp_path / ('previous' + suffix)
        target.write_bytes(b'previous')
        with pytest.raises(ValueError, match='不可用于立体校正'):
            writer(rig, target)
        assert target.read_bytes() == b'previous'


def test_overzoom_is_rejected_even_when_all_remapped_pixels_are_inside():
    pair = rig_for(intrinsics()).pairs['pair']
    for p in (pair.stereo.P1, pair.stereo.P2):
        p[0, 0] = p[1, 1] = 8000.
    health = check_stereo_health(pair.left_intrinsics, pair.right_intrinsics, pair.stereo)
    assert health['view']['common_valid_fraction'] == 1.
    assert health['status'] == 'failed'
    assert any('裁剪' in text for text in health['reasons'])


def test_invalid_generation_never_replaces_latest_success(tmp_path):
    from multicamera.io.session import CalibrationSession
    left, _, _ = calibrators()
    session = CalibrationSession.create(tmp_path, left.board, [])
    rig = rig_for(intrinsics())
    session.save_calibration_results(rig)
    previous = {p: p.read_bytes() for p in session.session_dir.rglob('*') if p.is_file()}
    intr = rig.pairs['pair'].left_intrinsics
    intr.camera_matrix[:2, :2] = [[463., 0.], [0., 463.]]
    intr.dist_coeffs = np.array([.058, -.034, .0002, .0001, -.0004])
    with pytest.raises(ValueError, match='不可用于立体校正'):
        session.save_calibration_results(rig)
    assert {p: p.read_bytes() for p in session.session_dir.rglob('*') if p.is_file()} == previous


def test_nonzero_tangential_fold_is_not_a_valid_export_fixture():
    intr = intrinsics()
    intr.dist_coeffs = np.array([.1, .2, .3, .4, 0.])
    assert check_intrinsics(intr)['status'] == 'failed'


def test_rational_denominator_pole_is_rejected():
    intr = intrinsics()
    intr.dist_coeffs = np.array([0., 0., 0., 0., 0., -2., 0., 0.])
    assert check_intrinsics(intr)['status'] == 'failed'


def test_integer_intrinsics_are_valid_numeric_inputs():
    intr = intrinsics()
    intr.camera_matrix = intr.camera_matrix.astype(int)
    intr.dist_coeffs = intr.dist_coeffs.astype(int)
    assert check_intrinsics(intr)['status'] == 'ok'


def test_small_sample_bad_pairing_fails_instead_of_publishing_success(app):
    from multicamera.ui.threads.calibration_worker import PairCalibrationWorker
    left, right, stereo = calibrators(8)
    order = [1, 0, 3, 2, 5, 4, 7, 6]
    stereo._img_points_right = [stereo._img_points_right[i] for i in order]
    right._all_corners = [right._all_corners[i] for i in order]
    right._all_ids = [right._all_ids[i] for i in order]
    with pytest.raises(ValueError, match='超过未验证解门槛'):
        calibrate_validated(left, right, stereo)
    worker = PairCalibrationWorker('pair', 'L', 'R', left, right, stereo, automatic_model=True)
    successes, failures = [], []
    worker.finished_ok.connect(lambda *args: successes.append(args))
    worker.failed.connect(lambda *args: failures.append(args))
    worker.run()
    assert not successes and len(failures) == 1
    assert '核对左右配对' in failures[0][1]


def test_healthy_small_sample_still_retains_standard_model():
    _, _, _, report = calibrate_validated(*calibrators(8))
    assert report['selected'] == 'standard'
    assert report['validation_status'] == 'insufficient_views'
    assert report['candidates']['standard']['validation_status'] == 'insufficient_views'
    assert '未完成留出验证' in report['reason']


@pytest.mark.parametrize('defect', ['zero_baseline', 'scale_baseline', 'reverse_baseline',
                                  'zero_q', 'q_scale_error', 'q_direction', 'rotation'])
def test_inconsistent_rectification_cannot_export_or_publish(tmp_path, defect):
    rig = rig_for(intrinsics())
    pair = rig.pairs['pair']
    stereo = pair.stereo
    if defect == 'zero_baseline':
        stereo.P2[0, 3] = 0.
    elif defect == 'scale_baseline':
        stereo.P2[0, 3] *= 2.
    elif defect == 'reverse_baseline':
        stereo.P2[0, 3] *= -1.
    elif defect == 'zero_q':
        stereo.Q[:] = 0.
    elif defect == 'q_scale_error':
        stereo.Q[3, 2] *= 2.
    elif defect == 'q_direction':
        stereo.Q[3, 2] *= -1.
    else:
        stereo.R2 = cv2.Rodrigues(np.array([0., .05, 0.]))[0]
    assert check_stereo_health(pair.left_intrinsics, pair.right_intrinsics, stereo)['status'] == 'failed'
    assert not summarize_quality(rig).valid
    for exporter, suffix in ((export_rig_json, 'json'), (export_rig_opencv_standard_yaml, 'yml')):
        path = tmp_path / f'previous.{suffix}'
        path.write_bytes(b'previous')
        with pytest.raises(ValueError, match='不可用于立体校正'):
            exporter(rig, path)
        assert path.read_bytes() == b'previous'
    from multicamera.io.session import CalibrationSession
    left, _, _ = calibrators()
    session = CalibrationSession.create(tmp_path / 'session', left.board, [])
    previous = {p: p.read_bytes() for p in session.session_dir.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='不可用于立体校正'):
        session.save_calibration_results(rig)
    assert {p: p.read_bytes() for p in session.session_dir.rglob('*') if p.is_file()} == previous


@pytest.mark.parametrize('vertical', [False, True])
@pytest.mark.parametrize('direction', [-1., 1.])
@pytest.mark.parametrize('zero_disparity', [False, True])
def test_native_rectification_geometry_handles_axis_sign_and_principal_offset(
        vertical, direction, zero_disparity):
    intr = intrinsics()
    right = deepcopy(intr)
    right.camera_matrix[:2, 2] += [17., -11.]
    translation = np.array([0., direction*.05, .001] if vertical else [direction*.05, .001, 0.])
    rotation = cv2.Rodrigues(np.array([.01, -.02, .005]))[0]
    stereo = StereoCalibResult(rotation, translation, np.eye(3), np.eye(3))
    stereo.R1, stereo.R2, stereo.P1, stereo.P2, stereo.Q, _, _ = cv2.stereoRectify(
        intr.camera_matrix, intr.dist_coeffs, right.camera_matrix, right.dist_coeffs,
        intr.image_size, rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY if zero_disparity else 0, alpha=0)
    stereo.Q *= 3.7 * direction  # Even a negative homogeneous scale is equivalent.
    health = check_stereo_health(intr, right, stereo)
    assert health['status'] == 'ok', health['reasons']
    assert health['geometry']['q_status'] == 'ok'
    assert health['geometry']['axis'] == ('vertical' if vertical else 'horizontal')


def test_standard_is_retained_without_unnecessary_model_switches():
    cals = calibrators()
    before = [p.copy() for p in cals[2]._img_points_left]
    left, right, stereo, report = calibrate_validated(*cals)
    assert report['selected'] == 'standard'
    assert set(report['candidates']) == {'standard'}
    assert report['validation_status'] == 'passed'
    assert len(report['candidates']['standard']['validation']['folds']) == 3
    assert check_stereo_health(left, right, stereo)['status'] == 'ok'
    for old, new in zip(before, cals[2]._img_points_left):
        np.testing.assert_array_equal(old, new)


def test_no_healthy_candidate_never_publishes_a_winner(monkeypatch):
    from multicamera.calibration import validated_selection
    monkeypatch.setattr(validated_selection, 'check_stereo_health',
                        lambda *args: {'status': 'failed', 'reasons': ['forced boundary failure']})
    with pytest.raises(ValueError, match='没有通过'):
        calibrate_validated(*calibrators())


def test_automatic_strategy_is_default_and_quality_report_explains_it(model_window, app):
    from test_real_pairs_ui import wait_for
    panel = model_window._calib_panel
    assert panel._combo_calib_strategy.currentData() == 'auto_validated'
    assert panel._calib_strategy_key == 'auto_validated'
    panel._calibrate_pair('test')
    wait_for(app, lambda: not panel.has_active_background_tasks())
    pair = panel._pair_calibs['test']
    assert pair.calibration_info['model_selection']['selected'] == 'standard'
    assert pair.calibration_info['health']['status'] == 'ok'
    assert '自动校验择模' in panel._quality_report.toPlainText()
    assert '有效性检查' in panel._quality_report.toPlainText()


def test_explicit_manual_strategy_does_not_silently_select_models(model_window, app):
    from test_real_pairs_ui import wait_for
    panel = model_window._calib_panel
    panel._combo_calib_strategy.setCurrentIndex(panel._combo_calib_strategy.findData('standard'))
    assert panel._calib_strategy_key == 'standard'
    panel._calibrate_pair('test')
    wait_for(app, lambda: not panel.has_active_background_tasks())
    assert 'model_selection' not in panel._pair_calibs['test'].calibration_info


@pytest.mark.parametrize('count,folds', [(9, 3), (15, 5)])
def test_low_error_standard_still_requires_disjoint_holdouts(count, folds):
    _, _, _, report = calibrate_validated(*calibrators(count))
    assert report['selected'] == 'standard'
    validation = report['candidates']['standard']['validation']
    assert len(validation['folds']) == folds
    held_out = []
    for fold in validation['folds']:
        assert not set(fold['train_ids']) & set(fold['test_ids'])
        held_out.extend(fold['test_ids'])
    assert sorted(held_out) == sorted(report['evaluated_ids'])


def test_low_error_standard_with_failed_holdout_uses_validated_fallback(monkeypatch):
    from multicamera.calibration import validated_selection as selection
    calls = []

    def validate(left, right, stereo, candidate, ids, log):
        calls.append(candidate)
        if candidate == 'standard':
            raise ValueError('simulated fold failure')
        return {'rms_px': .2 if candidate == 'constrained' else .3, 'folds': []}

    monkeypatch.setattr(selection, '_cross_validate', validate)
    _, _, _, report = calibrate_validated(*calibrators())
    assert calls == ['standard', 'constrained', 'rational']
    assert report['selected'] == 'constrained'
    assert report['candidates']['standard']['status'] == 'failed'
    assert 'fold failure' in report['candidates']['standard']['error']


@pytest.mark.parametrize('error', [float('nan'), float('inf'), -.1, 2.1])
def test_invalid_or_excessive_holdouts_cannot_use_low_training_rms(monkeypatch, error):
    from multicamera.calibration import validated_selection as selection
    monkeypatch.setattr(selection, '_cross_validate',
                        lambda *args: {'rms_px': error, 'folds': []})
    with pytest.raises(ValueError, match='没有通过'):
        calibrate_validated(*calibrators())


def test_fallback_requires_disjoint_training_without_reusing_final_seeds(monkeypatch):
    from multicamera.calibration import validated_selection as selection
    original = selection.check_stereo_health

    def reject_plain(left, right, stereo):
        if left.dist_coeffs.size < 8:
            return {'status': 'failed', 'reasons': ['plain model rejected']}
        return original(left, right, stereo)

    monkeypatch.setattr(selection, 'check_stereo_health', reject_plain)
    _, _, _, report = calibrate_validated(*calibrators())
    assert report['selected'] == 'rational'
    for fold in report['candidates']['rational']['validation']['folds']:
        assert not set(fold['train_ids']) & set(fold['test_ids'])
        assert set(fold['train_ids']) | set(fold['test_ids']) == set(report['evaluated_ids'])


def test_rational_stereo_keeps_denominators_and_finite_fit_metrics():
    left, right, stereo = calibrators()
    lintr, rintr = left.calibrate(), right.calibrate()
    for intr in (lintr, rintr):
        intr.dist_coeffs = np.pad(intr.dist_coeffs.ravel(), (0, 3))
        intr.dist_coeffs[5] = .01
    result = stereo.calibrate(lintr, rintr, flags=cv2.CALIB_FIX_INTRINSIC)
    assert lintr.dist_coeffs.size >= 8
    assert lintr.dist_coeffs.ravel()[5] == pytest.approx(.01)
    assert np.isfinite(lintr.rms_error) and np.isfinite(rintr.rms_error)
    assert np.isfinite(result.rms_error)


@pytest.mark.skipif(not os.environ.get('MULTICAMERA_HEALTH_SESSION'), reason='Real session opt-in')
def test_real_session_uses_validated_rational_and_preserves_originals(app, tmp_path):
    import hashlib
    from multicamera.board.charuco_board import CharucoBoard
    from multicamera.board.detector import CharucoDetector
    from multicamera.calibration.intrinsic import IntrinsicCalibrator
    from multicamera.calibration.stereo import StereoCalibrator
    from multicamera.ui.threads.calibration_worker import PairCalibrationWorker

    root = Path(os.environ['MULTICAMERA_HEALTH_SESSION'])
    paths = sorted(path for path in root.rglob('*') if path.is_file())
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    old = MultiCameraRig.from_dict(json.loads((root/'calibration_all_json.json').read_text()))
    assert not summarize_quality(old).valid
    board = CharucoBoard.from_dict(json.loads((root/'session.json').read_text())['board'])
    detector = CharucoDetector(board)
    left, right = IntrinsicCalibrator(board), IntrinsicCalibrator(board)
    stereo = StereoCalibrator(board)
    for lp in sorted((root/'images/rgb_pair_1').glob('left_*.png')):
        key = lp.stem.split('_')[1]
        dl = detector.detect(cv2.imread(str(lp)))
        dr = detector.detect(cv2.imread(str(lp.with_name('right_'+key+'.png'))))
        left.add_detection(dl, frame_id=key)
        right.add_detection(dr, frame_id=key)
        stereo.add_detection_pair(dl, dr, frame_id=key)
    worker = PairCalibrationWorker('pair', 'L', 'R', left, right, stereo, automatic_model=True)
    results, errors = [], []
    worker.finished_ok.connect(lambda *args: results.append(args[-1]))
    worker.failed.connect(lambda *args: errors.append(args))
    worker.run()
    assert results and not errors, errors
    pair = results[0]
    choice = pair.calibration_info['model_selection']
    assert choice['selected'] == 'rational'
    assert pair.calibration_info['health']['status'] == 'ok'
    folds = choice['candidates']['rational']['validation']['folds']
    assert len(folds) == 5
    for fold in folds:
        assert not set(fold['train_ids']) & set(fold['test_ids'])
    assert pair.stereo.rms_error < .3
    rig = MultiCameraRig(pairs={'pair': pair})
    export_rig_json(rig, tmp_path/'validated.json')
    export_rig_opencv_standard_yaml(rig, tmp_path/'validated.yml')
    from multicamera.io.session import CalibrationSession
    session = CalibrationSession.create(tmp_path/'sessions', board, [])
    files, failures = session.save_calibration_results(rig)
    assert {'rig.json', 'rig_opencv.yml', 'rig_opencv_standard.yml'} <= {p.name for p in files}
    assert len(failures) == 1 and failures[0].startswith('rig_kalibr.yaml:')
    assert session.load_rig().pairs['pair'].calibration_info['health']['status'] == 'ok'
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == value for p, value in hashes.items())
