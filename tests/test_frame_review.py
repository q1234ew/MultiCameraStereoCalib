"""Post-solve log warnings retain exact source IDs and never remove observations."""

from copy import deepcopy
import json
import logging
from pathlib import Path, PureWindowsPath

import numpy as np
import pytest

from multicamera.calibration.frame_review import frame_review_log_lines, review_pair_frames
from multicamera.calibration.models import (
    CameraIntrinsics, CameraModel, StereoCalibResult, StereoPairCalibration,
)
from multicamera.ui import main_window
from multicamera.ui.widgets import calib_panel
from multicamera.ui.widgets.calib_panel import _UndoCaptureEntry
from multicamera.io.session import CalibrationSession
from test_calibration_workflow import seed_observations, wait_until


@pytest.fixture
def pair():
    def intr(ids, errors):
        return CameraIntrinsics(
            np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]]),
            np.zeros(5), (640, 480), rms_error=.2, frame_ids=ids,
            per_frame_errors=errors, per_frame_error_metric='euclidean_rms_px',
        )
    return StereoPairCalibration(
        'rgb', 'left', 'right', intr(['0002', '0010'], [.2, 1.4]),
        intr(['0002', '0007'], [.3, .8]),
        StereoCalibResult(np.eye(3), np.array([[-.1], [0.], [0.]]), np.eye(3), np.eye(3)),
        calibration_info={
            'rectification': {'status': 'ok', 'per_frame': [
                {'frame_id': '0002', 'rms_px': .1, 'status': 'ok'},
                {'frame_id': '0010', 'rms_px': 1.2, 'status': 'ok'},
            ]},
            'selection': {
                'left': {'excluded_ids': ['0007']},
                'right': {'excluded_ids': ['0010']},
                'stereo': {'excluded_ids': ['0020', '0007']},
            },
        },
    )


def test_source_id_join_separates_sides_and_combines_pair_reasons_without_mutation(pair):
    before = deepcopy(pair.to_dict())
    report = review_pair_frames(pair)
    assert report['evaluated_pair_count'] == 3
    assert [row['frame_id'] for row in report['review_pairs']] == ['0007', '0010']
    assert '右内参 RMS=0.8000px' in report['review_pairs'][0]['reasons'][0]
    assert len(report['review_pairs'][1]['reasons']) == 2
    assert '左内参 RMS=1.4000px' in report['review_pairs'][1]['reasons'][0]
    assert '校正残差 RMS=1.2000px' in report['review_pairs'][1]['reasons'][1]
    assert report['excluded_pairs'] == [
        {'frame_id': '0007', 'stages': ['左内参', '双目外参']},
        {'frame_id': '0010', 'stages': ['右内参']},
        {'frame_id': '0020', 'stages': ['双目外参']},
    ]
    assert pair.to_dict() == before
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize('model,limit', [(CameraModel.PINHOLE, 1.), (CameraModel.FISHEYE, 1.5)])
def test_model_threshold_and_strict_boundary_do_not_use_stereo_rms(pair, model, limit):
    pair.left_intrinsics.model = model
    pair.left_intrinsics.rms_error = 2.
    pair.left_intrinsics.per_frame_errors = [limit, limit + .01]
    report = review_pair_frames(pair)
    assert report['thresholds']['left']['absolute_rms_px'] == limit
    reasons = {row['frame_id']: row['reasons'] for row in report['review_pairs']}
    assert '0002' not in reasons
    assert any('复核阈值' in message and '左内参' in message for message in reasons['0010'])


@pytest.mark.parametrize('error', [float('nan'), float('inf'), -.1, None, 'invalid'])
def test_invalid_error_has_pair_warning_not_zero_or_clean_bill(pair, error):
    pair.left_intrinsics.per_frame_errors[0] = error
    report = review_pair_frames(pair)
    assert '左内参逐帧 RMS 无效' in report['review_pairs'][0]['reasons']
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('mutation,fragment', [
    ('legacy', '缺少可靠源编号'), ('short', '完整逐帧误差'),
    ('duplicate', '源编号缺失或重复'), ('metric', '口径未确认'),
])
def test_incomplete_or_ambiguous_provenance_never_invents_a_source_id(pair, mutation, fragment):
    intr = pair.left_intrinsics
    if mutation == 'legacy':
        intr.frame_ids = None
    elif mutation == 'short':
        intr.per_frame_errors = [9.]
    elif mutation == 'duplicate':
        intr.frame_ids = ['2', '2']
    else:
        intr.per_frame_error_metric = None
    report = review_pair_frames(pair)
    assert any(fragment in line for line in report['limitations'])
    assert not any('左内参' in reason for row in report['review_pairs'] for reason in row['reasons'])
    assert not any(row['frame_id'] in {'1', '2'} for row in report['review_pairs'])


def test_no_records_is_not_reported_as_no_bad_images(pair):
    for intr in (pair.left_intrinsics, pair.right_intrinsics):
        intr.frame_ids = None
    pair.calibration_info = {}
    report = review_pair_frames(pair)
    lines = '\n'.join(frame_review_log_lines('rgb', report, {}))
    assert report['evaluated_pair_count'] == 0
    assert '无法逐对定位' in lines
    assert '未发现超过复核阈值' not in lines


def test_zero_numerical_noise_and_exact_rectification_limit_not_suspicious(pair):
    for intr in (pair.left_intrinsics, pair.right_intrinsics):
        intr.per_frame_errors = [0., .0001]
        intr.rms_error = .00001
    pair.calibration_info['rectification']['per_frame'][1]['rms_px'] = 1.
    report = review_pair_frames(pair)
    assert not report['review_pairs']
    lines = '\n'.join(frame_review_log_lines('rgb', report, {}))
    assert '未发现超过复核阈值' in lines
    assert '不代表全部图像或实际精度合格' in lines
    assert '策略取舍不等于图像不合格' in lines


def test_nonfinite_rectification_is_reported_with_exact_id(pair):
    pair.calibration_info['rectification']['per_frame'] = [
        {'frame_id': '0002', 'status': 'invalid', 'rms_px': None},
        {'frame_id': None, 'status': 'invalid', 'rms_px': None},
    ]
    report = review_pair_frames(pair)
    assert report['review_pairs'][0]['frame_id'] == '0002'
    assert report['review_pairs'][0]['reasons'] == ['同 ID 校正残差无效']
    assert any('校正残差缺少源编号' in line for line in report['limitations'])
    assert all(row['frame_id'] != 'None' for row in report['review_pairs'])


def test_distinct_padded_source_ids_are_not_merged(pair):
    pair.left_intrinsics.frame_ids = ['2', '0002']
    pair.left_intrinsics.per_frame_errors = [1.8, 1.9]
    report = review_pair_frames(pair)
    assert {'2', '0002'} <= {row['frame_id'] for row in report['review_pairs']}


def test_large_finite_rms_does_not_add_nonfinite_threshold_to_saved_json(pair):
    pair.left_intrinsics.rms_error = 1e308
    report = review_pair_frames(pair)
    assert report['thresholds']['left']['relative_rms_px'] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('root', [Path('/camera images/样本'), PureWindowsPath(r'E:\标定\images')])
def test_log_includes_exact_paths_reasons_units_and_missing_source_notice(pair, root):
    sources = {'0010': (str(root / '0010_left.png'), str(root / '0010_right.png'))}
    lines = frame_review_log_lines('rgb', review_pair_frames(pair), sources)
    warning = next(line for line in lines if 'rgb [0010]: 建议复核 pair' in line)
    assert f'L={sources["0010"][0]}' in warning and f'R={sources["0010"][1]}' in warning
    assert '1.4000px' in warning and '1.2000px' in warning
    assert any('rgb [0007]' in line and '原图路径未记录' in line for line in lines)
    assert '本次未自动剔除' in lines[-1]


@pytest.mark.parametrize('swapped', [False, True])
def test_offline_completion_logs_source_files_to_ui_and_file_log(app, tmp_path, monkeypatch,
                                                               caplog, pair, swapped):
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'sessions')
    win = main_window.MainWindow()
    panel = win._calib_panel
    pair.pair_name = 'offline'
    sources = (str(tmp_path / '拍摄0010_L.jpg'), str(tmp_path / '拍摄0010_R.jpg'))
    panel._offline_source_paths = {'0010': sources}
    panel._offline_lr_swapped = swapped
    # Exercise the real completion callback, including automatic export.
    try:
        with caplog.at_level(logging.INFO):
            panel._on_pair_calibration_finished('offline', 'left', 'right',
                                                pair.left_intrinsics, pair.right_intrinsics, pair)
        text = panel._log_text.toPlainText()
        left, right = sources[::-1] if swapped else sources
        assert 'offline [0010]: 建议复核 pair' in text
        assert f'L={left}；R={right}' in text
        assert f'L={left}；R={right}' in caplog.text
        restored = win._current_session.load_rig().pairs['offline']
        assert restored.calibration_info['frame_review'] == pair.calibration_info['frame_review']
    finally:
        win.close()


def test_live_paths_use_original_capture_session_not_current_session(app, tmp_path, monkeypatch, pair):
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'sessions')
    win = main_window.MainWindow()
    panel = win._calib_panel
    original = CalibrationSession(tmp_path / 'original')
    panel._session = CalibrationSession(tmp_path / 'other')
    panel._saved_capture_tokens.add('saved')
    panel._undo_stack = [([
        _UndoCaptureEntry('rgb', 'left', 'right', 10, 'saved', True, True, True, original),
        _UndoCaptureEntry('other_pair', 'left', 'right', 7, 'saved', True, True, True, original),
        _UndoCaptureEntry('rgb', 'left', 'right', 7, 'pending', True, True, True, original),
    ], False)]
    try:
        panel._log_pair_frame_review(pair)
        text = panel._log_text.toPlainText()
        assert str(original.session_dir / 'images' / 'rgb' / 'left_0010.png') in text
        assert str(panel._session.session_dir) not in text
        assert 'left_0007.png' not in text
    finally:
        win.close()


def test_review_failure_is_explicit_and_does_not_block_export(app, tmp_path, monkeypatch, pair):
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'sessions')
    win = main_window.MainWindow()
    panel = win._calib_panel

    def fail(_):
        raise ValueError('simulated review error')

    monkeypatch.setattr(calib_panel, 'review_pair_frames', fail)
    try:
        panel._on_pair_calibration_finished('rgb', 'left', 'right',
                                            pair.left_intrinsics, pair.right_intrinsics, pair)
        assert '逐对检查未完成（标定结果保留）' in panel._log_text.toPlainText()
        assert win._current_session.load_rig().pairs['rgb'].left_intrinsics.frame_ids == ['0002', '0010']
        assert panel._pair_calibs['rgb'] is pair
    finally:
        win.close()


@pytest.mark.parametrize('strategy', ['standard', 'auto_validated'])
def test_actual_worker_completion_logs_final_errors_and_persists_report(
        app, tmp_path, monkeypatch, strategy):
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'sessions')
    win = main_window.MainWindow()
    panel = win._calib_panel
    seed_observations(win)
    panel._combo_calib_strategy.setCurrentIndex(panel._combo_calib_strategy.findData(strategy))
    cal = panel._intr_cals['left']
    calibrate = type(cal).calibrate
    calls = []

    def with_review_error(self, flags=0):
        intr = calibrate(self, flags=flags)
        calls.append(self)
        if len(calls) == 1:
            intr.per_frame_errors[-1] = 1.7
        return intr

    monkeypatch.setattr(type(cal), 'calibrate', with_review_error)
    try:
        panel._calibrate_pair('rgb')
        wait_until(app, lambda: not panel.has_active_background_tasks())
        assert 'rgb [8]: 建议复核 pair' in panel._log_text.toPlainText()
        pair = win._current_session.load_rig().pairs['rgb']
        assert pair.calibration_info['frame_review']['review_pairs'][0]['frame_id'] == '8'
        assert panel._intr_cals['left'].num_frames == 8
        assert pair.left_intrinsics.per_frame_errors[-1] == 1.7
    finally:
        wait_until(app, lambda: not panel.has_active_background_tasks())
        win.close()
