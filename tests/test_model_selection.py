"""Model recommendations use detached data, balanced holdouts and explicit consent."""

from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QFileDialog, QMessageBox

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.board.detector import DetectionResult
from multicamera.calibration.intrinsic import IntrinsicCalibrator
from multicamera.calibration.model_selection import assess_camera_models
from multicamera.calibration.models import CameraModel
from multicamera.calibration.stereo import StereoCalibrator
from multicamera.streaming.stream_manager import CameraConfig, StereoPairConfig
from multicamera.ui import main_window
from test_adaptive_guidance import diverse, paired
from test_real_pairs_ui import wait_for


def calibrators(count=12, repeated=False):
    board = CharucoBoard(CharucoBoardConfig(cols=9, rows=7, square_length=.035, marker_length=.025))
    obs = paired(board, count=count) if repeated else diverse(board, count=count)
    left, right = [IntrinsicCalibrator(board, CameraModel.PINHOLE, 6) for _ in range(2)]
    stereo = StereoCalibrator(board, CameraModel.PINHOLE, 6)
    for a, b in zip(obs['L'], obs['R']):
        ds = [DetectionResult(o.corners, o.ids, [], None, o.image_size) for o in (a, b)]
        left.add_detection(ds[0], frame_id=a.frame_id)
        right.add_detection(ds[1], frame_id=b.frame_id)
        stereo.add_detection_pair(*ds, frame_id=a.frame_id)
    return left, right, stereo


def report():
    return {'recommended_model': 'fisheye', 'reason': 'test recommendation',
            'candidates': {'pinhole': {'rms_px': 5.}, 'fisheye': {'rms_px': .3}}}


def test_model_comparison_preserves_input_and_separates_holdouts():
    left, right, stereo = calibrators()
    before = [x.copy() for x in stereo._img_points_left]
    result = assess_camera_models(left, right, stereo)
    assert result['requires_confirmation']
    assert result['recommended_model'] != 'fisheye'  # Synthetic pinhole must not be called fisheye.
    assert left.model == right.model == stereo.model == CameraModel.PINHOLE
    for old, new in zip(before, stereo._img_points_left):
        np.testing.assert_array_equal(old, new)
    for name, candidate in result['candidates'].items():
        assert len(candidate['folds']) == 3
        for fold in candidate['folds']:
            assert not set(fold['train_ids']) & set(fold['test_ids'])
            assert set(fold['train_ids']) | set(fold['test_ids']) == set(result['evaluated_ids'])
            if fold['status'] == 'ok':
                assert len(fold['training_corner_counts']) == len(fold['train_ids'])
                if name == 'fisheye':
                    assert fold['training_corner_policy'] == 'cross_view_common_ids'
                    assert fold['shared_training_ids']
                    assert set(fold['training_corner_counts']) == {len(fold['shared_training_ids'])}
                else:
                    assert fold['training_corner_policy'] == 'per_pair_common_ids'
                    assert fold['shared_training_ids'] is None
    assert [f['test_ids'] for f in result['candidates']['pinhole']['folds']] == [
        f['test_ids'] for f in result['candidates']['fisheye']['folds']]


def test_insufficient_views_and_pose_diversity_are_not_guessed():
    with pytest.raises(ValueError, match='至少'):
        assess_camera_models(*calibrators(8))
    with pytest.raises(ValueError, match='倾角'):
        assess_camera_models(*calibrators(9, repeated=True))


def test_model_comparison_cancelled_does_not_publish_recommendation():
    with pytest.raises(RuntimeError, match='取消'):
        assess_camera_models(*calibrators(), cancelled=lambda: True)


@pytest.fixture
def model_window(app, monkeypatch, tmp_path):
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'sessions')
    win = main_window.MainWindow()
    pair = StereoPairConfig('test', CameraConfig('L', ''), CameraConfig('R', ''))
    win._stream_manager.add_virtual_stereo_pair(pair)
    left, right, stereo = calibrators()
    panel = win._calib_panel
    panel._intr_cals = {'L': left, 'R': right}
    panel._stereo_cals = {'test': stereo}
    panel.refresh_pairs()
    panel._model_report = report()
    panel._model_report_key = panel._model_snapshot_key('test')
    panel._refresh_model_choice_ui()
    yield win
    if panel._model_worker is not None:
        panel._model_worker.wait(30000)
        app.processEvents()
    win.close()
    win.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


def test_model_requires_consent_and_preserves_observations(model_window, monkeypatch):
    win, panel = model_window, model_window._calib_panel
    counts = [cal.num_frames for cal in (*panel._intr_cals.values(), *panel._stereo_cals.values())]
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.No)
    panel._request_model_choice()
    assert win._camera_model == panel._camera_model == CameraModel.PINHOLE
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.Yes)
    panel._request_model_choice()
    assert win._camera_model == panel._camera_model == CameraModel.FISHEYE
    assert counts == [cal.num_frames for cal in (*panel._intr_cals.values(), *panel._stereo_cals.values())]
    assert panel._combo_data_filter.currentData() == 'fisheye_common'
    assert not panel._pair_calibs


def test_old_recommendation_cannot_apply_after_data_changes(model_window, monkeypatch):
    panel = model_window._calib_panel
    panel._intr_cals['L'].pop_last_frame()
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: pytest.fail('Stale proposal must not ask consent'))
    panel._refresh_model_choice_ui()
    assert not panel._btn_apply_model.isVisible()
    panel._request_model_choice()
    assert panel._camera_model == CameraModel.PINHOLE


def test_model_metadata_failure_leaves_model_unchanged(model_window, monkeypatch):
    from multicamera.io.session import CalibrationSession

    win = model_window
    session = CalibrationSession.create(Path(win._session_manager.base_dir), win._board,
                                        list(win._stream_manager.stereo_pairs.values()))
    win._current_session = session
    win._calib_panel.set_session(session)
    before = deepcopy(session._metadata)

    def fail(*args):
        session._metadata['camera_model'] = 'fisheye'
        raise OSError('read only')

    monkeypatch.setattr(session, 'record_configuration', fail)
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.Yes)
    win._on_model_choice(CameraModel.FISHEYE)
    assert session._metadata == before
    assert win._camera_model == CameraModel.PINHOLE


def test_first_completion_preserves_assessment_and_repeat_persists_it(model_window, app):
    win, panel = model_window, model_window._calib_panel
    expected = deepcopy(panel._model_report)
    assert win._current_session is None
    for _ in range(2):
        panel._calibrate_pair('test')
        wait_for(app, lambda: not panel.has_active_background_tasks())
        assert panel._model_report_key == panel._model_snapshot_key('test')
        panel._refresh_model_choice_ui()
        assert '请重新评估' not in panel._lbl_model_choice.text()
        assert win._current_session.load_rig().pairs['test'].calibration_info['model_assessment'] == expected


@pytest.mark.parametrize('changed_data', [False, True])
def test_only_explicit_initial_session_rebind_can_preserve_assessment(model_window, changed_data):
    from multicamera.io.session import CalibrationSession

    win, panel = model_window, model_window._calib_panel
    first = CalibrationSession.create(Path(win._session_manager.base_dir), win._board, [])
    if changed_data:
        panel._intr_cals['L'].pop_last_frame()
    panel.set_session(first, preserve_model_assessment=True)
    assert (panel._model_report_key == panel._model_snapshot_key('test')) is not changed_data
    second = CalibrationSession.create(Path(win._session_manager.base_dir), win._board, [])
    panel.set_session(second, preserve_model_assessment=True)
    assert panel._model_report_key != panel._model_snapshot_key('test')
    assert '请重新评估' in panel._lbl_model_choice.text()


@pytest.mark.skipif(not os.environ.get('MULTICAMERA_REAL_PAIR_DIR'), reason='Real-image opt-in test')
def test_real_model_assessment_applies_without_reimport(app, monkeypatch, tmp_path):
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'sessions')
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args: os.environ['MULTICAMERA_REAL_PAIR_DIR'])
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.Yes)
    win = main_window.MainWindow()
    panel = win._calib_panel
    try:
        panel._btn_import.click()
        wait_for(app, lambda: not panel.has_active_background_tasks())
        count = panel._stereo_cals['offline'].num_frames
        panel._btn_assess_model.click()
        wait_for(app, lambda: not panel.has_active_background_tasks())
        panel._refresh_model_choice_ui()
        assert panel._model_report['recommended_model'] == 'fisheye', panel._lbl_model_choice.text()
        assert panel._model_report['candidates']['fisheye']['rms_px'] < 1.
        panel._btn_apply_model.click()
        assert panel._camera_model == win._camera_model == CameraModel.FISHEYE
        assert panel._stereo_cals['offline'].num_frames == count
        panel._btn_calib_all.click()
        wait_for(app, lambda: not panel.has_active_background_tasks())
        assert panel._pair_calibs['offline'].stereo.rms_error < 1.
        assert panel._pair_calibs['offline'].calibration_info['model_assessment']['recommended_model'] == 'fisheye'
        assert panel._model_report_key == panel._model_snapshot_key('offline')
        first_directory = win._current_session.result_directory
        first_bytes = (first_directory / 'rig.json').read_bytes()
        panel._btn_calib_all.click()
        wait_for(app, lambda: not panel.has_active_background_tasks())
        session = win._current_session
        assert session.result_directory != first_directory
        assert (first_directory / 'rig.json').read_bytes() == first_bytes
        assert session.load_rig().pairs['offline'].calibration_info['model_assessment']['recommended_model'] == 'fisheye'
    finally:
        wait_for(app, lambda: not panel.has_active_background_tasks())
        win.close()
        win.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()
