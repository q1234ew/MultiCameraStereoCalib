"""Participation, persistence and real worker coverage for the single-pair review loop."""

from copy import deepcopy
import json

import numpy as np
import pytest

from multicamera.calibration.review import (
    analyze_review, empty_state, excluded_ids, frame_fingerprint,
)
from multicamera.io.session import CalibrationSession
from test_calibration_workflow import seed_observations, wait_until

pytest_plugins = ['test_calibration_workflow']


@pytest.fixture
def dataset(window, monkeypatch):
    image, detections = seed_observations(window, count=10)
    panel = window._calib_panel
    panel._combo_pair.setCurrentText('rgb')
    assert panel._ensure_capture_session()
    panel._check_wb.setChecked(False)
    # Deterministic detector output; real solvers and asynchronous image saving remain active.
    for cid, detection in zip(('left', 'right'), detections):
        monkeypatch.setattr(panel._intr_cals[cid].detector, 'detect', lambda _, d=detection: d)
    return panel, image, detections


def analysis(panel):
    rows, result = panel._review_rows('rgb')
    intrinsics = (result.left_intrinsics, result.right_intrinsics) if result else (None, None)
    panel._review_report = analyze_review(panel._board, rows, intrinsics)
    panel._review_report_key = panel._review_key('rgb')
    return panel._review_report


def start_retake(panel, window, image, monkeypatch):
    window._stream_manager._virtual_pairs.discard('rgb')
    monkeypatch.setattr(window._stream_manager, 'get_sync_pair', lambda _: (image, image))
    # Pin a current measured source geometry. Raw admission remains enabled in the capture path.
    report = analysis(panel)
    row = next(row for row in report['rows'] if row['geometry'][0])
    assert panel._begin_review_task('rgb', row)
    return row['id']


def test_non_destructive_exclusion_applies_to_all_solver_roles_and_restores(dataset):
    panel, _, _ = dataset
    original = [list(cal.frame_ids) for cal in panel._effective_pair_calibrators('rgb')]
    fid = original[2][0]
    assert panel._toggle_review_participation('rgb', fid)
    assert all(fid not in cal.frame_ids for cal in panel._effective_pair_calibrators('rgb'))
    assert all(cal.num_frames == 10 for cal in [*panel._intr_cals.values(), panel._stereo_cals['rgb']])
    assert '有效配对 9 /' in panel._lbl_capture_count.text()
    assert panel._toggle_review_participation('rgb', fid)
    assert [cal.frame_ids for cal in panel._effective_pair_calibrators('rgb')] == original


def test_participation_is_persistent_and_bound_to_frame_evidence(dataset):
    panel, _, _ = dataset
    fid = panel._stereo_cals['rgb'].frame_ids[0]
    assert panel._toggle_review_participation('rgb', fid)
    path = panel._session.session_dir / 'review' / (panel._review_scope('rgb') + '.json')
    assert path.exists()
    panel._review_states.clear()
    assert fid in panel._review_excluded('rgb')
    # A different observation reusing the ID must never inherit a historical decision.
    panel._intr_cals['left']._all_corners[0] += .5
    panel._intr_cals['left']._data_revision += 1
    assert fid not in panel._review_excluded('rgb')


def test_persistence_failure_does_not_change_participation(dataset, monkeypatch):
    panel, _, _ = dataset
    before = deepcopy(panel._review_state('rgb'))
    def fail(*_):
        raise OSError('disk full')
    monkeypatch.setattr(panel._session, 'save_improvement', fail)
    assert not panel._toggle_review_participation('rgb', '1')
    assert panel._review_state('rgb') == before
    assert panel._effective_pair_calibrators('rgb')[2].num_frames == 10


def test_corrupt_state_blocks_solving_instead_of_reincluding_excluded_frames(dataset):
    panel, _, _ = dataset
    scope = panel._review_scope('rgb')
    folder = panel._session.session_dir / 'review'
    folder.mkdir(exist_ok=True)
    (folder / (scope + '.json')).write_text('{"schema_version":1}', encoding='utf-8')
    panel._review_states.clear()
    assert not panel._pair_data_ready('rgb')
    panel._calibrate_pair('rgb')
    assert panel._calib_worker is None


def test_below_minimum_is_not_calibratable(dataset):
    panel, _, _ = dataset
    for fid in list(panel._stereo_cals['rgb'].frame_ids)[:6]:
        assert panel._toggle_review_participation('rgb', fid)
    assert not panel._pair_data_ready('rgb')
    panel._calibrate_pair('rgb')
    assert panel._calib_worker is None


def test_analysis_uses_effective_observations_and_does_not_invent_pose(dataset):
    panel, _, _ = dataset
    report = analysis(panel)
    row = report['rows'][0]
    assert all(not geom['pose_available'] for geom in row['geometry'] if geom)
    fid = row['id']
    assert panel._toggle_review_participation('rgb', fid)
    updated = analysis(panel)
    assert not next(row for row in updated['rows'] if row['id'] == fid)['participates']
    assert sum(updated['zone_counts'][0].values()) == 9
    assert not any('倾角' in task['title'] for task in updated['tasks'])


def test_offline_retake_is_unavailable_but_review_and_exclusion_work(dataset):
    panel, _, _ = dataset
    row = analysis(panel)['rows'][0]
    assert not panel._begin_review_task('rgb', row)
    assert panel._toggle_review_participation('rgb', row['id'])


def test_new_candidate_is_excluded_until_saved_and_explicitly_applied(
        dataset, window, app, monkeypatch):
    panel, image, _ = dataset
    old = start_retake(panel, window, image, monkeypatch)
    monkeypatch.setattr(panel, '_review_capture_validator', lambda *args: (True, ''))
    panel._capture_impl()
    task = panel._review_task
    assert task['candidate'] is not None
    candidate = task['candidate']
    assert candidate in panel._review_excluded('rgb')
    assert old not in panel._review_excluded('rgb')
    assert not panel._apply_review_candidate()
    wait_until(app, lambda: not panel._save_workers and not panel._pending_save_tokens)
    assert task['token'] in panel._saved_capture_tokens
    source_paths = list((panel._session.session_dir / 'images' / 'rgb').glob('*.png'))
    assert len(source_paths) == 2
    source_bytes = [path.read_bytes() for path in source_paths]
    assert panel._apply_review_candidate()
    assert panel._review_task is None
    assert old in panel._review_excluded('rgb')
    assert candidate not in panel._review_excluded('rgb')
    assert panel._effective_pair_calibrators('rgb')[2].num_frames == 10
    assert not panel._undo_stack
    assert [path.read_bytes() for path in source_paths] == source_bytes
    assert panel._review_state('rgb')['replacements'][-1]['old_id'] == old


def test_failed_save_never_replaces_old_frame(dataset, window, app, monkeypatch):
    panel, image, _ = dataset
    old = start_retake(panel, window, image, monkeypatch)
    monkeypatch.setattr(panel, '_review_capture_validator', lambda *args: (True, ''))
    def fail(*args, **kwargs):
        raise OSError('write failure')
    monkeypatch.setattr(panel._session, 'save_frame_pair', fail)
    panel._capture_impl()
    wait_until(app, lambda: not panel._save_workers)
    assert panel.has_failed_saves
    assert not panel._apply_review_candidate()
    assert old not in panel._review_excluded('rgb')
    assert panel._review_task['candidate'] in panel._review_excluded('rgb')
    # Clean up the synthetic failed save so the shared window fixture can close.
    panel._failed_save_jobs.clear()


def test_cancel_preserves_old_participation_and_persisted_candidate_exclusion(
        dataset, window, app, monkeypatch):
    panel, image, _ = dataset
    old = start_retake(panel, window, image, monkeypatch)
    monkeypatch.setattr(panel, '_review_capture_validator', lambda *args: (True, ''))
    panel._capture_impl()
    wait_until(app, lambda: not panel._save_workers)
    candidate = panel._review_task['candidate']
    panel._cancel_review_task()
    wait_until(app, lambda: panel._review_worker is None)
    assert panel._review_task is None
    assert old not in panel._review_excluded('rgb')
    assert candidate in panel._review_excluded('rgb')
    panel._review_states.clear()
    assert candidate in panel._review_excluded('rgb')
    # Explicitly confirming a saved orphan candidate is still possible from review.
    assert panel._toggle_review_participation('rgb', candidate)
    assert candidate not in panel._review_excluded('rgb')
    if panel._review_worker:
        wait_until(app, lambda: panel._review_worker is None)


def test_real_recalibration_excludes_manual_frames_before_quality_selection(
        dataset, window, app):
    panel, _, _ = dataset
    panel._calibrate_pair('rgb')
    wait_until(app, lambda: panel._calib_worker is None)
    previous = panel._pair_calibs['rgb']
    previous_directory = panel._result_directory
    original_json = (previous_directory / 'rig.json').read_bytes()
    fid = panel._stereo_cals['rgb'].frame_ids[0]
    assert panel._toggle_review_participation('rgb', fid)
    assert panel._review_archive['rgb'].stereo.rms_error == previous.stereo.rms_error
    assert panel._rig is None
    assert '上次成功结果保留' in panel._lbl_result_decision.text()
    panel._combo_data_filter.setCurrentIndex(panel._combo_data_filter.findData('quality'))
    panel._calibrate_pair('rgb')
    wait_until(app, lambda: panel._calib_worker is None)
    current = panel._pair_calibs['rgb']
    for role in ('left', 'right', 'stereo'):
        selection = current.calibration_info['selection'][role]
        assert fid not in selection['used_ids']
        assert selection['input_count'] == 9
    before, after = panel._review_comparisons['rgb']
    assert before.calibration_info['selection']['stereo']['used_count'] == 10
    assert fid not in after.calibration_info['selection']['stereo']['used_ids']
    assert panel._result_directory != previous_directory
    assert (previous_directory / 'rig.json').read_bytes() == original_json


def test_failed_recalibration_preserves_previous_diagnostics(dataset, app, monkeypatch):
    panel, _, _ = dataset
    panel._calibrate_pair('rgb')
    wait_until(app, lambda: panel._calib_worker is None)
    old = panel._pair_calibs['rgb']
    def fail(*args, **kwargs):
        raise ValueError('cannot converge')
    from multicamera.calibration.intrinsic import IntrinsicCalibrator
    monkeypatch.setattr(IntrinsicCalibrator, 'calibrate', fail)
    panel._calibrate_pair('rgb')
    wait_until(app, lambda: panel._calib_worker is None)
    assert 'rgb' not in panel._pair_calibs
    assert panel._review_archive['rgb'].stereo.rms_error == old.stereo.rms_error
    assert panel._review_archive['rgb'].left_intrinsics.frame_ids == old.left_intrinsics.frame_ids
    assert not panel._review_comparisons


def test_review_window_refreshes_and_rejects_stale_report(dataset, app):
    panel, _, _ = dataset
    panel._show_calibration_review()
    wait_until(app, lambda: panel._review_worker is None)
    dialog = panel._review_dialog
    assert dialog.rows
    assert not dialog.retake.isEnabled()  # Virtual/offline group.
    dialog.toggle.click()
    wait_until(app, lambda: panel._review_worker is None)
    assert sum(row['participates'] for row in dialog.rows) == 9
    panel._reset_all()
    assert panel._review_dialog is None
    assert panel._review_report_key is None


def test_scope_and_state_validation(tmp_path):
    session = CalibrationSession(tmp_path)
    with pytest.raises(ValueError):
        session.save_improvement('../escape', empty_state())
    scope = 'a' * 64
    session.save_improvement(scope, empty_state())
    assert session.load_improvement(scope) == empty_state()
    stored = json.loads((tmp_path / 'review' / (scope + '.json')).read_text())
    assert stored['schema_version'] == 1
    assert excluded_ids({'excluded': {'1': 'old'}}, {'1': 'new'}) == set()
    assert frame_fingerprint(None, None) == frame_fingerprint(None, None)


def test_raw_review_capture_enforces_target_and_roundtrip_recalibration(
        dataset, window, app, monkeypatch):
    panel, image, detections = dataset
    panel._calibrate_pair('rgb')
    wait_until(app, lambda: panel._calib_worker is None)
    window._stream_manager._virtual_pairs.discard('rgb')
    monkeypatch.setattr(window._stream_manager, 'get_sync_pair', lambda _: (image, image))
    report = analysis(panel)
    source = next(row for row in report['rows'] if row['id'] == '10')
    panel._session.save_frame_pair('rgb', image, image, frame_idx=10)
    panel._next_session_frame_idx = panel._session.frame_count
    old_paths = list((panel._session.session_dir / 'images' / 'rgb').glob('*_0010.png'))
    old_bytes = [path.read_bytes() for path in old_paths]
    assert len(old_paths) == 2
    assert panel._begin_review_task('rgb', source)
    target = panel._review_task['target']
    assert window._stereo_capture._target() is target
    accepted, reason = window._stereo_capture.validate_review_capture(
        'rgb', image, image, *detections)
    assert accepted, reason
    old_result = panel._pair_calibs['rgb']
    panel._capture_impl()
    candidate = panel._review_task['candidate']
    assert candidate is not None
    # Pending/excluded additions have not changed the effective training set yet.
    assert panel._pair_calibs['rgb'] is old_result
    count = panel._stereo_cals['rgb'].num_frames
    panel._capture_impl()
    assert panel._stereo_cals['rgb'].num_frames == count
    wait_until(app, lambda: not panel._save_workers and not panel._pending_save_tokens)
    assert panel._apply_review_candidate()
    panel._calibrate_pair('rgb')
    wait_until(app, lambda: panel._calib_worker is None)
    result = panel._pair_calibs['rgb']
    used = result.calibration_info['selection']['stereo']['used_ids']
    assert '10' not in used and candidate in used
    assert result.calibration_info['review']['excluded_ids'] == ['10']
    assert result.calibration_info['review']['decisions']['replacements'][-1]['new_id'] == candidate
    assert panel._review_comparisons['rgb'][0].stereo.rms_error == old_result.stereo.rms_error
    assert [path.read_bytes() for path in old_paths] == old_bytes


def test_raw_retake_rejects_wrong_zone_before_registering_or_saving(
        dataset, window, monkeypatch):
    panel, image, detections = dataset
    start_retake(panel, window, image, monkeypatch)
    target = panel._review_task['target']
    target.zone_col, target.zone_row = 2, 0
    panel._review_task['reference_centers'] = [(0.95, 0.05), (0.95, 0.05)]
    accepted, _ = window._stereo_capture.validate_review_capture('rgb', image, image, *detections)
    assert not accepted
    panel._capture_impl()
    assert panel._review_task['candidate'] is None
    assert not panel._pending_save_tokens
    assert panel._stereo_cals['rgb'].num_frames == 10


def test_retry_failed_candidate_save_does_not_duplicate_observations(
        dataset, window, app, monkeypatch):
    panel, image, _ = dataset
    old = start_retake(panel, window, image, monkeypatch)
    monkeypatch.setattr(panel, '_review_capture_validator', lambda *args: (True, ''))
    save_pair = panel._session.save_frame_pair
    def fail(*args, **kwargs):
        raise OSError('temporary failure')
    monkeypatch.setattr(panel._session, 'save_frame_pair', fail)
    panel._capture_impl()
    wait_until(app, lambda: not panel._save_workers)
    candidate = panel._review_task['candidate']
    count = panel._stereo_cals['rgb'].num_frames
    assert not panel._apply_review_candidate()
    monkeypatch.setattr(panel._session, 'save_frame_pair', save_pair)
    panel._retry_failed_saves()
    wait_until(app, lambda: not panel._save_workers and not panel._pending_save_tokens)
    assert panel._stereo_cals['rgb'].num_frames == count
    assert panel._apply_review_candidate()
    assert old in panel._review_excluded('rgb') and candidate not in panel._review_excluded('rgb')


def test_replacement_persistence_failure_preserves_old_and_pending_candidate(
        dataset, window, app, monkeypatch):
    panel, image, _ = dataset
    old = start_retake(panel, window, image, monkeypatch)
    monkeypatch.setattr(panel, '_review_capture_validator', lambda *args: (True, ''))
    panel._capture_impl()
    wait_until(app, lambda: not panel._save_workers)
    before = deepcopy(panel._review_state('rgb'))
    def fail(*args):
        raise OSError('cannot commit decisions')
    monkeypatch.setattr(panel._session, 'save_improvement', fail)
    assert not panel._apply_review_candidate()
    assert panel._review_state('rgb') == before
    assert old not in panel._review_excluded('rgb')
    assert panel._review_task is not None


def test_effective_snapshot_preserves_quality_ranking_pixels(dataset):
    panel, _, _ = dataset
    assert panel._toggle_review_participation('rgb', '1')
    left, right, stereo = panel._effective_pair_calibrators('rgb', snapshot=True)
    for selected, original in ((left, panel._intr_cals['left']), (right, panel._intr_cals['right'])):
        assert all(image is not None for image in selected.stored_images)
        assert selected.stored_images[0] is original.stored_images[1]
        assert not np.shares_memory(selected._all_corners[0], original._all_corners[1])
    assert '1' not in stereo.frame_ids


def test_outlier_evidence_is_advisory_and_requires_enough_comparable_frames(dataset):
    panel, _, _ = dataset
    rows, _ = panel._review_rows('rgb')
    for i, row in enumerate(rows):
        row['left_rms'] = 5. if i == 0 else .3
        row['quality_reasons'] = []
    report = analyze_review(panel._board, rows, (None, None))
    assert any('RMS' in reason for reason in report['rows'][0]['reasons'])
    assert all(row['participates'] for row in report['rows'])
    small = analyze_review(panel._board, deepcopy(rows[:3]), (None, None))
    assert not any('RMS' in reason for row in small['rows'] for reason in row['reasons'])


def test_missing_right_observation_is_not_counted_as_stereo_participation(dataset):
    panel, _, _ = dataset
    panel._intr_cals['right'].pop_last_frame()
    panel._stereo_cals['rgb'].pop_last_frame()
    report = analysis(panel)
    row = next(row for row in report['rows'] if row['id'] == '10')
    assert not row['participates'] and not row['stereo_available']
    assert sum(row['participates'] for row in report['rows']) == 9
    assert any('共同角点' in reason for reason in row['reasons'])


def test_review_ui_and_optional_screenshots(dataset, window, app, monkeypatch, tmp_path):
    import os
    from pathlib import Path
    from multicamera.ui.theme import configure_application_style

    panel, image, _ = dataset
    configure_application_style(app)
    panel._live_preview_indices['rgb'] = [
        (i, i, fid) for i, fid in enumerate(panel._stereo_cals['rgb'].frame_ids)]
    panel._calibrate_pair('rgb')
    wait_until(app, lambda: panel._calib_worker is None)
    window.resize(1280, 850)
    window.show()
    panel.show_workflow('results')
    window._stream_view.show_review_images('rgb 1', {'left': image, 'right': image})
    app.processEvents()
    assert panel._btn_improve.isVisible() and panel._btn_improve.isEnabled()
    output = os.environ.get('MULTICAMERA_UI_QA_DIR')
    if output:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        assert window.grab().save(str(directory / 'calibration-results.png'))
    panel._btn_improve.click()
    wait_until(app, lambda: panel._review_worker is None)
    dialog = panel._review_dialog
    for width in (960, 640):
        dialog.resize(width, 700)
        app.processEvents()
        assert dialog.list.width() >= 100
        assert dialog.toggle.geometry().right() <= dialog.toggle.parentWidget().width()
        assert dialog.tabs.count() == 3
    if output:
        dialog.resize(960, 700)
        app.processEvents()
        assert dialog.grab().save(str(directory / 'calibration-review.png'))
    dialog.close()
    wait_until(app, lambda: panel._review_worker is None)
    window._stream_manager._virtual_pairs.discard('rgb')
    monkeypatch.setattr(window._stream_manager, 'get_sync_pair', lambda _: (image, image))
    report = analysis(panel)
    source = next(row for row in report['rows'] if row['id'] == '10')
    assert panel._begin_review_task('rgb', source)
    window._stereo_capture.refresh()
    app.processEvents()
    assert panel._review_task_box.isVisible()
    assert '定向补拍' in window._stereo_capture.instruction.text()
    target = window._stereo_capture._target()
    window._stereo_capture._apply_guidance_target()
    assert window._stereo_capture._target() is target
    assert not window._stereo_capture.auto_ready()
    assert not panel._btn_auto.isChecked()
    if output:
        import time
        for cid in ('left', 'right'):
            window._stream_manager.frame_received.emit(cid, image, time.time())
        app.processEvents()
        assert window.grab().save(str(directory / 'calibration-retake.png'))
    panel._review_task = None
    panel._refresh_review_controls()
    window.hide()
    # Wait for any guidance analysis started while the real workbench was visible.
    from PySide6.QtCore import QThreadPool
    QThreadPool.globalInstance().waitForDone(10000)
    app.processEvents()


def test_stale_review_action_cannot_apply_to_changed_data(dataset, app):
    panel, _, _ = dataset
    analysis(panel)
    panel._intr_cals['left']._all_corners[0] += 1
    panel._intr_cals['left']._data_revision += 1
    assert not panel._toggle_review_from_report('rgb', '1')
    assert not panel._review_excluded('rgb')
    wait_until(app, lambda: panel._review_worker is None)


def test_stale_calibration_completion_is_not_published(dataset, app):
    panel, _, _ = dataset
    panel._calibrate_pair('rgb')
    wait_until(app, lambda: panel._calib_worker is None)
    result = panel._pair_calibs['rgb']
    panel._invalidate_pair_result('rgb')
    panel._calibration_review_context = ('rgb', panel._review_key('rgb'))
    panel._intr_cals['left'].pop_last_frame()
    panel._on_pair_calibration_finished('rgb', 'left', 'right', result.left_intrinsics,
                                         result.right_intrinsics, result)
    assert 'rgb' not in panel._pair_calibs
    assert 'rgb' in panel._review_archive
    assert '未发布过期' in panel._log_text.toPlainText()


def test_compound_tilt_retake_uses_both_reference_angles(dataset, window, monkeypatch):
    panel, image, detections = dataset
    start_retake(panel, window, image, monkeypatch)
    guide = window._stereo_capture
    task = panel._review_task
    samples = [guide._detection_sample(det, image) for det in detections]
    task['reference_centers'] = [sample['center'] for sample in samples]
    task['reference_poses'] = [{'yaw': 25., 'pitch': 30.}, {'yaw': 26., 'pitch': 29.}]
    monkeypatch.setattr(guide, '_distance_status', lambda *args: (True, '距离到位'))
    monkeypatch.setattr(guide, '_pose_measurement', lambda *args: {'yaw': 24., 'pitch': 31.})
    assert guide.validate_review_capture('rgb', image, image, *detections)[0]
    monkeypatch.setattr(guide, '_pose_measurement', lambda *args: {'yaw': 50., 'pitch': 10.})
    assert not guide.validate_review_capture('rgb', image, image, *detections)[0]


def test_duplicate_advice_ignores_manually_excluded_observations(dataset, window):
    from multicamera.board.detector import DetectionResult
    panel, _, _ = dataset
    panel._live_preview_indices['rgb'] = [
        (i, i, fid) for i, fid in enumerate(panel._stereo_cals['rgb'].frame_ids)]
    raw = panel._review_raw('rgb')['1']
    detections = [DetectionResult(obs.corners, obs.ids, [], None, obs.image_size)
                  for obs in (raw['left'], raw['right'])]
    pair = window._stream_manager.stereo_pairs['rgb']
    assert window._stereo_capture._duplicate_source(pair, detections) == '1'
    assert panel._toggle_review_participation('rgb', '1')
    assert window._stereo_capture._duplicate_source(pair, detections) is None


def test_review_history_is_independent_without_copying_rectification_maps(dataset, app):
    panel, _, _ = dataset
    panel._calibrate_pair('rgb')
    wait_until(app, lambda: panel._calib_worker is None)
    result = panel._pair_calibs['rgb']
    assert result.stereo.map1_left is not None
    frozen = panel._review_result_snapshot(result)
    assert frozen.stereo.map1_left is None and frozen.stereo.map1_right is None
    assert frozen.left_intrinsics.frame_ids == result.left_intrinsics.frame_ids
    result.calibration_info['selection']['stereo']['used_ids'].clear()
    assert frozen.calibration_info['selection']['stereo']['used_ids']
    assert not np.shares_memory(frozen.left_intrinsics.camera_matrix,
                                result.left_intrinsics.camera_matrix)


def test_offline_failed_detection_does_not_shift_frame_quality_evidence(dataset):
    from multicamera.board.detector import DetectionResult
    panel, image, (left, right) = dataset
    bad = DetectionResult(None, None, [], None, (640, 480))
    panel._offline_import_records = [('bad', image, image, bad, right),
                                      ('good', image, image, left, right)]
    panel._load_offline_records(swapped=False)
    assert panel._intr_cals['offline_L'].frame_ids == ['good']
    quality = panel._review_quality_map('offline', 'offline_L')
    assert quality['bad'].num_corners == 0
    assert quality['good'].num_corners == left.num_corners
    selected = panel._effective_pair_calibrators('offline')[0]
    assert [quality[fid].num_corners for fid in selected.frame_ids] == [left.num_corners]
