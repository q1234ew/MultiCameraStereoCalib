"""Admission uses original images, not preview counts or successful disk writes."""

from copy import deepcopy
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.board.detector import CharucoDetector
from multicamera.calibration.capture_quality import assess_stereo_capture
from multicamera.calibration.models import CameraModel
from multicamera.ui.threads.import_worker import StereoImageImportWorker
from test_adaptive_workflow_followup import prepare_raw_capture
from test_multimodal_preview import LEFT, RIGHT
from test_real_pairs_ui import wait_for

pytest_plugins = ['test_stereo_capture_ui']


@pytest.fixture
def observation():
    board = CharucoBoard(CharucoBoardConfig())
    image = cv2.cvtColor(board.generate_image(pixel_size=40), cv2.COLOR_GRAY2BGR)
    detection = CharucoDetector(board).detect(image)
    assert detection.num_corners == board.num_corners
    return board, image, detection


def evaluate(observation, **kwargs):
    board, image, det = observation
    return assess_stereo_capture(board, CameraModel.PINHOLE, image, image, det, det, **kwargs)


def test_partial_charuco_accepted_but_strict_requires_full(observation):
    board, image, det = observation
    det.charuco_corners = det.charuco_corners[:-12]
    det.charuco_ids = det.charuco_ids[:-12]
    assert evaluate((board, image, det)).accepted
    assert evaluate((board, image, det), strict=True).state == '拒收'


@pytest.mark.parametrize('problem', ['dark', 'bright', 'blur', 'sync', 'cluster', 'size', 'ids'])
def test_uncertain_or_invalid_originals_not_admitted(observation, problem):
    board, image, det = observation
    options = {}
    if problem in ('dark', 'bright', 'blur'):
        image[:] = {'dark': 20, 'bright': 240, 'blur': 130}[problem]
    elif problem == 'sync':
        options['synchronized'] = False
    elif problem == 'cluster':
        det.charuco_corners = det.charuco_corners[:8]
        det.charuco_ids = det.charuco_ids[:8]
    elif problem == 'size':
        options['expected_size'] = (100, 100)
    else:
        det.charuco_ids[0] = board.num_corners + 1
    decision = evaluate((board, image, det), **options)
    assert not decision.accepted and decision.reasons


def test_fisheye_checks_existing_counts_without_cropping(observation):
    board, image, det = observation
    points = det.charuco_corners.copy()
    decision = assess_stereo_capture(board, CameraModel.FISHEYE, image, image, det, det,
                                     previous_common_counts=[board.num_corners - 2])
    assert decision.state == '待复核' and '鱼眼' in decision.text()
    np.testing.assert_array_equal(det.charuco_corners, points)
    decision = evaluate(observation, expected_intrinsic_sizes=(None, (100, 100)))
    assert decision.state == '拒收' and '内参观测' in decision.text()


@pytest.mark.parametrize('strict', [False, True])
def test_held_original_saved_but_never_advances_calibration(
        stereo_bench, monkeypatch, app, tmp_path, strict):
    win = stereo_bench
    panel, det, frame = prepare_raw_capture(win, monkeypatch)
    index = panel._capture_sequence.index if hasattr(panel._capture_sequence, 'index') else None
    if strict:
        panel._combo_capture_quality.setCurrentIndex(1)
        det = deepcopy(det)
        det.charuco_corners = det.charuco_corners[:-12]
        det.charuco_ids = det.charuco_ids[:-12]
    else:
        frame[:] = 130
    for cid in (LEFT, RIGHT):
        monkeypatch.setattr(panel._intr_cals[cid].detector, 'detect', lambda image: det)
    panel._capture_impl()
    wait_for(app, lambda: not panel._pending_save_tokens)
    assert panel._stereo_cals['rgb'].num_frames == 0
    assert all(c.num_frames == 0 for c in panel._intr_cals.values())
    assert not panel._undo_stack[-1][1]
    if index is not None:
        assert panel._capture_sequence.index == index
    pair = panel.preview_pairs()[-1]
    assert pair.quality_state == ('拒收' if strict else '待复核')
    assert pair.save_state == '已保存'
    assert '未计入有效样本' in panel._lbl_capture_decision.text()
    details = []
    monkeypatch.setattr('multicamera.ui.widgets.calib_panel.QMessageBox.information',
                        lambda parent, title, text: details.append(text))
    panel._btn_capture_decision_details.click()
    assert details and all(reason in details[-1] for reason in pair.quality_reasons)
    directory = Path(os.environ.get('MULTICAMERA_QUALITY_QA_DIR', tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    for width, height in ((1000, 680), (1280, 800), (1440, 1024)):
        win.resize(width, height)
        app.processEvents()
        for widget in (panel._lbl_capture_decision, panel._btn_capture, panel._btn_to_calibration):
            assert widget.isVisible()
            assert win.rect().contains(widget.mapTo(win, widget.rect().bottomRight()))
        win.grab().save(str(directory / f'quality-{strict}-{width}.png'))
    path = panel._session.session_dir / 'images/rgb/quality_0000.json'
    assert json.loads(path.read_text(encoding='utf-8'))['state'] == pair.quality_state
    worker = StereoImageImportWorker(str(path.parent), panel._board)
    imported, failures = [], []
    worker.finished_ok.connect(imported.append)
    worker.failed.connect(failures.append)
    worker.run()
    assert not imported and failures
    panel._on_undo_capture()
    assert not path.exists() and not panel.preview_pairs()


def test_accepted_partial_evidence_persists_across_import(stereo_bench, monkeypatch, app):
    panel, det, _ = prepare_raw_capture(stereo_bench, monkeypatch)
    det.charuco_corners = det.charuco_corners[:-12]
    det.charuco_ids = det.charuco_ids[:-12]
    for cid in (LEFT, RIGHT):
        monkeypatch.setattr(panel._intr_cals[cid].detector, 'detect', lambda image: det)
    panel._capture_impl()
    wait_for(app, lambda: not panel._pending_save_tokens)
    assert panel._stereo_cals['rgb'].num_frames == 1
    assert panel.preview_pairs()[-1].quality_state == '可采纳'
    assert '可采纳' in stereo_bench._gallery._latest_capture.text()
    worker = StereoImageImportWorker(
        str(panel._session.session_dir / 'images/rgb'), panel._board)
    imported = []
    worker.finished_ok.connect(imported.append)
    worker.run()
    assert len(imported) == 1 and len(imported[0]) == 1


def test_strict_auto_rejection_does_not_save_or_advance(stereo_bench, monkeypatch):
    panel, det, _ = prepare_raw_capture(stereo_bench, monkeypatch)
    panel._combo_capture_quality.setCurrentIndex(1)
    det.charuco_corners = det.charuco_corners[:-12]
    det.charuco_ids = det.charuco_ids[:-12]
    monkeypatch.setattr(panel, '_capture_validator', lambda *args: (True, ''))
    for cid in (LEFT, RIGHT):
        monkeypatch.setattr(panel._intr_cals[cid].detector, 'detect', lambda image: det)
    panel._on_auto_capture()
    assert not panel._pending_save_tokens and not panel._undo_stack
    assert panel._stereo_cals['rgb'].num_frames == 0
    assert '拒收' in panel._lbl_capture_decision.text()
    assert not list(panel._session.session_dir.rglob('left_*.png'))
