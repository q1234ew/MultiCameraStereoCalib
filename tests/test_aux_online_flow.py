"""Real panel request -> stable originals -> final acceptance and persistence."""

import time
import json
from copy import deepcopy
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import cv2

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.calibration.planar import PlanarPatternDetector
from multicamera.calibration.models import CameraIntrinsics
from multicamera.calibration.aux_capture_quality import assess_aux_frame
from multicamera.io.session import CalibrationSession
from multicamera.streaming.stream_manager import (
    AuxiliaryCameraConfig, CameraConfig, FrameEntry, StereoPairConfig, StreamManager,
)
from multicamera.ui.widgets.calib_panel import CalibrationPanel
from multicamera.ui.widgets.aux_capture import AuxCaptureGuide
from multicamera.calibration.offline_aux import confirm_offline_pair, offline_pair_review
from multicamera.ui.threads.import_worker import AuxImageImportWorker


def board_image():
    image = np.full((192, 256, 3), 255, np.uint8)
    for row in range(5):
        for col in range(7):
            if (row + col) % 2 == 0:
                image[40 + row*20:60 + row*20, 40 + col*20:60 + col*20] = 0
    return image


@pytest.fixture
def online(app, tmp_path, monkeypatch, finish_aux_tasks):
    manager = StreamManager()
    manager.add_virtual_stereo_pair(StereoPairConfig('rgb', CameraConfig('left', ''),
                                                    CameraConfig('right', '')))
    manager.set_auxiliary_camera(AuxiliaryCameraConfig('ir', 'http://device.test/ir'))
    board = CharucoBoard(CharucoBoardConfig())
    panel = CalibrationPanel(manager, board)
    panel.set_session(CalibrationSession.create(tmp_path, board,
        list(manager.stereo_pairs.values()), manager.auxiliary_camera))
    panel.show_workflow('capture', True)
    monkeypatch.setattr(panel, '_prepare_white_balance', lambda *args: True)
    monkeypatch.setattr(panel._aux_guide, '_request_preview', lambda: None)
    finish_aux_tasks(panel)
    yield panel
    panel._aux_live_gate.cancel()
    finish_aux_tasks(panel)
    manager.stop_all()
    panel.close()


def frames(panel, image):
    for cid in ('left', 'ir'):
        panel._sm._latest_frames[cid] = FrameEntry(image, time.time())


def stable(panel, image):
    guide = panel._aux_guide
    guide.invalidate_live()
    det = PlanarPatternDetector(panel._aux_board_config()).detect(image, adaptive_subpixel=True)
    preview = SimpleNamespace(valid=det.valid, num_corners=det.num_points,
                             charuco_corners=det.image_points, image_size=det.image_size)
    now, stamp = time.monotonic(), time.time()
    for offset in (1., .5, 0.):
        for cid in ('left', 'ir'):
            guide.observe(cid, preview, image, timestamp=stamp-offset, observed_at=now-offset)
    guide._preview_detections = [det, det]
    guide.last_qualities = [assess_aux_frame(image, det, panel._aux_board_config(), modality=m)
                            for m in ('rgb', 'ir')]
    guide._preview_result_at = now


def test_click_waits_until_stable_then_saves_one_original_pair(online, finish_aux_tasks):
    panel, image = online, board_image()
    frames(panel, image)
    panel._on_capture_aux()
    assert panel._aux_live_gate.waiting and panel._aux_saved_count == 0
    assert '尚未保存' in panel._aux_guide.receipt.text()
    assert panel._capture_status_pages.currentWidget() is panel._wb_progress
    assert panel._btn_cancel_wb.isEnabled()
    panel._on_capture_aux()  # A repeated click must not enqueue another request.
    stable(panel, image)
    panel._aux_live_gate.poll()
    assert not panel._aux_live_gate.waiting
    assert panel._aux_capture_worker is not None
    finish_aux_tasks(panel)
    panel._aux_live_gate.poll()
    assert panel._aux_saved_count == len(panel._aux_pairs) == 1
    assert panel._aux_quality_records[0]['stability']['verified']
    assert panel._aux_quality_records[0]['detection'][1]['method'].startswith('sb_')


def test_white_balance_completion_rebuilds_evidence_without_another_prepare(
        online, monkeypatch, finish_aux_tasks):
    panel, image = online, board_image()
    frames(panel, image)
    stable(panel, image)
    callbacks = []
    monkeypatch.setattr(panel, '_prepare_white_balance',
                        lambda pairs, resume: callbacks.append(resume) or False)
    panel._on_capture_aux()
    assert len(callbacks) == 1 and not panel._aux_live_gate.waiting
    callbacks[0]()
    assert panel._aux_live_gate.waiting
    assert not panel._aux_guide._samples  # No pre-AWB stability may be reused.
    panel._aux_live_gate.poll()
    assert panel._aux_saved_count == 0
    stable(panel, image)
    panel._aux_live_gate.poll()
    finish_aux_tasks(panel)
    assert len(callbacks) == 1
    assert panel._aux_saved_count == len(panel._aux_pairs) == 1


@pytest.mark.parametrize('ending', ['cancel', 'timeout', 'phase', 'connection', 'review'])
def test_unfinished_requests_never_save_or_advance(online, ending):
    panel, image = online, board_image()
    frames(panel, image)
    panel._on_capture_aux()
    gate = panel._aux_live_gate
    assert gate.waiting
    if ending == 'cancel':
        panel._cancel_white_balance()
    elif ending == 'timeout':
        gate._clock = lambda: gate.started_at + gate.TIMEOUT_SECONDS
    elif ending == 'phase':
        panel.show_workflow('calibration', True)
    elif ending == 'connection':
        panel._sm.camera_disconnected.emit('ir', 'test disconnect')
    else:
        panel._aux_guide._view = SimpleNamespace(review_key='old')
    gate.poll()
    assert not gate.waiting
    assert panel._aux_saved_count == 0 and not panel._aux_pairs
    assert '未保存' in panel._aux_guide.receipt.text()
    assert not list((panel._session.session_dir / 'multimodal').glob('*.png'))
    if ending == 'review':
        panel._aux_guide._view = None


def test_stale_sampler_result_uses_source_time_not_completion_time(online):
    panel, image = online, board_image()
    frames(panel, image)
    guide = panel._aux_guide
    stable(panel, image)
    detections, qualities = guide._preview_detections, guide.last_qualities
    guide.invalidate_live()
    started = guide._clock()
    guide._preview_ready((guide.context_key(), started, (time.time()-4, time.time()-4)),
                         (detections, qualities), '')
    assert not guide._samples and not guide.last_qualities


def test_three_slow_fresh_prechecks_reach_independent_original_acceptance(
        online, monkeypatch, finish_aux_tasks):
    panel, image = online, board_image()
    guide = panel._aux_guide
    clock, wall = [100.], [1000.]
    guide._clock = lambda: clock[0]
    panel._aux_live_gate._clock = lambda: clock[0]
    monkeypatch.setattr(time, 'time', lambda: wall[0])
    guide.invalidate_live()
    det = PlanarPatternDetector(panel._aux_board_config()).detect(image, adaptive_subpixel=True)
    qualities = [assess_aux_frame(image, det, panel._aux_board_config(), modality=m)
                 for m in ('rgb', 'ir')]
    frames(panel, image)
    panel._on_capture_aux()
    for i, started in enumerate((100., 102.5, 105.)):
        clock[0] = started + 2.2
        wall[0] = 1000. + clock[0] - 100.
        frames(panel, image)
        source_time = 1000. + started - 100.
        guide._preview_ready((guide.context_key(), started, (source_time, source_time)),
                             ((det, det), qualities), '')
        panel._aux_live_gate.poll()
        if i < 2:
            assert panel._aux_live_gate.waiting and panel._aux_saved_count == 0
    assert not panel._aux_live_gate.waiting and panel._aux_capture_worker is not None
    finish_aux_tasks(panel)
    assert panel._aux_saved_count == len(panel._aux_pairs) == 1
    assert panel._aux_quality_records[-1]['stability']['verified']


def test_raw_frame_change_after_ready_is_not_silently_accepted(online, finish_aux_tasks):
    panel, image = online, board_image()
    frames(panel, image)
    stable(panel, image)
    frames(panel, np.full_like(image, 100))
    panel._on_capture_aux()
    finish_aux_tasks(panel)
    assert panel._aux_saved_count == 1 and not panel._aux_pairs
    assert panel._aux_reviews[-1].quality_state == '待复核'
    assert '验收待复核' in panel._aux_guide.receipt.text()


def test_saved_online_photos_enter_explicit_review_not_legacy_pnp(online, finish_aux_tasks):
    panel, image = online, board_image()
    panel._intrinsics['left'] = CameraIntrinsics(
        np.array([[200., 0., 128.], [0., 200., 96.], [0., 0., 1.]]), np.zeros(5), (256, 192))
    for _ in range(6):
        frames(panel, image)
        stable(panel, image)
        panel._on_capture_aux()
        finish_aux_tasks(panel)
    assert panel._aux_saved_count == 6, panel._log_text.toPlainText()
    source = panel._session.session_dir / 'multimodal'
    originals = {path.name: path.read_bytes() for path in source.iterdir()}
    panel.show_workflow('calibration', True)
    panel._refresh_aux_readiness()
    assert panel._btn_calib_aux.text() == '复核已保存原图'
    assert panel._btn_calib_aux.isEnabled()
    panel._btn_calib_aux.click()
    finish_aux_tasks(panel)
    assert panel._aux_worker is None  # No old PnP-average result is silently accepted.
    assert len(panel._aux_offline_records) == 6
    assert not panel._aux_pairs  # Physical correspondence needs an explicit user decision.
    assert panel._aux_offline_dialog is not None and panel._aux_offline_dialog.isVisible()
    assert '至少需要 6 组' in panel._aux_calibration_problem()
    assert originals == {path.name: path.read_bytes() for path in source.iterdir()}
    panel._aux_offline_dialog.close()


def test_continuous_stream_runs_real_sampler_gate_and_raw_save(
        online, app, monkeypatch, finish_aux_tasks):
    panel, image = online, board_image()
    guide = panel._aux_guide
    monkeypatch.setattr(guide, '_request_preview', MethodType(AuxCaptureGuide._request_preview, guide))
    guide.invalidate_live()
    frames(panel, image)
    panel._on_capture_aux()
    assert panel._aux_live_gate.waiting and panel._aux_saved_count == 0
    deadline = time.monotonic() + 12
    while panel._aux_saved_count == 0 and time.monotonic() < deadline:
        frames(panel, image)
        app.processEvents()
        time.sleep(.02)
    finish_aux_tasks(panel)
    assert panel._aux_saved_count == len(panel._aux_pairs) == 1, panel._log_text.toPlainText()
    assert panel._aux_quality_records[-1]['stability']['verified']
    assert not panel._aux_live_gate.waiting


@pytest.mark.parametrize('kind', ['version', 'source', 'board'])
def test_old_confirmation_cannot_be_applied_to_replaced_dataset(online, tmp_path, kind):
    panel, image = online, board_image()
    board = panel._aux_board_config()
    det = PlanarPatternDetector(board).detect(image)
    paths = [tmp_path / 'rgb_left_0000.png', tmp_path / 'aux_0000.png']
    for path in paths:
        assert cv2.imwrite(str(path), image)
    current = offline_pair_review('0000', (image, image), (det, det), board, paths=paths)
    decision = confirm_offline_pair(current, 'identity', 'identity')
    panel._aux_offline_records = [('0000', image, image, det, det)]
    panel._aux_offline_reviews = {'0000': current}
    context = panel._aux_solver_context()
    if kind == 'version':
        panel._aux_dataset_revision += 1
    elif kind == 'source':
        decision['sources'][0]['path'] = str(tmp_path / 'other.png')
    else:
        decision['board']['square_size'] = .04
    before = deepcopy(panel._aux_offline_reviews)
    panel._apply_aux_offline_review({'0000': decision}, context)
    assert not panel._aux_pairs and not panel._aux_offline_confirmations
    assert before == panel._aux_offline_reviews
    assert '上下文已变化' in panel._log_text.toPlainText() or '不一致' in panel._log_text.toPlainText()


@pytest.mark.parametrize('recorded', [.04, 'invalid'])
def test_historical_board_scale_mismatch_is_not_readmitted(app, tmp_path, recorded):
    image = board_image()
    for prefix in ('rgb_left', 'aux'):
        assert cv2.imwrite(str(tmp_path / f'{prefix}_0000.png'), image)
    from multicamera.calibration.planar import PatternBoardConfig
    board = PatternBoardConfig.ir_chessboard()
    historical = board.to_dict()
    historical['square_size'] = recorded
    quality = tmp_path / 'quality_0000.json'
    quality.write_text(json.dumps({'state': '可采纳', 'board': historical}), encoding='utf-8')
    originals = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    worker = AuxImageImportWorker(str(tmp_path), board, offline_recheck=True)
    imported = []
    worker.finished_ok.connect(imported.extend)
    worker.run()
    assert len(imported) == 1
    assert not worker.offline_reviews['0000']['image_accepted']
    assert '历史标定板' in '；'.join(worker.offline_reviews['0000']['image_reasons'])
    with pytest.raises(ValueError, match='质量检查未通过'):
        confirm_offline_pair(worker.offline_reviews['0000'], 'identity', 'identity')
    assert originals == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
