"""One-shot live capture gating and independent source-frame evidence contracts."""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QObject, QThreadPool

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.io.session import CalibrationSession
from multicamera.streaming.stream_manager import (
    AuxiliaryCameraConfig, CameraConfig, FrameEntry, StereoPairConfig, StreamManager,
)
from multicamera.ui.threads.aux_capture_worker import AuxCaptureWorker
from multicamera.ui.widgets import aux_capture
from multicamera.ui.widgets import calib_panel
from multicamera.ui.widgets.aux_live_capture import AuxLiveCaptureGate
from multicamera.ui.widgets.calib_panel import CalibrationPanel


class FakeGuide:
    def __init__(self):
        self.revision = 1
        self.reviewing = False
        self.problem = '等待连续3次稳定检测'
        self.invalidate_calls = 0
        self.preview_requests = 0
        self.messages = []

    def context_key(self):
        return ('same-session', self.revision)

    def invalidate_live(self):
        self.invalidate_calls += 1
        self.problem = '等待新的稳定检测'

    def capture_problem(self):
        return self.problem

    def _request_preview(self):
        self.preview_requests += 1

    def show_receipt_message(self, message):
        self.messages.append(message)


class FakePanel(QObject):
    def __init__(self):
        super().__init__()
        self._aux_guide = FakeGuide()
        self._active_step = 2
        self._workflow_phase = 'capture'
        self._white_balance_guard = SimpleNamespace(_generation=0)
        self.checked = True
        self.mode = 'automatic'
        self._check_wb = SimpleNamespace(isChecked=lambda: self.checked)
        self._combo_wb_mode = SimpleNamespace(currentData=lambda: self.mode)
        self.frame = object()
        self.pairing_problem = ''
        self.captured = []
        self.saved_count = 0
        self.pair_count = 0
        self.logs = []
        self._prepare_white_balance = Mock()

    def _aux_live_frames(self):
        return (None, self.pairing_problem) if self.pairing_problem else (
            (self.frame, self.frame, .01), '软件配对')

    def _capture_aux_original(self):
        self.captured.append(self.frame)
        self.saved_count += 1
        self.pair_count += 1

    def _log(self, message):
        self.logs.append(message)


@pytest.fixture
def gate(app):
    panel = FakePanel()
    gate = AuxLiveCaptureGate(panel)
    clock = [100.]
    gate._clock = lambda: clock[0]
    gate.test_clock = clock
    gate.transitions = []
    gate.changed.connect(gate.transitions.append)
    yield gate
    gate.cancel('测试结束')
    gate._timer.stop()
    panel.deleteLater()
    app.processEvents()


def assert_not_saved(gate):
    panel = gate.panel
    assert panel.captured == []
    assert panel.saved_count == panel.pair_count == 0
    panel._prepare_white_balance.assert_not_called()


def test_waiting_request_is_idempotent_and_does_not_repeat_preparation(gate):
    gate.start()
    assert gate.waiting and gate._timer.isActive()
    assert gate.reason and gate.transitions == [True]
    started = gate.started_at
    for _ in range(30):
        gate.test_clock[0] += .01
        gate.start(reset=True)
    assert gate.started_at == started
    assert gate.panel._aux_guide.invalidate_calls == 0
    assert gate.panel._aux_guide.preview_requests == 1
    assert gate.transitions == [True]
    assert_not_saved(gate)


def test_ready_request_captures_once_then_late_callbacks_do_nothing(gate):
    gate.panel._aux_guide.problem = ''
    gate.start()
    assert not gate.waiting and not gate._timer.isActive()
    assert gate.transitions == [True, False]
    assert gate.panel.saved_count == gate.panel.pair_count == 1
    for _ in range(20):
        gate.poll()
        gate.cancel('迟到的取消')
    assert gate.panel.saved_count == 1 and not gate.panel.logs
    gate.panel._prepare_white_balance.assert_not_called()


def test_capture_resolves_latest_frame_not_frame_at_request_start(gate):
    initial = gate.panel.frame
    gate.start()
    assert_not_saved(gate)
    gate.panel.frame = object()
    latest = gate.panel.frame
    gate.panel._aux_guide.problem = ''
    gate.poll()
    assert gate.panel.captured == [latest]
    assert gate.panel.captured[0] is not initial
    assert not gate.waiting and gate.transitions == [True, False]


@pytest.mark.parametrize('kind', ['guide', 'step', 'phase', 'generation', 'policy', 'mode', 'review'])
def test_changed_context_cancels_even_if_cached_checks_become_ready(gate, kind):
    gate.start()
    panel = gate.panel
    if kind == 'guide':
        panel._aux_guide.revision += 1
    elif kind == 'step':
        panel._active_step = 1
    elif kind == 'phase':
        panel._workflow_phase = 'calibration'
    elif kind == 'generation':
        panel._white_balance_guard._generation += 1
    elif kind == 'policy':
        panel.checked = False
    elif kind == 'mode':
        panel.mode = 'manual'
    else:
        panel._aux_guide.reviewing = True
    panel._aux_guide.problem = ''
    gate.poll()
    assert not gate.waiting and not gate._timer.isActive()
    assert gate.transitions == [True, False]
    assert len(panel.logs) == len(panel._aux_guide.messages) == 1
    assert '未保存、未增加计数' in panel._aux_guide.messages[0]
    assert_not_saved(gate)


def test_missing_pair_waits_until_timeout_without_side_effects(gate):
    gate.panel.pairing_problem = '红外无新帧'
    gate.start()
    assert gate.reason == '红外无新帧'
    gate.test_clock[0] = gate.started_at + gate.TIMEOUT_SECONDS - .001
    gate.poll()
    assert gate.waiting
    gate.test_clock[0] += .001
    gate.poll()
    assert not gate.waiting and not gate._timer.isActive()
    assert '超时' in gate.panel.logs[0]
    assert '红外无新帧' in gate.panel.logs[0]
    gate.poll()
    assert len(gate.panel.logs) == 1
    assert_not_saved(gate)


@pytest.mark.parametrize('elapsed', [15., 16.])
def test_ready_result_after_deadline_does_not_resurrect_expired_request(gate, elapsed):
    gate.start()
    gate.test_clock[0] = gate.started_at + elapsed
    gate.panel._aux_guide.problem = ''
    gate.poll()
    assert not gate.waiting and not gate._timer.isActive()
    assert_not_saved(gate)
    assert '超时' in gate.panel.logs[0]


def test_cancel_is_once_only_and_new_request_gets_new_deadline(gate):
    gate.cancel()
    assert not gate.panel.logs
    gate.start()
    first = gate.started_at
    gate.cancel('用户取消')
    gate.cancel('重复取消')
    assert gate.transitions == [True, False]
    assert len(gate.panel.logs) == len(gate.panel._aux_guide.messages) == 1
    assert_not_saved(gate)
    gate.test_clock[0] += 4
    gate.start()
    assert gate.waiting and gate.started_at == first + 4
    assert gate.transitions == [True, False, True]


def test_reset_discards_cached_ready_evidence_once(gate):
    gate.panel._aux_guide.problem = ''
    gate.start(reset=True)
    assert gate.waiting and gate.panel._aux_guide.invalidate_calls == 1
    assert_not_saved(gate)
    gate.start(reset=True)
    assert gate.panel._aux_guide.invalidate_calls == 1
    gate.panel._aux_guide.problem = ''
    gate.poll()
    assert gate.panel.saved_count == 1


def test_synchronous_cancel_from_started_signal_leaves_no_running_timer(gate):
    gate.changed.connect(lambda waiting: gate.cancel('即时取消') if waiting else None)
    gate.start()
    assert not gate.waiting
    assert not gate._timer.isActive()
    assert gate.transitions == [True, False]
    assert_not_saved(gate)


def board_image():
    image = np.full((192, 256, 3), 255, np.uint8)
    for row in range(5):
        for col in range(7):
            if (row + col) % 2 == 0:
                image[40+row*20:60+row*20, 40+col*20:60+col*20] = 0
    return image


@pytest.fixture
def live_panel(app, tmp_path, monkeypatch, finish_aux_tasks):
    manager = StreamManager()
    manager.add_virtual_stereo_pair(StereoPairConfig('rgb', CameraConfig('left', ''),
                                                    CameraConfig('right', '')))
    manager.set_auxiliary_camera(AuxiliaryCameraConfig('ir', 'http://device.test/ir'))
    board = CharucoBoard(CharucoBoardConfig())
    panel = CalibrationPanel(manager, board)
    panel.set_session(CalibrationSession.create(tmp_path, board,
        list(manager.stereo_pairs.values()), manager.auxiliary_camera))
    panel.show_workflow('capture', True)
    finish_aux_tasks(panel)
    panel._aux_readiness_timer.stop()
    panel._readiness_timer.stop()
    clock = [100.]
    panel.test_clock = clock
    panel._aux_live_gate._clock = lambda: clock[0]
    guide = panel._aux_guide
    guide._clock = lambda: clock[0]
    guide._manual = True
    guide.invalidate_live()
    panel._white_balance_guard.ensure = Mock(side_effect=AssertionError('Unexpected device request'))
    yield panel
    panel._aux_live_gate.cancel('测试清理')
    panel._wb_progress_timer.stop()
    finish_aux_tasks(panel)
    assert QThreadPool.globalInstance().waitForDone(15000)
    manager.stop_all()
    panel.close()


def set_frames(panel, image, stamp):
    for camera in ('left', 'ir'):
        panel._sm._latest_frames[camera] = FrameEntry(image, stamp)


def detection_result(panel, image):
    worker = AuxCaptureWorker(None, (image, image), panel._aux_board_config(),
                               (None, None), 'ir', enhanced=True)
    results = []
    worker.signals.completed.connect(lambda key, result, error: results.append((result, error)))
    worker.run()
    assert results and not results[0][1]
    assert all(quality.accepted for quality in results[0][0][1])
    return results[0][0]


def feed_preview(panel, result, start, stamp, *, delivered_at=None):
    guide = panel._aux_guide
    panel.test_clock[0] = start if delivered_at is None else delivered_at
    guide._preview_ready((guide.context_key(), start, (stamp, stamp)), result, '')


def test_sampler_uses_start_time_and_repeated_source_never_proves_stability(live_panel):
    panel, image = live_panel, board_image()
    result = detection_result(panel, image)
    stamp = time.time() - .1
    for start in (100., 100.4, 100.8):
        feed_preview(panel, result, start, stamp, delivered_at=start+.1)
    guide = panel._aux_guide
    assert all(len(guide._samples[camera]) == 1 for camera in ('left', 'ir'))
    assert all(guide._samples[camera][0][0] == 100. for camera in ('left', 'ir'))
    assert guide.capture_problem()
    assert panel._aux_saved_count == 0 and not panel._aux_pairs
    panel._white_balance_guard.ensure.assert_not_called()


@pytest.mark.parametrize('count', [2, 3])
def test_both_sensors_require_three_distinct_stable_observations(live_panel, count):
    panel, image = live_panel, board_image()
    result = detection_result(panel, image)
    base = time.time() - .8
    offsets = (0., .7) if count == 2 else (0., .35, .7)
    for delta in offsets:
        feed_preview(panel, result, 100.+delta, base+delta)
    guide = panel._aux_guide
    assert all(len(guide._samples[camera]) == count for camera in ('left', 'ir'))
    assert bool(guide.capture_problem()) == (count < 3)
    assert panel._aux_saved_count == 0


@pytest.mark.parametrize('camera_index', [0, 1])
def test_one_sensor_reusing_source_stamp_does_not_pass_dual_stability(live_panel, camera_index):
    panel, image = live_panel, board_image()
    result = detection_result(panel, image)
    base = time.time() - .8
    guide = panel._aux_guide
    for delta in (0., .35, .7):
        panel.test_clock[0] = 100.+delta
        stamps = [base+delta, base+delta]
        stamps[camera_index] = base
        guide._preview_ready((guide.context_key(), 100.+delta, tuple(stamps)), result, '')
    repeated, advancing = ('left', 'ir') if camera_index == 0 else ('ir', 'left')
    assert len(guide._samples[repeated]) == 1
    assert len(guide._samples[advancing]) == 3
    assert guide.capture_problem()
    assert panel._aux_saved_count == 0


@pytest.mark.parametrize('kind', ['slow', 'stale_source', 'invalidated'])
def test_old_sampler_result_cannot_restore_ready_evidence(live_panel, kind):
    panel, image = live_panel, board_image()
    result = detection_result(panel, image)
    stamp = time.time() - (4 if kind == 'stale_source' else .1)
    delivered = 102. if kind == 'slow' else 100.1
    if kind == 'invalidated':
        panel.test_clock[0] = 100.05
        panel._aux_guide.invalidate_live()
    feed_preview(panel, result, 100., stamp, delivered_at=delivered)
    assert not panel._aux_guide.last_qualities
    assert not panel._aux_guide._samples
    assert panel._aux_guide.capture_problem()
    assert panel._aux_saved_count == 0 and not panel._aux_pairs


def test_sampler_has_one_inflight_request_and_uses_enhanced_detector(live_panel, monkeypatch):
    panel, image = live_panel, board_image()
    set_frames(panel, image, time.time() - .01)
    pool = SimpleNamespace(start=Mock())
    monkeypatch.setattr(aux_capture, 'QThreadPool', SimpleNamespace(globalInstance=lambda: pool))
    panel._aux_guide._preview_started = 98.
    panel._aux_guide._request_preview()
    worker = panel._aux_guide._preview_worker
    assert worker is not None and worker.enhanced
    for _ in range(20):
        panel.test_clock[0] += .1
        panel._aux_guide._request_preview()
    assert pool.start.call_count == 1
    assert panel._aux_guide._preview_worker is worker
    worker.run()
    assert panel._aux_guide._preview_worker is None
    assert panel._aux_saved_count == 0
    panel._white_balance_guard.ensure.assert_not_called()


def test_async_preparation_resumes_gate_without_repeating_prepare_or_ensure(live_panel, monkeypatch):
    panel, image = live_panel, board_image()
    set_frames(panel, image, time.time() - .01)
    callbacks = []
    def prepare(pairs, resume):
        callbacks.append(resume)
        return False
    prepare_mock = Mock(side_effect=prepare)
    monkeypatch.setattr(panel, '_prepare_white_balance', prepare_mock)
    monkeypatch.setattr(panel._aux_guide, '_request_preview', Mock())
    original = Mock()
    monkeypatch.setattr(panel, '_capture_aux_original', original)
    panel._on_capture_aux()
    assert prepare_mock.call_count == 1 and not panel._aux_live_gate.waiting
    callbacks[0]()
    assert panel._aux_live_gate.waiting and not panel._aux_guide._samples
    for _ in range(20):
        panel._on_capture_aux()
    assert prepare_mock.call_count == 1
    result = detection_result(panel, image)
    base = time.time() - 1.
    for delta in (0., .35, .7):
        feed_preview(panel, result, 100.+delta, base+delta)
    assert not panel._aux_guide.capture_problem()
    panel._aux_live_gate.poll()
    panel._aux_live_gate.poll()
    original.assert_called_once_with()
    assert prepare_mock.call_count == 1 and panel._aux_saved_count == 0
    panel._white_balance_guard.ensure.assert_not_called()


@pytest.mark.parametrize('change', ['generation', 'phase'])
def test_late_preparation_completion_cannot_restart_changed_context(live_panel, monkeypatch, change):
    panel = live_panel
    set_frames(panel, board_image(), time.time() - .01)
    callbacks = []
    def prepare(pairs, resume):
        callbacks.append(resume)
        return False
    monkeypatch.setattr(panel, '_prepare_white_balance', Mock(side_effect=prepare))
    original = Mock()
    monkeypatch.setattr(panel, '_capture_aux_original', original)
    panel._on_capture_aux()
    assert len(callbacks) == 1
    if change == 'generation':
        panel._white_balance_guard._generation += 1
    else:
        panel._workflow_phase = 'calibration'
    callbacks[0]()
    assert not panel._aux_live_gate.waiting
    original.assert_not_called()
    assert panel._aux_saved_count == 0 and not panel._aux_pairs
    panel._white_balance_guard.ensure.assert_not_called()


def test_actual_capture_saves_newest_original_once_after_stable_evidence(
        live_panel, monkeypatch, finish_aux_tasks):
    panel, image = live_panel, board_image()
    result = detection_result(panel, image)
    base = time.time() - 1.
    for delta in (0., .35, .7):
        set_frames(panel, image, base+delta)
        feed_preview(panel, result, 100.+delta, base+delta)
    latest = image.copy()
    latest[:3, :3] = [10, 20, 30]
    set_frames(panel, latest, time.time() - .01)
    prepare = Mock(return_value=True)
    monkeypatch.setattr(panel, '_prepare_white_balance', prepare)
    monkeypatch.setattr(panel._aux_guide, '_request_preview', Mock())
    enhanced = []
    def raw_worker(*args, **kwargs):
        enhanced.append(kwargs.get('enhanced', False))
        return AuxCaptureWorker(*args, **kwargs)
    monkeypatch.setattr(calib_panel, 'AuxCaptureWorker', raw_worker)
    panel._on_capture_aux()
    finish_aux_tasks(panel)
    assert not panel._aux_live_gate.waiting
    assert panel._aux_saved_count == 1 and len(panel._aux_pairs) == 1
    for _ in range(10):
        panel._aux_live_gate.poll()
    assert panel._aux_saved_count == 1 and prepare.call_count == 1
    assert enhanced == [True]
    folder = panel._session.session_dir / 'multimodal'
    for name in ('rgb_left_0000.png', 'aux_0000.png'):
        np.testing.assert_array_equal(cv2.imread(str(folder / name)), latest)
    assert len(list(folder.glob('quality_*.json'))) == 1
    panel._white_balance_guard.ensure.assert_not_called()
