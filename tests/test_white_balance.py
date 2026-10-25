"""AWB stabilization, freeze validation and capture-gate regression."""

import time
import threading

import numpy as np
import pytest

from multicamera.streaming.white_balance import WhiteBalancePolicy, camera_base, stabilize_and_lock
from multicamera.streaming.stream_manager import StreamManager
from multicamera.ui.threads import white_balance_worker


class Clock:
    now = 0.

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Client:
    def __init__(self, jump=False, unstable=False, unavailable=None):
        self.calls, self.modes = [], {}
        self.jump, self.unstable, self.unavailable = jump, unstable, unavailable
        self.samples = 0

    def config(self, base, payload=None):
        if base == self.unavailable:
            raise RuntimeError('接口不可用')
        if payload is not None:
            self.calls.append((base, payload.copy()))
            self.modes[base] = payload['white_balance_automatic']
        return {'ok': True, 'controls': {'white_balance_automatic': {'value': self.modes.get(base, 0)},
                                         'white_balance_temperature': {'value': 4254}}}

    def chroma(self, base):
        self.samples += 1
        if self.unstable:
            return np.array([.5 if self.samples % 2 else 1., .9])
        return np.array([.5 if self.jump and self.modes[base] == 0 else 1., .9])


def run(client, cancelled=lambda: False):
    clock = Clock()
    return stabilize_and_lock(['http://device/stream', 'http://device:8082/stream'], client,
                              WhiteBalancePolicy(warmup_seconds=.2, timeout_seconds=.8,
                                                 sample_interval=.1, window=2, lock_settle_seconds=.1),
                              cancelled=cancelled, clock=clock.clock, sleep=clock.sleep)


def test_warmup_freezes_both_sides_without_replaying_temperature():
    client = Client()
    result = run(client)
    assert set(result) == {'http://device/', 'http://device:8082/'}
    assert list(client.modes.values()) == [0, 0]
    assert all(set(payload) == {'white_balance_automatic'} for _, payload in client.calls)
    assert [p['white_balance_automatic'] for _, p in client.calls] == [1, 1, 0, 0]
    assert client.samples >= 8


@pytest.mark.parametrize('problem', ['jump', 'unstable', 'second_unavailable', 'cancel'])
def test_failed_prepare_never_returns_ready_and_recovers_automatic_mode(problem):
    client = Client(jump=problem == 'jump', unstable=problem == 'unstable',
                    unavailable='http://device:8082/' if problem == 'second_unavailable' else None)
    with pytest.raises(RuntimeError):
        run(client, cancelled=lambda: problem == 'cancel' and client.samples > 0)
    assert client.modes and all(value == 1 for value in client.modes.values())
    assert all('white_balance_temperature' not in payload for _, payload in client.calls)


def test_camera_base_dedup_and_empty_targets():
    assert camera_base('http://device/prefix/stream?cache=0') == 'http://device/prefix/'
    assert camera_base('http://device:8082/snapshot') == 'http://device:8082/'
    assert stabilize_and_lock([]) == {}


def wait_until(app, predicate, seconds=3):
    end = time.monotonic() + seconds
    while not predicate() and time.monotonic() < end:
        app.processEvents()
        time.sleep(.005)
    assert predicate()


def test_guard_waits_for_worker_and_new_frames_before_resuming(app, monkeypatch):
    manager = StreamManager()
    guard = white_balance_worker.WhiteBalanceGuard(manager)
    calls, resumed = [], []
    monkeypatch.setattr(white_balance_worker, 'stabilize_and_lock',
                        lambda urls, **kw: dict.fromkeys(urls, {'ok': True}))
    monkeypatch.setattr(manager, 'discard_frames_for_url', lambda url, **kw: calls.append(url))
    try:
        assert not guard.ensure(['http://device/stream'], lambda: resumed.append(True))
        assert guard.busy and not resumed
        wait_until(app, lambda: guard.worker is None)
        assert guard.busy and not resumed and calls == ['http://device/']
        wait_until(app, lambda: not guard.busy)
        assert resumed == [True]
        assert guard.ensure(['http://device/stream'], lambda: resumed.append(True))
        manager.camera_disconnected.emit('L', '断线')
        assert not guard.ready
    finally:
        guard.shutdown()
        if guard.worker is not None:
            guard.worker.wait(4000)
            app.processEvents()
        guard.deleteLater()


def test_guard_failure_does_not_resume_capture(app, monkeypatch):
    manager = StreamManager()
    guard = white_balance_worker.WhiteBalanceGuard(manager)
    resumed, messages = [], []

    def fail(*args, **kw):
        raise RuntimeError('颜色跳变')
    monkeypatch.setattr(white_balance_worker, 'stabilize_and_lock', fail)
    guard.message.connect(messages.append)
    try:
        guard.ensure(['http://device/stream'], lambda: resumed.append(True))
        wait_until(app, lambda: not guard.busy)
        assert not resumed and not guard.ready
        assert '未采图' in messages[-1]
    finally:
        guard.shutdown()
        if guard.worker is not None:
            guard.worker.wait(4000)
            app.processEvents()
        guard.deleteLater()


def test_guard_invalidates_settling_result_before_capture(app, monkeypatch):
    manager = StreamManager()
    guard = white_balance_worker.WhiteBalanceGuard(manager)
    resumed = []
    monkeypatch.setattr(white_balance_worker, 'stabilize_and_lock', lambda urls, **kw: dict.fromkeys(urls, {}))
    monkeypatch.setattr(manager, 'discard_frames_for_url', lambda *args, **kw: [])
    try:
        guard.ensure(['http://device/stream'], lambda: resumed.append(True))
        wait_until(app, lambda: guard.worker is None)
        assert guard.busy
        guard.invalidate()
        assert not guard.busy and not guard.ready
        app.processEvents()
        assert not resumed
    finally:
        guard.shutdown()
        if guard.worker is not None:
            guard.worker.wait(4000)
            app.processEvents()
        guard.deleteLater()


def test_guard_cancel_while_waiting_for_new_frames_never_resumes(app, monkeypatch):
    manager = StreamManager()
    guard = white_balance_worker.WhiteBalanceGuard(manager)
    resumed, messages = [], []
    monkeypatch.setattr(white_balance_worker, 'stabilize_and_lock', lambda urls, **kw: dict.fromkeys(urls, {}))
    monkeypatch.setattr(manager, 'discard_frames_for_url', lambda *args, **kw: [])
    guard.message.connect(messages.append)
    try:
        guard.ensure(['http://device/stream'], lambda: resumed.append(True))
        wait_until(app, lambda: guard.worker is None)
        assert guard.stage == '等待新画面'
        guard.cancel()
        assert not guard._timer.isActive()
        assert not guard.busy and not guard.ready and not resumed
        assert '保持不变' in messages[-1]
    finally:
        guard.shutdown()
        guard.deleteLater()


def test_core_reports_preparation_stages():
    clock, stages = Clock(), []
    stabilize_and_lock(['http://device/stream'], Client(),
                       WhiteBalancePolicy(warmup_seconds=.2, timeout_seconds=.8,
                                          sample_interval=.1, window=2, lock_settle_seconds=.1),
                       clock=clock.clock, sleep=clock.sleep, progress=stages.append)
    assert stages[0] == '读取相机参数'
    assert '锁定白平衡' in stages and stages[-1] == '复核锁定后颜色'


def test_cancel_running_worker_does_not_resume_or_destroy_thread(app, monkeypatch):
    manager = StreamManager()
    guard = white_balance_worker.WhiteBalanceGuard(manager)
    entered, release = threading.Event(), threading.Event()
    resumed = []

    def prepare(urls, cancelled, progress):
        progress('检查颜色稳定')
        entered.set()
        release.wait(3)
        if cancelled():
            raise RuntimeError('用户取消')
        return dict.fromkeys(urls, {})

    monkeypatch.setattr(white_balance_worker, 'stabilize_and_lock', prepare)
    try:
        guard.ensure(['http://device/stream'], lambda: resumed.append(True))
        wait_until(app, entered.is_set)
        guard.cancel()
        assert guard.busy and guard.worker is not None
        assert guard.cancel_requested
        release.set()
        wait_until(app, lambda: not guard.busy)
        assert not guard.ready and not resumed and guard.worker is None
    finally:
        release.set()
        guard.shutdown()
        if guard.worker is not None:
            guard.worker.wait(4000)
            app.processEvents()
        guard.deleteLater()
