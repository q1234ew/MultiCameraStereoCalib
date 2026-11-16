"""Control validation, persistence, optimistic concurrency, and pause boundaries."""
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.jetson.tracking_options import DEFAULT_OPTIONS, OptionStore, RevisionConflict
from scripts.jetson.tracking_service import TrackingRuntime


@pytest.mark.parametrize('patch', [
    {'enabled': 'false'}, {'camera_fps': 0}, {'camera_fps': 26},
    {'camera_fps': float('nan')},
    {'camera_fps': True}, {'imgsz': 641}, {'tracker': 'arbitrary'},
    {'new_track_conf': .9}, {'jpeg_quality': 100}, {'draw_boxes': 1}, {'path': '/tmp/x'},
])
def test_invalid_control_does_not_change_disk_or_revision(tmp_path, patch):
    path = tmp_path / 'runtime.json'
    path.write_text(json.dumps(DEFAULT_OPTIONS))
    store = OptionStore(path)
    revision, before = store.revision, path.read_bytes()
    with pytest.raises(ValueError):
        store.update(patch, revision)
    assert store.options == DEFAULT_OPTIONS
    assert store.revision == revision and path.read_bytes() == before


def test_persistence_and_stale_writer_rejection(tmp_path):
    path = tmp_path / 'runtime.json'
    store = OptionStore(path)
    old = store.revision
    store.update({'enabled': False, 'imgsz': 320}, old)
    assert not OptionStore(path).options['enabled']
    assert OptionStore(path).options['imgsz'] == 320
    with pytest.raises(RevisionConflict):
        store.update({'enabled': True}, old)
    assert not store.options['enabled']


def test_high_rate_setting_persists_without_changing_camera_resolution(tmp_path):
    path = tmp_path / 'runtime.json'
    store = OptionStore(path)
    store.update({'camera_fps': 20, 'imgsz': 480}, store.revision)
    reloaded = OptionStore(path)
    assert reloaded.options['camera_fps'] == 20
    assert reloaded.options['imgsz'] == 480


def test_persistence_failure_does_not_apply_partial_update(tmp_path, monkeypatch):
    path = tmp_path / 'runtime.json'
    store = OptionStore(path)
    revision = store.revision

    def failed_replace(*args):
        raise OSError('Disk unavailable')

    monkeypatch.setattr(Path, 'replace', failed_replace)
    with pytest.raises(OSError):
        store.update({'enabled': False}, revision)
    assert store.revision == revision and store.options['enabled']


def test_pause_clears_current_session_but_display_changes_preserve_ids(tmp_path):
    runtime = TrackingRuntime({'devices': [{'id': 'device_1'}]}, tmp_path / 'unused.pt')
    runtime.engine_ready, runtime.state = True, 'running'
    camera = runtime.cameras['device_1']
    camera.update(jpeg=b'old', tracks=[{'local_id': 1}], frame=('old',), processed_mono=1)
    session, epoch = runtime.session_id, camera['epoch']
    result = runtime.configure({'jpeg_quality': 70}, runtime.options.revision)
    assert not result['session_reset'] and camera['epoch'] == epoch
    result = runtime.configure({'enabled': False}, runtime.options.revision)
    assert result['session_reset'] and runtime.state == 'paused'
    assert runtime.session_id != session and camera['epoch'] != epoch
    assert camera['jpeg'] is None and camera['tracks'] == [] and camera['frame'] is None
    assert runtime.snapshot()['tracks'] == []
    runtime.configure({'enabled': True}, runtime.options.revision)
    assert runtime.state == 'running'


def test_pause_discards_an_already_inflight_camera_response(tmp_path, monkeypatch):
    import cv2

    body = cv2.imencode('.jpg', np.zeros((480, 640, 3), np.uint8))[1].tobytes()
    entered, release = threading.Event(), threading.Event()

    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            entered.set()
            assert release.wait(2)
            return body

    monkeypatch.setattr('urllib.request.build_opener',
                        lambda *_: SimpleNamespace(open=lambda *args, **kwargs: Response()))
    device = {'id': 'device_1', 'host': 'fixture', 'raw_ports': {'left': 80},
              'resolution': {'left': [640, 480]}}
    runtime = TrackingRuntime({'devices': [device]}, tmp_path / 'unused.pt')
    thread = threading.Thread(target=runtime.capture_loop, args=(device,), daemon=True)
    thread.start()
    try:
        assert entered.wait(2)
        runtime.configure({'enabled': False}, runtime.options.revision)
        release.set()
        runtime.stop.set()
        thread.join(2)
        assert not thread.is_alive()
        assert runtime.cameras['device_1']['frame'] is None
    finally:
        release.set()
        runtime.stop.set()


def test_noop_does_not_reset_tracking_session(tmp_path):
    runtime = TrackingRuntime({'devices': []}, tmp_path / 'unused.pt')
    session = runtime.session_id
    result = runtime.configure({'imgsz': 640}, runtime.options.revision)
    assert runtime.session_id == session and not result['session_reset']


def test_native_dependencies_are_initialized_before_capture_threads(tmp_path, monkeypatch):
    import sys

    initialized = set()
    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace(
        setNumThreads=lambda _: initialized.add('cv2')))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        set_num_threads=lambda _: initialized.add('torch')))
    factory = object()
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=factory))
    started = []

    class Thread:
        def __init__(self, **kwargs):
            self.target = kwargs['target']

        def start(self):
            assert initialized == {'cv2', 'torch'}
            started.append(self.target)

    monkeypatch.setattr('scripts.jetson.tracking_service.threading.Thread', Thread)
    runtime = TrackingRuntime({'devices': [{'id': 'device_1'}]}, tmp_path/'unused.pt')
    runtime.start()
    assert runtime.model_factory is factory
    assert len(started) == 2


def test_exact_preview_encodes_lazily_and_does_not_hold_runtime_lock(tmp_path, monkeypatch):
    import time
    import cv2

    runtime = TrackingRuntime({'devices': [{'id': 'device_1'}]}, tmp_path / 'unused.pt')
    runtime.state = 'running'
    camera = runtime.cameras['device_1']
    camera.update(annotated=np.zeros((48, 64, 3), np.uint8), processed_mono=time.monotonic(),
                  processed_sequence=1)
    entered, release = threading.Event(), threading.Event()
    encode = cv2.imencode

    def slow_encode(*args):
        entered.set()
        assert release.wait(2)
        return encode(*args)

    monkeypatch.setattr(cv2, 'imencode', slow_encode)
    results = []
    thread = threading.Thread(target=lambda: results.append(runtime.jpeg('device_1')))
    thread.start()
    try:
        assert entered.wait(2)
        # Pausing must remain responsive while a client requests an exact JPEG.
        runtime.configure({'enabled': False}, runtime.options.revision)
        release.set()
        thread.join(2)
        assert results == [None]
        assert camera['annotated'] is None and camera['jpeg'] is None
    finally:
        release.set()
        thread.join(2)


def test_exact_preview_is_a_valid_cached_jpeg(tmp_path):
    import time
    import cv2

    runtime = TrackingRuntime({'devices': [{'id': 'device_1'}]}, tmp_path / 'unused.pt')
    runtime.state = 'running'
    camera = runtime.cameras['device_1']
    camera.update(annotated=np.zeros((48, 64, 3), np.uint8), processed_mono=time.monotonic(),
                  processed_sequence=1)
    body = runtime.jpeg('device_1')
    assert body == runtime.jpeg('device_1')
    assert cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR).shape == (48, 64, 3)
    assert runtime.jpeg('unknown') is None
