"""Post-commit cleanup and cached-intrinsic provenance must fail consistently."""

from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.board.detector import DetectionResult
from multicamera.calibration.intrinsic import IntrinsicCalibrator
from multicamera.calibration.models import (
    CameraIntrinsics, MultiCameraRig, StereoCalibResult, StereoPairCalibration,
)
from multicamera.calibration.stereo import StereoCalibrator
from multicamera.io import session as session_module
from multicamera.ui.threads.calibration_worker import PairCalibrationWorker


@pytest.fixture
def session(tmp_path):
    value = session_module.CalibrationSession(tmp_path / 'session')
    value._metadata = {'frame_count': 0, 'multimodal_frame_count': 0}
    value._save_metadata()
    return value


@pytest.fixture
def rig():
    intr = CameraIntrinsics(np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]]),
                            np.zeros(5), (640, 480), rms_error=.2)
    stereo = StereoCalibResult(np.eye(3), np.array([[-.12], [0.], [0.]]),
                              np.eye(3), np.eye(3), rms_error=.2)
    return MultiCameraRig(pairs={'rgb': StereoPairCalibration('rgb', 'L', 'R', intr, intr, stereo)},
                          reference_camera='L')


def fail_cleanup(monkeypatch):
    original = session_module.TemporaryDirectory

    class CleanupFailure:
        def __init__(self, **kwargs):
            self.directory = original(**kwargs)
            self.name = self.directory.name

        def cleanup(self):
            self.directory.cleanup()
            raise PermissionError('injected staging cleanup failure')

    monkeypatch.setattr(session_module, 'TemporaryDirectory', CleanupFailure)


@pytest.mark.parametrize('kind', ['stereo', 'multimodal'])
def test_cleanup_after_capture_commit_keeps_disk_memory_and_images(session, monkeypatch, kind, caplog):
    fail_cleanup(monkeypatch)
    image = np.zeros((8, 8, 3), np.uint8)
    if kind == 'stereo':
        assert session.save_frame_pair('rgb', image, image) == 0
        counter = 'frame_count'
    else:
        assert session.save_multimodal_pair(image, image) == 0
        counter = 'multimodal_frame_count'
    assert session.metadata[counter] == 1
    assert json.loads((session.session_dir / 'session.json').read_text())[counter] == 1
    assert len(list(session.session_dir.rglob('*.png'))) == 2
    assert 'Failed to clean staging directory' in caplog.text


def test_cleanup_does_not_mask_original_capture_failure(session, monkeypatch):
    fail_cleanup(monkeypatch)

    def fail(*args):
        raise OSError('original image write failure')

    monkeypatch.setattr(cv2, 'imencode', fail)
    with pytest.raises(OSError, match='original image write failure'):
        session.save_frame_pair('rgb', np.zeros((8, 8), np.uint8), np.zeros((8, 8), np.uint8))
    assert session.frame_count == 0
    assert not list(session.session_dir.rglob('*.png'))


def test_configuration_cleanup_preserves_commit(session, monkeypatch):
    fail_cleanup(monkeypatch)
    session.record_configuration(CharucoBoard(CharucoBoardConfig()), [], None, 'fisheye')
    restored = session_module.CalibrationSession.load(session.session_dir)
    assert restored.metadata == session.metadata
    assert restored.metadata['camera_model'] == 'fisheye'


def test_result_and_manifest_cleanup_are_not_export_failures(session, rig, monkeypatch):
    fail_cleanup(monkeypatch)
    saved, errors = session.save_calibration_results(rig)
    assert not errors
    assert len(saved) == 5 and all(path.is_file() for path in saved)
    manifest = json.loads(saved[-1].read_text())
    assert {entry['name'] for entry in manifest['files']} == {path.name for path in saved[:-1]}


@pytest.mark.parametrize('failure', ['hash', 'manifest_write'])
def test_manifest_failure_returns_successful_outputs(session, rig, monkeypatch, failure):
    if failure == 'hash':
        original = Path.read_bytes

        def read(path):
            if path.name == 'rig.json':
                raise PermissionError('injected manifest hash failure')
            return original(path)

        monkeypatch.setattr(Path, 'read_bytes', read)
    else:
        original = session._atomic_json

        def write(path, data):
            if path.name == 'calibration_manifest.json':
                raise OSError('injected manifest write failure')
            return original(path, data)

        monkeypatch.setattr(session, '_atomic_json', write)
    saved, errors = session.save_calibration_results(rig)
    assert len(saved) == 4 and all(path.is_file() for path in saved)
    assert len(errors) == 1 and errors[0].startswith('calibration_manifest.json:')
    assert not (saved[0].parent / 'calibration_manifest.json').exists()


@pytest.fixture
def worker_data():
    board = CharucoBoard(CharucoBoardConfig())
    left, right = [IntrinsicCalibrator(board, min_frames=5) for _ in range(2)]
    stereo = StereoCalibrator(board, min_frames=5)
    k = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
    ids = np.arange(board.num_corners, dtype=np.int32)[:, None]
    for i in range(10):
        rv = np.array([.3 * np.sin(i), .2 * np.cos(i), .03 * i])
        tv = np.array([-.2 + .01 * i, -.15, .9 + .02 * i])
        detections = [DetectionResult(
            cv2.projectPoints(board.object_points, rv, tv + [offset, 0., 0.], k, None)[0]
            .astype(np.float32), ids, [], None, (640, 480),
        ) for offset in (0., -.12)]
        source_id = f'{i + 1:04d}'
        assert left.add_detection(detections[0], frame_id=source_id)
        assert right.add_detection(detections[1], frame_id=source_id)
        assert stereo.add_detection_pair(*detections, frame_id=source_id)
    intrinsics = [cal.calibrate(flags=cv2.CALIB_FIX_K3) for cal in (left, right)]
    return left, right, stereo, intrinsics


def run_worker(data, flags):
    left, right, stereo, intrinsics = data
    worker = PairCalibrationWorker('rgb', 'L', 'R', left, right, stereo,
                                   *intrinsics, stereo_flags=flags, data_filter_key='quality')
    success, failure = [], []
    worker.finished_ok.connect(lambda *args: success.append(args[-1]))
    worker.failed.connect(lambda *args: failure.append(args[-1]))
    worker.run()
    return success, failure


@pytest.mark.parametrize('known_ids', [True, False])
def test_fixed_provided_intrinsics_keep_actual_or_unknown_provenance(worker_data, monkeypatch, known_ids):
    if not known_ids:
        for intr in worker_data[3]:
            intr.frame_ids = None

    def unexpected(*args, **kwargs):
        raise AssertionError('Cached fixed intrinsics must not be recalibrated')

    monkeypatch.setattr(IntrinsicCalibrator, 'calibrate', unexpected)
    monkeypatch.setattr(IntrinsicCalibrator, 'recompute_errors', unexpected)
    success, failure = run_worker(worker_data, cv2.CALIB_FIX_INTRINSIC)
    assert not failure and len(success) == 1
    pair = success[0]
    for side, actual, original in zip(('left', 'right'),
                                      (pair.left_intrinsics, pair.right_intrinsics), worker_data[3]):
        info = pair.calibration_info['selection'][side]
        assert info['source'] == 'provided_intrinsics'
        assert info['frame_ids_known'] is known_ids
        assert info['used_ids'] == (original.frame_ids if known_ids else [])
        assert info['used_count'] == (10 if known_ids else None)
        assert info['selected_observations']['used_count'] == 8
        assert len(actual.per_frame_errors) == 10
        assert actual is not original
        assert not np.shares_memory(actual.camera_matrix, original.camera_matrix)


def test_joint_refinement_records_recomputed_ids_and_preserves_caller(worker_data):
    previous = deepcopy(worker_data[3])
    success, failure = run_worker(worker_data, cv2.CALIB_USE_INTRINSIC_GUESS)
    assert not failure and len(success) == 1
    pair = success[0]
    for side, intr in (('left', pair.left_intrinsics), ('right', pair.right_intrinsics)):
        info = pair.calibration_info['selection'][side]
        assert info['source'] == 'refined'
        assert info['initial_source'] == 'provided_intrinsics'
        assert info['initial_frame_ids'] == worker_data[3][0].frame_ids
        assert info['used_ids'] == intr.frame_ids == worker_data[0].frame_ids[:8]
        assert info['used_count'] == len(intr.per_frame_errors) == 8
        assert np.isfinite(intr.rms_error)
    for original, old in zip(worker_data[3], previous):
        np.testing.assert_array_equal(original.camera_matrix, old.camera_matrix)
        assert original.to_dict() == old.to_dict()


def test_joint_failure_leaves_caller_intrinsics_intact(worker_data, monkeypatch):
    previous = deepcopy(worker_data[3])

    def fail(*args):
        raise ValueError('injected recompute failure')

    monkeypatch.setattr(IntrinsicCalibrator, 'recompute_errors', fail)
    success, failure = run_worker(worker_data, cv2.CALIB_USE_INTRINSIC_GUESS)
    assert not success and len(failure) == 1 and 'injected recompute failure' in failure[0]
    for original, old in zip(worker_data[3], previous):
        assert original.to_dict() == old.to_dict()
        for field in ('rvecs', 'tvecs'):
            np.testing.assert_array_equal(getattr(original, field), getattr(old, field))
