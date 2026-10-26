"""Failed persistence must preserve previous files and successful counters."""

import hashlib
import json

import cv2
import numpy as np
import pytest

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.calibration.models import CameraIntrinsics, MultiCameraRig, StereoCalibResult, StereoPairCalibration
from multicamera.io.session import CalibrationSession


@pytest.fixture
def session(tmp_path):
    session = CalibrationSession(tmp_path / 'session')
    session._metadata = {'frame_count': 0, 'multimodal_frame_count': 0}
    session._save_metadata()
    return session


@pytest.fixture
def rig():
    intr = CameraIntrinsics(np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]]),
                            np.zeros(5), (640, 480), rms_error=.2)
    stereo = StereoCalibResult(np.eye(3), np.array([[-.1], [0.], [0.]]), np.eye(3), np.eye(3), rms_error=.2)
    return MultiCameraRig(pairs={'rgb': StereoPairCalibration('rgb', 'left', 'right', intr, intr, stereo)},
                          reference_camera='left')


@pytest.mark.parametrize('kind', ['stereo', 'multimodal'])
@pytest.mark.parametrize('failure', ['false', 'exception', 'metadata'])
def test_capture_failure_does_not_advance_or_leave_half_pair(session, monkeypatch, kind, failure):
    previous = (session.session_dir / 'session.json').read_bytes()
    image = np.zeros((16, 24, 3), np.uint8)
    original = cv2.imwrite
    calls = []

    def write(path, img):
        calls.append(path)
        if len(calls) == 2:
            if failure == 'false':
                return False
            if failure == 'exception':
                raise OSError('simulated image write error')
        return original(path, img)

    monkeypatch.setattr(cv2, 'imwrite', write)
    if failure == 'metadata':
        def fail():
            raise PermissionError('session metadata is read-only')
        monkeypatch.setattr(session, '_save_metadata', fail)
    with pytest.raises(OSError):
        if kind == 'stereo':
            session.save_frame_pair('rgb', image, image)
        else:
            session.save_multimodal_pair(image, image)
    assert session.frame_count == session.metadata['multimodal_frame_count'] == 0
    assert not list(session.session_dir.rglob('*.png'))
    assert (session.session_dir / 'session.json').read_bytes() == previous


def test_existing_capture_never_overwritten(session):
    image = np.zeros((16, 24, 3), np.uint8)
    session.save_frame_pair('rgb', image, image, frame_idx=0)
    path = session.session_dir / 'images/rgb/left_0000.png'
    previous = path.read_bytes()
    with pytest.raises(FileExistsError):
        session.save_frame_pair('rgb', image + 255, image + 255, frame_idx=0)
    assert path.read_bytes() == previous and session.frame_count == 1


def test_metadata_write_error_preserves_previous_json(session, monkeypatch):
    from pathlib import Path

    path = session.session_dir / 'session.json'
    previous = path.read_bytes()

    def partial(path, *args, **kwargs):
        raise OSError('disk full')

    monkeypatch.setattr(Path, 'write_text', partial)
    session._metadata['frame_count'] = 7
    with pytest.raises(OSError):
        session._save_metadata()
    assert path.read_bytes() == previous


def test_same_second_sessions_get_unique_directories(tmp_path):
    board = CharucoBoard(CharucoBoardConfig())
    sessions = [CalibrationSession.create(tmp_path, board, [], name='calib') for _ in range(4)]
    assert len({session.session_dir for session in sessions}) == 4
    assert all(session.metadata['name'] == session.session_dir.name for session in sessions)


def test_initial_session_persists_selected_camera_model(tmp_path):
    session = CalibrationSession.create(tmp_path, CharucoBoard(CharucoBoardConfig()), [],
                                        camera_model='fisheye')
    assert CalibrationSession.load(session.session_dir).metadata['camera_model'] == 'fisheye'


@pytest.mark.parametrize('data', [None, [], {'frame_count': -1}, {'frame_count': True}, {'frame_count': 1.5}])
def test_load_rejects_missing_or_invalid_session_before_creating_subfolders(tmp_path, data):
    directory = tmp_path / 'selected'
    directory.mkdir()
    if data is not None:
        (directory / 'session.json').write_text(json.dumps(data))
    with pytest.raises(ValueError):
        CalibrationSession.load(directory)
    assert not (directory / 'results').exists()


def test_generation_manifest_distinguishes_historical_exports(session, rig):
    saved, errors = session.save_calibration_results(rig)
    assert not errors
    first = json.loads((saved[-1]).read_text())
    pair = rig.pairs['rgb']
    rig.pairs['second'] = StereoPairCalibration('second', 'L2', 'R2', pair.left_intrinsics,
                                               pair.right_intrinsics, pair.stereo)
    saved, errors = session.save_calibration_results(rig)
    assert any('rig_kalibr.yaml' in error for error in errors)
    manifest = json.loads(saved[-1].read_text())
    assert manifest['generation_id'] != first['generation_id']
    assert manifest['pair_names'] == ['rgb', 'second']
    assert 'rig_opencv_standard.yml' not in {record['name'] for record in manifest['files']}
    assert (session.session_dir / 'results/rig_opencv_standard.yml').is_file()
    assert 'rig_kalibr.yaml' not in {record['name'] for record in manifest['files']}
    for record in manifest['files']:
        path = session.session_dir / 'results' / record['name']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']


def test_rig_json_refuses_nan_without_replacing_previous(session, rig):
    path = session.save_rig(rig)
    previous = path.read_bytes()
    rig.pairs['rgb'].stereo.rms_error = float('nan')
    with pytest.raises(ValueError):
        session.save_rig(rig)
    assert path.read_bytes() == previous
