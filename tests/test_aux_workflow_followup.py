"""Auxiliary workflow state, source-photo persistence and result lifetime contracts."""

import json
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.calibration.models import (
    AuxCameraCalibration, CameraIntrinsics, MultiCameraRig, StereoCalibResult,
    StereoPairCalibration,
)
from multicamera.calibration.planar import PatternBoardConfig, PlanarPatternDetector
from multicamera.io.session import CalibrationSession
from multicamera.streaming.stream_manager import (
    AuxiliaryCameraConfig, CameraConfig, FrameEntry, StereoPairConfig, StreamManager,
)
from multicamera.ui.configuration import ConfigurationImpact
from multicamera.ui.threads.import_worker import AuxImageImportWorker
from multicamera.ui.widgets.calib_panel import CalibrationPanel


def board_image():
    image = np.full((192, 256, 3), 255, np.uint8)
    for row in range(5):
        for col in range(7):
            if (row + col) % 2 == 0:
                image[40 + row * 20:60 + row * 20, 40 + col * 20:60 + col * 20] = 0
    return image


@pytest.fixture
def panel(app, tmp_path, monkeypatch, finish_aux_tasks):
    manager = StreamManager()
    manager.add_virtual_stereo_pair(StereoPairConfig(
        'rgb', CameraConfig('left', ''), CameraConfig('right', ''),
    ))
    manager.set_auxiliary_camera(AuxiliaryCameraConfig('ir', 'http://device.test/ir'))
    board = CharucoBoard(CharucoBoardConfig())
    value = CalibrationPanel(manager, board)
    value.set_session(CalibrationSession.create(
        tmp_path, board, list(manager.stereo_pairs.values()), manager.auxiliary_camera,
    ))
    value.show_workflow('capture', True)
    monkeypatch.setattr(value, '_prepare_white_balance', lambda *args: True)
    yield value
    finish_aux_tasks(value)
    value._aux_readiness_timer.stop()
    manager.stop_all()
    value.close()


def import_pairs(panel, count=3, *, invalid_last=False, folder=None):
    image = board_image()
    detector = PlanarPatternDetector(panel._aux_board_config())
    detection = detector.detect(image)
    assert detection.valid and detection.num_points == 24
    rows = [(f'{i:04d}', image.copy(), image.copy(), detection, detection) for i in range(count)]
    if invalid_last:
        blank = np.zeros_like(image)
        rows[-1] = (rows[-1][0], image, blank, detection, detector.detect(blank))
    panel._aux_import_worker = SimpleNamespace(
        board_config=panel._aux_board_config(), folder=folder,
    )
    panel._on_import_aux_images_ready(rows)
    return rows


def add_result(panel, tmp_path):
    intr = CameraIntrinsics(np.array([[200., 0., 128.], [0., 200., 96.], [0., 0., 1.]]),
                            np.zeros((1, 5)), (256, 192), rms_error=.2)
    stereo = StereoCalibResult(np.eye(3), np.array([[-.1], [0.], [0.]]), np.eye(3), np.eye(3))
    pair = StereoPairCalibration('rgb', 'left', 'right', intr, intr, stereo)
    panel._intrinsics.update(left=intr, right=intr)
    panel._pair_calibs['rgb'] = pair
    panel._aux_intrinsics = intr
    rig = MultiCameraRig(pairs={'rgb': pair}, reference_camera='left', aux_cameras={
        'ir': AuxCameraCalibration('ir', 'ir', intr, np.eye(4), .2, 'chessboard', 3),
    })
    panel.set_result_model(rig)
    saved, errors = panel._session.save_calibration_results(rig)
    panel.set_result_storage(saved[0].parent, saved, errors)
    return rig, intr, saved


def live_frames(panel, monkeypatch, image=None):
    image = board_image() if image is None else image
    monkeypatch.setattr(panel, '_aux_live_frames', lambda: ((image, image, .01), 'ready'))
    monkeypatch.setattr(panel._aux_guide, 'stability_reasons', lambda detections, snapshot=None: [])
    return image


def test_replacement_import_does_not_claim_guided_poses(panel):
    panel._aux_capture_sequence.advance()
    import_pairs(panel, 8)
    assert panel._aux_capture_sequence.done_count == 0
    assert panel._aux_capture_sequence.current_index == 0
    assert len(panel._aux_pairs) == 8
    import_pairs(panel, 3, invalid_last=True)
    assert len(panel._aux_pairs) == 2
    assert len(panel._aux_images) == 3
    assert panel._aux_capture_sequence.done_count == 0
    assert '引导采用' in panel._aux_cap_progress.format()
    assert '验收通过 2 组' in panel._aux_guide.counts.text()
    assert '不代表姿态覆盖或标定精度' in panel._lbl_aux_pair_count.toolTip()


def test_real_offline_worker_preserves_source_not_session_copy(panel, tmp_path):
    folder = tmp_path / 'offline'
    folder.mkdir()
    image = board_image()
    for i in range(3):
        assert cv2.imwrite(str(folder / f'rgb_left_{i:04d}.png'), image)
        assert cv2.imwrite(str(folder / f'aux_{i:04d}.png'), image)
    originals = {p.name: p.read_bytes() for p in folder.iterdir()}
    worker = AuxImageImportWorker(str(folder), PatternBoardConfig.ir_chessboard())
    panel._aux_import_worker = worker
    worker.finished_ok.connect(panel._on_import_aux_images_ready)
    failures = []
    worker.failed.connect(failures.append)
    worker.run()
    assert not failures
    assert len(panel._aux_pairs) == 3
    assert '离线 3 组保留在源目录' in panel._lbl_raw_save.text()
    assert str(folder) in panel._lbl_raw_save.toolTip()
    assert panel._session.metadata.get('multimodal_frame_count', 0) == 0
    assert not list((panel._session.session_dir / 'multimodal').glob('*.png'))
    assert originals == {p.name: p.read_bytes() for p in folder.iterdir()}


def test_import_without_session_never_claims_missing_source_photos(panel, tmp_path):
    panel._session = None
    import_pairs(panel, folder=str(tmp_path))
    assert '离线 3 组保留在源目录' in panel._lbl_raw_save.text()
    assert '未落盘' not in panel._lbl_raw_save.text()
    assert panel._session is None


def test_capture_before_rgb_intrinsics_saves_raw_but_cannot_calibrate(panel, monkeypatch, finish_aux_tasks):
    image = live_frames(panel, monkeypatch)
    panel._on_capture_aux()
    finish_aux_tasks(panel)
    assert len(panel._aux_pairs) == 1
    assert panel._aux_saved_count == 1
    assert '实时原图已落盘 1 组' in panel._lbl_raw_save.text()
    assert '请先完成 RGB 左目' in panel._aux_calibration_problem()
    assert not panel._btn_calib_aux.isEnabled()
    session = CalibrationSession.load(panel._session.session_dir)
    assert session.metadata['multimodal_frame_count'] == 1
    for prefix in ('rgb_left', 'aux'):
        path = session.session_dir / session.MULTIMODAL_DIR / f'{prefix}_0000.png'
        np.testing.assert_array_equal(cv2.imread(str(path)), image)
    assert '24' in panel._lbl_aux_board.text()
    assert panel._aux_board_config() == PatternBoardConfig.ir_chessboard()


def test_failed_capture_does_not_claim_saved_or_invalidate_result(panel, tmp_path, monkeypatch, finish_aux_tasks):
    rig, _, saved = add_result(panel, tmp_path)
    live_frames(panel, monkeypatch)

    def fail(*args):
        raise OSError('disk full')

    monkeypatch.setattr(panel._session, 'save_multimodal_pair', fail)
    panel._on_capture_aux()
    finish_aux_tasks(panel)
    assert panel._aux_saved_count == 0
    assert not panel._aux_pairs
    assert panel.rig is rig and 'ir' in rig.aux_cameras
    assert panel.result_directory == saved[0].parent
    assert panel._session.metadata.get('multimodal_frame_count', 0) == 0


@pytest.mark.parametrize('operation', ['capture', 'import', 'configuration'])
def test_aux_updates_keep_rgb_and_historical_directory(panel, tmp_path, monkeypatch, operation, finish_aux_tasks):
    rig, intr, saved = add_result(panel, tmp_path)
    previous = saved[0].parent
    originals = {p: p.read_bytes() for p in saved}
    if operation == 'capture':
        live_frames(panel, monkeypatch)
        panel._on_capture_aux()
        finish_aux_tasks(panel)
    elif operation == 'import':
        import_pairs(panel)
    else:
        panel._sm.set_auxiliary_camera(AuxiliaryCameraConfig('renamed_ir', ''))
        panel.apply_configuration(panel._board, panel._camera_model, ConfigurationImpact(
            False, frozenset(), frozenset(), True, True,
        ))
    assert not rig.aux_cameras
    assert panel._aux_intrinsics is None
    assert panel._intrinsics['left'] is intr
    assert 'rgb' in panel._pair_calibs
    assert panel._results_outdated
    assert panel._previous_result_directory == previous
    assert panel.result_directory is None
    assert str(previous) in panel._lbl_result_storage.text()
    assert '旧结果已过期' in panel._lbl_result_storage.text()
    assert originals == {p: p.read_bytes() for p in saved}
    # The published rig is shared with the main window: RGB remains exportable, AUX is absent.
    exported = panel._session.save_rig(panel.rig)
    assert not json.loads(exported.read_text())['aux_cameras']


def test_offline_then_live_counts_only_live_files_as_saved(panel, tmp_path, monkeypatch, finish_aux_tasks):
    import_pairs(panel, folder=str(tmp_path / 'source'))
    live_frames(panel, monkeypatch)
    panel._on_capture_aux()
    finish_aux_tasks(panel)
    assert len(panel._aux_pairs) == 4
    assert '离线 3 组' in panel._lbl_raw_save.text()
    assert '实时原图已落盘 1 组' in panel._lbl_raw_save.text()
    assert panel._session.metadata['multimodal_frame_count'] == 1
    assert panel._aux_capture_sequence.done_count == 0  # Counts do not certify guided poses.
    import_pairs(panel, 2)
    assert panel._aux_saved_count == 0
    assert '实时原图已落盘' not in panel._lbl_raw_save.text()
    assert panel._session.metadata['multimodal_frame_count'] == 1  # Old photos are untouched.


def test_frame_ids_remain_unique_after_import_and_supplement(panel, monkeypatch, finish_aux_tasks):
    image = live_frames(panel, monkeypatch)
    det = PlanarPatternDetector(panel._aux_board_config()).detect(image)
    panel._on_import_aux_images_ready([('live_0000', image, image, det, det)])
    panel._on_capture_aux()
    finish_aux_tasks(panel)
    assert [key for key, _, _ in panel._aux_pairs] == ['live_0000', 'live_0001']


def test_software_pairing_does_not_claim_hardware_sync(panel):
    image = board_image()
    now = time.time()
    panel._sm._latest_frames.update(left=FrameEntry(image, now), ir=FrameEntry(image, now))
    frames, status = panel._aux_live_frames()
    assert frames is not None
    assert '软件配对' in status and '非硬件同步' in status
    panel._sm._latest_frames['ir'] = FrameEntry(image, now - 5)
    assert panel._aux_live_frames()[0] is None


def test_aux_result_finish_republishes_with_unchanged_transform_direction(panel, tmp_path):
    _, intr, _ = add_result(panel, tmp_path)
    import_pairs(panel)
    transform = np.eye(4)
    transform[:3, 3] = [.04, -.01, .02]
    board = panel._aux_board_config()
    panel._pending_aux_context = ('ir', 'ir', board, 'left')
    published = []
    panel.calibration_finished.connect(published.append)
    panel._on_aux_calibration_finished(intr, SimpleNamespace(
        T_rgb_left_aux=transform, rms_error=.3, frame_count=3, per_frame_errors=[.3] * 3,
    ))
    assert len(published) == 1
    np.testing.assert_array_equal(published[0].aux_cameras['ir'].T_rgb_left_aux, transform)
    saved, _ = panel._session.save_calibration_results(published[0])
    exported = json.loads(saved[0].read_text())
    assert exported['aux_cameras']['ir']['board_config'] == board.to_dict()
    assert CalibrationSession.load(panel._session.session_dir).metadata['multimodal_result']['valid_pairs'] == 3
