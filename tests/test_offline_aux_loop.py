"""Offline reviews, fixed-RGB refinement and result recovery contracts."""

from copy import deepcopy
import base64
import hashlib
import json

import cv2
import numpy as np
import pytest

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.calibration import joint_aux
from multicamera.calibration.models import (AuxCameraCalibration, CameraIntrinsics, CameraModel,
    MultiCameraRig, StereoCalibResult, StereoPairCalibration)
from multicamera.calibration.offline_aux import (confirm_offline_pair, offline_pair_review,
    reordered_detection, validate_session_rig, verify_review_sources)
from multicamera.calibration.planar import PatternBoardConfig, PlanarDetectionResult, PlanarPatternDetector
from multicamera.io.session import CalibrationSession
from multicamera.streaming.stream_manager import CameraConfig, StereoPairConfig
from multicamera.ui.widgets.aux_offline_review import AuxOfflineReviewDialog


BOARD = PatternBoardConfig.ir_chessboard()


def synthetic_pairs(model=CameraModel.PINHOLE):
    rgb = CameraIntrinsics(np.array([[740., 0., 640.], [0., 745., 360.], [0., 0., 1.]]),
                           np.zeros(5), (1280, 720))
    aux = CameraIntrinsics(np.array([[180., 0., 128.], [0., 182., 96.], [0., 0., 1.]]),
                           np.zeros(4 if model == CameraModel.FISHEYE else 5), (256, 192), model)
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(np.array([.01, -.02, .005]))[0]
    transform[:3, 3] = [.025, .001, .004]
    objects = PlanarPatternDetector._make_object_points(BOARD).astype(float)
    records = []
    for i in range(9):
        r = cv2.Rodrigues(np.array([-.3 + i*.07, -.2 + (i % 3)*.2, -.05+i*.01]))[0]
        xyz = objects @ r.T + [-.25+(i % 3)*.04, -.15+(i % 2)*.05, .9+(i % 4)*.15]
        aux_xyz = (xyz - transform[:3, 3]) @ transform[:3, :3]
        records.append((f'{i:04d}',
            PlanarDetectionResult(True, objects.copy(), joint_aux.project(xyz, rgb), rgb.image_size,
                                  BOARD.pattern_type),
            PlanarDetectionResult(True, objects.copy(), joint_aux.project(aux_xyz, aux), aux.image_size,
                                  BOARD.pattern_type)))
    return rgb, aux, transform, records


@pytest.mark.parametrize('model,key', [(CameraModel.PINHOLE, 'pinhole_constrained'),
                                      (CameraModel.FISHEYE, 'fisheye_2')])
def test_joint_synthetic_fixed_rgb_and_disjoint_holdouts(model, key):
    rgb, _, expected, records = synthetic_pairs(model)
    before = deepcopy(rgb.to_dict())
    intr, result = joint_aux.calibrate_offline_aux(records, BOARD, rgb, key)
    np.testing.assert_allclose(result.T_rgb_left_aux, expected, atol=2e-4)
    assert rgb.to_dict() == before
    assert intr.model == model
    report = result.calibration_info
    assert report['status'] == 'offline_candidate_pending_physical_validation'
    assert report['metric'] == 'cross_camera_transfer_native_pixels'
    assert report['heldout']['aux_transfer_rms_px'] < .01
    assert report['heldout']['rgb_transfer_rms_px'] < .03
    heldout = []
    for fold in report['folds']:
        assert not set(fold['train_ids']) & set(fold['heldout_ids'])
        assert fold['fit']['health']['status'] == 'ok'
        heldout.extend(fold['heldout_ids'])
    assert sorted(heldout) == [row[0] for row in records]


def test_joint_does_not_accept_small_dataset_or_unknown_model():
    rgb, _, _, records = synthetic_pairs()
    with pytest.raises(ValueError, match='至少需要'):
        joint_aux.calibrate_offline_aux(records[:5], BOARD, rgb, 'pinhole')
    with pytest.raises(ValueError, match='独立'):
        joint_aux.calibrate_offline_aux(records, BOARD, rgb, 'automatic_unknown')


def test_joint_health_failure_cannot_be_masked_by_low_training_error(monkeypatch):
    rgb, _, _, records = synthetic_pairs()
    checks = iter([{'status': 'ok'}, {'status': 'failed', 'reasons': ['synthetic invalid edge']}])
    monkeypatch.setattr(joint_aux, 'check_intrinsics', lambda intr: next(checks))
    with pytest.raises(ValueError, match='全视场有效性'):
        joint_aux.calibrate_offline_aux(records, BOARD, rgb, 'pinhole_constrained')


def test_cross_transfer_detects_bad_transform_even_with_good_single_sensor_pnp():
    rgb, aux, transform, records = synthetic_pairs()
    correct = joint_aux.transfer_metrics(records, rgb, aux, transform)
    wrong = transform.copy()
    wrong[0, 3] += .05
    broken = joint_aux.transfer_metrics(records, rgb, aux, wrong)
    assert correct['aux_transfer_rms_px'] < 1e-5
    assert broken['aux_transfer_rms_px'] > 5


def test_low_training_rms_cannot_bypass_heldout_cross_camera_rejection(monkeypatch):
    rgb, aux, transform, records = synthetic_pairs()
    wrong = transform.copy()
    wrong[0, 3] += .05
    monkeypatch.setattr(joint_aux, '_fit', lambda *args: (aux, wrong,
        {'joint_training_rgb_rms_px': 0., 'joint_training_aux_rms_px': 0.}))
    with pytest.raises(ValueError, match='离线留出检查失败'):
        joint_aux.calibrate_offline_aux(records, BOARD, rgb, 'pinhole_constrained')


def test_fold_initialization_receives_training_only_and_full_fit_runs_last(monkeypatch):
    rgb, aux, transform, records = synthetic_pairs()
    calls = []
    def fit(training, *args):
        calls.append([row[0] for row in training])
        return aux, transform, {'converged': True}
    monkeypatch.setattr(joint_aux, '_fit', fit)
    _, result = joint_aux.calibrate_offline_aux(records, BOARD, rgb, 'pinhole_constrained')
    assert len(calls) == 4
    for fold, call in zip(result.calibration_info['folds'], calls):
        assert call == fold['train_ids']
        assert not set(call) & set(fold['heldout_ids'])
    assert calls[-1] == [row[0] for row in records]


def board_image():
    image = np.full((192, 256, 3), 255, np.uint8)
    for r in range(5):
        for c in range(7):
            if (r+c) % 2 == 0:
                image[40+r*20:60+r*20, 40+c*20:60+c*20] = 0
    return image


def review_fixture(tmp_path):
    image = board_image()
    detector = PlanarPatternDetector(BOARD)
    det = detector.detect(image)
    paths = [tmp_path / 'rgb_left_0000.png', tmp_path / 'aux_0000.png']
    for path in paths:
        assert cv2.imwrite(str(path), image)
    prior = dict(state='待复核', reasons=['历史稳定性未通过'], stability={'verified': False})
    review = offline_pair_review('0000', (image, image), (det, det), BOARD,
                                 paths=paths, capture_review=prior)
    return image, det, review, prior


def test_explicit_review_preserves_capture_evidence_and_detects_source_changes(tmp_path):
    _, _, review, prior = review_fixture(tmp_path)
    before = deepcopy(review)
    confirmed = confirm_offline_pair(review, 'identity', 'reverse_both')
    assert review == before and prior['stability']['verified'] is False
    assert confirmed['capture_review'] == prior
    assert confirmed['temporal_stability'] == 'not_verified_offline'
    assert confirmed['synchronization'] == 'not_verified_offline'
    assert verify_review_sources([confirmed])
    with pytest.raises(ValueError, match='未逐对确认'):
        verify_review_sources([review])
    path = tmp_path / 'aux_0000.png'
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='原图已变化'):
        verify_review_sources([confirmed])


@pytest.mark.parametrize('changed', [False, True])
def test_offline_import_hashes_the_bytes_actually_decoded(app, tmp_path, monkeypatch, changed):
    from multicamera.ui.threads.import_worker import AuxImageImportWorker
    from multicamera.calibration.planar import PlanarPatternDetector
    image, _, _, _ = review_fixture(tmp_path)
    monkeypatch.setattr(cv2, 'imread', lambda *args: pytest.fail('Offline input must decode hashed bytes'))
    if changed:
        original = PlanarPatternDetector.detect_offline
        def detect(self, frame, **kwargs):
            result = original(self, frame, **kwargs)
            path = tmp_path / 'aux_0000.png'
            path.write_bytes(path.read_bytes() + b'external-change')
            return result
        monkeypatch.setattr(PlanarPatternDetector, 'detect_offline', detect)
    worker = AuxImageImportWorker(str(tmp_path), BOARD, offline_recheck=True)
    results, errors = [], []
    worker.finished_ok.connect(results.append)
    worker.failed.connect(errors.append)
    worker.run()
    if changed:
        assert errors and '原图已变化' in errors[0] and not results
    else:
        assert results and not errors
        np.testing.assert_array_equal(results[0][0][1], image)
        assert worker.offline_reviews['0000']['sources'][0]['sha256'] == hashlib.sha256(
            (tmp_path / 'rgb_left_0000.png').read_bytes()).hexdigest()


@pytest.mark.parametrize('order', ['identity', 'reverse_rows', 'reverse_cols', 'reverse_both'])
def test_corner_reordering_is_explicit_and_does_not_move_board_object_frame(order):
    rgb, _, _, records = synthetic_pairs()
    det = records[0][1]
    before = det.image_points.copy()
    ordered = reordered_detection(det, BOARD, order)
    np.testing.assert_array_equal(det.image_points, before)
    np.testing.assert_array_equal(ordered.object_points, det.object_points)
    restored = reordered_detection(ordered, BOARD, order)
    np.testing.assert_array_equal(restored.image_points, before)
    assert rgb.image_size == ordered.image_size


def test_resize_mapping_obeys_pixel_centres_and_roi_offset():
    points = np.array([[[.5, 2.5]], [[10.5, 20.5]]])
    result = PlanarPatternDetector.map_scaled_points(points, 2, (20, 30))
    np.testing.assert_allclose(result, [[20., 31.], [25., 40.]])


def test_low_resolution_subpixel_window_stays_within_cell(monkeypatch):
    image = board_image()
    original = cv2.cornerSubPix
    windows = []
    def subpix(gray, corners, window, *args):
        windows.append(window)
        return original(gray, corners, window, *args)
    monkeypatch.setattr(cv2, 'cornerSubPix', subpix)
    assert PlanarPatternDetector(BOARD).detect(image, adaptive_subpixel=True).valid
    assert windows and max(windows[0]) <= 5
    PlanarPatternDetector(BOARD).detect(image)
    assert windows[-1] == (11, 11)  # Online/default path remains unchanged.


def test_native_sb_is_used_before_any_ir_enlargement(monkeypatch):
    detector = PlanarPatternDetector(BOARD)
    shapes = []
    original = detector._detect_sb
    def detect(gray, scale, roi, invert):
        shapes.append((gray.shape, scale, roi))
        return original(gray, scale, roi, invert)
    monkeypatch.setattr(detector, '_detect_sb', detect)
    result = detector.detect_offline(board_image(), modality='ir')
    assert result.valid and result.image_size == (256, 192)
    assert shapes[0] == ((192, 256), 1, None)
    assert result.detection_info['coordinate_space'] == 'original_pixels'


def test_dialog_requires_each_pair_confirmation_and_invalidates_changed_order(app, tmp_path):
    image, det, review, _ = review_fixture(tmp_path)
    dialog = AuxOfflineReviewDialog([('0000', image, image, det, det)], {'0000': review}, BOARD)
    assert not dialog.decisions
    dialog.confirm.click()
    assert dialog.decisions['0000']['physical_corner_order'] == 'manually_confirmed_per_pair'
    dialog.orders[1].setCurrentIndex(3)
    assert not dialog.decisions
    dialog.confirm.click()
    assert dialog.decisions['0000']['corner_orders']['aux'] == 'reverse_both'
    assert not dialog.binding.isChecked()
    restored = AuxOfflineReviewDialog([('0000', image, image, det, det)], {'0000': review},
        BOARD, decisions=dialog.decisions)
    assert restored.orders[1].currentData() == 'reverse_both'
    assert '已确认纳入' in restored.list.item(0).text()
    restored.resize(760, 600)
    restored.show()
    app.processEvents()
    for view in restored.views:
        assert view.pixmap().width() <= view.width()
        assert view.pixmap().height() <= view.height()
    restored.close()
    dialog.close()


def make_session(tmp_path):
    configs = [StereoPairConfig('rgb', CameraConfig('left', ''), CameraConfig('right', ''))]
    session = CalibrationSession.create(tmp_path, CharucoBoard(CharucoBoardConfig()), configs)
    rgb, _, _, _ = synthetic_pairs()
    stereo = StereoCalibResult(np.eye(3), np.array([[-.05], [0.], [0.]]), np.eye(3), np.eye(3), .2)
    rig = MultiCameraRig(pairs={'rgb': StereoPairCalibration('rgb', 'left', 'right', rgb, rgb, stereo)},
                          reference_camera='left')
    session.save_calibration_results(rig)
    return session, configs, rig


@pytest.mark.parametrize('kind', ['missing', 'ids', 'lr', 'model', 'resolution', 'health'])
def test_restore_rejects_missing_or_incompatible_rgb_before_ui_mutation(tmp_path, kind):
    session, configs, rig = make_session(tmp_path)
    if kind == 'missing':
        session = CalibrationSession(tmp_path / 'empty')
    elif kind == 'ids':
        configs[0].left.camera_id = 'other'
    elif kind == 'lr':
        configs[0].left, configs[0].right = configs[0].right, configs[0].left
    elif kind == 'model':
        session._metadata['camera_model'] = 'fisheye'
    else:
        rig.pairs['rgb'].right_intrinsics = deepcopy(rig.pairs['rgb'].right_intrinsics)
        if kind == 'resolution':
            rig.pairs['rgb'].right_intrinsics.image_size = (640, 480)
        else:
            rig.pairs['rgb'].right_intrinsics.camera_matrix[0, 0] = -1
        session.save_rig(rig, directory=session.result_directory)
    with pytest.raises(ValueError):
        validate_session_rig(session, configs)


def test_json_opencv_roundtrip_carries_candidate_state_models_transform_and_lineage(tmp_path):
    session, configs, rig = make_session(tmp_path)
    assert validate_session_rig(session, configs).pairs['rgb'].left_id == 'left'
    _, aux, transform, _ = synthetic_pairs(CameraModel.FISHEYE)
    info = {'status': 'offline_candidate_pending_physical_validation',
            'heldout': {'rgb_transfer_rms_px': .4, 'aux_transfer_rms_px': .3},
            'sources': [{'sha256': hashlib.sha256(b'source').hexdigest()}],
            'large_unicode_path': 'C:\\标定数据\\原图目录\\"quoted"\\image.png' * 1000}
    rig.aux_cameras['ir'] = AuxCameraCalibration('ir', 'ir', aux, transform,
        rms_error=.3, frame_count=9, board_config=BOARD.to_dict(), calibration_info=info)
    saved, _ = session.save_calibration_results(rig)
    loaded = session.load_rig().aux_cameras['ir']
    assert loaded.calibration_info == info and loaded.intrinsics.model == CameraModel.FISHEYE
    np.testing.assert_array_equal(loaded.T_rgb_left_aux, transform)
    yaml = next(path for path in saved if path.name == 'rig_opencv.yml')
    fs = cv2.FileStorage(str(yaml), cv2.FILE_STORAGE_READ)
    try:
        assert fs.getNode('aux_ir_calibration_status').string() == info['status']
        chunks = fs.getNode('aux_ir_calibration_info_json_base64_chunks')
        encoded = ''.join(chunks.at(i).string() for i in range(chunks.size()))
        assert json.loads(base64.b64decode(encoded)) == info
        np.testing.assert_array_equal(fs.getNode('aux_ir_T_rgb_left_aux').mat(), transform)
        assert fs.getNode('camera_ir_model').string() == 'fisheye'
    finally:
        fs.release()


def test_real_main_window_restores_rgb_and_missing_result_preserves_state(app, tmp_path, monkeypatch):
    from multicamera.ui import main_window
    session, configs, _ = make_session(tmp_path / 'source')
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'sessions')
    win = main_window.MainWindow()
    manager, panel = win._stream_manager, win._calib_panel
    for name in list(manager.stereo_pairs):
        manager.remove_stereo_pair(name)
    manager.add_virtual_stereo_pair(configs[0])
    panel.refresh_pairs()
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in session.session_dir.rglob('*') if path.is_file()}
    monkeypatch.setattr(main_window.QFileDialog, 'getExistingDirectory',
                         lambda *a: str(session.session_dir))
    win._on_load_session()
    panel = win._calib_panel
    assert panel._intrinsics['left'].image_size == (1280, 720)
    assert 'rgb' in panel._pair_calibs and win._rig is panel.rig
    assert win._act_export.isEnabled()
    assert hashes == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in hashes}
    previous, rig = win._current_session, win._rig
    empty = CalibrationSession.create(tmp_path / 'empty', win._board, configs)
    monkeypatch.setattr(main_window.QFileDialog, 'getExistingDirectory', lambda *a: str(empty.session_dir))
    win._on_load_session()
    assert win._current_session is previous and win._rig is rig
    assert '已保留当前会话' in win.statusBar().currentMessage()
    win.close()


def test_multimodal_offline_preview_survives_workflow_refresh_without_live_url(app):
    from multicamera.streaming.stream_manager import AuxiliaryCameraConfig, StreamManager
    from multicamera.ui.widgets.stream_view import MultiStreamView
    manager = StreamManager()
    manager.add_virtual_stereo_pair(StereoPairConfig('rgb', CameraConfig('left', ''),
                                                    CameraConfig('right', '')))
    manager.set_auxiliary_camera(AuxiliaryCameraConfig('ir', ''))
    view = MultiStreamView(manager, CharucoBoard(CharucoBoardConfig()))
    view.set_multimodal_mode(True, 'left')
    image = board_image()
    view.show_multimodal_images({'left': image, 'ir': image})
    assert view._views['ir']._last_frame is not None
    view.set_multimodal_mode(True, 'left')
    assert view._views['ir']._last_frame is not None
    assert '未配置' not in view._views['ir']._info.text()
    view.close()
