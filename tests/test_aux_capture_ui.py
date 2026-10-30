"""Raw quality, stable guidance layout and completion contracts in the real Qt panel."""

import json
import time
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QThreadPool

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.calibration.aux_guidance import AuxGuidanceAnalysis, AuxTarget
from multicamera.calibration.models import CameraIntrinsics
from multicamera.calibration.planar import PlanarPatternDetector
from multicamera.io.session import CalibrationSession
from multicamera.streaming.stream_manager import (
    AuxiliaryCameraConfig, CameraConfig, FrameEntry, StereoPairConfig, StreamManager,
)
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
    manager.add_virtual_stereo_pair(StereoPairConfig('rgb', CameraConfig('left', ''),
                                                    CameraConfig('right', '')))
    manager.set_auxiliary_camera(AuxiliaryCameraConfig('ir', 'http://device.test/ir'))
    board = CharucoBoard(CharucoBoardConfig())
    panel = CalibrationPanel(manager, board)
    panel.set_session(CalibrationSession.create(tmp_path, board, list(manager.stereo_pairs.values()),
                                                manager.auxiliary_camera))
    panel.show_workflow('capture', True)
    monkeypatch.setattr(panel, '_prepare_white_balance', lambda *args: True)
    # Isolate raw/stability contracts from the optional periodic preview sampler.
    monkeypatch.setattr(panel._aux_guide, '_request_preview', lambda: None)
    finish_aux_tasks(panel)
    yield panel
    finish_aux_tasks(panel)
    QThreadPool.globalInstance().waitForDone(15000)
    panel._aux_readiness_timer.stop()
    manager.stop_all()
    panel.close()


def live(panel, monkeypatch, image):
    monkeypatch.setattr(panel, '_aux_live_frames', lambda: ((image, image, .01), 'software pair'))


def seed_stability(panel, image):
    det = PlanarPatternDetector(panel._aux_board_config()).detect(image)
    preview = SimpleNamespace(valid=det.valid, num_corners=det.num_points,
                             charuco_corners=det.image_points, image_size=det.image_size)
    now = time.monotonic()
    clock = panel._aux_guide._clock
    for at in (now - 1., now - .5, now):
        panel._aux_guide._clock = lambda: at
        for camera in ('left', 'ir'):
            panel._aux_guide.observe(camera, preview, image)
    panel._aux_guide._clock = clock
    return det


def test_stable_original_saved_with_quality_evidence_and_no_126_gate(panel, monkeypatch, finish_aux_tasks):
    image = board_image()
    live(panel, monkeypatch, image)
    seed_stability(panel, image)
    panel._on_capture_aux()
    assert panel._aux_capture_worker is not None  # Computation is off the UI thread.
    finish_aux_tasks(panel)
    assert len(panel._aux_pairs) == 1
    assert panel._aux_guide.analysis.unique_count == 1
    assert 'RGB 24 / IR 24' in panel._aux_guide.receipt.text()
    path = panel._session.session_dir / 'multimodal/quality_0000.json'
    evidence = json.loads(path.read_text())
    assert evidence['state'] == '可采纳'
    assert evidence['pair_delta_ms'] == 10 and evidence['pairing'] == 'software_not_hardware'
    assert evidence['metrics'][1]['thresholds']['min_grid_spacing_px'] == 4
    np.testing.assert_array_equal(cv2.imread(str(path.with_name('aux_0000.png'))), image)


def test_unstable_photo_retained_but_not_counted_and_reimport_does_not_promote(panel, monkeypatch, finish_aux_tasks):
    image = board_image()
    live(panel, monkeypatch, image)
    panel._on_capture_aux()
    finish_aux_tasks(panel)
    assert panel._aux_saved_count == 1 and not panel._aux_pairs
    assert '稳定性未通过' in panel._aux_guide.receipt.text()
    folder = panel._session.session_dir / 'multimodal'
    original = {p: p.read_bytes() for p in folder.iterdir()}
    worker = AuxImageImportWorker(str(folder), panel._aux_board_config())
    panel._aux_import_worker = worker
    worker.finished_ok.connect(panel._on_import_aux_images_ready)
    worker.run()
    finish_aux_tasks(panel)
    assert not panel._aux_pairs and len(panel._aux_reviews) == 1
    assert '原采图验收未通过' in '；'.join(panel._aux_reviews[0].quality_reasons)
    assert original == {p: p.read_bytes() for p in folder.iterdir()}


def test_low_contrast_or_blur_not_hidden_by_successful_grid_detection(panel, monkeypatch, finish_aux_tasks):
    image = np.clip(board_image().astype(float) / 255 * 3 + 100, 0, 255).astype(np.uint8)
    live(panel, monkeypatch, image)
    monkeypatch.setattr(panel._aux_guide, 'stability_reasons', lambda detections, snapshot=None: [])
    panel._on_capture_aux()
    finish_aux_tasks(panel)
    assert not panel._aux_pairs
    assert panel._aux_quality_records[0]['state'] == '待复核'
    assert panel._aux_saved_count == 1
    assert panel._aux_guide.analysis.unique_count == 0


def test_duplicate_raw_photos_do_not_advance_unique_progress(panel, monkeypatch, finish_aux_tasks):
    image = board_image()
    live(panel, monkeypatch, image)
    seed_stability(panel, image)
    for _ in range(2):
        panel._on_capture_aux()
        finish_aux_tasks(panel)
    assert len(panel._aux_pairs) == 2 and panel._aux_saved_count == 2
    assert panel._aux_guide.analysis.unique_count == 1
    assert panel._aux_guide.analysis.duplicate_count == 1
    assert '非重复 1' in panel._aux_guide.counts.text()
    assert panel._aux_capture_sequence.done_count == 0


def test_stale_background_capture_does_not_save_to_new_context(panel, monkeypatch, finish_aux_tasks):
    live(panel, monkeypatch, board_image())
    panel._on_capture_aux()
    panel._aux_guide._revision += 1
    finish_aux_tasks(panel)
    assert not panel._aux_images and panel._aux_saved_count == 0
    assert panel._session.metadata.get('multimodal_frame_count', 0) == 0


def test_save_failure_does_not_advance_or_clear_existing_aux_result(panel, monkeypatch, finish_aux_tasks):
    image = board_image()
    live(panel, monkeypatch, image)
    seed_stability(panel, image)
    monkeypatch.setattr(panel._session, 'save_multimodal_pair',
                        lambda *args: (_ for _ in ()).throw(OSError('disk full')))
    panel._on_capture_aux()
    finish_aux_tasks(panel)
    assert not panel._aux_images and not panel._aux_pairs
    assert panel._aux_guide.analysis.unique_count == 0
    assert '保存失败' in panel._aux_guide.receipt.text()


def test_symmetry_requires_explicit_per_dataset_correspondence_confirmation(panel, monkeypatch, finish_aux_tasks):
    image = board_image()
    det = PlanarPatternDetector(panel._aux_board_config()).detect(image)
    panel._intrinsics['left'] = CameraIntrinsics(np.array([[200., 0., 128.], [0., 200., 96.],
                                                          [0., 0., 1.]]), np.zeros(5), (256, 192))
    panel._on_import_aux_images_ready([(str(i), image, image, det, det) for i in range(3)])
    finish_aux_tasks(panel)
    assert '物理角点对应' in panel._aux_calibration_problem()
    panel._check_aux_order.setChecked(True)
    assert panel._aux_calibration_problem() == ''
    panel._on_import_aux_images_ready([('new', image, image, det, det)] * 3)
    assert not panel._check_aux_order.isChecked()


@pytest.mark.parametrize('width', [360, 480, 680])
def test_initial_repair_completion_and_receipt_keep_fixed_geometry(panel, app, width, finish_aux_tasks):
    guide = panel._aux_guide
    finish_aux_tasks(panel)
    panel.resize(width, 1000)
    guide.resize(width - 32, guide.sizeHint().height())
    guide.show()
    app.processEvents()
    before = [(label.geometry().width(), label.height()) for label in
              (guide.phase, guide.action, guide.counts, guide.receipt)]
    for phase, count, target in (('initial', 8, AuxTarget()), ('repair', 17, AuxTarget(scale='large')),
                                 ('complete', 21, None)):
        guide.analysis = AuxGuidanceAnalysis(phase, target, count, 0, 0, {}, ('large', 'small'))
        guide._key = guide.context_key()
        guide.render()
        guide.receipt.setText('本次原图已保存。' * 30)
        app.processEvents()
        assert before == [(label.geometry().width(), label.height()) for label in
                          (guide.phase, guide.action, guide.counts, guide.receipt)]
    assert '本轮采集完成' in guide.phase.text()
    assert '待验证距离' in guide.action.text()
    assert panel._aux_seq_widget._large_guide.completion_title == '本轮采集完成'
    guide.refresh()
    assert '本轮采集完成' in guide.phase.text()  # Not a disappearing toast.
    assert panel._btn_capture_aux.text()  # Manual supplement remains available.


def test_rejected_stale_stability_and_cumulative_drift_do_not_pass(panel):
    image = board_image()
    det = seed_stability(panel, image)
    assert not panel._aux_guide.stability_reasons([det, det])
    shifted = replace(det, image_points=det.image_points + [5., 0.])
    assert panel._aux_guide.stability_reasons([shifted, det])
    panel._aux_guide._clock = lambda: time.monotonic() + 5
    assert len(panel._aux_guide.stability_reasons([det, det])) == 2


@pytest.mark.parametrize('case', ['short', 'stale', 'size', 'drift', 'missing',
                                'quality_stale', 'stable'])
def test_preview_ready_and_raw_stability_share_full_acceptance(panel, case):
    from multicamera.calibration.aux_guidance import guidance_action
    from multicamera.ui.threads.aux_capture_worker import AuxCaptureWorker

    guide = panel._aux_guide
    image = board_image()
    intr = CameraIntrinsics(np.array([[200., 0., 128.], [0., 200., 96.], [0., 0., 1.]]),
                            np.zeros(5), (256, 192))
    guide.analysis = AuxGuidanceAnalysis('initial', AuxTarget(), 0, 0, 0, {}, ())
    guide._key = guide.context_key()
    guide._clock = lambda: 10.
    worker = AuxCaptureWorker((guide.context_key(), 10.), (image, image),
                              panel._aux_board_config(), (intr, intr), 'ir')
    worker.signals.completed.connect(guide._preview_ready)
    worker.run()
    assert guidance_action(panel._aux_board_config(), AuxTarget(), guide.last_qualities)[0] == 'ready'
    detections = guide._preview_detections
    for camera, det in zip(('left', 'ir'), detections):
        times = [9., 9.5, 10.] if case != 'short' else [9.5, 9.6, 10.]
        points = det.image_points.copy()
        size = det.image_size
        if camera == 'ir':
            if case == 'size':
                size = (512, 384)
            if case == 'drift':
                points += [5., 0.]
        guide._samples[camera] = [(at, points.copy(), size) for at in times]
    if case == 'missing':
        guide._preview_detections = []
    if case == 'stale':
        guide._clock = lambda: 12.
    if case == 'quality_stale':
        guide._preview_result_at = 8.
    guide.render()
    assert ('可以采集' in guide.action.text()) == (case == 'stable')
    assert bool(guide.stability_reasons(guide._preview_detections)) == (
        case not in ('stable', 'quality_stale'))


def test_aux_fitted_area_reaches_schematic_without_changing_progress(panel):
    guide = panel._aux_guide
    guide.analysis = AuxGuidanceAnalysis('initial', AuxTarget(scale='large', target_area=.20),
                                        5, 0, 0, {}, ('large',))
    guide._key = guide.context_key()
    guide.render()
    seq = panel._aux_capture_sequence
    assert seq.current.target_area == .20
    assert seq.done_count == 0 and guide.analysis.unique_count == 5


def test_repeated_sampling_of_same_source_frame_never_proves_stability(panel):
    image = board_image()
    now = time.time()
    for camera in ('left', 'ir'):
        panel._sm._latest_frames[camera] = FrameEntry(image, now)
    det = seed_stability(panel, image)
    assert all(len(history) == 1 for history in panel._aux_guide._samples.values())
    assert len(panel._aux_guide.stability_reasons([det, det])) == 2


def test_review_write_failure_preserves_raw_but_never_admits(panel, monkeypatch, finish_aux_tasks):
    image = board_image()
    live(panel, monkeypatch, image)
    seed_stability(panel, image)
    monkeypatch.setattr(panel._session, 'save_multimodal_review',
                        lambda *args: (_ for _ in ()).throw(OSError('review denied')))
    panel._on_capture_aux()
    finish_aux_tasks(panel)
    assert panel._aux_saved_count == 1 and not panel._aux_pairs
    assert panel._aux_quality_records[0]['state'] == '待复核'
    assert '验收记录写入失败' in panel._aux_guide.receipt.text()
    assert panel._session.metadata['multimodal_frame_count'] == 1
    folder = panel._session.session_dir / 'multimodal'
    pending = json.loads((folder / 'quality_0000.json').read_text())
    assert pending['state'] == '待复核'
    worker = AuxImageImportWorker(str(folder), panel._aux_board_config())
    imported = []
    worker.finished_ok.connect(imported.append)
    worker.run()
    assert imported and worker.frame_reviews['0000']['state'] == '待复核'


def test_initial_review_is_transactional_with_both_images(panel, monkeypatch):
    image = board_image()
    original = panel._session._atomic_json

    def fail(path, payload):
        if path.name.startswith('quality_'):
            raise OSError('cannot stage review')
        return original(path, payload)

    monkeypatch.setattr(panel._session, '_atomic_json', fail)
    with pytest.raises(OSError):
        panel._session.save_multimodal_pair(image, image)
    folder = panel._session.session_dir / 'multimodal'
    assert not list(folder.glob('*.png')) and not list(folder.glob('quality_*.json'))
    assert panel._session.metadata.get('multimodal_frame_count', 0) == 0


def test_stability_belongs_to_click_snapshot_not_background_finish_time(panel, monkeypatch, finish_aux_tasks):
    image = board_image()
    live(panel, monkeypatch, image)
    seed_stability(panel, image)
    panel._on_capture_aux()
    # The user can move after the clicked image has been captured. Later preview
    # expiry must not turn that already stable image into an invalid capture.
    panel._aux_guide._samples.clear()
    panel._aux_guide._clock = lambda: time.monotonic() + 10
    finish_aux_tasks(panel)
    assert len(panel._aux_pairs) == 1
    assert panel._aux_quality_records[0]['stability']['evaluated_at'] == 'capture_snapshot'


def test_lost_pairing_revokes_ready_text_but_preserves_last_receipt(panel, monkeypatch):
    guide = panel._aux_guide
    guide._manual = False
    guide.analysis = AuxGuidanceAnalysis('initial', AuxTarget(), 1, 0, 0, {}, ('large',))
    guide._key = guide.context_key()
    guide.action.setText('保持静止，可以采集')
    guide.receipt.setText('最近原图：已保存')
    # Exercise the actual sampler (the fixture disables it by default).
    from multicamera.ui.widgets.aux_capture import AuxCaptureGuide
    monkeypatch.setattr(panel, '_aux_live_frames', lambda: (None, '无新帧'))
    guide._preview_started = -10
    AuxCaptureGuide._request_preview(guide)
    assert '可以采集' not in guide.action.text()
    assert guide.receipt.text() == '最近原图：已保存'


def test_actual_import_button_to_completion_and_manual_supplement(panel, monkeypatch, tmp_path,
                                                                 finish_aux_tasks):
    from PySide6.QtWidgets import QFileDialog

    source = tmp_path / 'source'
    source.mkdir()
    image = board_image()
    for i in range(21):
        matrix = np.array([[1., 0., (i - 10) * .9], [0., 1., 0.]])
        moved = cv2.warpAffine(image, matrix, (256, 192), borderValue=(255, 255, 255))
        for prefix in ('rgb_left', 'aux'):
            assert cv2.imwrite(str(source / f'{prefix}_{i:04d}.png'), moved)
    originals = {p.name: p.read_bytes() for p in source.iterdir()}
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args: str(source))
    panel._btn_import_aux.click()
    finish_aux_tasks(panel)
    assert len(panel._aux_pairs) == 21
    assert panel._aux_guide.analysis.unique_count == 21
    assert panel._aux_guide.analysis.phase == 'complete'
    assert '本轮采集完成' in panel._aux_guide.phase.text()
    assert '源目录 21' in panel._aux_guide.counts.text()
    assert panel._aux_capture_sequence.done_count == 0  # No invented pose verification.
    assert all('建议' in action.text() for action in panel._aux_seq_widget._step_actions)
    live(panel, monkeypatch, image)
    seed_stability(panel, image)
    panel._refresh_aux_readiness()
    assert panel._btn_capture_aux.isEnabled()
    panel._btn_capture_aux.click()
    finish_aux_tasks(panel)
    assert panel._aux_saved_count == 1
    assert panel._aux_guide.analysis.phase == 'complete'
    assert panel._aux_guide.analysis.unique_count == 21
    assert '重复构图' in panel._aux_guide.receipt.text()
    assert originals == {p.name: p.read_bytes() for p in source.iterdir()}


def test_expired_preview_result_cannot_restore_a_revoked_ready_state(panel):
    guide = panel._aux_guide
    guide.analysis = AuxGuidanceAnalysis('initial', AuxTarget(), 1, 0, 0, {}, ('large',))
    guide._key = guide.context_key()
    guide.action.setText('保持静止，可以采集')
    guide.expire('ir')
    assert '可以采集' not in guide.action.text()
    guide._preview_ready((guide.context_key(), guide._clock() - 5), None, '')
    assert '可以采集' not in guide.action.text() and not guide.last_qualities


def test_completion_does_not_disable_manual_supplement(panel, monkeypatch, finish_aux_tasks):
    image = board_image()
    live(panel, monkeypatch, image)
    seed_stability(panel, image)
    panel._aux_guide.analysis = AuxGuidanceAnalysis('complete', None, 21, 0, 0, {}, ('large',))
    panel._aux_guide._key = panel._aux_guide.context_key()
    panel._aux_guide.render()
    panel._refresh_aux_readiness()
    assert panel._btn_capture_aux.isEnabled()
    panel._on_capture_aux()
    finish_aux_tasks(panel)
    assert panel._aux_saved_count == 1
