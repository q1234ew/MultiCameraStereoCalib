"""Saved-original receipts must not masquerade as current preview detections."""

import time

import numpy as np
import pytest

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.board.detector import DetectionResult
from multicamera.streaming.stream_manager import AuxiliaryCameraConfig, StreamManager
from multicamera.ui.widgets.stream_view import MultiStreamView


@pytest.fixture
def view(app, monkeypatch):
    manager = StreamManager()
    manager.set_auxiliary_camera(AuxiliaryCameraConfig('ir', 'http://device.test/ir'))
    widget = MultiStreamView(manager, CharucoBoard(CharucoBoardConfig()))
    widget.set_multimodal_mode(True, 'left')
    widget._show_guide = False
    monkeypatch.setattr(widget, '_detect_for_preview',
                        lambda image: DetectionResult(None, None, [], None, image.shape[1::-1]))
    widget.resize(800, 400)
    widget.show()
    widget._on_frame('ir', np.zeros((192, 256, 3), np.uint8), time.time())
    app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()


def receipt(view, state='可采纳', key='0005'):
    payload = dict(key=key, points=24, total=24, state=state, method='SB/ROI')
    view.show_aux_original_receipt('ir', payload)
    return payload


@pytest.mark.parametrize('state, label', [('可采纳', '验收通过'), ('待复核', '待复核')])
def test_preview_and_original_evidence_coexist_without_replacing_points(view, state, label):
    tile = view._views['ir']
    frame = tile._last_frame
    detection = view._last_detection['ir']
    payload = receipt(view, state)
    payload['points'] = 0
    first, second = tile._info.text().split('\n')
    assert '预览检测 0/24' in first
    assert second == f'最近原图 0005 24/24 {label}'
    assert '方法 SB/ROI' in tile._info.toolTip()
    assert tile._last_frame is frame
    assert view._last_detection['ir'] is detection
    assert view.displayed_corner_counts()['ir'] == 0


def test_fps_frames_and_detection_expiry_keep_original_receipt(view):
    receipt(view)
    view._on_fps('ir', 12.5)
    view._last_render_ts.clear()
    view._on_frame('ir', np.zeros((192, 256, 3), np.uint8), time.time())
    assert '12.5 fps' in view._views['ir']._info.text()
    assert '最近原图 0005 24/24 验收通过' in view._views['ir']._info.text()
    view._last_detect_ts['ir'] = time.monotonic() - 2
    view._expire_preview_detections()
    assert '已过期' in view._views['ir']._info.text()
    assert '最近原图 0005 24/24 验收通过' in view._views['ir']._info.text()
    assert view.displayed_corner_counts()['ir'] is None


@pytest.mark.parametrize('operation', [
    lambda v: v.begin_capture_session(),
    lambda v: v.clear_camera_frames(['ir']),
    lambda v: v._views['ir'].clear_frame(),
    lambda v: v.update_board(v._board),
    lambda v: v.reset_trackers(),
    lambda v: v.show_connecting({'ir': 'http://device.test/ir'}),
    lambda v: v.set_multimodal_mode(False),
    lambda v: v.clear_all(),
])
def test_session_configuration_and_cache_resets_discard_receipts(view, operation):
    receipt(view)
    operation(view)
    for tile in view._views.values():
        assert '最近原图' not in tile._info.text()
        tile.update_info('刷新')
        assert '最近原图' not in tile._info.toolTip()


@pytest.mark.parametrize('review', [True, False])
def test_different_offline_image_does_not_inherit_saved_evidence(view, review):
    receipt(view)
    images = {'ir': np.full((192, 256, 3), 100, np.uint8)}
    if review:
        view.show_review_images('0099', images)
        receipt(view)  # Delayed live receipts must not contaminate review images either.
    else:
        view.show_multimodal_images(images)
    assert '最近原图' not in view._views['ir']._info.text()
    assert view.displayed_corner_counts()['ir'] == 0


@pytest.mark.parametrize('width', [240, 400, 800])
def test_two_rows_remain_fixed_and_each_row_is_independently_elided(view, app, width):
    tile = view._views['ir']
    tile._info.setFixedWidth(width)
    before = tile._info.height(), tile.minimumSizeHint().width()
    receipt(view, key='original_' * 30)
    app.processEvents()
    assert (tile._info.height(), tile.minimumSizeHint().width()) == before
    rows = tile._info.visible_text().split('\n')
    assert len(rows) == 2
    assert rows[0].startswith('预览检测 0/24')
    assert rows[1].startswith('最近原图')
    for row in rows:
        assert tile._info.fontMetrics().horizontalAdvance(row) <= width - 8
