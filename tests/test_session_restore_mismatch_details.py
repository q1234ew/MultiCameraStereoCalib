"""Explain offline isolation without connecting historical addresses or weakening rollback."""

from copy import deepcopy
import time
from unittest.mock import Mock

import numpy as np
import pytest

from multicamera.board.charuco_board import CharucoBoard
from multicamera.calibration.models import CameraModel
from multicamera.io import session_restore
from multicamera.streaming.stream_manager import FrameEntry
from multicamera.ui import main_window
from multicamera.ui.widgets.config_dialog import ConfigDialog
import test_session_restore_workspace as restore


window = restore.window


def configure_matching(window, session, pair):
    manager = window._stream_manager
    # Represent a configured live source without creating or starting network threads.
    manager.add_virtual_stereo_pair(pair)
    manager._virtual_pairs.discard(pair.name)
    window._board = CharucoBoard.from_dict(session.metadata['board'])
    window._camera_model = CameraModel(session.metadata.get('camera_model', 'pinhole'))
    plan = session_restore.prepare_session_restore(session)
    if plan.auxiliary is not None:
        manager.set_auxiliary_camera(plan.auxiliary)
    window._calib_panel.refresh_pairs()
    return manager


def cancel_prompt(monkeypatch):
    question = Mock(return_value=main_window.QMessageBox.No)
    monkeypatch.setattr(main_window.QMessageBox, 'question', question)
    return question


@pytest.mark.parametrize('change,expected', [
    ('left_url', 'rgb 左目视频地址不同'),
    ('right_url', 'rgb 右目视频地址不同'),
    ('left_id', 'rgb 左目 ID：当前 other_left；会话 left'),
    ('right_id', 'rgb 右目 ID：当前 other_right；会话 right'),
    ('swap', 'rgb 左目 ID：当前 right；会话 left'),
    ('missing_group', '相机组 rgb：当前未配置'),
    ('stream_type', 'rgb 左目视频类型：当前 ir；会话 rgb'),
    ('model', 'RGB 镜头模型：当前 fisheye；会话 pinhole'),
    ('cols', 'RGB 标定板 列格数：当前 10；会话 9'),
    ('rows', 'RGB 标定板 行格数：当前 8；会话 7'),
    ('square_length', 'RGB 标定板 方格边长(m)：当前 0.05；会话 0.04'),
    ('marker_length', 'RGB 标定板 标记边长(m)：当前 0.02；会话 0.03'),
    ('dict_type', 'RGB 标定板 字典：当前 DICT_4X4_250；会话 DICT_4X4_1000'),
    ('size', 'left 实时图像尺寸：当前 640×480；会话内参 1280×720'),
    ('aux_missing', '辅助相机：当前未配置；会话 thermal（ir）'),
    ('aux_id', '辅助相机 ID：当前 other_aux；会话 thermal'),
    ('aux_modality', '辅助相机 模态：当前 acoustic_rgb；会话 ir'),
    ('aux_board', '辅助标定板 点间距(m)：当前 0.09；会话 0.08'),
    ('aux_url', '辅助相机视频地址不同'),
])
def test_each_mismatch_is_explained_without_mutating_session_or_streams(
        window, tmp_path, monkeypatch, change, expected):
    session, pair, _ = restore.make_session(tmp_path / 'source', auxiliary=change.startswith('aux'))
    before = restore.hashes(session)
    manager = configure_matching(window, session, pair)
    if change.endswith('_url') and not change.startswith('aux'):
        getattr(pair, change.split('_')[0]).url = 'http://user:secret@new-device.test/video?token=private'
    elif change in ('left_id', 'right_id'):
        getattr(pair, change.split('_')[0]).camera_id = 'other_' + change.split('_')[0]
    elif change == 'swap':
        pair.left.camera_id, pair.right.camera_id = pair.right.camera_id, pair.left.camera_id
    elif change == 'missing_group':
        manager.remove_stereo_pair(pair.name)
    elif change == 'stream_type':
        pair.left.stream_type = 'ir'
    elif change == 'model':
        window._camera_model = CameraModel.FISHEYE
    elif change in ('cols', 'rows', 'square_length', 'marker_length', 'dict_type'):
        board = window._board.to_dict()
        board[change] = {'cols': 10, 'rows': 8, 'square_length': .05,
                         'marker_length': .02, 'dict_type': 'DICT_4X4_250'}[change]
        window._board = CharucoBoard.from_dict(board)
    elif change == 'size':
        manager._latest_frames['left'] = FrameEntry(np.zeros((480, 640, 3), np.uint8), time.time())
    elif change == 'aux_missing':
        manager.clear_auxiliary_camera()
    elif change == 'aux_id':
        manager.auxiliary_camera.camera_id = 'other_aux'
    elif change == 'aux_modality':
        manager.auxiliary_camera.modality = 'acoustic_rgb'
    elif change == 'aux_board':
        manager.auxiliary_camera.board_config.square_size = .09
    elif change == 'aux_url':
        manager.auxiliary_camera.url = 'http://user:secret@new-device.test/ir?token=private'
    stop = Mock(wraps=manager.stop_all)
    start = Mock()
    monkeypatch.setattr(manager, 'stop_all', stop)
    monkeypatch.setattr(manager, 'start_all', start)
    question = cancel_prompt(monkeypatch)
    restore.choose_session(monkeypatch, session)
    window._on_load_session()
    question.assert_called_once()
    text = question.call_args.args[2]
    assert expected in text
    assert '不是在线续拍确认' in text and '不能证明实物设备及镜头相同' in text
    assert 'secret' not in text and 'private' not in text and 'new-device.test' not in text
    assert window._current_session is None and window._stream_manager is manager
    assert restore.hashes(session) == before
    start.assert_not_called()
    stop.assert_not_called()


def test_auxiliary_difference_alone_explains_same_rgb_pair_prompt(window, tmp_path, monkeypatch):
    session, pair, _ = restore.make_session(tmp_path / 'source', auxiliary=True)
    manager = configure_matching(window, session, pair)
    manager.clear_auxiliary_camera()
    plan = session_restore.prepare_session_restore(session)
    assert window._session_workspace_differences(session, plan) == [
        '辅助相机：当前未配置；会话 thermal（ir）']
    question = cancel_prompt(monkeypatch)
    assert not window._load_session_result(session, plan)
    text = question.call_args.args[2]
    assert '结果页“继续 RGB + 红外采集”' in text
    assert '普通“相机配置”修改地址' in text


def test_uncomputed_session_also_checks_live_url_before_restoring_observations(
        window, tmp_path, monkeypatch):
    session, pair, _ = restore.make_session(tmp_path / 'source')
    configure_matching(window, session, pair)
    (session.session_dir / 'results/latest.json').unlink()
    assert session.load_rig() is None
    pair.left.url = 'http://changed-device.test/left'
    before = restore.hashes(session)
    question = cancel_prompt(monkeypatch)
    restore.choose_session(monkeypatch, session)
    window._on_load_session()
    assert 'rgb 左目视频地址不同' in question.call_args.args[2]
    assert window._current_session is None
    assert restore.hashes(session) == before


def test_all_differences_are_reported_even_when_url_already_requires_isolation(
        window, tmp_path):
    session, pair, _ = restore.make_session(tmp_path / 'source', auxiliary=True)
    manager = configure_matching(window, session, pair)
    manager.clear_auxiliary_camera()
    pair.left.url = 'http://changed-device.test/left'
    manager._latest_frames['left'] = FrameEntry(np.zeros((480, 640, 3), np.uint8), time.time())
    differences = window._session_workspace_differences(
        session, session_restore.prepare_session_restore(session))
    assert len(differences) == 3
    assert any('左目视频地址不同' in item for item in differences)
    assert any('辅助相机：当前未配置' in item for item in differences)
    assert any('实时图像尺寸' in item for item in differences)


def test_matching_live_configuration_keeps_manager_and_streams(window, tmp_path, monkeypatch):
    session, pair, _ = restore.make_session(tmp_path / 'source')
    manager = configure_matching(window, session, pair)
    stop = Mock(wraps=manager.stop_all)
    monkeypatch.setattr(manager, 'stop_all', stop)
    monkeypatch.setattr(manager, 'start_all', lambda: pytest.fail('Loading cannot connect devices'))
    monkeypatch.setattr(main_window.QMessageBox, 'question',
                        lambda *a: pytest.fail('Matching configuration needs no isolation prompt'))
    before = restore.hashes(session)
    restore.choose_session(monkeypatch, session)
    window._on_load_session()
    assert window._stream_manager is manager
    assert window._current_session.session_dir == session.session_dir
    assert not manager.is_virtual_pair('rgb')
    assert restore.hashes(session) == before
    stop.assert_not_called()


@pytest.mark.parametrize('failure', ['preflight', 'restore', 'publish'])
def test_failure_after_diagnosis_keeps_original_workspace_files_and_live_frames(
        window, tmp_path, monkeypatch, failure):
    current, pair, rig = restore.make_session(tmp_path / 'current')
    incoming, _, _ = restore.make_session(tmp_path / 'incoming', auxiliary=True)
    manager = configure_matching(window, current, pair)
    window._current_session = current
    window._calib_panel.set_session(current)
    window._rig = rig
    frame = FrameEntry(np.zeros((720, 1280, 3), np.uint8), time.time())
    manager._latest_frames['left'] = frame
    central, panel = window.centralWidget(), window._calib_panel
    configuration = deepcopy(window._session_configuration())
    current_files, incoming_files = restore.hashes(current), restore.hashes(incoming)
    stop = Mock(wraps=manager.stop_all)
    monkeypatch.setattr(manager, 'stop_all', stop)
    monkeypatch.setattr(manager, 'start_all', lambda: pytest.fail('Loading cannot connect devices'))
    monkeypatch.setattr(main_window.QMessageBox, 'question', lambda *a: main_window.QMessageBox.Yes)
    if failure == 'preflight':
        owner, name = session_restore, 'prepare_session_result'
    elif failure == 'restore':
        owner, name = main_window.CalibrationPanel, 'restore_session_result'
    else:
        owner, name = window, 'setCentralWidget'
    original = getattr(owner, name)

    def fail(*args, **kwargs):
        if failure == 'publish':
            original(*args, **kwargs)
        raise OSError('injected session restore failure')

    monkeypatch.setattr(owner, name, fail)
    restore.choose_session(monkeypatch, incoming)
    window._on_load_session()
    assert 'injected session restore failure' in window.statusBar().currentMessage()
    assert window._stream_manager is manager and manager._latest_frames['left'] is frame
    assert window._current_session is current and window._calib_panel is panel
    assert panel._session is current and window._rig is rig
    assert window.centralWidget() is central
    assert window._session_configuration() == configuration
    assert restore.hashes(current) == current_files and restore.hashes(incoming) == incoming_files
    stop.assert_not_called()


def test_offline_open_does_not_connect_but_dedicated_continuation_requires_confirmation(
        window, app, tmp_path, monkeypatch):
    session, _, _ = restore.make_session(tmp_path / 'source', auxiliary=True)
    before = restore.hashes(session)
    answers = [main_window.QMessageBox.Yes]
    prompts = []

    def answer(*args):
        prompts.append(args[1])
        return answers.pop(0)

    monkeypatch.setattr(main_window.QMessageBox, 'question', answer)
    monkeypatch.setattr(main_window.StreamManager, 'start_all',
                        lambda *a: pytest.fail('Neither offline load nor canceled continuation may connect'))
    restore.choose_session(monkeypatch, session)
    window._on_load_session()
    restore.wait_restore(app, window._calib_panel)
    assert not window._stream_manager.configured_camera_urls()
    assert not window._stream_manager._grabbers
    assert restore.hashes(session) == before
    assert prompts == ['按会话信息离线打开']
    answers.append(main_window.QMessageBox.No)
    monkeypatch.setattr(ConfigDialog, 'exec',
                        lambda *a: pytest.fail('Device/lens attestation must precede configuration'))
    window._on_continue_aux_capture()
    assert prompts == ['按会话信息离线打开', '确认沿用双目标定']
    assert not window._stream_manager.configured_camera_urls()
    assert restore.hashes(session) == before


def test_invalid_result_is_rejected_before_offering_any_rebinding(window, tmp_path, monkeypatch):
    session, _, _ = restore.make_session(tmp_path / 'source')
    session._metadata['pairs'][0]['left_id'] = 'wrong_left'
    session._save_metadata()
    before = restore.hashes(session)
    monkeypatch.setattr(main_window.QMessageBox, 'question',
                        lambda *a: pytest.fail('Invalid results are not an isolation choice'))
    restore.choose_session(monkeypatch, session)
    window._on_load_session()
    assert window._current_session is None
    assert 'ID 不一致' in window.statusBar().currentMessage()
    assert restore.hashes(session) == before


@pytest.mark.parametrize('side', ['rgb', 'aux'])
def test_unknown_serialized_board_field_still_requires_isolation(window, tmp_path, monkeypatch, side):
    session, pair, _ = restore.make_session(tmp_path / 'source', auxiliary=True)
    manager = configure_matching(window, session, pair)
    board = window._board if side == 'rgb' else manager.auxiliary_camera.pattern_board_config()
    original = board.to_dict()
    monkeypatch.setattr(board, 'to_dict', lambda: {**original, 'future_board_option': None})
    if side == 'aux':
        monkeypatch.setattr(manager.auxiliary_camera, 'pattern_board_config', lambda: board)
    differences = window._session_workspace_differences(
        session, session_restore.prepare_session_restore(session))
    label = 'RGB 标定板' if side == 'rgb' else '辅助标定板'
    assert f'{label} 扩展参数 future_board_option 不同' in differences
