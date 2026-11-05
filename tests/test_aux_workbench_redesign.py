"""AUX workbench presentation follows real capture, review and evidence state."""

from types import SimpleNamespace

import numpy as np
import pytest

from multicamera.calibration.aux_capture_quality import AuxFrameQuality
from multicamera.calibration.aux_guidance import AuxGuidanceAnalysis, AuxTarget
from multicamera.ui.theme import SUCCESS, app_stylesheet
from multicamera.ui.widgets.capture_guide import BoardTilt
from test_aux_capture_ui import board_image, live, panel as panel, seed_stability


def set_analysis(panel, phase='initial', target=AuxTarget(), count=0):
    guide = panel._aux_guide
    guide.analysis = AuxGuidanceAnalysis(phase, target, count, 0, 0, {}, ())
    guide._key = guide.context_key()
    guide.render()
    return guide


def ready_preview(panel, monkeypatch):
    image = board_image()
    live(panel, monkeypatch, image)
    det = seed_stability(panel, image)
    guide = panel._aux_guide
    guide._preview_detections = [det, det]
    guide.last_qualities = [AuxFrameQuality(True, (), {'area': .11},
        {'distance_reliable': True, 'yaw': 0., 'pitch': 0.}),
        AuxFrameQuality(True, (), {'area': .11}, None)]
    guide._preview_result_at = guide._clock()
    return set_analysis(panel)


@pytest.mark.parametrize('width', [300, 360, 480])
def test_current_action_next_step_and_target_have_stable_geometry(
        panel, app, monkeypatch, width, tmp_path):
    panel._aux_readiness_timer.stop()
    guide = ready_preview(panel, monkeypatch)
    guide.setStyleSheet(app_stylesheet())
    guide.setParent(None)
    guide.resize(width, guide.sizeHint().height())
    guide.show()
    app.processEvents()
    labels = (guide.phase, guide.action, guide.target_info, guide.next_step)
    geometry = [label.geometry() for label in labels]
    assert guide.action.fontMetrics().height() > guide.phase.fontMetrics().height()
    assert '采集配对图像' in guide.next_step.visible_text()
    assert '检测点 6×4' in guide.target_info.visible_text()
    assert guide.grab().save(str(tmp_path / f'aux-current-action-{width}.png'))
    for phase, count, target in (('repair', 17, AuxTarget(scale='large')),
                                  ('complete', 21, None), ('blocked', 0, None)):
        set_analysis(panel, phase, target, count)
        app.processEvents()
        assert [label.geometry() for label in labels] == geometry
        assert guide.next_step.isVisible()
        assert guide.next_step.visible_text().startswith('下一步：')
        for first, second in zip(labels, labels[1:]):
            assert first.geometry().bottom() < second.geometry().top()
    guide.hide()
    guide.setParent(panel)


def test_goal_preserves_recommended_tilt_while_current_schematic_stays_frontal(panel):
    guide = set_analysis(panel, target=AuxTarget(tilt='left'))
    assert panel._aux_capture_sequence.current.tilt == BoardTilt.FRONT
    assert '左边缘靠近镜头' in guide.target_info.text()
    assert '检测点 6×4' in guide.target_info.text()


def test_reference_browsing_is_not_a_capture_or_a_completed_stage(panel, monkeypatch):
    live(panel, monkeypatch, board_image())
    guide = set_analysis(panel)
    before = (guide.counts.text(), panel._aux_cap_progress.value(),
              len(panel._aux_pairs), panel._aux_saved_count)
    panel._aux_seq_widget._btn_next.click()
    guide.refresh()
    assert '浏览参考动作 2/17' in guide.phase.text()
    assert '非采集进度' in guide.phase.text()
    assert '仅浏览参考' in guide.next_step.text()
    assert '采图后恢复推荐' in guide.next_step.text()
    assert '浏览参考' in guide.target_info.text()
    assert before == (guide.counts.text(), panel._aux_cap_progress.value(),
                      len(panel._aux_pairs), panel._aux_saved_count)


def test_review_overrides_ready_preview_without_promoting_saved_images(panel, monkeypatch):
    guide = ready_preview(panel, monkeypatch)
    assert '可以采集' in guide.action.text()
    guide._view = SimpleNamespace(review_key='原图 live_0000')
    guide.refresh()
    assert '正在回看原图' in guide.phase.text()
    assert '结束回看' in guide.next_step.text()
    assert '可以采集' not in guide.action.text()
    assert all('回看中' in value.text() for value in guide.check_values.values())
    assert panel._aux_saved_count == 0
    guide._view = None
    guide.render()
    assert '采集配对图像' in guide.next_step.text()


def test_actual_capture_waits_for_check_and_save_then_updates_next_step(
        panel, monkeypatch, finish_aux_tasks):
    live(panel, monkeypatch, board_image())
    panel._on_capture_aux()
    guide = panel._aux_guide
    assert panel._aux_capture_worker is not None
    assert '尚未确认保存' in guide.action.text()
    assert '暂勿重复采集' in guide.next_step.text()
    assert '会话已保存 0 组' in guide.counts.text()
    finish_aux_tasks(panel)
    assert panel._aux_saved_count == 1
    assert '等待本次' not in guide.next_step.text()
    assert '会话已保存 1 组' in guide.counts.text()
    assert '验收通过 0 组' in guide.counts.text()


def test_save_failure_stays_visible_until_a_new_capture_receipt(panel, monkeypatch):
    guide = ready_preview(panel, monkeypatch)
    guide.show_receipt_message('live_0000 · 原图保存失败\n未计入验收和引导，请重新采图')
    for _ in range(3):
        guide.render()
        guide.refresh()
    assert '原图保存失败' in guide.action.text()
    assert '处理失败原因后重新采图' in guide.next_step.text()
    assert '可以采集' not in guide.action.text()
    assert panel._aux_saved_count == 0
    guide.record_receipt('live_0001', '可采纳', [], guide._preview_detections)
    guide.refresh_workbench()
    assert '保存失败' not in guide.action.text()
    assert '验收通过' in guide.next_step.text()
    assert '正在更新引导进度' in guide.next_step.text()


@pytest.mark.parametrize('operation', ['white_balance', 'import', 'capture'])
def test_busy_states_override_ready_action_and_restore_it(panel, monkeypatch, operation):
    guide = ready_preview(panel, monkeypatch)
    with monkeypatch.context() as busy:
        if operation == 'white_balance':
            busy.setattr(panel._white_balance_guard, 'busy', True)
        elif operation == 'import':
            busy.setattr(panel, '_aux_import_busy', True)
        else:
            busy.setattr(panel, '_aux_capture_worker', object())
        guide.refresh_workbench()
        assert '可以采集' not in guide.action.text()
        assert '等待' in guide.next_step.text()
        assert '保存与验收后自动更新推荐' not in guide.next_step.text()
    guide.refresh_workbench()
    assert '可以采集' in guide.action.text()
    assert '保存与验收后自动更新推荐' in guide.next_step.text()


def test_fresh_preview_checks_are_separate_from_capture_admission(panel, monkeypatch):
    guide = ready_preview(panel, monkeypatch)
    before = (panel._btn_capture_aux.isEnabled(), panel._btn_calib_aux.isEnabled(),
              panel._aux_saved_count, len(panel._aux_pairs), panel._aux_cap_progress.value())
    guide.refresh_workbench()
    assert guide.check_values['board'].text() == '双路完整可见'
    assert guide.check_values['quality'].text() == '预检通过'
    assert guide.check_values['stability'].text() == '连续检测稳定'
    assert guide.check_values['pairing'].text() == '软件已配对'
    assert SUCCESS not in guide.check_values['pairing'].styleSheet()
    assert '不是本次采图验收' in guide.check_values['quality'].toolTip()
    assert before == (panel._btn_capture_aux.isEnabled(), panel._btn_calib_aux.isEnabled(),
                      panel._aux_saved_count, len(panel._aux_pairs), panel._aux_cap_progress.value())
    monkeypatch.setattr(guide, '_clock', lambda: guide._preview_result_at + 5)
    guide.refresh_workbench()
    assert guide.check_values['quality'].text() == '等待检测'
    assert '等待双路连续稳定检测' in guide.next_step.text()
    assert '可以采集' not in guide.action.text()


def test_missing_frames_never_display_fresh_quality_results(panel, monkeypatch):
    guide = ready_preview(panel, monkeypatch)
    monkeypatch.setattr(panel, '_aux_live_frames', lambda: (None, '两路时间差过大'))
    guide.refresh_workbench()
    assert guide.check_values['pairing'].text() == '未配对'
    assert '时间差过大' in guide.check_values['pairing'].toolTip()
    assert guide.check_values['quality'].text() == '等待检测'
    assert '连接两路实时图像' in guide.next_step.text()


@pytest.mark.parametrize('problem', ['', '请逐组核对两路0点及行方向', '请先完成 RGB 左目所在双目标定'])
def test_completed_guidance_uses_existing_calibration_gate(panel, monkeypatch, problem):
    monkeypatch.setattr(panel, '_aux_calibration_problem', lambda: problem)
    guide = set_analysis(panel, 'complete', None, 21)
    assert '前往标定评估' in guide.next_step.text()
    assert problem in guide.next_step.text()
    assert '本轮引导结束' in guide.phase.text()
    assert panel._aux_capture_sequence.done_count == 0
    assert not panel._check_aux_order.isChecked()


def test_blocked_analysis_failure_and_pending_have_concrete_next_steps(panel, finish_aux_tasks):
    guide = set_analysis(panel, 'blocked', None)
    assert '调整标定板或分辨率' in guide.next_step.text()
    guide._analyzed(guide.context_key(), None, 'controlled failure')
    assert '查看日志' in guide.next_step.text()
    guide.data_changed()
    assert '等待采图建议更新' in guide.next_step.text()
    assert '等待采图建议更新' in guide.target_info.text()
    finish_aux_tasks(panel)


def test_acoustic_auxiliary_uses_its_real_modality_in_target_and_checks(panel, monkeypatch):
    panel._sm.auxiliary_camera.modality = 'acoustic_rgb'
    guide = set_analysis(panel)
    assert '声像仪RGB' in guide.target_info.toolTip()
    assert '红外' not in guide.target_info.toolTip()
    guide.last_qualities = [AuxFrameQuality(False, ('low_board_contrast',), {}, None)] * 2
    guide._preview_result_at = guide._clock()
    live(panel, monkeypatch, np.zeros((192, 256, 3), np.uint8))
    guide.refresh_workbench()
    assert '声像仪RGB' in guide.check_values['quality'].toolTip()
    assert '红外' not in guide.check_values['quality'].toolTip()
