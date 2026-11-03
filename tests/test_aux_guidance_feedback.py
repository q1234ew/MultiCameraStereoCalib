"""Timely reference progress and readable feedback without changing admission."""

from dataclasses import replace
import threading
import time

from PySide6.QtCore import QThreadPool

from multicamera.calibration.aux_guidance import AuxTarget, summarize_aux_progress
from multicamera.calibration.planar import PlanarPatternDetector
from multicamera.ui.widgets import aux_capture
from test_aux_capture_ui import board_image, panel as panel


def wait(app, predicate):
    deadline = time.monotonic()+5
    while not predicate():
        app.processEvents()
        assert time.monotonic() < deadline
        time.sleep(.002)
    app.processEvents()


def test_progress_and_next_reference_advance_while_full_analysis_is_blocked(
        panel, app, monkeypatch, finish_aux_tasks):
    release = threading.Event()
    original = aux_capture.AuxGuidanceWorker
    class SlowWorker(original):
        def run(self):
            release.wait(10)
            super().run()
    monkeypatch.setattr(aux_capture, 'AuxGuidanceWorker', SlowWorker)
    pool = QThreadPool.globalInstance()
    previous_limit = pool.maxThreadCount()
    pool.setMaxThreadCount(1)
    guide = panel._aux_guide
    det = PlanarPatternDetector(panel._aux_board_config()).detect(board_image())
    try:
        panel._aux_pairs = [('first', det, det)]
        guide.data_changed()
        wait(app, lambda: guide._key == guide.context_key())
        assert guide._worker is not None and guide.analysis.provisional
        assert guide.analysis.unique_count == 1
        assert panel._aux_capture_sequence.current_index == 1
        assert '参考动作 2/17' in panel._aux_seq_widget._large_guide.accessibleName()
        shifted = replace(det, image_points=det.image_points+[5., 0.])
        panel._aux_pairs.append(('second', shifted, shifted))
        guide.data_changed()
        wait(app, lambda: guide._key == guide.context_key())
        assert guide._worker is not None and guide.analysis.provisional
        assert guide.analysis.unique_count == 2
        assert panel._aux_capture_sequence.current_index == 2
        assert '参考动作 3/17' in panel._aux_seq_widget._large_guide.accessibleName()
    finally:
        release.set()
        pool.setMaxThreadCount(previous_limit)
        finish_aux_tasks(panel)
    assert not guide.analysis.provisional
    assert guide.analysis.unique_count == 2
    assert panel._aux_capture_sequence.current_index == 2


def test_capture_receipt_is_readable_for_five_seconds_and_explains_no_progress(
        panel, app):
    guide = panel._aux_guide
    clock = [100.]
    guide._clock = lambda: clock[0]
    det = PlanarPatternDetector(panel._aux_board_config()).detect(board_image())
    guide.record_receipt('bad_0000', '待复核', ['红外：未检测到完整网格'], (det, det))
    for at in (100., 101., 103., 104.9):
        clock[0] = at
        guide.render()
        app.processEvents()
        assert 'bad_0000' in guide.action.text()
        assert '验收待复核' in guide.action.text()
        assert '本组未计入引导' in guide.next_step.text()
        assert '未检测到完整网格' in guide.next_step.text()
        assert panel._aux_capture_sequence.current_index == 0
    clock[0] = 105.1
    guide.render()
    assert 'bad_0000' not in guide.action.text()
    assert 'bad_0000' in guide.receipt.text()  # Persistent receipt remains after hold expires.


def test_new_capture_or_failure_overrides_held_receipt_immediately(panel, monkeypatch):
    guide = panel._aux_guide
    det = PlanarPatternDetector(panel._aux_board_config()).detect(board_image())
    guide.record_receipt('first', '可采纳', [], (det, det))
    with monkeypatch.context() as changed:
        changed.setattr(panel, '_aux_capture_worker', object())
        guide.refresh_workbench()
        assert '尚未确认保存' in guide.action.text()
    guide.show_receipt_message('new · 原图保存失败\n本次未保存')
    guide.refresh_workbench()
    assert '保存失败' in guide.action.text()
    assert 'first' not in guide.action.text()


def test_duplicate_capture_explains_why_reference_does_not_advance(panel):
    guide = panel._aux_guide
    det = PlanarPatternDetector(panel._aux_board_config()).detect(board_image())
    panel._aux_pairs = [('one', det, det), ('duplicate', det, det)]
    guide.analysis = summarize_aux_progress(panel._aux_board_config(), panel._aux_pairs)
    guide._key = guide.context_key()
    guide.record_receipt('duplicate', '可采纳', [], (det, det))
    guide.render()
    assert panel._aux_capture_sequence.current_index == 1
    assert '重复构图未推进' in guide.action.text()
    assert '重复构图' in guide.next_step.text()
    assert '引导采用 1 组' in guide.counts.text()
    before = panel._log_text.toPlainText().count('重复构图，参考不推进')
    guide._announce_progress(guide.analysis)
    guide._announce_progress(guide.analysis)
    log = panel._log_text.toPlainText()
    assert log.count('重复构图，参考不推进') == before+1
    assert '建议参考 2/17' in log


def test_directional_hysteresis_never_keeps_a_stale_capture_permission(panel):
    guide, clock = panel._aux_guide, [100.]
    guide._clock = lambda: clock[0]
    target = AuxTarget()
    assert guide._stable_instruction('distance', '靠近镜头', target) == ('distance', '靠近镜头')
    clock[0] = 101.
    assert guide._stable_instruction('tilt', '左边靠近镜头', target) == ('distance', '靠近镜头')
    clock[0] = 102.
    assert guide._stable_instruction('tilt', '左边靠近镜头', target) == ('tilt', '左边靠近镜头')
    assert guide._stable_instruction('ready', '可以采集', target) == ('ready', '可以采集')
    clock[0] = 102.1
    assert guide._stable_instruction('quality', '整板未完整可见', target) == ('quality', '整板未完整可见')


def test_provisional_full_count_waits_for_distribution_before_repairs(panel):
    det = PlanarPatternDetector(panel._aux_board_config()).detect(board_image())
    pairs = [(str(i), replace(det, image_points=det.image_points+[i*3., 0.]),
              replace(det, image_points=det.image_points+[i*3., 0.])) for i in range(17)]
    guide = panel._aux_guide
    guide.analysis = summarize_aux_progress(panel._aux_board_config(), pairs)
    guide._key = guide.context_key()
    guide.render()
    assert guide.analysis.phase == 'refining'
    assert '正在评估距离与倾角分布' in guide.phase.text()
    assert '本轮引导结束' not in guide.phase.text()
    assert '当前尚未结束引导' in guide.next_step.text()
    assert '姿态与距离统计待更新' in guide.counts.toolTip()
