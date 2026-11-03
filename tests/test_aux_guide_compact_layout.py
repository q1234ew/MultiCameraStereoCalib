"""Compact AUX instructions keep usable text and stable native Qt geometry."""

import os
from pathlib import Path

import pytest
from PySide6.QtWidgets import QStyleFactory

from multicamera.calibration.aux_guidance import AuxGuidanceAnalysis, AuxTarget
from multicamera.ui.theme import app_stylesheet
from test_aux_capture_ui import panel as panel


@pytest.mark.parametrize('width', [300, 360, 430, 520])
@pytest.mark.parametrize('style_name', ['Fusion', 'Windows'])
def test_compact_instruction_budget_and_stable_transitions(
        panel, app, width, style_name, tmp_path):
    style = QStyleFactory.create(style_name)
    if style is None:
        pytest.skip(f'{style_name} style is unavailable')
    panel._aux_readiness_timer.stop()
    guide = panel._aux_guide
    guide.setParent(None)
    guide.setStyle(style)
    guide._test_style_reference = style
    guide.setStyleSheet(app_stylesheet())
    guide.analysis = AuxGuidanceAnalysis('initial', AuxTarget(), 0, 0, 0, {}, ())
    guide._key = guide.context_key()
    guide.render()
    guide.resize(width, guide.sizeHint().height())
    guide.show()
    app.processEvents()
    labels = (guide.heading, guide.phase, guide.action, guide.target_info,
              guide.next_title, guide.next_step)
    old_budget = (guide.heading.sizeHint().height() + guide.next_title.sizeHint().height()
                  + sum(label.fontMetrics().lineSpacing()*lines + 4
                        for label, lines in ((guide.phase, 2), (guide.action, 3),
                                             (guide.target_info, 3), (guide.next_step, 3)))
                  + 5*6)
    reclaimed = sum(label.fontMetrics().lineSpacing()
                    for label in (guide.action, guide.target_info, guide.next_step))
    assert guide.height() <= old_budget-reclaimed
    assert guide.action._lines == guide.target_info._lines == guide.next_step._lines == 2
    assert guide.action.contentsMargins().left() >= 8
    assert guide.action.fontMetrics().height() > guide.phase.fontMetrics().height()
    assert '非实测' in guide.phase.visible_text()
    assert '检测点 6×4' in guide.target_info.visible_text()
    assert '…' not in guide.action.visible_text()
    assert '…' not in guide.target_info.visible_text()
    original = [label.geometry() for label in labels]
    for phase, target, count in (('repair', AuxTarget(scale='large'), 17),
                                  ('complete', None, 21), ('blocked', None, 0)):
        guide.analysis = AuxGuidanceAnalysis(phase, target, count, 0, 0, {}, ())
        guide.render()
        app.processEvents()
        assert [label.geometry() for label in labels] == original
        for first, second in zip(labels, labels[1:]):
            assert first.geometry().bottom() < second.geometry().top()
        assert guide.next_step.visible_text().startswith('下一步：')
        assert guide.next_step.toolTip() == guide.next_step.text()
        assert guide.next_step.accessibleDescription() == guide.next_step.text()
    directory = Path(os.environ.get('MULTICAMERA_UI_QA_DIR', tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    assert guide.grab().save(str(directory / f'aux-compact-{style_name}-{width}.png'))
    guide.hide()
    guide.setParent(panel)
