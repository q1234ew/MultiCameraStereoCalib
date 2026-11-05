"""Historical review data does not turn current live capture into offline UI."""

from copy import deepcopy

from multicamera.calibration.aux_guidance import initial_plan
import test_aux_online_flow as flow
from test_aux_online_flow import board_image, frames, stable

online = flow.online


def test_live_intent_keeps_reference_and_live_checks_with_history(online, app, finish_aux_tasks):
    panel = online
    panel._aux_offline_records = [('history',)]
    panel._aux_pairs = [('history', None, None)]
    assert panel.aux_offline_review_mode
    panel.show()
    panel.set_live_sources_active(True)
    finish_aux_tasks(panel)
    assert not panel.aux_offline_review_mode
    assert panel.aux_guidance_pairs == []
    assert len(panel._aux_pairs) == 1  # Never alter historical solver admissions.
    assert not panel._aux_seq_widget.isHidden()
    assert '离线候选复核' not in panel._aux_guide.phase.text()
    assert '离线未验证同步' not in panel._aux_guide.check_values['pairing'].text()
    assert len(initial_plan()) == panel._aux_capture_sequence.total
    panel.set_live_sources_active(False)
    finish_aux_tasks(panel)
    panel._aux_guide.render()
    assert panel.aux_offline_review_mode and panel._aux_seq_widget.isHidden()
    assert '离线候选复核' in panel._aux_guide.phase.text()


def test_restored_total_and_new_images_are_not_double_counted(online):
    panel = online
    panel._aux_saved_count = 17
    panel._aux_imported_count = 17
    panel._aux_source_folder = panel._session.session_dir / 'multimodal'
    assert panel.aux_storage_text() == '会话已保存 17 组（历史 17 · 本次新增 0）'
    panel._aux_saved_count += 1
    panel._aux_new_saved_indices.add(17)
    panel._aux_guide._update_counts()
    assert panel._aux_guide.counts.text().startswith('会话已保存 18 组（历史 17 · 本次新增 1）')
    assert '源目录 17' not in panel._aux_guide.counts.text()
    panel._aux_source_folder = panel._session.session_dir / 'external'
    assert '外部源目录 17 组（未复制）' in panel.aux_storage_text()
    panel._reset_all()
    assert not panel._aux_new_saved_indices and not panel._aux_live_capture_active


def test_saved_original_receipts_are_emitted_for_both_sources(online, finish_aux_tasks):
    panel = online
    received = []
    panel.aux_original_checked.connect(lambda cid, payload: received.append((cid, deepcopy(payload))))
    image = board_image()
    frames(panel, image)
    stable(panel, image)
    panel._on_capture_aux()
    finish_aux_tasks(panel)
    assert {cid for cid, _ in received} == {'left', 'ir'}
    assert all(row['points'] == row['total'] == 24 and row['state'] == '验收通过'
               and row['method'] for _, row in received)
    assert len(panel._aux_new_saved_indices) == 1


def test_bad_original_has_separate_review_receipt_without_guidance_credit(online, finish_aux_tasks):
    import numpy as np
    panel = online
    received = []
    panel.aux_original_checked.connect(lambda cid, payload: received.append(payload))
    image = board_image()
    frames(panel, image)
    stable(panel, image)
    frames(panel, np.full_like(image, 90))
    panel._on_capture_aux()
    finish_aux_tasks(panel)
    assert len(received) == 2 and all(row['state'] == '待复核' for row in received)
    assert not panel._aux_live_guidance_pairs
    assert len(panel._aux_new_saved_indices) == 1
