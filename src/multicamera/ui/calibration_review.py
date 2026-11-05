"""Single-pair review orchestration; raw observations remain authoritative."""

from copy import deepcopy

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from ..calibration.adaptive_guidance import GuidanceObservation
from ..calibration.models import StereoPairCalibration
from ..calibration.review import empty_state, excluded_ids, frame_fingerprint, scope_key
from .threads.review_worker import ReviewWorker
from .widgets.calibration_review import CalibrationReviewDialog, PairImages
from .widgets.capture_guide import BoardTilt, CaptureTarget


class CalibrationReviewMixin:
    def _init_review(self):
        self._review_states = {}
        self._review_archive = {}
        self._review_comparisons = {}
        self._review_dialog = None
        self._review_worker = None
        self._review_report = None
        self._review_report_key = None
        self._review_task = None
        self._review_error = ''
        self._review_raw_cache = {}
        self._review_effective_cache = {}

    @staticmethod
    def _review_result_snapshot(pair):
        # Persisted result facts suffice for review; do not duplicate full-resolution rectification maps.
        return StereoPairCalibration.from_dict(deepcopy(pair.to_dict()))

    def _reset_review(self):
        if self._review_dialog:
            self._review_dialog.close()
        self._review_states.clear()
        self._review_archive.clear()
        self._review_comparisons.clear()
        self._review_raw_cache.clear()
        self._review_effective_cache.clear()
        self._review_report_key = None
        self._review_report = None
        self._review_task = None

    def _review_scope(self, name):
        return scope_key(self._board, self._camera_model, self._sm.stereo_pairs[name],
                         self._offline_source_folder if name == 'offline' else '',
                         self._offline_lr_swapped if name == 'offline' else False)

    def _review_state(self, name):
        scope = self._review_scope(name)
        key = (id(self._session), scope)
        if key not in self._review_states:
            self._review_states[key] = (self._session.load_improvement(scope)
                                        if self._session else None) or empty_state()
        return self._review_states[key]

    def _store_review_state(self, name, state):
        if self._session is None:
            raise ValueError('请先创建会话以保存图像参与记录')
        scope = self._review_scope(name)
        # Commit on disk before publishing any changed participation in memory.
        self._session.save_improvement(scope, state)
        self._review_states[id(self._session), scope] = state

    def _review_raw(self, name):
        cfg = self._sm.stereo_pairs[name]
        cals = [self._intr_cals.get(camera.camera_id) for camera in (cfg.left, cfg.right)]
        stereo = self._stereo_cals.get(name)
        key = (self._review_scope(name), tuple((id(c), c.data_revision) if c else None for c in cals),
               id(stereo), tuple(stereo.frame_ids) if stereo else ())
        cached = self._review_raw_cache.get(name)
        if cached and cached[0] == key:
            return cached[1]
        observations = []
        for cal in cals:
            if cal is not None and len(cal.frame_ids) != len(set(cal.frame_ids)):
                raise ValueError('源帧 ID 重复，不能可靠应用图像复核决定')
            observations.append({str(fid): GuidanceObservation(
                corners.copy(), ids.copy(), cal._image_size, str(fid))
                for fid, corners, ids in zip(cal.frame_ids, cal._all_corners, cal._all_ids)}
                if cal else {})
        ids = list(dict.fromkeys([*observations[0], *observations[1]]))
        rows = {fid: {'left': observations[0].get(fid), 'right': observations[1].get(fid)}
                for fid in ids}
        for row in rows.values():
            row['fingerprint'] = frame_fingerprint(row['left'], row['right'])
        self._review_raw_cache[name] = key, rows
        return rows

    def _review_excluded(self, name):
        rows = self._review_raw(name)
        return excluded_ids(self._review_state(name),
                            {fid: row['fingerprint'] for fid, row in rows.items()})

    def _effective_pair_calibrators(self, name, *, snapshot=False):
        cfg = self._sm.stereo_pairs[name]
        cals = [self._intr_cals.get(cfg.left.camera_id), self._intr_cals.get(cfg.right.camera_id),
                self._stereo_cals.get(name)]
        if any(cal is None for cal in cals):
            return cals
        excluded = self._review_excluded(name)
        if not excluded and not snapshot:
            return cals
        cache_key = (self._review_raw_cache[name][0], tuple(sorted(excluded)),
                     tuple(cal.min_frames for cal in cals))
        cached = self._review_effective_cache.get(name)
        if not snapshot and cached and cached[0] == cache_key:
            return cached[1]
        indices = [[i for i, fid in enumerate(cal.frame_ids) if str(fid) not in excluded]
                   for cal in cals]
        selected = [cals[0].copy_with_indices(indices[0], keep_images=False),
                    cals[1].copy_with_indices(indices[1], keep_images=False),
                    cals[2].copy_with_indices(indices[2])]
        if snapshot:
            # Workers only read pixels for quality ranking; avoid copying full-resolution photos.
            for original, clone, take in zip(cals[:2], selected[:2], indices[:2]):
                clone._images = [original.stored_images[i] if i < len(original.stored_images)
                                 else None for i in take]
        else:
            self._review_effective_cache[name] = cache_key, selected
        return selected

    def _review_key(self, name):
        if name not in self._sm.stereo_pairs or name not in self._stereo_cals:
            return None
        cfg = self._sm.stereo_pairs[name]
        state = self._review_state(name)
        cals = [self._intr_cals.get(cid) for cid in (cfg.left.camera_id, cfg.right.camera_id)]
        return (id(self._session), self._review_scope(name),
                tuple((id(c), c.data_revision) if c else None for c in cals),
                id(self._stereo_cals[name]), tuple(self._stereo_cals[name].frame_ids),
                tuple(sorted(state['excluded'].items())), tuple(sorted(state['pending'].items())),
                id(self._pair_calibs.get(name) or self._review_archive.get(name)),
                tuple(sorted(self._pending_save_tokens)), tuple(sorted(self._failed_save_jobs)))

    def _review_mutable(self):
        return not (self._calib_worker or self._model_worker or self._aux_worker
                    or self._stereo_import_worker or self._aux_import_worker
                    or self._pending_save_tokens or self._failed_save_jobs
                    or self._white_balance_guard.busy or self._auto_timer.isActive())

    def _review_name(self):
        names = list(self._selected_pairs())
        if len(names) == 1:
            return names[0]
        return next(iter(self._stereo_cals), None) if len(self._stereo_cals) == 1 else None

    def _review_previews(self, name):
        return [p for p in self.preview_pairs() if p.key.startswith(name + ' ')]

    def _review_quality_map(self, name, camera_id):
        qualities = self._fqs.get(camera_id, [])
        # Offline quality rows include failed detections, whereas calibrators only keep accepted ones.
        ids = ([str(record[0]) for record in self._offline_import_records] if name == 'offline' else
               self._intr_cals[camera_id].frame_ids)
        return dict(zip(ids, qualities))

    def _review_rows(self, name):
        raw = self._review_raw(name)
        excluded = self._review_excluded(name)
        result = self._pair_calibs.get(name) or self._review_archive.get(name)
        error_maps = []
        for intr in ((result.left_intrinsics, result.right_intrinsics) if result else (None, None)):
            error_maps.append(dict(zip(intr.frame_ids or [], intr.per_frame_errors
                                      if intr.per_frame_errors is not None else []))
                              if intr and intr.per_frame_error_metric == 'euclidean_rms_px' else {})
        alignment = {str(row['frame_id']): row.get('rms_px') for row in
                     (result.calibration_info.get('rectification', {}).get('per_frame', [])
                      if result else [])}
        previews = self._review_previews(name)
        known = set(raw)
        rows = []
        for fid, observation in raw.items():
            matches = [p for p in previews if p.key == f'{name} {fid}']
            if not matches and fid.isdecimal() and not any(
                    other != fid and other.isdecimal() and int(other) == int(fid) for other in known):
                matches = [p for p in previews if p.key[len(name) + 1:].isdecimal()
                           and int(p.key[len(name) + 1:]) == int(fid)]
            preview = matches[0] if len(matches) == 1 else None
            left, right = observation['left'], observation['right']
            row = dict(observation, id=fid, preview=preview, excluded=fid in excluded,
                       stereo_available=fid in self._stereo_cals[name].frame_ids,
                       participates=fid not in excluded and fid in self._stereo_cals[name].frame_ids,
                       left_rms=error_maps[0].get(fid), right_rms=error_maps[1].get(fid),
                       alignment_rms=alignment.get(fid),
                       common_count=len(np.intersect1d(left.ids, right.ids)) if left and right else 0,
                       quality_reasons=list(preview.quality_reasons) if preview and
                       preview.quality_state in ('拒收', '待复核') else [])
            cfg = self._sm.stereo_pairs[name]
            for cid, label in ((cfg.left.camera_id, '左'), (cfg.right.camera_id, '右')):
                quality = self._review_quality_map(name, cid).get(fid)
                if quality is not None:
                    if quality.is_blurry:
                        row['quality_reasons'].append(label + '整图清晰度偏低，请结合板面原图复核')
                    if quality.is_overexposed or quality.is_underexposed:
                        row['quality_reasons'].append(label + '整图亮度异常，请结合板面原图复核')
            rows.append(row)
        return rows, result

    def _show_calibration_review(self, tab=0):
        name = self._review_name()
        if name is None or name not in self._stereo_cals:
            self._log('请先选择一个具有有效观测的双目组，再查看改进建议')
            return
        if self._review_dialog is not None:
            self._review_dialog.raise_()
            self._review_dialog.activateWindow()
            self._review_dialog.tabs.setCurrentIndex(tab)
            return
        result = self._pair_calibs.get(name) or self._review_archive.get(name)
        dialog = CalibrationReviewDialog(name, result, self._review_previews(name), self)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.reanalysis_requested.connect(lambda: self._request_review_analysis(name, force=True))
        dialog.participation_requested.connect(lambda fid: self._toggle_review_from_report(name, fid))
        dialog.retake_requested.connect(lambda row: self._begin_review_task(name, row))
        dialog.task_requested.connect(lambda task: self._begin_review_task(name, task, replacing=False))
        dialog.preview_requested.connect(self.diagnostic_preview_requested)
        dialog.finished.connect(lambda: setattr(self, '_review_dialog', None))
        self._review_dialog = dialog
        dialog.tabs.setCurrentIndex(tab)
        dialog.show()
        self._request_review_analysis(name)

    def _request_review_analysis(self, name, *, force=False):
        if force:
            self._review_report_key = None
        dialog = self._review_dialog
        try:
            key = self._review_key(name)
            if key is None:
                raise ValueError('该组观测已清除，请重新采集或导入')
            if self._review_report_key == key:
                self._display_review_report(name)
                return
            if dialog:
                dialog.set_busy()
            if self._review_worker is not None:
                return
            rows, result = self._review_rows(name)
            intrinsics = ((result.left_intrinsics, result.right_intrinsics) if result else (None, None))
            worker = ReviewWorker(key, self._board, rows, deepcopy(intrinsics), self)
            self._review_worker = worker
            def completed(received_key, report, error):
                try:
                    if self._review_key(name) == received_key:
                        self._review_report_key, self._review_report = received_key, report
                        if error and self._review_dialog:
                            self._review_dialog.set_busy('分析失败：' + error)
                        elif report:
                            self._display_review_report(name)
                except (ValueError, KeyError) as exc:
                    self._log('图像复核分析已过期：' + str(exc))
            def finished():
                self._review_worker = None
                if self._review_dialog is not None:
                    QTimer.singleShot(0, lambda: self._request_review_analysis(self._review_dialog.name)
                                      if self._review_dialog else None)
            worker.completed.connect(completed)
            worker.finished.connect(finished)
            worker.finished.connect(worker.deleteLater)
            worker.start()
        except (ValueError, OSError, KeyError) as exc:
            self._review_error = str(exc)
            if dialog:
                dialog.set_busy('不能应用复核记录：' + str(exc))
            self._log('图像复核失败：' + str(exc))

    def _display_review_report(self, name):
        dialog = self._review_dialog
        if dialog and dialog.name == name and self._review_report:
            dialog.set_analysis(self._review_report, self._review_previews(name),
                                not self._sm.is_virtual_pair(name),
                                self._review_mutable() and self._review_task is None)
        self._refresh_review_controls()

    def _review_changed(self, name):
        self._review_report_key = None
        self._invalidate_pair_result(name)
        self._sync_pair_states()
        self._update_capture_stats()
        self._refresh_preview_pairs()
        self._refresh_review_controls()
        if self._review_dialog:
            self._request_review_analysis(name)

    def _toggle_review_from_report(self, name, fid):
        try:
            if self._review_report_key != self._review_key(name):
                self._request_review_analysis(name)
                return False
        except (ValueError, OSError, KeyError):
            return False
        return self._toggle_review_participation(name, fid)

    def _toggle_review_participation(self, name, fid):
        if not self._review_mutable() or self._review_task is not None:
            return False
        try:
            if not self._ensure_capture_session():
                return False
            raw = self._review_raw(name)
            if fid not in raw:
                return False
            state = deepcopy(self._review_state(name))
            if fid in state['pending']:
                candidate = next((row for row in self._review_rows(name)[0] if row['id'] == fid), None)
                preview = candidate.get('preview') if candidate else None
                if preview is None or preview.save_state not in ('已保存', '源目录原图'):
                    raise ValueError('新图尚未成功保存，不能确认参与')
                state['pending'].pop(fid)

            if fid in self._review_excluded(name):
                state['excluded'].pop(fid, None)
            else:
                state['excluded'][fid] = raw[fid]['fingerprint']
            self._store_review_state(name, state)
            self._review_changed(name)
            return True
        except (ValueError, OSError) as exc:
            self._log('更新图像参与状态失败，原状态保留：' + str(exc))
            return False

    def _begin_review_task(self, name, item, replacing=True):
        if not self._review_mutable() or self._review_task or self._sm.is_virtual_pair(name):
            self._log('定向补拍仅支持空闲的单个实时双目组；离线数据可复核与排除')
            return False
        try:
            if not self._ensure_capture_session():
                return False
            if replacing:
                # Only a current analysis can turn a source image into a capture target.
                if self._review_report_key != self._review_key(name):
                    self._request_review_analysis(name)
                    return False
                current = next(row for row in self._review_report['rows'] if row['id'] == item['id'])
                geometry = current['geometry'][0]
                if geometry is None or not all(current['geometry']):
                    raise ValueError('双路原图构图不可解，请从补拍清单选择目标')
                target = dict(geometry)
                source, fingerprint = current['id'], current['fingerprint']
            else:
                if self._review_report_key != self._review_key(name) or item not in self._review_report['tasks']:
                    raise ValueError('补拍建议已过期，请重新分析')
                target, source, fingerprint = dict(item), None, None
            tilts = dict(zip(('front', 'left', 'right', 'up', 'down'), BoardTilt))
            capture_target = CaptureTarget(target['row'], target['col'], tilts[target['tilt']],
                                           scale=target['scale'])
            self._review_task = dict(name=name, scope=self._review_scope(name),
                                     source=source, fingerprint=fingerprint, target=capture_target,
                                     candidate=None, token=None,
                                     reference_centers=[g['center'] if g else None
                                                        for g in current['geometry']]
                                     if replacing else None,
                                     reference_poses=[g.get('pose') if g else None
                                                      for g in current['geometry']]
                                     if replacing else None)
            if self._review_dialog:
                self._review_dialog.close()
            self._combo_pair.setCurrentText(name)
            self._btn_auto.setChecked(False)
            self.show_workflow('capture')
            self.capture_started.emit()  # Leave gallery review and return to the live stream.
            self._refresh_review_controls()
            return True
        except (ValueError, OSError, StopIteration) as exc:
            self._log('无法开始补拍：' + str(exc))
            return False

    def _register_review_candidate(self, name, fid, det_l, det_r):
        task = self._review_task
        if not task or task['name'] != name:
            return True
        try:
            if task['scope'] != self._review_scope(name) or task['candidate']:
                raise ValueError('补拍上下文已变化或已有新图待确认')
            observations = [GuidanceObservation(np.asarray(det.charuco_corners, np.float32),
                                                np.asarray(det.charuco_ids, np.int32),
                                                det.image_size, fid) for det in (det_l, det_r)]
            state = deepcopy(self._review_state(name))
            state['pending'][fid] = frame_fingerprint(*observations)
            self._store_review_state(name, state)
            task['candidate'] = fid
            return True
        except (ValueError, OSError) as exc:
            self._log('补拍候选记录失败，本次未采集：' + str(exc))
            return False

    def _confirm_review_candidate(self):
        task = self._review_task
        if not task or not task['candidate'] or task['token'] not in self._saved_capture_tokens:
            self._log('新图尚未成功保存，不能应用替换')
            return
        previews = self._review_previews(task['name'])
        preview = next((p for p in previews if p.save_token == task['token']), None)
        if preview is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('确认补拍新图')
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"新图 {task['candidate']} · 原图验收通过并已保存"))
        layout.addWidget(PairImages(preview))
        note = QLabel(f"应用后，新图参与求解，旧图 {task['source']} 暂不参与；两份原图均保留。"
                      if task['source'] else '应用后，新图加入本次标定；已有图像参与状态不变。')
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QHBoxLayout()
        apply = QPushButton('应用替换，保留旧图' if task['source'] else '加入本次标定')
        apply.setProperty('class', 'primary')
        apply.clicked.connect(lambda: dialog.accept() if self._apply_review_candidate() else None)
        buttons.addWidget(apply)
        cancel = QPushButton('稍后确认')
        cancel.clicked.connect(dialog.reject)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        dialog.exec()

    def _apply_review_candidate(self):
        task = self._review_task
        if not task or not self._review_mutable() or task['token'] not in self._saved_capture_tokens:
            return False
        try:
            name, fid = task['name'], task['candidate']
            if task['scope'] != self._review_scope(name):
                raise ValueError('补拍配置已变化')
            raw = self._review_raw(name)
            state = deepcopy(self._review_state(name))
            if fid not in raw or state['pending'].get(fid) != raw[fid]['fingerprint']:
                raise ValueError('新图观测已变化或不再有效')
            source = task['source']
            if source and (source not in raw or raw[source]['fingerprint'] != task['fingerprint']):
                raise ValueError('旧图观测已变化，不能替换')
            state['pending'].pop(fid)
            if source:
                state['excluded'][source] = task['fingerprint']
            state['replacements'].append({'old_id': source, 'new_id': fid,
                                          'fingerprint': raw[fid]['fingerprint']})
            self._store_review_state(name, state)
            self._review_task = None
            self._undo_stack.clear()  # Historical replacements must never use destructive capture undo.
            self._refresh_undo_btn()
            self._review_changed(name)
            self.show_workflow('calibration')
            self._log(f'{name}: 新图 {fid} 已参与，旧图保留；请重新标定')
            return True
        except (ValueError, OSError) as exc:
            self._log('替换未应用，原状态保留：' + str(exc))
            return False

    def _cancel_review_task(self):
        task = self._review_task
        if task is None or self._pending_save_tokens:
            return
        if task['candidate']:
            # A saved but unconfirmed candidate remains excluded, including after restart.
            self._log('补拍新图保留为待确认，不参与求解；旧图参与状态不变')
        self._review_task = None
        self._refresh_review_controls()
        self._refresh_capture_source_ui()
        self.show_workflow('results')
        self._show_calibration_review(2)

    def _refresh_review_controls(self):
        if not hasattr(self, '_btn_improve'):
            return
        name = self._review_name()
        available = bool(name in self._stereo_cals) if name else False
        self._btn_improve.setEnabled(available and not self._calib_worker)
        report = self._review_report
        self._lbl_improvement.setText(
            f"建议复核 {report['flagged']} 对 · 补拍 {len(report['tasks'])} 类构图"
            if report and self._review_report_key is not None else '复核问题图像，补足构图后重新标定')
        self._btn_compare_review.setEnabled(bool(name in self._review_comparisons))
        task = self._review_task
        self._review_task_box.setVisible(task is not None)
        if task:
            self._btn_auto.setEnabled(False)
            self._btn_capture.setEnabled(not task['candidate'] and self._review_mutable())
        if task:
            candidate = task['candidate']
            self._lbl_review_task.setText(
                f"{'替换 ' + task['source'] if task['source'] else '补拍构图'} · "
                + task['target'].label + ('\n新图 ' + candidate + ' 待确认' if candidate else '')
                + '\n旧图状态不变；新图保存后再确认参与。')
            ready = bool(candidate and task['token'] in self._saved_capture_tokens)
            self._btn_confirm_review.setVisible(candidate is not None)
            self._btn_confirm_review.setEnabled(ready and self._review_mutable())
            self._btn_cancel_review.setEnabled(not self._pending_save_tokens)
        if self._review_dialog and self._review_worker is None and name:
            try:
                if self._review_report_key != self._review_key(self._review_dialog.name):
                    self._review_dialog.set_busy('数据已变化，请点击重新分析')
            except (ValueError, OSError, KeyError):
                self._review_dialog.set_busy('数据范围已变化，请重新打开复核')

    def _compare_review_results(self):
        name = self._review_name()
        comparison = self._review_comparisons.get(name)
        if comparison is None:
            return
        before, after = comparison
        def describe(pair):
            return (f'左 / 右内参 RMS：{pair.left_intrinsics.rms_error:.3f} / '
                    f'{pair.right_intrinsics.rms_error:.3f} px\n双目 RMS：'
                    f'{pair.stereo.rms_error:.3f} px\n'
                    + '参与配对：' + str(pair.calibration_info.get('selection', {})
                                         .get('stereo', {}).get('used_count', '未记录'))
                    + '\n同 ID 校正残差 RMS：'
                    + str(pair.calibration_info.get('rectification', {}).get('rms_px', '未记录'))
                    + ' px\n实际选用观测的 4×4 视场覆盖：'
                    + ' / '.join(f'{value:.0%}' if value is not None else '未记录'
                                 for value in pair.calibration_info.get('review', {}).get(
                                     'image_coverage', [None, None])))
        dialog = QDialog(self)
        dialog.setWindowTitle('与上次结果比较')
        layout = QVBoxLayout(dialog)
        for title, pair in (('上次成功结果', before), ('本次成功结果', after)):
            label = QLabel(title + '\n' + describe(pair))
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(label)
        note = QLabel('输入数据可能已变化。训练误差下降不能证明精度改善。\n固定验证集：未执行。')
        note.setWordWrap(True)
        layout.addWidget(note)
        close = QPushButton('关闭')
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)
        dialog.exec()
