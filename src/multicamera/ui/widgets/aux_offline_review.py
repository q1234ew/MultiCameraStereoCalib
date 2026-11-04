"""Per-pair physical correspondence decisions over immutable offline images."""

from copy import deepcopy

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
                              QLabel, QListWidget, QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from ...calibration.offline_aux import (ORDER_OPTIONS, confirm_offline_pair,
                                       detection_fingerprint, reordered_detection,
                                       review_order_check)
from ..theme import app_icon


class _ReviewImage(QLabel):
    def __init__(self):
        super().__init__()
        self._source = QPixmap()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(160, 150)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)

    def set_source(self, image):
        self._source = QPixmap.fromImage(image)
        self._fit()

    def _fit(self):
        if not self._source.isNull():
            self.setPixmap(self._source.scaled(self.size(), Qt.KeepAspectRatio,
                                               Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()


class AuxOfflineReviewDialog(QDialog):
    check_requested = Signal()

    def __init__(self, records, reviews, board, parent=None, *, decisions=None):
        super().__init__(parent)
        self.setWindowTitle('RGB + IR / 辅助 · 离线逐对复核')
        self.resize(1100, 760)
        self.records = list(records)
        self.reviews = deepcopy(reviews)
        self.board = deepcopy(board)
        self.decisions = deepcopy(decisions or {})
        self._draft_orders = {key: deepcopy(value.get('corner_orders', {}))
                              for key, value in self.decisions.items()}
        self._index = 0
        self._candidates = {}
        self._candidate_basis = {}
        self._correspondence_report = None
        self._requires_check = sum(review['image_accepted'] for review in self.reviews.values()) >= 3
        self._input_fingerprints = {str(key): [detection_fingerprint(rgb),
                                              detection_fingerprint(aux)]
                                    for key, _, _, rgb, aux in self.records}
        self._applied_candidates = {key: deepcopy(value['correspondence_aid'])
            for key, value in self.decisions.items() if value.get('correspondence_aid')}
        self._checking = False
        outer = QVBoxLayout(self)
        self.summary = QLabel('离线候选复核 · 不改写原采图验收 · 时间稳定性和同步未验证')
        self.summary.setWordWrap(True)
        outer.addWidget(self.summary)
        body = QHBoxLayout()
        self.list = QListWidget()
        self.list.setMaximumWidth(190)
        self.list.setAccessibleName('待复核图像对')
        for key, *_ in self.records:
            self.list.addItem(f'{key} · ' + ('已确认纳入' if str(key) in self.decisions else '未纳入'))
        body.addWidget(self.list)
        images = QHBoxLayout()
        self.views, self.orders = [], []
        for name in ('RGB 左目', 'IR / 辅助'):
            column = QVBoxLayout()
            column.addWidget(QLabel(name))
            image = _ReviewImage()
            column.addWidget(image, 1)
            order = QComboBox()
            for key, text in ORDER_OPTIONS.items():
                order.addItem(text, key)
            order.setAccessibleName(f'{name}角点排列')
            column.addWidget(order)
            self.views.append(image)
            self.orders.append(order)
            pane = QWidget()
            pane.setLayout(column)
            images.addWidget(pane, 1)
        body.addLayout(images, 1)
        outer.addLayout(body, 1)
        self.details = QLabel()
        self.details.setWordWrap(True)
        outer.addWidget(self.details)
        self.attestation = QLabel('确认内容：两图 0 点、X 行方向、Y 列方向对应同一物理板点。'
                                 '对称板无法仅凭检测或低 RMS 自动证明；无法辨认时不要纳入。')
        self.attestation.setWordWrap(True)
        outer.addWidget(self.attestation)
        inspection = QHBoxLayout()
        self.check = QPushButton('检查原始编号')
        self.check.setIcon(app_icon('check'))
        self.check.setToolTip('只检查原始检测编号，不应用排列；采用候选后仍须核对0点和X/Y方向并确认')
        self.check.clicked.connect(self.check_requested.emit)
        self.apply_candidate = QPushButton('采用本对候选排列')
        self.apply_candidate.clicked.connect(self._apply_candidate)
        self.apply_candidate.setEnabled(False)
        inspection.addWidget(self.check)
        inspection.addWidget(self.apply_candidate)
        inspection.addStretch()
        outer.addLayout(inspection)
        self.candidate_status = QLabel('对应候选尚未检查；请逐对核对物理0点及X/Y方向。')
        self.candidate_status.setWordWrap(True)
        outer.addWidget(self.candidate_status)
        self.selection_status = QLabel()
        self.selection_status.setWordWrap(True)
        outer.addWidget(self.selection_status)
        commands = QHBoxLayout()
        self.confirm = QPushButton('确认本对物理对应并纳入')
        self.confirm.setIcon(app_icon('check'))
        self.confirm.clicked.connect(self.confirm_current)
        self.exclude = QPushButton('不纳入本对')
        self.exclude.clicked.connect(self.exclude_current)
        commands.addWidget(self.confirm)
        commands.addWidget(self.exclude)
        outer.addLayout(commands)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.buttons = buttons
        self.binding = QCheckBox('确认本批 RGB 来自同一左相机、分辨率和镜头 / 焦距未改变，且标定板规格与当前配置一致')
        self.binding.setToolTip('缺少历史板记录的原图也须核对实物板型、检测点数和物理间距；图像本身不能证明毫米尺度。')
        outer.addWidget(self.binding)
        buttons.button(QDialogButtonBox.Save).setText('应用已确认观测')
        buttons.button(QDialogButtonBox.Cancel).setText('取消')
        buttons.button(QDialogButtonBox.Save).setEnabled(False)
        self.binding.toggled.connect(self._update_apply_state)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self.list.currentRowChanged.connect(self._select)
        for order in self.orders:
            order.currentIndexChanged.connect(self._order_changed)
        self.list.setCurrentRow(0)

    def _select(self, index):
        if not 0 <= index < len(self.records):
            return
        self._index = index
        key = str(self.records[index][0])
        orders = self._draft_orders.get(key, {})
        for side, combo in zip(('rgb', 'aux'), self.orders):
            combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(orders.get(side, 'identity')))
            combo.blockSignals(False)
        self._render()

    def set_checking(self, checking):
        self._checking = checking
        self.check.setEnabled(not checking)
        if checking:
            self.candidate_status.setText('正在后台检查对应，原图和纳入决定保持不变…')
        self._render()

    def set_correspondence_report(self, report):
        self._correspondence_report = deepcopy(report)
        self._candidates = {row['source_id']: row for row in report.get('frames', [])}
        self._candidate_basis = {key: deepcopy(value) for key, value in report.items()
                                 if key != 'frames'}
        if report['status'] == 'ambiguous':
            text = '位姿共识不明确，未提供排列候选；请核对实物标记，补充不同倾角观测。'
            if report.get('refinement_error'):
                text += ' 几何细化未通过，具体原因见操作日志；请同时核对辅助模型。'
        else:
            suspects = report.get('suspect_ids', [])
            text = ('原始编号与几何候选不同：' + '、'.join(suspects) if suspects
                    else '原始检测排列未发现明显冲突')
            if report.get('unresolved_ids'):
                text += '；未获得可靠候选：' + '、'.join(report['unresolved_ids'])
            text += '。这是原始检测记录；当前选择及冲突见下方。几何候选不证明物理0点，仍须逐对核对并确认。'
            if report.get('diagnostic_refinement'):
                text += ' 本批候选已通过算法几何细化；其参数不复用到训练折。'
        self.candidate_status.setText(text)
        self.set_checking(False)

    def set_check_error(self, message):
        self.candidate_status.setText('对应检查未完成：' + message
                                     + '；已有检查和编号冲突保留，未应用任何调整。')
        self.set_checking(False)

    def correspondence_check(self):
        if self._correspondence_report is None:
            return None
        return dict(report=deepcopy(self._correspondence_report),
                    input_fingerprints=deepcopy(self._input_fingerprints))

    def _apply_candidate(self):
        candidate = self._candidates.get(str(self.records[self._index][0]))
        if candidate is None or self._checking:
            return
        # Suggestions refer to raw detections, not the currently selected ordering.
        for combo, order in zip(self.orders, ('identity', candidate['aux_order'])):
            combo.setCurrentIndex(combo.findData(order))
        self._order_changed()
        self._applied_candidates[str(self.records[self._index][0])] = {
            **deepcopy(self._candidate_basis), 'candidate': deepcopy(candidate)}

    def _order_changed(self):
        if not self.records:
            return
        key = str(self.records[self._index][0])
        self._draft_orders[key] = {side: combo.currentData()
                                  for side, combo in zip(('rgb', 'aux'), self.orders)}
        self.decisions.pop(key, None)
        self._applied_candidates.pop(key, None)
        self.list.item(self._index).setText(f'{key} · 未纳入')
        self._render()

    def _render(self):
        key, rgb, aux, rgb_det, aux_det = self.records[self._index]
        review = self.reviews[str(key)]
        self._update_apply_state()
        self.confirm.setEnabled(review['image_accepted'] and not self._checking)
        candidate = self._candidates.get(str(key))
        self.apply_candidate.setEnabled(bool(candidate) and not self._checking)
        self.apply_candidate.setToolTip('本对候选：RGB原顺序，辅助' +
            ORDER_OPTIONS[candidate['aux_order']] if candidate else '本对没有可靠的排列候选')
        self.details.setText(f'{key} · 原图质量：' + ('通过' if review['image_accepted'] else
            '未通过：' + '；'.join(review['image_reasons'])) + '\n'
            + ('原实时验收：' + str((review.get('capture_review') or {}).get('state', '无记录')))
            + ' · 离线不验证历史稳定性/同步')
        current_orders = {side: combo.currentData()
                          for side, combo in zip(('rgb', 'aux'), self.orders)}
        state = review_order_check({str(key): {'corner_orders': current_orders}},
                                   self._correspondence_report, self.board)
        if state['invalid_order_ids']:
            selected = '当前板型不支持此非刚性排列；请恢复合法排列后重新确认'
        elif state['conflict_ids']:
            selected = '与几何候选冲突；请调整排列并重新确认，或不纳入本对'
        elif state['matched_ids']:
            selected = '与几何候选一致；仍须核对物理0点和X/Y方向'
        else:
            selected = '无可靠几何候选；仅凭检测不能证明物理对应'
        self.details.setText(self.details.text() + '\n当前选择：RGB '
            + ORDER_OPTIONS[current_orders['rgb']] + ' / IR辅助 '
            + ORDER_OPTIONS[current_orders['aux']] + '；' + selected)
        for image, det, combo, label in zip((rgb, aux), (rgb_det, aux_det), self.orders, self.views):
            canvas = image.copy()
            if det.valid:
                ordered = reordered_detection(det, self.board, combo.currentData())
                points = ordered.image_points
                scale = max(.4, image.shape[1] / 1000)
                origin = tuple(points[0].round().astype(int))
                for index, color, text in ((1, (0, 220, 255), 'X'),
                                            (self.board.cols, (255, 210, 0), 'Y')):
                    end = tuple(points[index].round().astype(int))
                    cv2.arrowedLine(canvas, origin, end, color, max(1, round(scale * 2)))
                    cv2.putText(canvas, text, end, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
                cv2.circle(canvas, origin, max(3, round(scale * 6)), (0, 255, 255), 2)
                cv2.putText(canvas, '0', origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)
            rgb_image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            h, w = rgb_image.shape[:2]
            qimage = QImage(rgb_image.data, w, h, rgb_image.strides[0], QImage.Format_RGB888)
            label.set_source(qimage)

    def confirm_current(self):
        if self._checking or not self.reviews[str(self.records[self._index][0])]['image_accepted']:
            return
        key = str(self.records[self._index][0])
        self.decisions[key] = confirm_offline_pair(self.reviews[key],
            self.orders[0].currentData(), self.orders[1].currentData())
        aid = self._applied_candidates.get(key)
        if aid and self.decisions[key]['corner_orders'] == {
                'rgb': 'identity', 'aux': aid['candidate']['aux_order']}:
            self.decisions[key]['correspondence_aid'] = deepcopy(aid)
        self._draft_orders[key] = deepcopy(self.decisions[key]['corner_orders'])
        self.list.item(self._index).setText(f'{key} · 已确认纳入')
        self._update_apply_state()
        if self._index + 1 < len(self.records):
            self.list.setCurrentRow(self._index + 1)

    def exclude_current(self):
        key = str(self.records[self._index][0])
        self.decisions.pop(key, None)
        self.list.item(self._index).setText(f'{key} · 不纳入')
        self._update_apply_state()

    def _update_apply_state(self):
        state = review_order_check(self.decisions, self._correspondence_report, self.board)
        conflicts = state['conflict_ids']
        needs_check = self._requires_check and self._correspondence_report is None
        self.selection_status.setText(
            f'已确认 {len(self.decisions)} 组 · 当前编号冲突 {len(conflicts)} 组'
            f' · 无可靠几何候选 {len(state["unchecked_ids"])} 组'
            + ('\n未应用：请先点击“检查原始编号”，检查本批候选。' if needs_check else
               '\n未应用：请复核 ' + '、'.join(conflicts) + '，或将其设为不纳入。'
               if conflicts else '\n排列调整仅在“应用已确认观测”后送入标定；取消不生效。'))
        for index, (key, *_) in enumerate(self.records):
            key = str(key)
            text = f'{key} · ' + ('已确认纳入' if key in self.decisions else '未纳入')
            if key in conflicts:
                text = f'{key} · 当前编号冲突'
            elif key in state['matched_ids']:
                text = f'{key} · 候选一致'
            self.list.item(index).setText(text)
            orders = self._draft_orders.get(key, {})
            self.list.item(index).setToolTip(text + '\nRGB：'
                + ORDER_OPTIONS.get(orders.get('rgb', 'identity'), '未知')
                + '\nIR辅助：' + ORDER_OPTIONS.get(orders.get('aux', 'identity'), '未知')
                + '\n是否纳入以本批已确认观测为准；几何候选不证明物理对应。')
        self.buttons.button(QDialogButtonBox.Save).setEnabled(
            bool(self.decisions) and self.binding.isChecked() and not self._checking
            and not conflicts and not needs_check)

    def accept(self):
        if self._checking or not self.binding.isChecked() or not self.decisions:
            self.candidate_status.setText('未应用：请至少确认纳入一对图像，并确认设备及标定板信息；检查期间请等待。')
            return
        if self._requires_check and self._correspondence_report is None:
            self.candidate_status.setText('未应用：请先点击“检查原始编号”，再核对并应用本批观测。')
            return
        conflicts = review_order_check(self.decisions, self._correspondence_report,
                                       self.board)['conflict_ids']
        if conflicts:
            self.candidate_status.setText('未应用：当前编号仍与几何候选冲突：' + '、'.join(conflicts)
                + '。仅点击检查不会调整编号；请核对后重新确认，或不纳入冲突观测。')
            self.list.setCurrentRow(next(index for index, row in enumerate(self.records)
                                         if str(row[0]) == conflicts[0]))
            return
        super().accept()
