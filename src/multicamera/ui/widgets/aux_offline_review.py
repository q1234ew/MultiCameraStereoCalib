"""Per-pair physical correspondence decisions over immutable offline images."""

from copy import deepcopy

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
                              QLabel, QListWidget, QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from ...calibration.offline_aux import (ORDER_OPTIONS, confirm_offline_pair,
                                       reordered_detection)
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
        self.check = QPushButton('检查对应')
        self.check.setIcon(app_icon('check'))
        self.check.setToolTip('根据固定相机之间的位姿一致性检查排列候选；不自动纳入观测')
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
        self._candidates = {row['source_id']: row for row in report.get('frames', [])}
        self._candidate_basis = {key: deepcopy(value) for key, value in report.items()
                                 if key != 'frames'}
        if report['status'] == 'ambiguous':
            text = '位姿共识不明确，未提供排列候选；请核对实物标记，补充不同倾角观测。'
        else:
            suspects = report.get('suspect_ids', [])
            text = ('原始角点编号方向待核对：' + '、'.join(suspects) if suspects else '原始检测排列未发现明显冲突')
            if report.get('unresolved_ids'):
                text += '；未获得可靠候选：' + '、'.join(report['unresolved_ids'])
            text += '。图像配对不变；几何候选不证明物理0点，采用后仍须逐对核对并确认纳入。'
        self.candidate_status.setText(text)
        self.set_checking(False)
        for index, (key, *_) in enumerate(self.records):
            text = f'{key} · ' + ('已确认纳入' if str(key) in self.decisions else '未纳入')
            if self._candidates.get(str(key), {}).get('suspect'):
                text += ' · 原排列待核对'
            self.list.item(index).setText(text)

    def set_check_error(self, message):
        self._candidates = {}
        self.candidate_status.setText('对应检查未完成：' + message)
        self.set_checking(False)

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
        self.confirm.setEnabled(review['image_accepted'])
        candidate = self._candidates.get(str(key))
        self.apply_candidate.setEnabled(bool(candidate) and not self._checking)
        self.apply_candidate.setToolTip('本对候选：RGB原顺序，辅助' +
            ORDER_OPTIONS[candidate['aux_order']] if candidate else '本对没有可靠的排列候选')
        self.details.setText(f'{key} · 原图质量：' + ('通过' if review['image_accepted'] else
            '未通过：' + '；'.join(review['image_reasons'])) + '\n'
            + ('原实时验收：' + str((review.get('capture_review') or {}).get('state', '无记录')))
            + ' · 离线不验证历史稳定性/同步')
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
        self.buttons.button(QDialogButtonBox.Save).setEnabled(
            bool(self.decisions) and self.binding.isChecked() and not self._checking)

    def accept(self):
        if self._checking or not self.binding.isChecked() or not self.decisions:
            self.candidate_status.setText('未应用：请至少确认纳入一对图像，并确认设备及标定板信息；检查期间请等待。')
            return
        super().accept()
