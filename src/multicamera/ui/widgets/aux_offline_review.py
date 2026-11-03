"""Per-pair physical correspondence decisions over immutable offline images."""

from copy import deepcopy

import cv2
from PySide6.QtCore import Qt
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
    def __init__(self, records, reviews, board, parent=None, *, decisions=None):
        super().__init__(parent)
        self.setWindowTitle('RGB + IR / 辅助 · 离线逐对复核')
        self.resize(1100, 760)
        self.records = list(records)
        self.reviews = deepcopy(reviews)
        self.board = deepcopy(board)
        self.decisions = deepcopy(decisions or {})
        self._index = 0
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
        self.binding = QCheckBox('确认本批 RGB 来自同一左相机、分辨率和镜头 / 焦距未改变，且标定板规格与当前配置一致')
        self.binding.setToolTip('缺少历史板记录的原图也须核对实物板型、检测点数和物理间距；图像本身不能证明毫米尺度。')
        outer.addWidget(self.binding)
        buttons.button(QDialogButtonBox.Save).setText('应用已确认观测')
        buttons.button(QDialogButtonBox.Cancel).setText('取消')
        buttons.button(QDialogButtonBox.Save).setEnabled(False)
        self.binding.toggled.connect(buttons.button(QDialogButtonBox.Save).setEnabled)
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
        decision = self.decisions.get(key, {})
        for side, combo in zip(('rgb', 'aux'), self.orders):
            combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(decision.get('corner_orders', {}).get(side, 'identity')))
            combo.blockSignals(False)
        self._render()

    def _order_changed(self):
        if not self.records:
            return
        key = str(self.records[self._index][0])
        self.decisions.pop(key, None)
        self.list.item(self._index).setText(f'{key} · 未纳入')
        self._render()

    def _render(self):
        key, rgb, aux, rgb_det, aux_det = self.records[self._index]
        review = self.reviews[str(key)]
        self.confirm.setEnabled(review['image_accepted'])
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
        self.list.item(self._index).setText(f'{key} · 已确认纳入')
        if self._index + 1 < len(self.records):
            self.list.setCurrentRow(self._index + 1)

    def exclude_current(self):
        key = str(self.records[self._index][0])
        self.decisions.pop(key, None)
        self.list.item(self._index).setText(f'{key} · 不纳入')
