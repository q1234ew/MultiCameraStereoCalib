"""Camera group configuration and ChArUco board parameter dialog."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from urllib.parse import urlsplit

from PySide6.QtCore import QEvent, Qt, Slot, QSignalBlocker
from PySide6.QtWidgets import (
    QAbstractItemView,
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStyledItemDelegate,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...board.charuco_board import ArucoDictType, CharucoBoard, CharucoBoardConfig
from ...calibration.models import CameraModel
from ...calibration.planar import PatternBoardConfig, PatternType
from ...io.board_config import load_pattern_board_config
from ..configuration import pair_signature
from ..theme import TEXT_DIM, DANGER, app_icon
from ...streaming.discovery import DiscoveredService, DiscoveryWorker
from ...streaming.stream_manager import (
    AuxiliaryCameraConfig,
    CameraConfig,
    StereoPairConfig,
    StreamManager,
)

logger = logging.getLogger(__name__)

_SOURCE_LABELS = {"mdns": "mDNS", "probe": "扫描"}
_STREAM_TYPE_LABELS = {
    "rgb": "RGB",
    "ir": "红外",
    "acoustic_rgb": "声像仪 RGB",
    "control": "控制台",
    "unknown": "未知",
}
_EYE_LABELS = {
    "left": "左目",
    "right": "右目",
    "unknown": "未知",
}


class ConfigurationError(ValueError):
    def __init__(self, message, widget, page):
        super().__init__(message)
        self.widget, self.page = widget, page


class StreamTypeDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        editor = QComboBox(parent)
        editor.addItems(["RGB", "红外"])
        return editor

    def setEditorData(self, editor, index):
        editor.setCurrentText(index.data())

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText(), Qt.EditRole)


class _ResponsiveRow(QWidget):
    """Keep complete fields together, stacking them before their minimum widths collide."""

    def __init__(self, items):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._items = [widget for widget, _ in items]
        self._box = QBoxLayout(QBoxLayout.LeftToRight, self)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(8)
        for widget, stretch in items:
            self._box.addWidget(widget, stretch)

    def _reflow(self):
        margins = self._box.contentsMargins()
        required = sum(max(widget.minimumSizeHint().width(), widget.minimumWidth())
                       for widget in self._items if not widget.isHidden())
        count = sum(not widget.isHidden() for widget in self._items)
        required += max(0, count - 1) * self._box.spacing()
        available = self.width() - margins.left() - margins.right()
        direction = (QBoxLayout.LeftToRight if available >= required
                     else QBoxLayout.TopToBottom)
        if self._box.direction() != direction:
            self._box.setDirection(direction)

    def event(self, event):
        result = super().event(event)
        if hasattr(self, '_box') and event.type() in (
                QEvent.Resize, QEvent.LayoutRequest, QEvent.Show, QEvent.FontChange):
            self._reflow()
        return result


class ConfigDialog(QDialog):
    """Dialog for configuring stereo pairs, camera model, and ChArUco board."""

    def __init__(
        self,
        stream_manager: StreamManager,
        board: CharucoBoard,
        camera_model: CameraModel = CameraModel.PINHOLE,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("相机与标定板配置")
        self.setMinimumSize(900, 480)
        self.resize(1080, 760)
        self._stream_manager = stream_manager
        self._board = board
        self._camera_model = camera_model
        self.before_apply = None
        self.streams_changed = False
        self._has_scanned = False

        self._init_ui()
        self._load_current()
        try:
            self._initial_configuration = self._configuration_signature(self._proposed_configuration())
        except ConfigurationError:
            self._initial_configuration = None
        self._connect_validation()
        self._refresh_validation()

    @property
    def board(self) -> CharucoBoard:
        return self._board

    @property
    def camera_model(self) -> CameraModel:
        return self._camera_model

    def focus_auxiliary(self):
        self._tabs.setCurrentIndex(1)
        if not self._combo_fusion_mode.currentData():
            self._combo_fusion_mode.setCurrentIndex(self._combo_fusion_mode.findData("ir"))
        self._edit_aux_url.setFocus()

    def prefill_session_urls(self, metadata):
        """Fill only empty transports for exact existing logical camera bindings."""
        pairs = {pair['name']: pair for pair in metadata.get('pairs', [])}
        for row in range(self._table.rowCount()):
            binding = pairs.get(self._item_text(row, 0))
            if binding is None or (self._item_text(row, 2), self._item_text(row, 4)) != (
                    binding.get('left_id'), binding.get('right_id')):
                continue
            for column, key in ((3, 'left_url'), (5, 'right_url')):
                if not self._item_text(row, column) and binding.get(key):
                    self._table.item(row, column).setText(binding[key])
        aux = metadata.get('auxiliary_camera') or {}
        if self._edit_aux_id.text().strip() == aux.get('camera_id') and not self._edit_aux_url.text().strip():
            self._edit_aux_url.setText(aux.get('url', ''))
        self._refresh_validation()

    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._tabs = QTabWidget()
        outer.addWidget(self._tabs, 1)
        pairs = self._create_pairs_group()
        auxiliary = self._create_multimodal_group()
        self._add_page("相机连接", [pairs], stretch=True)
        self._add_page("融合相机", [auxiliary])
        self._board_columns = QWidget()
        columns = QHBoxLayout(self._board_columns)
        self._board_columns_layout = columns
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(16)
        self._rgb_board_group = self._create_board_group()
        columns.addWidget(self._rgb_board_group, 1)
        columns.addWidget(self._aux_board_group, 1)
        self._board_scroll = self._add_page("标定板", [self._create_model_group(), self._board_columns])
        self._board_viewport = self._board_scroll.viewport()
        self._board_viewport.installEventFilter(self)
        self._tabs.currentChanged.connect(self._update_board_columns)

        footer = QFrame()
        footer.setObjectName("actionFooter")
        row = QVBoxLayout(footer)
        row.setContentsMargins(16, 12, 16, 12)
        row.setSpacing(8)
        self._lbl_config_status = QLabel()
        self._lbl_config_status.setWordWrap(True)
        self._lbl_config_status.setTextFormat(Qt.PlainText)
        row.addWidget(self._lbl_config_status)
        self._lbl_next_step = QLabel()
        self._lbl_next_step.setWordWrap(True)
        self._lbl_next_step.setStyleSheet(f"color:{TEXT_DIM}; background:transparent;")
        row.addWidget(self._lbl_next_step)
        actions = QHBoxLayout()
        self._btn_next_page = QPushButton("下一步：标定板")
        self._btn_next_page.setIcon(app_icon("chevron-right"))
        self._btn_next_page.setAutoDefault(False)
        self._btn_next_page.clicked.connect(lambda: self._tabs.setCurrentIndex(2))
        actions.addWidget(self._btn_next_page)
        actions.addStretch()
        self._buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self._buttons.button(QDialogButtonBox.Save).setText("应用配置")
        self._buttons.button(QDialogButtonBox.Save).setIcon(app_icon("check"))
        self._buttons.button(QDialogButtonBox.Save).setProperty("class", "primary")
        self._buttons.button(QDialogButtonBox.Save).setDefault(True)
        self._buttons.button(QDialogButtonBox.Cancel).setText("取消")
        self._buttons.button(QDialogButtonBox.Cancel).setIcon(app_icon("x"))
        self._buttons.button(QDialogButtonBox.Cancel).setAutoDefault(False)
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        actions.addWidget(self._buttons)
        row.addLayout(actions)
        outer.addWidget(footer)
        self._tabs.currentChanged.connect(self._update_next_step)
        self._update_next_step()

    def _update_next_step(self, *_):
        page = self._tabs.currentIndex()
        self._btn_next_page.setVisible(page != 2)
        self._lbl_next_step.setText(
            "下一步：核对标定板参数，应用配置后返回采集。" if page != 2 else
            "下一步：应用配置并返回采集；取消不保留本次修改。"
        )

    def _add_page(self, title, widgets, stretch=False):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        for widget in widgets:
            layout.addWidget(widget, 1 if stretch else 0)
        if not stretch:
            layout.addStretch()
        scroll.setWidget(body)
        self._tabs.addTab(scroll, title)
        return scroll

    def _update_board_columns(self, *_):
        # Use the viewport, not an already-overflowing content widget's width.
        margins = self._board_scroll.widget().layout().contentsMargins()
        available = self._board_viewport.width() - margins.left() - margins.right()
        required = (self._rgb_board_group.minimumSizeHint().width()
                    + self._aux_board_group.minimumSizeHint().width()
                    + self._board_columns_layout.spacing())
        direction = (QBoxLayout.LeftToRight if available >= required + 24
                     else QBoxLayout.TopToBottom)
        if self._board_columns_layout.direction() != direction:
            self._board_columns_layout.setDirection(direction)

    def eventFilter(self, watched, event):
        if watched is getattr(self, '_board_viewport', None) and event.type() == QEvent.Resize:
            self._update_board_columns()
        return super().eventFilter(watched, event)

    @staticmethod
    def _configuration_form(group):
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.DontWrapRows)
        form.setFormAlignment(Qt.AlignTop)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setContentsMargins(12, 10, 12, 12)
        form.setSpacing(8)
        return form

    def _create_model_group(self) -> QGroupBox:
        group = QGroupBox("相机模型")
        form = self._configuration_form(group)

        self._combo_model = QComboBox()
        self._combo_model.addItem("针孔 (Pinhole)", CameraModel.PINHOLE)
        self._combo_model.addItem("鱼眼 (Fisheye)", CameraModel.FISHEYE)
        form.addRow("镜头类型:", self._combo_model)

        self._lbl_model_hint = QLabel("双目与辅助内参共用镜头模型")
        self._lbl_model_hint.setStyleSheet(f"color:{TEXT_DIM};")
        self._lbl_model_hint.setWordWrap(True)
        form.addRow(self._lbl_model_hint)

        return group

    def _create_multimodal_group(self) -> QGroupBox:
        group = QGroupBox("融合辅助单目")
        form = self._configuration_form(group)

        self._combo_fusion_mode = QComboBox()
        self._combo_fusion_mode.addItem("不启用", "")
        self._combo_fusion_mode.addItem("RGB 双目 + IR 单目", "ir")
        self._combo_fusion_mode.addItem("RGB 双目 + 声像仪 RGB 单目", "acoustic_rgb")
        self._combo_fusion_mode.currentIndexChanged.connect(self._update_aux_enabled)
        self._combo_fusion_mode.currentIndexChanged.connect(self._refresh_aux_dropdown)
        form.addRow("融合模式:", self._combo_fusion_mode)

        self._combo_aux_camera = QComboBox()
        self._combo_aux_camera.setPlaceholderText("暂无匹配辅助流")
        self._combo_aux_camera.setMinimumContentsLength(18)
        self._combo_aux_camera.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._btn_use_aux_service = QPushButton("使用发现服务")
        self._btn_use_aux_service.clicked.connect(self._use_selected_aux_service)
        self._btn_discover_aux = QPushButton("发现 / 刷新")
        self._btn_discover_aux.setIcon(app_icon("scan-line"))
        self._btn_discover_aux.clicked.connect(self._on_scan)
        self._aux_service_picker = _ResponsiveRow([
            (self._combo_aux_camera, 1), (self._btn_use_aux_service, 0),
            (self._btn_discover_aux, 0),
        ])
        form.addRow("已发现辅助流:", self._aux_service_picker)
        self._lbl_aux_discovery = QLabel("尚未扫描设备")
        self._lbl_aux_discovery.setWordWrap(True)
        self._lbl_aux_discovery.setStyleSheet(f"color:{TEXT_DIM}; font-size:12px;")
        form.addRow(self._lbl_aux_discovery)

        self._edit_aux_id = QLineEdit()
        self._edit_aux_id.setPlaceholderText("aux_cam")
        form.addRow("辅助相机 ID:", self._edit_aux_id)

        self._edit_aux_url = QLineEdit()
        self._edit_aux_url.setPlaceholderText("http://...")
        self._edit_aux_url.setToolTip("离线标定可留空，不会创建视频连接")
        self._edit_aux_url.textChanged.connect(self._refresh_aux_dropdown)
        form.addRow("辅助相机 URL:", self._edit_aux_url)

        self._lbl_aux_reference = QLineEdit()
        self._lbl_aux_reference.setReadOnly(True)
        self._lbl_aux_reference.setMinimumHeight(32)
        self._lbl_aux_reference.setAccessibleName("RGB 左目参考相机（只读）")
        form.addRow("参考相机:", self._lbl_aux_reference)
        self._btn_aux_board_page = QPushButton("辅助标定板设置...")
        self._btn_aux_board_page.setIcon(app_icon("chevron-right"))
        self._btn_aux_board_page.clicked.connect(lambda: self._tabs.setCurrentIndex(2))
        form.addRow(self._btn_aux_board_page)

        self._aux_board_group = QGroupBox("红外 / 辅助平面板（独立配置）")
        form = self._configuration_form(self._aux_board_group)

        self._combo_aux_board = QComboBox()
        self._combo_aux_board.addItem("棋盘格 (chessboard)", "chessboard")
        self._combo_aux_board.addItem("圆点阵 (circles_grid)", "circles_grid")
        form.addRow("平面板类型:", self._combo_aux_board)

        self._combo_aux_count_mode = QComboBox()
        self._combo_aux_count_mode.addItem("格子数量", "squares")
        self._combo_aux_count_mode.addItem("内角点数量", "points")
        self._aux_display_offset = 1
        self._combo_aux_count_mode.currentIndexChanged.connect(self._update_aux_dimensions)
        form.addRow("棋盘计数方式:", self._combo_aux_count_mode)

        points_row = QHBoxLayout()
        self._spin_aux_cols = QSpinBox()
        self._spin_aux_cols.setRange(2, 101)
        self._spin_aux_rows = QSpinBox()
        self._spin_aux_rows.setRange(2, 101)
        points_row.addWidget(QLabel("列:"))
        points_row.addWidget(self._spin_aux_cols)
        points_row.addSpacing(12)
        points_row.addWidget(QLabel("行:"))
        points_row.addWidget(self._spin_aux_rows)
        self._lbl_aux_dimensions = QLabel("格子数量:")
        form.addRow(self._lbl_aux_dimensions, points_row)

        self._spin_aux_spacing = QDoubleSpinBox()
        self._spin_aux_spacing.setRange(0.001, 10000)
        self._spin_aux_spacing.setDecimals(3)
        self._spin_aux_spacing.setSuffix(" mm")
        form.addRow("格子边长 / 点间距:", self._spin_aux_spacing)

        self._check_aux_asymmetric = QCheckBox("非对称圆点阵")
        self._check_aux_asymmetric.toggled.connect(self._refresh_aux_board_summary)
        form.addRow(self._check_aux_asymmetric)

        board_buttons = QHBoxLayout()
        self._btn_aux_board_preset = QPushButton("红外 7×5 / 80 mm")
        self._btn_aux_board_preset.setToolTip("恢复红外预设：7 列 × 5 行格子，每格 80 mm，6×4 内角点")
        self._btn_aux_board_preset.clicked.connect(
            lambda: self._set_aux_board_values(PatternBoardConfig.ir_chessboard())
        )
        self._btn_import_aux_board = QPushButton("导入板参数...")
        self._btn_import_aux_board.clicked.connect(self._on_import_aux_board)
        board_buttons.addWidget(self._btn_aux_board_preset)
        board_buttons.addWidget(self._btn_import_aux_board)
        form.addRow(board_buttons)

        self._lbl_aux_board_summary = QLabel()
        self._lbl_aux_board_summary.setWordWrap(True)
        form.addRow(self._lbl_aux_board_summary)
        for spin in (self._spin_aux_cols, self._spin_aux_rows, self._spin_aux_spacing):
            spin.valueChanged.connect(self._refresh_aux_board_summary)
        self._combo_aux_board.currentIndexChanged.connect(self._update_aux_enabled)
        self._combo_aux_board.currentIndexChanged.connect(self._refresh_aux_board_summary)
        self._set_aux_board_values(PatternBoardConfig.ir_chessboard())

        return group

    def _update_aux_dimensions(self, *_):
        chessboard = self._combo_aux_board.currentData() == "chessboard"
        offset = int(chessboard and self._combo_aux_count_mode.currentData() == "squares")
        difference = offset - self._aux_display_offset
        for spin in (self._spin_aux_cols, self._spin_aux_rows):
            with QSignalBlocker(spin):
                spin.setValue(spin.value() + difference)
        self._aux_display_offset = offset
        self._lbl_aux_dimensions.setText("格子数量:" if offset else "内角点数量:" if chessboard else "圆心数量:")
        self._combo_aux_count_mode.setEnabled(bool(self._combo_fusion_mode.currentData()) and chessboard)
        self._refresh_aux_board_summary()

    def _aux_board_values(self) -> PatternBoardConfig:
        pattern = self._combo_aux_board.currentData()
        return PatternBoardConfig(
            cols=self._spin_aux_cols.value() - self._aux_display_offset,
            rows=self._spin_aux_rows.value() - self._aux_display_offset,
            square_size=self._spin_aux_spacing.value() / 1000.0,
            pattern_type=pattern,
            asymmetric=pattern == "circles_grid" and self._check_aux_asymmetric.isChecked(),
        )

    def _set_aux_board_values(self, board: PatternBoardConfig):
        # Validate before replacing the current values; an invalid import changes nothing.
        board = PatternBoardConfig.from_dict(board.to_dict())
        spacing_mm = board.square_size * 1000.0
        if not 0.001 <= spacing_mm <= 10000 or not math.isclose(
            spacing_mm, round(spacing_mm, 3), rel_tol=0, abs_tol=1e-8,
        ):
            raise ValueError("间距必须在 0.001–10000 mm 范围内，精度为 0.001 mm")
        with QSignalBlocker(self._combo_aux_board):
            self._combo_aux_board.setCurrentIndex(self._combo_aux_board.findData(board.pattern_type.value))
        self._aux_display_offset = int(board.pattern_type == PatternType.CHESSBOARD and
                                       self._combo_aux_count_mode.currentData() == "squares")
        self._spin_aux_cols.setValue(board.cols + self._aux_display_offset)
        self._spin_aux_rows.setValue(board.rows + self._aux_display_offset)
        self._spin_aux_spacing.setValue(spacing_mm)
        self._check_aux_asymmetric.setChecked(board.asymmetric)
        self._update_aux_enabled()
        self._refresh_aux_board_summary()

    def _refresh_aux_board_summary(self, *_):
        cols = self._spin_aux_cols.value() - self._aux_display_offset
        rows = self._spin_aux_rows.value() - self._aux_display_offset
        spacing = self._spin_aux_spacing.value()
        if self._combo_aux_board.currentData() == "chessboard":
            text = f"棋盘格：{cols + 1} 列 × {rows + 1} 行格子；{cols} × {rows} 内角点，共 {cols * rows} 点"
        else:
            text = f"圆点阵：{cols} 列 × {rows} 行圆心，共 {cols * rows} 点"
        self._lbl_aux_board_summary.setText(f"{text}；间距 {spacing:g} mm")

    def _on_import_aux_board(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "导入辅助/红外标定板参数", "",
            "板参数 JSON / OpenCV YAML (*.json *.yaml *.yml)",
        )
        if not path:
            return
        try:
            board = load_pattern_board_config(path)
            self._set_aux_board_values(board)
        except Exception as exc:
            logger.warning("Auxiliary board import failed: %s", path, exc_info=True)
            QMessageBox.warning(self, "标定板参数导入失败", str(exc).splitlines()[0])
            return
        self._lbl_aux_board_summary.setToolTip(f"导入来源：{Path(path).name}")
        logger.info("Imported auxiliary board: file=%s board=%s", path, board.to_dict())

    def _create_board_group(self) -> QGroupBox:
        group = QGroupBox("ChArUco 标定板")
        form = self._configuration_form(group)

        row_size = QHBoxLayout()
        self._spin_cols = QSpinBox()
        self._spin_cols.setRange(3, 30)
        row_size.addWidget(QLabel("列数:"))
        row_size.addWidget(self._spin_cols)
        row_size.addSpacing(12)
        self._spin_rows = QSpinBox()
        self._spin_rows.setRange(3, 30)
        row_size.addWidget(QLabel("行数:"))
        row_size.addWidget(self._spin_rows)
        form.addRow("格子数量:", row_size)

        self._spin_sq_len = QDoubleSpinBox()
        self._spin_sq_len.setDecimals(3)
        self._spin_sq_len.setRange(0.001, 1000.0)
        self._spin_sq_len.setSuffix(" mm")
        self._spin_sq_len.setSingleStep(1)
        form.addRow("方格边长:", self._spin_sq_len)
        self._spin_mk_len = QDoubleSpinBox()
        self._spin_mk_len.setDecimals(3)
        self._spin_mk_len.setRange(0.001, 1000.0)
        self._spin_mk_len.setSuffix(" mm")
        self._spin_mk_len.setSingleStep(1)
        form.addRow("Marker 边长:", self._spin_mk_len)

        self._combo_dict = QComboBox()
        for d in ArucoDictType:
            self._combo_dict.addItem(d.name, d)
        form.addRow("ArUco 字典:", self._combo_dict)

        self._lbl_rgb_board_summary = QLabel()
        self._lbl_rgb_board_summary.setWordWrap(True)
        form.addRow(self._lbl_rgb_board_summary)

        return group

    def _create_pairs_group(self) -> QGroupBox:
        group = QGroupBox("双目相机组")
        group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        # ── discovery controls ──
        disc_row = QHBoxLayout()
        self._btn_scan = QPushButton("发现设备")
        self._btn_scan.setIcon(app_icon("scan-line"))
        self._btn_scan.setToolTip("通过 mDNS/Zeroconf 和子网扫描发现局域网中的推流服务")
        self._btn_scan.clicked.connect(self._on_scan)
        disc_row.addWidget(self._btn_scan)

        self._chk_mdns = QCheckBox("mDNS")
        self._chk_mdns.setChecked(True)
        self._chk_mdns.setToolTip("使用 mDNS/Zeroconf 协议被动发现已广播的服务（推荐）")
        disc_row.addWidget(self._chk_mdns)
        self._chk_probe = QCheckBox("子网扫描")
        self._chk_probe.setChecked(True)
        self._chk_probe.setToolTip("主动探测局域网内常见端口（当设备不支持 mDNS 时使用）")
        disc_row.addWidget(self._chk_probe)

        self._scan_progress = QProgressBar()
        self._scan_progress.setMaximumHeight(16)
        self._scan_progress.setTextVisible(True)
        self._scan_progress.setFormat("%v/%m 主机")
        self._scan_progress.setVisible(False)
        disc_row.addWidget(self._scan_progress, 1)
        disc_row.addStretch()

        self._lbl_scan_status = QLabel("")
        layout.addLayout(disc_row)
        self._lbl_scan_status.setWordWrap(True)
        self._lbl_scan_status.setStyleSheet(f"color:{TEXT_DIM};")
        layout.addWidget(self._lbl_scan_status)

        # ── discovered camera dropdowns ──
        self._combo_stream_type = QComboBox()
        self._combo_stream_type.addItem("RGB 双目", "rgb")
        self._combo_stream_type.addItem("红外双目", "ir")
        self._combo_stream_type.currentIndexChanged.connect(self._refresh_camera_dropdowns)
        self._combo_left_camera = QComboBox()
        self._combo_left_camera.setPlaceholderText("暂无左目服务")
        self._combo_left_camera.setMinimumContentsLength(12)
        self._combo_left_camera.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._combo_right_camera = QComboBox()
        self._combo_right_camera.setPlaceholderText("暂无右目服务")
        self._combo_right_camera.setMinimumContentsLength(12)
        self._combo_right_camera.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)

        def field(title, control):
            widget = QWidget()
            row = QHBoxLayout(widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            row.addWidget(QLabel(title))
            row.addWidget(control, 1)
            return widget

        self._stereo_service_picker = _ResponsiveRow([
            (field("类型:", self._combo_stream_type), 0),
            (field("左目:", self._combo_left_camera), 1),
            (field("右目:", self._combo_right_camera), 1),
        ])

        self._btn_add_selected_pair = QPushButton("添加相机组")
        self._btn_add_selected_pair.clicked.connect(self._add_pair_from_dropdowns)

        self._btn_auto_pair = QPushButton("自动配对")
        self._btn_auto_pair.setToolTip("仅配对同一主机、同类型且唯一的左右目；不推测未知目别")
        self._btn_auto_pair.clicked.connect(self._auto_pair_discovered)
        self._btn_auto_pair.setEnabled(False)
        self._btn_add_selected_pair.setEnabled(False)
        layout.addWidget(self._stereo_service_picker)
        pair_actions = QHBoxLayout()
        pair_actions.addStretch()
        pair_actions.addWidget(self._btn_add_selected_pair)
        pair_actions.addWidget(self._btn_auto_pair)
        layout.addLayout(pair_actions)

        # ── tables: splitter between discovery and pairs ──
        self._table_splitter = QSplitter(Qt.Vertical)

        # discovered services table
        self._discovered_table = QTableWidget(0, 6)
        self._discovered_table.setHorizontalHeaderLabels(
            ["名称", "地址", "端口/路径", "类型", "目别", "来源"]
        )
        self._configure_scrollable_table(self._discovered_table)
        self._discovered_table.verticalHeader().setVisible(False)
        self._discovered_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._discovered_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._discovered_table.setAlternatingRowColors(True)
        self._discovered_table.setMinimumHeight(80)
        self._discovered_table.setVisible(False)
        self._table_splitter.addWidget(self._discovered_table)

        # stereo pair table
        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["组名称", "类型", "左相机 ID", "左相机 URL", "右相机 ID", "右相机 URL"]
        )
        self._configure_scrollable_table(self._table)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setItemDelegateForColumn(1, StreamTypeDelegate(self._table))
        self._table.setMinimumHeight(120)
        self._table_splitter.addWidget(self._table)

        self._table_splitter.setStretchFactor(0, 2)
        self._table_splitter.setStretchFactor(1, 3)
        self._table_splitter.setChildrenCollapsible(False)
        self._table_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._table_splitter, 1)

        btn_layout = QHBoxLayout()
        self._btn_add = QPushButton("添加")
        self._btn_add.setText("手动添加")
        self._btn_add.setIcon(app_icon("camera"))
        self._btn_add.clicked.connect(self._add_row)
        btn_layout.addWidget(self._btn_add)

        self._btn_remove = QPushButton("删除选中")
        self._btn_remove.setIcon(app_icon("circle-minus"))
        self._btn_remove.setEnabled(False)
        self._btn_remove.clicked.connect(self._remove_row)
        btn_layout.addWidget(self._btn_remove)

        self._btn_swap_sources = QPushButton("交换左右图像源")
        self._btn_swap_sources.setIcon(app_icon("arrow-left-right"))
        self._btn_swap_sources.setEnabled(False)
        self._table.itemSelectionChanged.connect(self._update_row_actions)
        self._btn_swap_sources.setToolTip(
            "选中相机组后仅交换左右 URL，保留左/右相机 ID；"
            "用于设备服务把左右画面输出反了的情况"
        )
        self._btn_swap_sources.clicked.connect(self._swap_selected_pair_sources)
        btn_layout.addWidget(self._btn_swap_sources)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self._discovery_worker: DiscoveryWorker | None = None
        self._discovered: list[DiscoveredService] = []

        return group

    def closeEvent(self, event):
        self._stop_discovery()
        super().closeEvent(event)

    def reject(self):
        self._stop_discovery()
        super().reject()

    def _stop_discovery(self):
        if self._discovery_worker is None:
            return
        worker = self._discovery_worker
        self._discovery_worker = None
        # Detach from dialog so its destructor won't cascade-delete a running QThread.
        worker.setParent(None)
        if worker.isRunning():
            worker.stop()
        if worker.isRunning():
            # Thread didn't stop in time; let it self-destruct when it finishes.
            worker.finished.connect(worker.deleteLater)
        else:
            worker.deleteLater()

    def _configure_scrollable_table(self, table: QTableWidget) -> None:
        """Columns fill full width; user can still drag to resize individual columns."""
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.setWordWrap(False)
        hdr = table.horizontalHeader()
        hdr.setStretchLastSection(True)
        hdr.setMinimumSectionSize(60)
        hdr.setSectionResizeMode(QHeaderView.Stretch)

    # ── Network discovery ──────────────────────────────────────

    @Slot()
    def _on_scan(self):
        if self._discovery_worker is not None and self._discovery_worker.isRunning():
            return

        use_mdns = self._chk_mdns.isChecked()
        use_probe = self._chk_probe.isChecked()
        if not use_mdns and not use_probe:
            self._lbl_scan_status.setText("请至少选择一种发现方式")
            self._lbl_aux_discovery.setText("未启动扫描：请在相机连接页选择至少一种发现方式")
            self._lbl_scan_status.setStyleSheet("font-size:11px; color:#e07040;")
            return

        self._btn_scan.setEnabled(False)
        self._btn_discover_aux.setEnabled(False)
        self._lbl_aux_discovery.setText("正在发现设备…")
        self._btn_scan.setText("发现中...")
        self._scan_progress.setValue(0)
        self._scan_progress.setVisible(use_probe)
        self._lbl_scan_status.setText("mDNS 监听中..." if use_mdns else "")
        self._lbl_scan_status.setStyleSheet("font-size:11px; color:#888;")

        self._stop_discovery()
        self._discovery_worker = DiscoveryWorker(
            enable_mdns=use_mdns, enable_probe=use_probe,
        )
        self._discovery_worker.progress.connect(self._on_scan_progress)
        self._discovery_worker.service_found.connect(self._on_service_found)
        self._discovery_worker.finished.connect(self._on_scan_finished)
        self._discovery_worker.start()

    @Slot(int, int)
    def _on_scan_progress(self, current: int, total: int):
        self._scan_progress.setMaximum(total)
        self._scan_progress.setValue(current)
        self._lbl_scan_status.setText(f"子网扫描 {current}/{total}")

    @Slot(object)
    def _on_service_found(self, svc):
        """Real-time callback when mDNS discovers a service during scan."""
        if svc.key not in {s.key for s in self._discovered}:
            self._discovered.append(svc)
            self._append_discovered_row(svc)
            self._refresh_camera_dropdowns()
            self._discovered_table.setVisible(True)
            self._lbl_scan_status.setText(f"已发现 {len(self._discovered)} 个服务...")
            self._lbl_scan_status.setStyleSheet("font-size:11px; color:#52b788;")

    @Slot(list)
    def _on_scan_finished(self, results: list):
        self._has_scanned = True
        self._btn_scan.setEnabled(True)
        self._btn_discover_aux.setEnabled(True)
        self._btn_scan.setText("发现设备")
        self._scan_progress.setVisible(False)

        existing_keys = {s.key for s in self._discovered}
        for svc in results:
            if svc.key not in existing_keys:
                existing_keys.add(svc.key)
                self._discovered.append(svc)
                self._append_discovered_row(svc)

        has = len(self._discovered) > 0
        self._discovered_table.setVisible(has)
        self._btn_auto_pair.setVisible(len(self._discovered) >= 2)
        self._refresh_camera_dropdowns()

        if has:
            camera_count = sum(
                1
                for s in self._discovered
                if s.stream_type in {"rgb", "ir", "acoustic_rgb"}
            )
            n_mdns = sum(1 for s in self._discovered if s.source == "mdns")
            n_probe = sum(1 for s in self._discovered if s.source == "probe")
            parts = []
            if n_mdns:
                parts.append(f"mDNS {n_mdns}")
            if n_probe:
                parts.append(f"扫描 {n_probe}")
            self._lbl_scan_status.setText(
                f"发现 {len(self._discovered)} 个服务，相机流 {camera_count} 个（{', '.join(parts)}）"
            )
            self._lbl_scan_status.setStyleSheet("font-size:11px; color:#52b788;")
        else:
            self._lbl_scan_status.setText("未发现推流服务")
            self._lbl_scan_status.setStyleSheet("font-size:11px; color:#e07040;")

    def _append_discovered_row(self, svc: DiscoveredService):
        row = self._discovered_table.rowCount()
        self._discovered_table.insertRow(row)
        self._discovered_table.setRowHeight(row, 30)
        self._discovered_table.setItem(row, 0, QTableWidgetItem(svc.name))
        self._discovered_table.setItem(row, 1, QTableWidgetItem(svc.host))
        self._discovered_table.setItem(
            row, 2, QTableWidgetItem(f"{svc.port}{svc.path}")
        )
        self._discovered_table.setItem(
            row, 3, QTableWidgetItem(_STREAM_TYPE_LABELS.get(svc.stream_type, svc.stream_type))
        )
        self._discovered_table.setItem(
            row, 4, QTableWidgetItem(_EYE_LABELS.get(svc.eye, svc.eye))
        )
        source_label = _SOURCE_LABELS.get(svc.source, svc.source)
        item = QTableWidgetItem(source_label)
        if svc.source == "mdns":
            item.setForeground(Qt.cyan)
        self._discovered_table.setItem(row, 5, item)

    def _refresh_camera_dropdowns(self, *_):
        stream_type = self._combo_stream_type.currentData() or "rgb"
        candidates = [
            svc
            for svc in self._discovered
            if svc.stream_type == stream_type and svc.stream_type in {"rgb", "ir"}
        ]
        self._fill_camera_combo(self._combo_left_camera, candidates, preferred_eye="left")
        self._fill_camera_combo(self._combo_right_camera, candidates, preferred_eye="right")
        self._btn_add_selected_pair.setEnabled(bool(self._combo_left_camera.count() and self._combo_right_camera.count()))
        self._btn_auto_pair.setEnabled(len(getattr(self, "_discovered", [])) >= 2)
        self._refresh_aux_dropdown()

    def _fill_camera_combo(
        self,
        combo: QComboBox,
        services: list[DiscoveredService],
        preferred_eye: str,
    ):
        combo.blockSignals(True)
        selected = combo.currentData()
        selected_key = selected.key if selected is not None else None
        combo.clear()
        preferred: list[DiscoveredService] = []
        fallback: list[DiscoveredService] = []
        for svc in services:
            if svc.eye == preferred_eye:
                preferred.append(svc)
            elif svc.eye == "unknown":
                fallback.append(svc)

        for svc in preferred + fallback:
            combo.addItem(self._service_combo_label(svc), svc)
            combo.setItemData(combo.count() - 1, svc.url, Qt.ToolTipRole)
        for i in range(combo.count()):
            if combo.itemData(i).key == selected_key:
                combo.setCurrentIndex(i)
                break
        if combo.count() and combo.currentIndex() < 0:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _service_combo_label(self, svc: DiscoveredService) -> str:
        eye = _EYE_LABELS.get(svc.eye, svc.eye)
        stype = _STREAM_TYPE_LABELS.get(svc.stream_type, svc.stream_type)
        return f"{stype} / {eye}  {svc.name}  {svc.url}"

    def _refresh_aux_dropdown(self, *_):
        if not hasattr(self, "_combo_aux_camera"):
            return
        modality = self._combo_fusion_mode.currentData() or ""
        self._combo_aux_camera.blockSignals(True)
        selected = self._combo_aux_camera.currentData()
        selected_key = selected.key if selected is not None else None
        self._combo_aux_camera.clear()
        if not modality:
            self._combo_aux_camera.blockSignals(False)
            self._btn_use_aux_service.setEnabled(False)
            self._lbl_aux_discovery.setText("未启用辅助相机")
            return

        candidates = [
            svc
            for svc in getattr(self, "_discovered", [])
            if self._service_matches_aux_mode(svc, modality)
        ]
        for svc in candidates:
            self._combo_aux_camera.addItem(self._service_combo_label(svc), svc)
            self._combo_aux_camera.setItemData(self._combo_aux_camera.count() - 1, svc.url, Qt.ToolTipRole)
            if svc.key == selected_key:
                self._combo_aux_camera.setCurrentIndex(self._combo_aux_camera.count() - 1)
        if candidates and self._combo_aux_camera.currentIndex() < 0:
            self._combo_aux_camera.setCurrentIndex(0)
        self._combo_aux_camera.blockSignals(False)
        self._btn_use_aux_service.setEnabled(bool(candidates))
        if candidates:
            text = f"已发现 {len(candidates)} 路匹配辅助流"
        elif not self._btn_scan.isEnabled():
            text = "正在发现设备…"
        else:
            text = "扫描完成，未发现匹配辅助流" if self._has_scanned else "尚未扫描设备"
            if self._edit_aux_url.text().strip():
                text += " · 已填写地址，连接状态未验证"
        self._lbl_aux_discovery.setText(text)

    def _service_matches_aux_mode(self, svc: DiscoveredService, modality: str) -> bool:
        if modality == "ir":
            return svc.stream_type == "ir"
        if modality == "acoustic_rgb":
            if svc.stream_type == "acoustic_rgb":
                return True
            haystack = f"{svc.name} {svc.url} {svc.path}".lower()
            return svc.stream_type == "rgb" and any(
                token in haystack for token in ("acoustic", "sound", "声像仪")
            )
        return False

    def _use_selected_aux_service(self):
        svc = self._combo_aux_camera.currentData()
        if svc is None:
            self._lbl_scan_status.setText("请先发现并选择辅助相机服务")
            self._lbl_scan_status.setStyleSheet("font-size:11px; color:#e09f3e;")
            return
        modality = self._combo_fusion_mode.currentData() or svc.stream_type
        prefix = "aux_ir" if modality == "ir" else "aux_acoustic"
        if not self._edit_aux_id.text().strip():
            self._edit_aux_id.setText(self._camera_id_from_service(svc, prefix))
        self._edit_aux_url.setText(svc.url)
        self._lbl_scan_status.setText(f"已选择辅助服务: {svc.name}")
        self._lbl_scan_status.setStyleSheet("font-size:11px; color:#52b788;")

    def _add_pair_from_dropdowns(self):
        left = self._combo_left_camera.currentData()
        right = self._combo_right_camera.currentData()
        stream_type = self._combo_stream_type.currentData() or "unknown"
        if left is None or right is None:
            self._lbl_scan_status.setText("请先在左目和右目下拉框中选择相机服务")
            self._lbl_scan_status.setStyleSheet("font-size:11px; color:#e09f3e;")
            return
        if left.key == right.key:
            self._lbl_scan_status.setText("左目和右目不能选择同一个服务")
            self._lbl_scan_status.setStyleSheet("font-size:11px; color:#e07040;")
            return
        self._add_auto_pair(stream_type, left, right)

    def _auto_pair_discovered(self):
        camera_streams = [s for s in self._discovered if s.stream_type in {"rgb", "ir"}]
        if len(camera_streams) < 2:
            self._lbl_scan_status.setText("相机视频流不足 2 个，无法自动配对")
            self._lbl_scan_status.setStyleSheet("font-size:11px; color:#e09f3e;")
            return
        pair_count = 0
        for stream_type, host in sorted({(s.stream_type, s.host) for s in camera_streams}):
            typed = [s for s in camera_streams if s.stream_type == stream_type and s.host == host]
            lefts = [s for s in typed if s.eye == "left"]
            rights = [s for s in typed if s.eye == "right"]

            if len(lefts) == len(rights) == 1:
                pair_count += int(self._add_auto_pair(stream_type, lefts[0], rights[0]))

        if pair_count == 0:
            self._lbl_scan_status.setText("没有新增配对：服务已添加，或未找到同一主机内唯一的左右目；可手动选择。")
            self._lbl_scan_status.setStyleSheet("font-size:11px; color:#e09f3e;")
        else:
            self._lbl_scan_status.setText(f"已新增 {pair_count} 组相机，尚未保存")

    def _add_auto_pair(self, stream_type: str, left: DiscoveredService, right: DiscoveredService):
        urls = {left.url.rstrip("/"), right.url.rstrip("/")}
        if len(urls) != 2 or any(
            urls == {self._item_text(row, 3).rstrip("/"), self._item_text(row, 5).rstrip("/")}
            for row in range(self._table.rowCount())
        ):
            self._lbl_scan_status.setText("该配对已存在，或左右视频来源相同")
            return False
        row = self._table.rowCount()
        self._table.insertRow(row)
        prefix = stream_type if stream_type in {"rgb", "ir"} else "cam"
        pair_idx = self._available_pair_index(prefix)
        self._table.setItem(row, 0, QTableWidgetItem(f"{prefix}_pair_{pair_idx}"))
        self._table.setItem(
            row, 1, QTableWidgetItem(_STREAM_TYPE_LABELS.get(stream_type, stream_type))
        )
        self._table.setItem(row, 2, QTableWidgetItem(f"{prefix}_cam_{pair_idx}_L"))
        self._table.setItem(row, 3, QTableWidgetItem(left.url))
        self._table.setItem(row, 4, QTableWidgetItem(f"{prefix}_cam_{pair_idx}_R"))
        self._table.setItem(row, 5, QTableWidgetItem(right.url))
        self._table.selectRow(row)
        return True

    def _available_pair_index(self, prefix):
        names = {self._item_text(row, 0) for row in range(self._table.rowCount())}
        ids = {self._item_text(row, col) for row in range(self._table.rowCount()) for col in (2, 4)}
        index = 1
        while f"{prefix}_pair_{index}" in names or any(
            f"{prefix}_cam_{index}_{eye}" in ids for eye in ("L", "R")
        ):
            index += 1
        return index

    def _load_current(self):
        idx = self._combo_model.findData(self._camera_model)
        if idx >= 0:
            self._combo_model.setCurrentIndex(idx)

        cfg = self._board.config
        self._spin_cols.setValue(cfg.cols)
        self._spin_rows.setValue(cfg.rows)
        self._spin_sq_len.setValue(cfg.square_length * 1000)
        self._spin_mk_len.setValue(cfg.marker_length * 1000)

        idx = self._combo_dict.findData(cfg.dict_type)
        if idx >= 0:
            self._combo_dict.setCurrentIndex(idx)

        for name, pair in self._stream_manager.stereo_pairs.items():
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(pair.name))
            self._table.setItem(
                row,
                1,
                QTableWidgetItem(_STREAM_TYPE_LABELS.get(pair.left.stream_type, pair.left.stream_type)),
            )
            self._table.setItem(row, 2, QTableWidgetItem(pair.left.camera_id))
            self._table.setItem(row, 3, QTableWidgetItem(pair.left.url))
            self._table.setItem(row, 4, QTableWidgetItem(pair.right.camera_id))
            self._table.setItem(row, 5, QTableWidgetItem(pair.right.url))

        aux = self._stream_manager.auxiliary_camera
        if aux is not None:
            idx = self._combo_fusion_mode.findData(aux.modality)
            if idx >= 0:
                self._combo_fusion_mode.setCurrentIndex(idx)
            self._edit_aux_id.setText(aux.camera_id)
            self._edit_aux_url.setText(aux.url)
            self._set_aux_board_values(aux.pattern_board_config())
        self._update_aux_enabled()
        self._refresh_aux_dropdown()

    def _add_row(self):
        index = self._available_pair_index("rgb")
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(f"rgb_pair_{index}"))
        self._table.setItem(row, 1, QTableWidgetItem("RGB"))
        self._table.setItem(row, 2, QTableWidgetItem(f"rgb_cam_{index}_L"))
        self._table.setItem(row, 3, QTableWidgetItem(""))
        self._table.setItem(row, 4, QTableWidgetItem(f"rgb_cam_{index}_R"))
        self._table.setItem(row, 5, QTableWidgetItem(""))
        self._table.selectRow(row)
        self._table.scrollToItem(self._table.item(row, 0))

    def _update_row_actions(self):
        selected = bool(self._table.selectedIndexes())
        self._btn_remove.setEnabled(selected)
        self._btn_swap_sources.setEnabled(selected)

    def _remove_row(self):
        rows = set(idx.row() for idx in self._table.selectedIndexes())
        for row in sorted(rows, reverse=True):
            self._table.removeRow(row)

    def _swap_selected_pair_sources(self):
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()})
        if not rows and self._table.currentRow() >= 0:
            rows = [self._table.currentRow()]
        if not rows:
            self._lbl_scan_status.setText("请先选中需要交换左右图像源的相机组")
            self._lbl_scan_status.setStyleSheet("font-size:11px; color:#e09f3e;")
            return

        for row in rows:
            left_url = self._item_text(row, 3)
            right_url = self._item_text(row, 5)
            logger.info(
                "Swap stereo pair sources in config row=%d left_url=%s right_url=%s",
                row,
                left_url,
                right_url,
            )
            self._table.setItem(row, 3, QTableWidgetItem(right_url))
            self._table.setItem(row, 5, QTableWidgetItem(left_url))

        self._lbl_scan_status.setText(f"已交换 {len(rows)} 组左右图像源")
        self._lbl_scan_status.setStyleSheet("font-size:11px; color:#52b788;")

    def _proposed_configuration(self):
        modality = self._combo_fusion_mode.currentData() or ""
        aux_id = self._edit_aux_id.text().strip() or "aux_cam"
        aux_url = self._edit_aux_url.text().strip()
        stereo_ids = {self._item_text(row, col) for row in range(self._table.rowCount())
                      for col in (2, 4)}
        stereo_urls = {self._item_text(row, col).rstrip("/")
                       for row in range(self._table.rowCount()) for col in (3, 5)}
        if modality and (aux_id in stereo_ids or (aux_url and aux_url.rstrip("/") in stereo_urls)):
            raise ConfigurationError(
                "红外/辅助相机必须使用独立的相机 ID 和视频流，不能复用 RGB 左右目。",
                self._edit_aux_id, 1,
            )
        try:
            aux_board = self._aux_board_values() if modality else None
        except ValueError as exc:
            raise ConfigurationError(f"辅助标定板参数无效：{exc}", self._spin_aux_cols, 2) from exc
        square, marker = self._spin_sq_len.value(), self._spin_mk_len.value()
        if marker >= square:
            raise ConfigurationError("RGB 标定板：Marker 边长必须小于方格边长。", self._spin_mk_len, 2)
        dictionary = self._combo_dict.currentData()
        marker_count = self._spin_cols.value() * self._spin_rows.value() // 2
        if marker_count > int(dictionary.name.rsplit("_", 1)[-1]):
            raise ConfigurationError("RGB 标定板：字典容量不足，请选择更大的 ArUco 字典。", self._combo_dict, 2)
        board_cfg = CharucoBoardConfig(
            cols=self._spin_cols.value(),
            rows=self._spin_rows.value(),
            square_length=square / 1000,
            marker_length=marker / 1000,
            dict_type=self._combo_dict.currentData(),
        )
        pairs = []
        for row in range(self._table.rowCount()):
            name = self._item_text(row, 0)
            stream_type = self._stream_type_from_label(self._item_text(row, 1))
            left_id = self._item_text(row, 2)
            left_url = self._item_text(row, 3)
            right_id = self._item_text(row, 4)
            right_url = self._item_text(row, 5)

            if not any((name, left_id, right_id, left_url, right_url)):
                continue
            if not name or not left_id or not right_id or bool(left_url) != bool(right_url):
                raise ConfigurationError(f"第 {row + 1} 组：需填写名称和左右 ID；视频地址需同时填写或同时留空。",
                                         self._table, 0)
            if stream_type not in {"rgb", "ir"}:
                raise ConfigurationError(f"第 {row + 1} 组：类型需为 RGB 或红外。", self._table, 0)
            for url in (left_url, right_url):
                self._validate_stream_url(url, self._table, 0)
            if left_url and left_url.rstrip("/") == right_url.rstrip("/"):
                raise ConfigurationError(f"第 {row + 1} 组：左右视频来源不能相同。", self._table, 0)
            pair = StereoPairConfig(
                name=name,
                left=CameraConfig(
                    camera_id=left_id,
                    url=left_url,
                    role="left",
                    group=name,
                    stream_type=stream_type,
                ),
                right=CameraConfig(
                    camera_id=right_id,
                    url=right_url,
                    role="right",
                    group=name,
                    stream_type=stream_type,
                ),
            )
            pairs.append(pair)
        cameras = {}
        conflict = False
        for pair in pairs:
            conflict |= pair.left.camera_id == pair.right.camera_id
            for cam in (pair.left, pair.right):
                source = (cam.url, cam.stream_type)
                conflict |= cam.camera_id in cameras and cameras[cam.camera_id] != source
                cameras[cam.camera_id] = source
        if len({p.name for p in pairs}) != len(pairs) or conflict:
            raise ConfigurationError("相机组名不能重复；左右 ID 需不同，同一 ID 不能指向不同来源。", self._table, 0)
        if modality:
            self._validate_stream_url(aux_url, self._edit_aux_url, 1)
        auxiliary = AuxiliaryCameraConfig(
            camera_id=aux_id, url=aux_url, modality=modality,
            board_pattern=self._combo_aux_board.currentData() or "chessboard",
            stream_type="ir" if modality == "ir" else "acoustic_rgb", board_config=aux_board,
        ) if modality else None
        model = self._combo_model.currentData()
        return board_cfg, model, pairs, auxiliary

    @staticmethod
    def _configuration_signature(configuration):
        cfg, model, pairs, auxiliary = configuration
        return cfg, model, [pair_signature(p) for p in pairs], auxiliary.to_dict() if auxiliary else None

    @staticmethod
    def _validate_stream_url(url, widget, page):
        if not url:
            return
        try:
            parsed = urlsplit(url)
            valid = parsed.scheme in {"http", "https"} and bool(parsed.hostname) and not parsed.fragment
            valid = valid and not any(char.isspace() for char in url)
            valid = valid and (parsed.port is None or 1 <= parsed.port <= 65535)
        except ValueError:
            valid = False
        if not valid:
            raise ConfigurationError("视频地址需为有效的 HTTP(S) URL；离线数据可留空。", widget, page)

    def _connect_validation(self):
        self._table.itemChanged.connect(self._refresh_validation)
        self._table.model().rowsRemoved.connect(self._refresh_validation)
        for spin in (self._spin_cols, self._spin_rows, self._spin_sq_len, self._spin_mk_len,
                     self._spin_aux_cols, self._spin_aux_rows, self._spin_aux_spacing):
            spin.valueChanged.connect(self._refresh_validation)
        for combo in (self._combo_dict, self._combo_model, self._combo_fusion_mode,
                      self._combo_aux_board, self._combo_aux_count_mode):
            combo.currentIndexChanged.connect(self._refresh_validation)
        self._check_aux_asymmetric.toggled.connect(self._refresh_validation)
        self._edit_aux_id.textChanged.connect(self._refresh_validation)
        self._edit_aux_url.textChanged.connect(self._refresh_validation)

    def _refresh_validation(self, *_):
        cols, rows = self._spin_cols.value(), self._spin_rows.value()
        self._lbl_rgb_board_summary.setText(f"{cols} × {rows} 格子 · {(cols - 1) * (rows - 1)} 内角点 · "
                                             f"板面 {cols * self._spin_sq_len.value():g} × "
                                             f"{rows * self._spin_sq_len.value():g} mm（不含边距）")
        reference = next((self._item_text(row, 2) for row in range(self._table.rowCount())
                          if self._stream_type_from_label(self._item_text(row, 1)) == "rgb"), None)
        self._lbl_aux_reference.setText(f"RGB 左目 · {reference}" if reference else "尚未配置 RGB 双目参考相机")
        self._lbl_aux_reference.setToolTip(self._lbl_aux_reference.text())
        self._lbl_aux_reference.setCursorPosition(0)
        try:
            configuration = self._proposed_configuration()
            _, _, pairs, auxiliary = configuration
        except ConfigurationError as exc:
            text, color, valid = str(exc), DANGER, False
        else:
            offline = sum(not p.left.url for p in pairs)
            dirty = self._configuration_signature(configuration) != self._initial_configuration
            text = f"双目 {len(pairs)} 组（离线 {offline} 组） · 辅助{'已启用' if auxiliary else '未启用'} · " + (
                "有未应用修改" if dirty else "未修改"
            )
            color, valid = TEXT_DIM, True
        self._lbl_config_status.setText(text)
        self._lbl_config_status.setStyleSheet(f"color:{color}; background:transparent;")
        self._buttons.button(QDialogButtonBox.Save).setEnabled(valid)
        self._buttons.button(QDialogButtonBox.Save).setToolTip("" if valid else text)

    def _on_accept(self):
        try:
            cfg, model, pairs, auxiliary = self._proposed_configuration()
            board = CharucoBoard(cfg)
        except Exception as exc:
            logger.warning("Configuration validation failed: %s", exc)
            if isinstance(exc, ConfigurationError):
                self._tabs.setCurrentIndex(exc.page)
                exc.widget.setFocus()
            self._refresh_validation()
            QMessageBox.warning(self, "配置无效", str(exc).splitlines()[0])
            return
        if self.before_apply is not None and not self.before_apply(board, model, pairs, auxiliary):
            return
        self._stop_discovery()
        old_aux = self._stream_manager.auxiliary_camera
        self.streams_changed = (
            [pair_signature(p) for p in pairs] !=
            [pair_signature(p) for p in self._stream_manager.stereo_pairs.values()] or
            (old_aux.to_dict() if old_aux else None) != (auxiliary.to_dict() if auxiliary else None)
        )
        self._board, self._camera_model = board, model
        if not self.streams_changed:
            self.accept()
            return
        self._stream_manager.stop_all()
        for name in list(self._stream_manager.stereo_pairs):
            self._stream_manager.remove_stereo_pair(name)
        self._stream_manager.clear_auxiliary_camera()
        for pair in pairs:
            if pair.left.url:
                self._stream_manager.add_stereo_pair(pair)
            else:
                self._stream_manager.add_virtual_stereo_pair(pair)
            logger.info(
                "Configured stereo pair: name=%s type=%s left=%s url=%s right=%s url=%s",
                pair.name, pair.left.stream_type, pair.left.camera_id, pair.left.url,
                pair.right.camera_id, pair.right.url,
            )

        if auxiliary is not None:
            self._stream_manager.set_auxiliary_camera(auxiliary)

        self.accept()

    @staticmethod
    def _stream_type_from_label(label: str) -> str:
        lowered = label.strip().lower()
        if lowered in {"ir", "红外", "infrared"}:
            return "ir"
        if lowered in {"rgb", "color", "彩色"}:
            return "rgb"
        return "unknown"

    def _item_text(self, row: int, col: int) -> str:
        item = self._table.item(row, col)
        return item.text().strip() if item is not None else ""

    def _update_aux_enabled(self, *_):
        enabled = bool(self._combo_fusion_mode.currentData())
        self._edit_aux_id.setEnabled(enabled)
        self._edit_aux_url.setEnabled(enabled)
        self._combo_aux_board.setEnabled(enabled)
        self._combo_aux_camera.setEnabled(enabled)
        self._btn_use_aux_service.setEnabled(enabled and self._combo_aux_camera.count() > 0)
        for widget in (self._spin_aux_cols, self._spin_aux_rows, self._spin_aux_spacing,
                       self._btn_aux_board_preset, self._btn_import_aux_board):
            widget.setEnabled(enabled)
        self._check_aux_asymmetric.setEnabled(
            enabled and self._combo_aux_board.currentData() == PatternType.CIRCLES_GRID.value
        )
        self._btn_aux_board_page.setEnabled(enabled)
        self._update_aux_dimensions()

    @staticmethod
    def _camera_id_from_service(svc: DiscoveredService, fallback_prefix: str) -> str:
        raw = svc.name or svc.host or fallback_prefix
        cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in raw).strip("_")
        return cleaned or fallback_prefix
