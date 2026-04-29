"""Camera group configuration and ChArUco board parameter dialog."""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...board.charuco_board import ArucoDictType, CharucoBoard, CharucoBoardConfig
from ...calibration.models import CameraModel
from ...streaming.discovery import DiscoveredService, DiscoveryWorker
from ...streaming.stream_manager import CameraConfig, StereoPairConfig, StreamManager

_SOURCE_LABELS = {"mdns": "mDNS", "probe": "扫描"}
_STREAM_TYPE_LABELS = {
    "rgb": "RGB",
    "ir": "红外",
    "control": "控制台",
    "unknown": "未知",
}
_EYE_LABELS = {
    "left": "左目",
    "right": "右目",
    "unknown": "未知",
}


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
        self.setMinimumSize(980, 700)
        self.resize(1080, 860)
        self._stream_manager = stream_manager
        self._board = board
        self._camera_model = camera_model

        self._init_ui()
        self._load_current()

    @property
    def board(self) -> CharucoBoard:
        return self._board

    @property
    def camera_model(self) -> CameraModel:
        return self._camera_model

    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        top_row.addWidget(self._create_model_group(), 1)
        top_row.addWidget(self._create_board_group(), 2)
        layout.addLayout(top_row)
        layout.addWidget(self._create_pairs_group(), 1)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        scroll.setWidget(body)
        outer.addWidget(scroll)

    def _create_model_group(self) -> QGroupBox:
        group = QGroupBox("相机模型")
        form = QFormLayout(group)
        form.setContentsMargins(10, 6, 10, 6)
        form.setSpacing(4)

        self._combo_model = QComboBox()
        self._combo_model.addItem("针孔 (Pinhole)", CameraModel.PINHOLE)
        self._combo_model.addItem("鱼眼 (Fisheye)", CameraModel.FISHEYE)
        form.addRow("镜头类型:", self._combo_model)

        self._lbl_model_hint = QLabel("FOV < 120° 选针孔，> 120° 选鱼眼")
        self._lbl_model_hint.setStyleSheet("color: #888; font-size: 11px;")
        self._lbl_model_hint.setWordWrap(True)
        form.addRow(self._lbl_model_hint)

        return group

    def _create_board_group(self) -> QGroupBox:
        group = QGroupBox("ChArUco 标定板")
        form = QFormLayout(group)
        form.setContentsMargins(10, 6, 10, 6)
        form.setSpacing(4)

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
        form.addRow(row_size)

        row_len = QHBoxLayout()
        self._spin_sq_len = QDoubleSpinBox()
        self._spin_sq_len.setDecimals(4)
        self._spin_sq_len.setRange(0.001, 1.0)
        self._spin_sq_len.setSuffix(" m")
        self._spin_sq_len.setSingleStep(0.005)
        row_len.addWidget(QLabel("方格边长:"))
        row_len.addWidget(self._spin_sq_len)
        row_len.addSpacing(12)
        self._spin_mk_len = QDoubleSpinBox()
        self._spin_mk_len.setDecimals(4)
        self._spin_mk_len.setRange(0.001, 1.0)
        self._spin_mk_len.setSuffix(" m")
        self._spin_mk_len.setSingleStep(0.005)
        row_len.addWidget(QLabel("Marker:"))
        row_len.addWidget(self._spin_mk_len)
        form.addRow(row_len)

        self._combo_dict = QComboBox()
        for d in ArucoDictType:
            self._combo_dict.addItem(d.name, d)
        form.addRow("ArUco 字典:", self._combo_dict)

        return group

    def _create_pairs_group(self) -> QGroupBox:
        group = QGroupBox("双目相机组")
        group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        # ── discovery controls ──
        disc_row = QHBoxLayout()
        self._btn_scan = QPushButton("🔍 发现设备")
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

        self._lbl_scan_status = QLabel("")
        self._lbl_scan_status.setMinimumWidth(180)
        self._lbl_scan_status.setStyleSheet("font-size:11px; color:#888;")
        disc_row.addWidget(self._lbl_scan_status)
        layout.addLayout(disc_row)

        # ── discovered camera dropdowns ──
        picker_row = QHBoxLayout()
        picker_row.addWidget(QLabel("类型:"))
        self._combo_stream_type = QComboBox()
        self._combo_stream_type.addItem("RGB 双目", "rgb")
        self._combo_stream_type.addItem("红外双目", "ir")
        self._combo_stream_type.currentIndexChanged.connect(self._refresh_camera_dropdowns)
        picker_row.addWidget(self._combo_stream_type)

        picker_row.addWidget(QLabel("左目:"))
        self._combo_left_camera = QComboBox()
        self._combo_left_camera.setMinimumWidth(220)
        picker_row.addWidget(self._combo_left_camera, 1)

        picker_row.addWidget(QLabel("右目:"))
        self._combo_right_camera = QComboBox()
        self._combo_right_camera.setMinimumWidth(220)
        picker_row.addWidget(self._combo_right_camera, 1)

        self._btn_add_selected_pair = QPushButton("添加相机组")
        self._btn_add_selected_pair.clicked.connect(self._add_pair_from_dropdowns)
        picker_row.addWidget(self._btn_add_selected_pair)

        self._btn_auto_pair = QPushButton("⚡ 自动配对")
        self._btn_auto_pair.setToolTip("按类型和左右目自动配成双目相机组")
        self._btn_auto_pair.clicked.connect(self._auto_pair_discovered)
        picker_row.addWidget(self._btn_auto_pair)
        layout.addLayout(picker_row)

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
        self._discovered_table.setAlternatingRowColors(True)
        self._discovered_table.setMinimumHeight(80)
        self._discovered_table.setVisible(False)
        self._table_splitter.addWidget(self._discovered_table)

        # stereo pair table
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["组名称", "类型", "左相机 ID", "左相机 URL", "右相机 URL"]
        )
        self._configure_scrollable_table(self._table)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setMinimumHeight(120)
        self._table_splitter.addWidget(self._table)

        self._table_splitter.setStretchFactor(0, 2)
        self._table_splitter.setStretchFactor(1, 3)
        self._table_splitter.setChildrenCollapsible(False)
        self._table_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._table_splitter, 1)

        btn_layout = QHBoxLayout()
        self._btn_add = QPushButton("添加")
        self._btn_add.clicked.connect(self._add_row)
        btn_layout.addWidget(self._btn_add)

        self._btn_remove = QPushButton("删除选中")
        self._btn_remove.clicked.connect(self._remove_row)
        btn_layout.addWidget(self._btn_remove)

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
            self._lbl_scan_status.setStyleSheet("font-size:11px; color:#e07040;")
            return

        self._btn_scan.setEnabled(False)
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
        self._btn_scan.setEnabled(True)
        self._btn_scan.setText("🔍 发现设备")
        self._scan_progress.setVisible(False)

        existing_keys = {s.key for s in self._discovered}
        for svc in results:
            if svc.key not in existing_keys:
                self._discovered.append(svc)
                self._append_discovered_row(svc)

        has = len(self._discovered) > 0
        self._discovered_table.setVisible(has)
        self._btn_auto_pair.setVisible(len(self._discovered) >= 2)
        self._refresh_camera_dropdowns()

        if has:
            camera_count = sum(1 for s in self._discovered if s.stream_type in {"rgb", "ir"})
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

    def _fill_camera_combo(
        self,
        combo: QComboBox,
        services: list[DiscoveredService],
        preferred_eye: str,
    ):
        combo.blockSignals(True)
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
        combo.blockSignals(False)

    def _service_combo_label(self, svc: DiscoveredService) -> str:
        eye = _EYE_LABELS.get(svc.eye, svc.eye)
        stype = _STREAM_TYPE_LABELS.get(svc.stream_type, svc.stream_type)
        return f"{stype} / {eye}  {svc.name}  {svc.url}"

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
        for stream_type in ("rgb", "ir"):
            typed = [s for s in camera_streams if s.stream_type == stream_type]
            lefts = [s for s in typed if s.eye == "left"]
            rights = [s for s in typed if s.eye == "right"]

            pairs: list[tuple[DiscoveredService, DiscoveredService]] = []
            pairs.extend(zip(lefts, rights))

            if not pairs:
                unknown = [s for s in typed if s.eye == "unknown"]
                if len(unknown) >= 2:
                    pairs.append((unknown[0], unknown[1]))

            for left, right in pairs:
                self._add_auto_pair(stream_type, left, right)
                pair_count += 1

        if pair_count == 0:
            self._lbl_scan_status.setText("未找到可自动配对的同类型左右目服务，请手动选择左右相机")
            self._lbl_scan_status.setStyleSheet("font-size:11px; color:#e09f3e;")

    def _add_auto_pair(self, stream_type: str, left: DiscoveredService, right: DiscoveredService):
        row = self._table.rowCount()
        self._table.insertRow(row)
        pair_idx = row + 1
        prefix = stream_type if stream_type in {"rgb", "ir"} else "cam"
        self._table.setItem(row, 0, QTableWidgetItem(f"{prefix}_pair_{pair_idx}"))
        self._table.setItem(
            row, 1, QTableWidgetItem(_STREAM_TYPE_LABELS.get(stream_type, stream_type))
        )
        self._table.setItem(row, 2, QTableWidgetItem(f"{prefix}_cam_{pair_idx}"))
        self._table.setItem(row, 3, QTableWidgetItem(left.url))
        self._table.setItem(row, 4, QTableWidgetItem(right.url))

    def _load_current(self):
        idx = self._combo_model.findData(self._camera_model)
        if idx >= 0:
            self._combo_model.setCurrentIndex(idx)

        cfg = self._board.config
        self._spin_cols.setValue(cfg.cols)
        self._spin_rows.setValue(cfg.rows)
        self._spin_sq_len.setValue(cfg.square_length)
        self._spin_mk_len.setValue(cfg.marker_length)

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
            self._table.setItem(row, 4, QTableWidgetItem(pair.right.url))

    def _add_row(self):
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(f"pair_{row + 1}"))
        self._table.setItem(row, 1, QTableWidgetItem("RGB"))
        self._table.setItem(row, 2, QTableWidgetItem(f"cam_{row * 2 + 1}"))
        self._table.setItem(row, 3, QTableWidgetItem("http://"))
        self._table.setItem(row, 4, QTableWidgetItem("http://"))

    def _remove_row(self):
        rows = set(idx.row() for idx in self._table.selectedIndexes())
        for row in sorted(rows, reverse=True):
            self._table.removeRow(row)

    def _on_accept(self):
        self._stop_discovery()
        self._camera_model = self._combo_model.currentData()

        board_cfg = CharucoBoardConfig(
            cols=self._spin_cols.value(),
            rows=self._spin_rows.value(),
            square_length=self._spin_sq_len.value(),
            marker_length=self._spin_mk_len.value(),
            dict_type=self._combo_dict.currentData(),
        )
        self._board = CharucoBoard(board_cfg)

        self._stream_manager.stop_all()
        for name in list(self._stream_manager.stereo_pairs.keys()):
            self._stream_manager.remove_stereo_pair(name)

        for row in range(self._table.rowCount()):
            name = self._item_text(row, 0)
            stream_type = self._stream_type_from_label(self._item_text(row, 1))
            left_id = self._item_text(row, 2)
            left_url = self._item_text(row, 3)
            right_url = self._item_text(row, 4)
            right_id = f"{left_id}_R"

            if not name or not left_url or not right_url:
                continue

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
            self._stream_manager.add_stereo_pair(pair)

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
