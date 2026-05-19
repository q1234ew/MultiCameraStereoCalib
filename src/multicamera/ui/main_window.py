"""Main application window — modern dark-themed layout."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..board.charuco_board import CharucoBoard, CharucoBoardConfig
from ..calibration.models import CameraModel, MultiCameraRig
from ..io.session import CalibrationSession, SessionManager
from ..runtime_paths import logo_png_path, sessions_dir
from ..streaming.stream_manager import StreamManager
from .widgets.calib_panel import CalibrationPanel
from .widgets.config_dialog import ConfigDialog
from .widgets.stream_view import MultiStreamView

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MultiCamera Stereo Calibration")
        self.resize(1500, 920)

        logo_path = logo_png_path()
        if logo_path is not None:
            self.setWindowIcon(QIcon(str(logo_path)))

        self._stream_manager = StreamManager(parent=self)
        self._board = CharucoBoard(CharucoBoardConfig())
        self._camera_model = CameraModel.PINHOLE
        self._session_manager = SessionManager(sessions_dir())
        self._current_session: CalibrationSession | None = None
        self._rig: MultiCameraRig | None = None

        self._init_widgets()
        self._init_menus()
        self._init_toolbar()
        self._init_statusbar()
        self._connect_signals()

    # ── UI construction ───────────────────────────────────────

    def _init_widgets(self):
        # Left: camera views  |  Right: control panel + 3D viewer (tabs)
        self._stream_view = MultiStreamView(self._stream_manager, self._board)

        self._calib_panel = CalibrationPanel(
            self._stream_manager, self._board
        )
        self._stream_view.set_capture_sequence(self._calib_panel._capture_sequence)

        self._cloud_viewer = None
        self._cloud_placeholder = self._create_cloud_placeholder()

        # Right side: tabs for control panel and 3D viewer
        self._right_tabs = QTabWidget()
        self._right_tabs.setTabPosition(QTabWidget.North)
        self._right_tabs.addTab(self._calib_panel, "标定控制")
        self._cloud_tab_index = self._right_tabs.addTab(
            self._cloud_placeholder, "3D 点云"
        )

        # Splitter for main content
        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.addWidget(self._stream_view)
        self._splitter.addWidget(self._right_tabs)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([1050, 450])

        self.setCentralWidget(self._splitter)

    def _create_cloud_placeholder(self) -> QWidget:
        placeholder = QWidget()
        layout = QVBoxLayout(placeholder)
        label = QLabel("完成标定并生成点云后此处显示 3D 点云")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        return placeholder

    def _init_menus(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件(&F)")

        self._act_config = QAction("相机配置(&C)...", self)
        self._act_config.setShortcut(QKeySequence("Ctrl+K"))
        file_menu.addAction(self._act_config)

        self._act_new_session = QAction("新建标定会话(&N)", self)
        self._act_new_session.setShortcut(QKeySequence.New)
        file_menu.addAction(self._act_new_session)

        self._act_load_session = QAction("加载会话(&L)...", self)
        self._act_load_session.setShortcut(QKeySequence.Open)
        file_menu.addAction(self._act_load_session)

        file_menu.addSeparator()

        self._act_export = QAction("导出标定结果(&E)...", self)
        self._act_export.setShortcut(QKeySequence("Ctrl+E"))
        self._act_export.setEnabled(False)
        file_menu.addAction(self._act_export)

        file_menu.addSeparator()

        self._act_exit = QAction("退出(&Q)", self)
        self._act_exit.setShortcut(QKeySequence.Quit)
        file_menu.addAction(self._act_exit)

        tools_menu = menubar.addMenu("工具(&T)")
        self._act_gen_board = QAction("生成标定板图像(&G)...", self)
        tools_menu.addAction(self._act_gen_board)

    def _init_toolbar(self):
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        toolbar.setIconSize(toolbar.iconSize())
        self.addToolBar(toolbar)

        self._act_start_streams = QAction("▶  连接相机", self)
        toolbar.addAction(self._act_start_streams)

        self._act_stop_streams = QAction("⏹  断开相机", self)
        self._act_stop_streams.setEnabled(False)
        toolbar.addAction(self._act_stop_streams)

        toolbar.addSeparator()

        self._act_config_tb = QAction("⚙  相机配置", self)
        toolbar.addAction(self._act_config_tb)

        toolbar.addSeparator()

        self._act_gen_board_tb = QAction("🖨  生成标定板", self)
        toolbar.addAction(self._act_gen_board_tb)

    def _init_statusbar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("就绪  |  请先通过 ⚙ 相机配置 添加相机，然后点击 ▶ 连接")

    def _connect_signals(self):
        self._act_config.triggered.connect(self._on_config)
        self._act_config_tb.triggered.connect(self._on_config)
        self._act_new_session.triggered.connect(self._on_new_session)
        self._act_load_session.triggered.connect(self._on_load_session)
        self._act_export.triggered.connect(self._on_export)
        self._act_exit.triggered.connect(self.close)
        self._act_start_streams.triggered.connect(self._on_start_streams)
        self._act_stop_streams.triggered.connect(self._on_stop_streams)
        self._act_gen_board.triggered.connect(self._on_gen_board)
        self._act_gen_board_tb.triggered.connect(self._on_gen_board)

        self._calib_panel.calibration_finished.connect(self._on_calib_finished)
        self._calib_panel.pointcloud_ready.connect(self._on_pointcloud)
        self._calib_panel.frame_captured.connect(self._stream_view.notify_capture)
        self._calib_panel.sequence_updated.connect(
            self._stream_view.set_capture_sequence
        )
        self._calib_panel.images_imported.connect(self._stream_view.show_static_images)

        self._stream_manager.camera_connected.connect(self._on_camera_connected)
        self._stream_manager.camera_disconnected.connect(
            lambda cid, reason: self._statusbar.showMessage(
                f"✗  相机 {cid} 断开: {reason}"
            )
        )

    # ── Slots ─────────────────────────────────────────────────

    @Slot()
    def _on_config(self):
        dlg = ConfigDialog(
            self._stream_manager, self._board, self._camera_model, parent=self
        )
        if dlg.exec():
            self._board = dlg.board
            self._camera_model = dlg.camera_model
            self._stream_view.update_board(self._board)
            self._calib_panel.update_board(self._board)
            self._calib_panel.set_camera_model(self._camera_model)
            self._calib_panel.refresh_pairs()
            model_name = "针孔" if self._camera_model == CameraModel.PINHOLE else "鱼眼"
            self._statusbar.showMessage(
                f"✓  已配置 {len(self._stream_manager.stereo_pairs)} 组相机 [{model_name}模型]"
            )

    @Slot()
    def _on_new_session(self):
        pairs = list(self._stream_manager.stereo_pairs.values())
        if not pairs:
            QMessageBox.warning(self, "提示", "请先配置相机")
            return
        self._current_session = self._session_manager.create_session(
            self._board,
            pairs,
            auxiliary=self._stream_manager.auxiliary_camera,
            name="calib",
        )
        self._calib_panel.set_session(self._current_session)
        self._statusbar.showMessage(f"✓  新建会话: {self._current_session.name}")

    @Slot()
    def _on_load_session(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择会话目录", str(sessions_dir())
        )
        if dir_path:
            self._current_session = CalibrationSession.load(Path(dir_path))
            self._calib_panel.set_session(self._current_session)
            self._statusbar.showMessage(f"✓  加载会话: {self._current_session.name}")

    @Slot()
    def _on_export(self):
        if self._rig is None:
            return
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出标定结果",
            "calibration_result.json",
            "JSON (*.json);;OpenCV YAML (*.yml *.yaml);;Kalibr camchain YAML (*.yaml *.yml)",
        )
        if path:
            from ..io.export import (
                export_rig_json,
                export_rig_kalibr_yaml,
                export_rig_opencv_yaml,
            )

            if "Kalibr" in selected_filter:
                export_rig_kalibr_yaml(self._rig, path)
            elif "OpenCV" in selected_filter or path.endswith((".yml", ".yaml")):
                export_rig_opencv_yaml(self._rig, path)
            else:
                export_rig_json(self._rig, path)
            self._statusbar.showMessage(f"✓  已导出: {path}")

    @Slot(str)
    def _on_camera_connected(self, cid: str):
        self._statusbar.showMessage(f"✓  相机 {cid} 已连接")
        self._calib_panel.refresh_pairs()

    @Slot()
    def _on_start_streams(self):
        self._stream_manager.start_all()
        self._act_start_streams.setEnabled(False)
        self._act_stop_streams.setEnabled(True)
        self._calib_panel.refresh_pairs()
        self._statusbar.showMessage("相机连接中...")

    @Slot()
    def _on_stop_streams(self):
        self._stream_manager.stop_all()
        self._act_start_streams.setEnabled(True)
        self._act_stop_streams.setEnabled(False)
        self._statusbar.showMessage("相机已断开")

    @Slot()
    def _on_gen_board(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存标定板图像",
            "charuco_board.png",
            "PNG (*.png);;JPEG (*.jpg)",
        )
        if path:
            self._board.save_image(path)
            self._statusbar.showMessage(f"✓  标定板已保存: {path}")

    @Slot(object)
    def _on_calib_finished(self, rig: MultiCameraRig):
        self._rig = rig
        self._act_export.setEnabled(True)
        if self._current_session:
            self._current_session.save_rig(rig)
        self._statusbar.showMessage("✓  标定完成！可在 3D 点云 标签页查看结果")

    @Slot(object)
    def _on_pointcloud(self, pcd):
        if self._cloud_viewer is None:
            from .widgets.cloud_viewer import PointCloudViewer

            self._cloud_viewer = PointCloudViewer()
            self._right_tabs.removeTab(self._cloud_tab_index)
            self._cloud_tab_index = self._right_tabs.addTab(
                self._cloud_viewer, "3D 点云"
            )
        self._cloud_viewer.set_pointcloud(pcd)
        self._right_tabs.setCurrentIndex(self._cloud_tab_index)

    def closeEvent(self, event):
        self._stream_manager.stop_all()
        super().closeEvent(event)
