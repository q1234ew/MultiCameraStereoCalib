"""Main application window — modern dark-themed layout."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Slot, QSignalBlocker, QSize
from PySide6.QtGui import QDesktopServices
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QButtonGroup,
    QBoxLayout,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QLayout,
    QMessageBox,
    QSplitter,
    QScrollArea,
    QStatusBar,
    QStyle,
    QSizePolicy,
    QTabBar,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..board.charuco_board import CharucoBoard, CharucoBoardConfig
from ..build_info import build_label
from ..calibration.models import CameraModel, MultiCameraRig
from ..io.session import CalibrationSession, SessionManager
from ..runtime_paths import logo_png_path, logs_dir, sessions_dir
from ..streaming.stream_manager import StreamManager
from .widgets.calib_panel import CalibrationPanel
from .widgets.config_dialog import ConfigDialog
from .widgets.export_dialog import ExportDialog
from .widgets.rgb_settings_dialog import DEFAULT_RGB_SETTINGS_URL, RgbSettingsDialog
from .widgets.stream_view import MultiStreamView
from .widgets.pair_gallery import PairGallery
from .widgets.stereo_capture import StereoCaptureWorkspace
from .configuration import configuration_impact
from .theme import TEXT, TEXT_DIM, SUCCESS, WARNING, BORDER, standard_icon, app_icon

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MultiCamera Stereo Calibration")
        self.resize(1440, 960)
        self.setMinimumSize(1000, 680)

        logo_path = logo_png_path()
        if logo_path is not None:
            self.setWindowIcon(QIcon(str(logo_path)))

        self._stream_manager = StreamManager(parent=self)
        self._board = CharucoBoard(CharucoBoardConfig())
        self._camera_model = CameraModel.PINHOLE
        self._session_manager = SessionManager(sessions_dir())
        self._current_session: CalibrationSession | None = None
        self._rig: MultiCameraRig | None = None
        self._connection_expected: set[str] = set()
        self._connection_connected: set[str] = set()
        self._connection_failures_shown: set[str] = set()

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
        self._right_tabs.tabBar().hide()
        self._right_tabs.setStyleSheet("QTabWidget::pane {border:none;}")

        self._workspace = QWidget()
        workspace = QVBoxLayout(self._workspace)
        workspace.setContentsMargins(6, 8, 6, 8)
        workspace.setSpacing(8)
        self._stereo_capture = StereoCaptureWorkspace(
            self._calib_panel, self._stream_view, self._board,
        )
        workspace.addWidget(self._stereo_capture.banner)
        self._review_bar = QWidget()
        review = QHBoxLayout(self._review_bar)
        review.setContentsMargins(6, 0, 6, 0)
        self._lbl_review = QLabel()
        self._lbl_review.setTextFormat(Qt.PlainText)
        self._lbl_review.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._lbl_review.setStyleSheet(f"color:{WARNING};")
        review.addWidget(self._lbl_review, 1)
        self._btn_end_review = QToolButton()
        self._btn_end_review.setText("结束回看")
        self._btn_end_review.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._btn_end_review.setIcon(app_icon('undo-2'))
        self._btn_end_review.clicked.connect(self._stream_view.end_review)
        review.addWidget(self._btn_end_review)
        workspace.addWidget(self._review_bar)
        self._review_bar.hide()
        self._preview_body = QWidget()
        preview_body = QVBoxLayout(self._preview_body)
        preview_body.setContentsMargins(0, 0, 0, 0)
        self._stream_view.setMinimumHeight(230)
        self._capture_preview_splitter = QSplitter(Qt.Horizontal)
        self._capture_preview_splitter.setChildrenCollapsible(False)
        self._capture_preview_splitter.setHandleWidth(5)
        self._capture_preview_splitter.addWidget(self._stream_view)
        self._guide_scroll = QScrollArea()
        self._guide_scroll.setFrameShape(QFrame.NoFrame)
        self._guide_scroll.setWidgetResizable(True)
        self._guide_scroll.setMinimumWidth(300)
        self._guide_scroll.setMaximumWidth(440)
        self._guide_scroll.setWidget(self._stereo_capture)
        self._capture_preview_splitter.addWidget(self._guide_scroll)
        self._capture_preview_splitter.setStretchFactor(0, 1)
        self._capture_preview_splitter.setStretchFactor(1, 0)
        preview_body.addWidget(self._capture_preview_splitter, 1)
        self._preview_scroll = QScrollArea()
        self._preview_scroll.setFrameShape(QFrame.NoFrame)
        self._preview_scroll.setWidgetResizable(True)
        self._preview_scroll.setWidget(self._preview_body)
        workspace.addWidget(self._preview_scroll, 1)
        self._capture_bottom = QWidget()
        self._capture_bottom.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self._capture_bottom_layout = QHBoxLayout(self._capture_bottom)
        self._capture_bottom_layout.setContentsMargins(0, 0, 0, 0)
        self._capture_actions = QWidget()
        self._capture_actions.setMinimumWidth(310)
        self._capture_actions.setMaximumWidth(400)
        self._capture_actions_layout = QVBoxLayout(self._capture_actions)
        self._capture_actions_layout.setContentsMargins(0, 0, 0, 0)
        self._capture_actions_layout.setSizeConstraint(QLayout.SetMinimumSize)
        self._gallery = PairGallery()
        self._gallery._toggle.toggled.connect(self._on_gallery_expansion_changed)
        self._capture_bottom_layout.addWidget(self._gallery, 1, Qt.AlignTop)
        self._capture_bottom_layout.addWidget(self._capture_actions)
        workspace.addWidget(self._capture_bottom)
        self._calib_panel._capture_readiness = self._stereo_capture.auto_ready
        self._calib_panel._capture_validator = self._stereo_capture.validate_capture
        self._capture_layout_active = False
        self._capture_section_layout = self._calib_panel._sections[0].parentWidget().layout()
        self._capture_section_index = self._capture_section_layout.indexOf(self._calib_panel._sections[0])

        log_header = QHBoxLayout()
        self._toggle_logs = QToolButton()
        self._toggle_logs.setText("操作日志")
        self._toggle_logs.setArrowType(Qt.RightArrow)
        self._toggle_logs.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle_logs.setCheckable(True)
        log_header.addWidget(self._toggle_logs)
        self._lbl_latest_log = QLabel()
        self._lbl_latest_log.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._lbl_latest_log.setStyleSheet(f"font-size:12px; color:{TEXT_DIM};")
        log_header.addWidget(self._lbl_latest_log, 1)
        workspace.addLayout(log_header)
        self._logs_widget = self._calib_panel.take_log_widget()
        self._logs_widget.setMaximumHeight(150)
        self._logs_widget.setMinimumHeight(100)
        workspace.addWidget(self._logs_widget)
        self._logs_workspace_index = workspace.indexOf(self._logs_widget)
        self._aux_panel_dialogs = {}
        self._logs_widget.hide()
        self._toggle_logs.toggled.connect(self._toggle_log_panel)

        self._inspector = QFrame()
        self._inspector.setObjectName("inspectorColumn")
        self._inspector.setStyleSheet(f"QFrame#inspectorColumn {{border-left:1px solid {BORDER};}}")
        self._inspector.setMinimumWidth(360)
        self._inspector.setMaximumWidth(600)
        inspector = QVBoxLayout(self._inspector)
        inspector.setContentsMargins(0, 0, 0, 0)
        inspector.setSpacing(6)
        self._comparison_host = QWidget()
        self._comparison_host.setFixedHeight(238)
        comparison = QVBoxLayout(self._comparison_host)
        comparison.setContentsMargins(4, 4, 4, 0)
        self._stream_view.set_comparison_host(comparison)
        inspector.addWidget(self._comparison_host)
        self._comparison_host.hide()
        inspector.addWidget(self._right_tabs, 1)

        # Splitter for main content
        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.addWidget(self._workspace)
        self._splitter.addWidget(self._inspector)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(1)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([1000, 430])

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_workbench_header())
        layout.addWidget(self._splitter, 1)
        self.setCentralWidget(central)
        self._set_stereo_capture_layout(True)

    def _set_stereo_capture_layout(self, active):
        panel = self._calib_panel
        if active != self._capture_layout_active:
            section = panel._sections[0]
            footer = panel._action_footer
            if active:
                self._capture_section_layout.removeWidget(section)
                self._stereo_capture.details.setWidget(section)
                panel.layout().removeWidget(footer)
                self._capture_actions_layout.addWidget(footer)
                footer.layout().setContentsMargins(10, 8, 10, 8)
            else:
                self._stereo_capture.details.takeWidget()
                self._capture_section_layout.insertWidget(self._capture_section_index, section)
                self._capture_actions_layout.removeWidget(footer)
                panel.layout().addWidget(footer)
                footer.layout().setContentsMargins(16, 12, 16, 16)
            section.setVisible(active or panel._active_step == 0)
            footer.show()
            self._capture_layout_active = active
        self._stereo_capture.setVisible(active)
        self._guide_scroll.setVisible(active)
        self._stereo_capture.banner.setVisible(active)
        self._capture_actions.setVisible(active)
        self._inspector.setVisible(not active)
        self._view_aux.setVisible(not active)
        self._view_split.setVisible(not active)
        self._sync_capture_density()
        self._sync_auxiliary_panels()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_capture_layout_active'):
            self._sync_capture_density()
        if hasattr(self, '_toolbar'):
            self._sync_toolbar_density()
        if hasattr(self, '_aux_panel_dialogs'):
            self._sync_auxiliary_panels()

    def _sync_capture_density(self):
        compact = self._capture_layout_active
        if compact == getattr(self, '_compact_capture', None):
            return
        self._compact_capture = compact
        self._stereo_capture.set_compact(compact)
        self._calib_panel.set_capture_footer_compact(compact)
        self._stream_view.setMinimumHeight(170 if compact else 230)
        self._capture_actions.setMaximumWidth(16777215 if compact else 400)
        self._capture_bottom_layout.setDirection(
            QBoxLayout.TopToBottom if compact else QBoxLayout.LeftToRight,
        )
        if compact:
            self._gallery_before_compact = self._gallery._toggle.isChecked()
            self._changing_gallery_density = True
            self._gallery._toggle.setChecked(False)
            self._changing_gallery_density = False
        elif hasattr(self, '_gallery_before_compact'):
            self._changing_gallery_density = True
            self._gallery._toggle.setChecked(self._gallery_before_compact)
            self._changing_gallery_density = False
        self._capture_preview_splitter.setOrientation(Qt.Horizontal)
        self._capture_preview_splitter.setSizes([750, 350])

    def _on_gallery_expansion_changed(self, expanded):
        if getattr(self, '_compact_capture', False) and not getattr(self, '_changing_gallery_density', False):
            self._gallery_before_compact = expanded
        if hasattr(self, '_aux_panel_dialogs'):
            self._sync_auxiliary_panels()

    def _sync_auxiliary_panels(self):
        """Keep low-height windows usable without compressing the video area."""
        floating = self.height() < 900
        panels = (
            ('gallery', '配对图像', self._gallery._body, self._gallery.layout(), 1,
             self._gallery._toggle, bool(self._gallery._pairs)),
            ('logs', '操作日志', self._logs_widget, self._workspace.layout(),
             self._logs_workspace_index, self._toggle_logs, True),
        )
        for key, title, widget, home, index, toggle, available in panels:
            toggle.setToolTip(f'在独立窗口查看{title}' if floating else f'展开或收起{title}')
            dialog = self._aux_panel_dialogs.get(key)
            expanded = toggle.isChecked() and available
            if floating and (expanded or dialog is not None):
                if dialog is None:
                    dialog = QDialog(self, Qt.Tool)
                    dialog.setWindowTitle(title)
                    box = QVBoxLayout(dialog)
                    box.setContentsMargins(8, 8, 8, 8)
                    dialog.resize(min(900, self.width() - 80), 170)
                    dialog.finished.connect(lambda _result, t=toggle: t.setChecked(False))
                    self._aux_panel_dialogs[key] = dialog
                if widget.parentWidget() is not dialog:
                    home.removeWidget(widget)
                    dialog.layout().addWidget(widget)
                widget.setVisible(expanded)
                dialog.setVisible(expanded and self.isVisible())
            else:
                if dialog is not None and widget.parentWidget() is dialog:
                    dialog.layout().removeWidget(widget)
                    home.insertWidget(index, widget)
                if dialog is not None:
                    dialog.hide()
                widget.setVisible(expanded)

    def _build_workbench_header(self):
        header = QWidget()
        header.setObjectName("workbenchHeader")
        row = QHBoxLayout(header)
        row.setContentsMargins(12, 0, 12, 0)
        row.setSpacing(12)
        row.addWidget(QLabel("任务"))
        self._combo_task = QComboBox()
        self._combo_task.addItem("RGB 双目", False)
        self._combo_task.addItem("RGB + 红外/辅助", True)
        self._combo_task.setMinimumWidth(164)
        self._combo_task.setMaximumWidth(260)
        row.addWidget(self._combo_task)
        self._workflow_tabs = QTabBar()
        self._workflow_tabs.setObjectName("workflowTabs")
        self._workflow_tabs.setExpanding(False)
        for label in ("采集", "标定", "结果"):
            self._workflow_tabs.addTab(label)
        row.addWidget(self._workflow_tabs)
        row.addStretch()
        self._view_group = QButtonGroup(self)
        self._view_split = QToolButton()
        self._view_split.setText("并排")
        self._view_aux = QToolButton()
        self._view_aux.setText("红外优先")
        self._view_aux.setToolTip("放大辅助画面，RGB 左目保留在右上角")
        for index, button in enumerate((self._view_split, self._view_aux)):
            button.setCheckable(True)
            button.setMinimumHeight(30)
            self._view_group.addButton(button, index)
            row.addWidget(button)
        self._view_split.setChecked(True)
        self._view_aux.setEnabled(False)
        return header

    def _create_cloud_placeholder(self) -> QWidget:
        placeholder = QWidget()
        layout = QVBoxLayout(placeholder)
        label = QLabel("完成标定并生成点云后此处显示 3D 点云")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        return placeholder

    def _create_action(self, text):
        action = QAction(self)
        action.setText(text)
        return action

    def _init_menus(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件(&F)")

        self._act_config = self._create_action("相机配置(&C)...")
        self._act_config.setShortcut(QKeySequence("Ctrl+K"))
        file_menu.addAction(self._act_config)

        self._act_new_session = self._create_action("新建标定会话(&N)")
        self._act_new_session.setShortcut(QKeySequence.New)
        file_menu.addAction(self._act_new_session)

        self._act_load_session = self._create_action("加载会话(&L)...")
        self._act_load_session.setShortcut(QKeySequence.Open)
        file_menu.addAction(self._act_load_session)

        file_menu.addSeparator()

        self._act_export = self._create_action("导出标定结果(&E)...")
        self._act_export.setShortcut(QKeySequence("Ctrl+E"))
        self._act_export.setEnabled(False)
        file_menu.addAction(self._act_export)

        file_menu.addSeparator()

        self._act_exit = self._create_action("退出(&Q)")
        self._act_exit.setShortcut(QKeySequence.Quit)
        file_menu.addAction(self._act_exit)

        view_menu = menubar.addMenu("视图(&V)")
        self._act_restore_previews = self._create_action("恢复已关闭预览")
        self._act_restore_previews.setEnabled(False)
        view_menu.addAction(self._act_restore_previews)

        tools_menu = menubar.addMenu("工具(&T)")
        self._act_gen_board = self._create_action("生成标定板图像(&G)...")
        tools_menu.addAction(self._act_gen_board)
        self._act_rgb_settings = self._create_action("RGB 参数(&R)...")
        tools_menu.addAction(self._act_rgb_settings)
        tools_menu.addSeparator()
        self._act_open_logs = self._create_action("打开日志目录(&L)")
        tools_menu.addAction(self._act_open_logs)
        self._act_cloud_tools = self._create_action("点云工具")
        tools_menu.addAction(self._act_cloud_tools)

    def _init_toolbar(self):
        toolbar = QToolBar("主工具栏")
        self._toolbar = toolbar
        toolbar.setObjectName('mainToolbar')
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toolbar.setIconSize(toolbar.iconSize())
        self.addToolBar(toolbar)
        toolbar.setIconSize(QSize(18, 18))
        brand = QLabel("MultiCamera")
        brand.setStyleSheet(f"font-size:17px; font-weight:bold; color:{TEXT}; padding:0 10px;")
        toolbar.addWidget(brand)
        self._lbl_connections = QLabel("未连接")
        self._lbl_connections.setMinimumWidth(100)

        self._act_start_streams = self._create_action("连接相机")
        self._act_start_streams.setIcon(standard_icon(self, QStyle.SP_MediaPlay))
        self._act_start_streams.setToolTip('连接已配置的相机')
        toolbar.addAction(self._act_start_streams)

        self._act_stop_streams = self._create_action("断开")
        self._act_stop_streams.setIcon(standard_icon(self, QStyle.SP_MediaStop))
        self._act_stop_streams.setEnabled(False)
        self._act_stop_streams.setToolTip('断开全部相机')
        toolbar.addAction(self._act_stop_streams)
        toolbar.widgetForAction(self._act_stop_streams).setToolButtonStyle(Qt.ToolButtonIconOnly)
        toolbar.addWidget(self._lbl_connections)

        toolbar.addSeparator()

        self._act_config_tb = self._create_action("相机配置")
        self._act_config_tb.setIcon(standard_icon(self, QStyle.SP_FileDialogDetailedView))
        self._act_config_tb.setToolTip('配置相机连接、融合相机和标定板')
        toolbar.addAction(self._act_config_tb)

        self._act_gen_board_tb = self._create_action("生成标定板")

        self._act_rgb_settings_tb = self._create_action("RGB 参数")
        self._act_rgb_settings_tb.setIcon(app_icon('sliders-horizontal'))
        self._act_rgb_settings_tb.setToolTip('同步调整左右 RGB 相机参数')
        toolbar.addAction(self._act_rgb_settings_tb)
        toolbar.addSeparator()
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        self._session_button = QToolButton()
        self._session_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._session_button.setIcon(app_icon('folder-open'))
        self._session_button.setText("未创建会话")
        self._session_button.setToolTip("新建采集会话")
        self._session_button.setAccessibleName('当前会话与会话操作')
        self._session_button.setPopupMode(QToolButton.MenuButtonPopup)
        self._session_menu = QMenu(self._session_button)
        self._session_menu.addAction(self._act_new_session)
        self._session_menu.addAction(self._act_load_session)
        self._session_button.setMenu(self._session_menu)
        toolbar.addWidget(self._session_button)
        self._sync_toolbar_density()

    def _sync_toolbar_density(self):
        compact = self.width() < 1200
        style = Qt.ToolButtonIconOnly if compact else Qt.ToolButtonTextBesideIcon
        for action in (self._act_config_tb, self._act_rgb_settings_tb):
            button = self._toolbar.widgetForAction(action)
            button.setToolButtonStyle(style)
            button.setAccessibleName(action.text())
        self._session_button.setMinimumWidth(140 if compact else 180)
        self._session_button.setMaximumWidth(180 if compact else 250)
        self._refresh_session_button()

    def _refresh_session_button(self):
        if self._current_session is not None:
            name = self._current_session.name
            budget = self._session_button.maximumWidth() - 70
            self._session_button.setText(self._session_button.fontMetrics().elidedText(name, Qt.ElideMiddle, budget))
            self._session_button.setToolTip(f'{name}\n{self._current_session.session_dir.resolve()}')

    def _init_statusbar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("就绪")
        self._statusbar.setToolTip(self._statusbar.currentMessage())
        self._statusbar.messageChanged.connect(self._statusbar.setToolTip)
        identity = build_label()
        self._lbl_build = QLabel()
        self._lbl_build.setMaximumWidth(210)
        self._lbl_build.setText(self._lbl_build.fontMetrics().elidedText(identity, Qt.ElideMiddle, 200))
        self._lbl_build.setToolTip(identity + '\n源码修改不等同已发布安装包')
        self._lbl_build.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._statusbar.addPermanentWidget(self._lbl_build)

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
        self._act_rgb_settings.triggered.connect(self._on_rgb_settings)
        self._act_rgb_settings_tb.triggered.connect(self._on_rgb_settings)
        self._act_open_logs.triggered.connect(self._on_open_logs)
        self._act_restore_previews.triggered.connect(self._stream_view.restore_closed_previews)
        self._stream_view.preview_visibility_changed.connect(
            lambda: self._act_restore_previews.setEnabled(self._stream_view.has_closed_previews)
        )

        self._calib_panel.calibration_finished.connect(self._on_calib_finished)
        self._calib_panel.calibration_reset.connect(self._on_calib_reset)
        self._calib_panel.pointcloud_ready.connect(self._on_pointcloud)
        self._calib_panel.frame_captured.connect(self._stream_view.notify_capture)
        self._calib_panel.sequence_updated.connect(
            self._stream_view.set_capture_sequence
        )
        self._calib_panel.images_imported.connect(self._stream_view.show_static_images)
        self._calib_panel.aux_images_imported.connect(self._stream_view.show_multimodal_images)
        self._calib_panel.preview_mode_changed.connect(self._on_preview_mode_changed)
        self._calib_panel.configure_aux_requested.connect(lambda: self._on_config(True))
        self._calib_panel.configure_stereo_requested.connect(lambda: self._on_config(False))
        self._calib_panel.model_choice_requested.connect(self._on_model_choice)
        self._calib_panel.data_source_changed.connect(self._update_connection_summary)
        self._calib_panel.offline_imported.connect(self._gallery.show_first_page)
        self._calib_panel.diagnostic_preview_requested.connect(self._on_gallery_pair_selected)
        self._calib_panel.workflow_changed.connect(self._sync_workbench_header)
        self._calib_panel.preview_pairs_changed.connect(self._on_preview_pairs_changed)
        self._calib_panel.log_added.connect(self._on_log_added)
        self._calib_panel.export_requested.connect(self._on_export)
        self._calib_panel.open_results_requested.connect(self._open_result_directory)
        self._calib_panel.capture_started.connect(self._stream_view.end_review)
        self._calib_panel.create_session_requested.connect(self._on_new_session)
        self._gallery.pair_selected.connect(self._on_gallery_pair_selected)
        self._stream_view.review_changed.connect(self._on_review_changed)
        self._stream_view.presentation_changed.connect(self._on_presentation_changed)
        self._stream_view.comparison_visible_changed.connect(self._comparison_host.setVisible)
        self._view_split.clicked.connect(lambda: self._stream_view.set_presentation("split"))
        self._view_aux.clicked.connect(lambda: self._stream_view.set_presentation("aux_focus"))
        self._combo_task.currentIndexChanged.connect(self._on_workflow_navigation)
        self._workflow_tabs.currentChanged.connect(self._on_workflow_navigation)
        self._session_button.clicked.connect(self._open_session_directory)
        self._act_cloud_tools.triggered.connect(self._show_cloud_tools)

        self._stream_manager.camera_connected.connect(self._on_camera_connected)
        self._stream_manager.camera_disconnected.connect(
            self._on_camera_disconnected
        )

    # ── Slots ─────────────────────────────────────────────────

    @Slot(int)
    def _on_workflow_navigation(self, index):
        self._right_tabs.setCurrentIndex(0)
        phase = ("capture", "calibration", "results")[self._workflow_tabs.currentIndex()]
        self._calib_panel.show_workflow(phase, bool(self._combo_task.currentData()))

    @Slot(str, bool)
    def _sync_workbench_header(self, phase: str, multimodal: bool):
        self._set_stereo_capture_layout(phase == "capture" and not multimodal)
        self._view_aux.setVisible(multimodal)
        self._view_split.setVisible(multimodal)
        aux = self._stream_manager.auxiliary_camera
        label = "RGB + 声像仪 RGB" if aux is not None and aux.modality == "acoustic_rgb" else "RGB + 红外"
        with QSignalBlocker(self._combo_task), QSignalBlocker(self._workflow_tabs):
            generic = any(pair.left.stream_type != "rgb" for pair in self._stream_manager.stereo_pairs.values())
            self._combo_task.setItemText(0, "双目相机组" if generic else "RGB 双目")
            self._combo_task.setItemText(1, label)
            self._combo_task.setCurrentIndex(1 if multimodal else 0)
            self._workflow_tabs.setCurrentIndex({"capture": 0, "calibration": 1, "results": 2, "pointcloud": 2}[phase])
        self._view_aux.setText("辅助优先" if aux is not None and aux.modality == "acoustic_rgb" else "红外优先")
        self._on_presentation_changed(self._stream_view.presentation_mode)

    @Slot(str)
    def _on_presentation_changed(self, mode):
        with QSignalBlocker(self._view_split), QSignalBlocker(self._view_aux):
            self._view_split.setChecked(mode == "split")
            self._view_aux.setChecked(mode == "aux_focus")
        self._view_aux.setEnabled(self._stream_view.can_focus_aux)

    @Slot(object)
    def _on_preview_pairs_changed(self, pairs):
        if self._stream_view.review_key:
            current = next((p for p in pairs if p.key == self._stream_view.review_key), None)
            if current is None or not self._stream_view.review_matches(
                {current.left_id: current.left, current.right_id: current.right}
            ):
                self._stream_view.end_review()
            else:
                self._show_review_pair(current)
        self._gallery.set_pairs(pairs)
        self._sync_auxiliary_panels()

    @Slot(str, bool)
    def _on_review_changed(self, key, active):
        if not active:
            self._gallery.clear_selection()
        self._lbl_review.setText(f"回看 · {key}")
        self._lbl_review.setToolTip(key)
        self._review_bar.setVisible(active)
        self._stereo_capture.refresh()

    @Slot(str, object)
    def _on_gallery_pair_selected(self, key, images):
        self._calib_panel.select_offline_preview(key)
        pair = next((p for p in self._calib_panel.preview_pairs() if p.key == key), None)
        if pair is not None:
            self._show_review_pair(pair)
        else:
            self._stream_view.show_review_images(key, images)

    def _show_review_pair(self, pair):
        recorded = pair.left_detection is not None and pair.right_detection is not None
        self._stream_view.show_review_images(
            pair.key, {pair.left_id: pair.left, pair.right_id: pair.right},
            detections={pair.left_id: pair.left_detection, pair.right_id: pair.right_detection} if recorded else None,
            detection_origin=pair.detection_origin if recorded else '预览重检',
            corner_total=pair.corner_total,
            save_state=' · '.join(filter(None, (pair.save_state, pair.quality_state))))

    @Slot(bool)
    def _toggle_log_panel(self, expanded):
        self._toggle_logs.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._logs_widget.setVisible(expanded)
        self._sync_auxiliary_panels()

    @Slot(str)
    def _on_log_added(self, message):
        self._toggle_logs.setText(f"操作日志（{self._logs_widget.document().blockCount()}）")
        self._lbl_latest_log.setText(self._lbl_latest_log.fontMetrics().elidedText(
            message, Qt.ElideRight, max(120, self._lbl_latest_log.width()),
        ))
        self._lbl_latest_log.setToolTip(message)
        if "失败" in message or "无法连接" in message:
            self._toggle_logs.setChecked(True)
        self._refresh_session_button()

    def _update_connection_summary(self):
        count, expected = len(self._connection_connected), len(self._connection_expected)
        text = f"已连接 {count}/{expected} 路" if expected else (f"已连接 {count} 路" if count else "未连接")
        if not count and not expected and self._calib_panel.offline_selection:
            text = '离线数据'
        self._lbl_connections.setText(text)
        self._lbl_connections.setToolTip(text)
        color = SUCCESS if count and (not expected or count == expected) else (WARNING if expected else TEXT_DIM)
        self._lbl_connections.setStyleSheet(f"color:{color};")

    @Slot()
    def _open_session_directory(self):
        if self._current_session is None:
            self._on_new_session()
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._current_session.session_dir.resolve())))

    @Slot()
    def _open_result_directory(self):
        directory = self._calib_panel.result_directory
        if directory is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    @Slot()
    def _show_cloud_tools(self):
        self._right_tabs.setCurrentIndex(0)
        self._calib_panel.show_workflow("pointcloud")

    @Slot()
    def _on_config(self, auxiliary: bool = False):
        logger.info("Open camera configuration dialog")
        if self._calib_panel.has_active_operations or self._calib_panel.has_failed_saves:
            QMessageBox.information(self, "配置暂不可修改", "请先停止自动采集，等待后台任务完成；原图保存失败时请重试或撤销对应采集。")
            return
        was_streaming = self._act_stop_streams.isEnabled()
        preview_auxiliary = auxiliary or self._calib_panel._active_step == 2
        dlg = ConfigDialog(
            self._stream_manager, self._board, self._camera_model, parent=self
        )
        old_pairs = dict(self._stream_manager.stereo_pairs)
        old_aux = self._stream_manager.auxiliary_camera
        impact = None

        def confirm(board, model, pairs, aux):
            nonlocal impact
            impact = configuration_impact(self._board, board, self._camera_model, model,
                                          old_pairs, pairs, old_aux, aux)
            if not self._calib_panel.has_affected_work(impact):
                return True
            effects = []
            if impact.reset_stereo:
                effects.append("全部双目采图缓存和标定结果（板规格或镜头模型改变）")
            elif impact.changed_pairs:
                effects.append("以下组的采图缓存和标定结果：" + "、".join(sorted(impact.changed_pairs)))
            if impact.reset_aux:
                effects.append("辅助相机配对缓存和融合标定结果")
            approved = QMessageBox.question(
                self, "确认配置变更影响", "本次修改将失效：\n" + "\n".join(effects) +
                "\n\n未受影响的相机组将保留。磁盘历史图片和文件不会删除。是否应用？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            ) == QMessageBox.Yes
            if not approved:
                logger.info("Configuration rejected before apply")
            return approved

        dlg.before_apply = confirm
        if auxiliary:
            dlg.focus_auxiliary()
        if dlg.exec():
            if impact is None:
                return
            self._board = dlg.board
            self._camera_model = CameraModel(dlg.camera_model)
            if impact.streams_changed or impact.reset_stereo:
                affected = set(impact.camera_ids)
                if impact.reset_aux:
                    affected.update(a.camera_id for a in (old_aux, self._stream_manager.auxiliary_camera) if a)
                trackers = {cid: t for cid, t in self._stream_view._trackers.items()
                            if cid not in affected}
                mode_trackers = {mode: {cid: t for cid, t in items.items() if cid not in affected}
                                 for mode, items in self._stream_view._mode_trackers.items()}
                connection_state = dict(self._stream_view._connection_state)
                self._stream_view.clear_all()
                if not impact.streams_changed:
                    self._stream_view._connection_state = connection_state
                if impact.reset_stereo:
                    self._stream_view.update_board(self._board)
                else:
                    self._stream_view._trackers = trackers
                    self._stream_view._mode_trackers = mode_trackers
            self._calib_panel.apply_configuration(self._board, self._camera_model, impact)
            logger.info("Configuration impact applied: reset_stereo=%s pairs=%s reset_aux=%s reconnect=%s",
                        impact.reset_stereo, sorted(impact.changed_pairs), impact.reset_aux, impact.streams_changed)
            self._rig = self._calib_panel.rig
            self._act_export.setEnabled(self._rig is not None)
            if impact.streams_changed:
                self._act_start_streams.setEnabled(True)
                self._act_stop_streams.setEnabled(False)
                self._connection_expected.clear()
                self._connection_connected.clear()
                self._connection_failures_shown.clear()
                self._update_connection_summary()
            if impact.reset_stereo and not impact.streams_changed:
                self._stream_view.set_multimodal_mode(
                    self._calib_panel._active_step == 2, self._calib_panel._rgb_left_camera_id(),
                    force_refresh=True,
                )
            if self._current_session is not None and (impact.streams_changed or impact.reset_stereo):
                try:
                    self._current_session.record_configuration(
                        self._board, list(self._stream_manager.stereo_pairs.values()),
                        self._stream_manager.auxiliary_camera, camera_model=self._camera_model.value,
                    )
                except OSError as exc:
                    self._calib_panel.append_log(f"配置已应用，但会话配置记录失败：{exc}")
            if preview_auxiliary:
                self._calib_panel._activate_step(2)
            model_name = "针孔" if self._camera_model == CameraModel.PINHOLE else "鱼眼"
            logger.info(
                "Configuration accepted: pairs=%d aux=%s model=%s board=%s",
                len(self._stream_manager.stereo_pairs),
                bool(self._stream_manager.auxiliary_camera),
                self._camera_model.value,
                self._board.config,
            )
            self._statusbar.showMessage(
                f"✓  已配置 {len(self._stream_manager.stereo_pairs)} 组相机 [{model_name}模型]"
            )
            if was_streaming and impact.streams_changed:
                self._on_start_streams(preserve_trackers=True)

    @Slot(bool)
    def _on_preview_mode_changed(self, multimodal: bool):
        rgb_left_id = self._calib_panel._rgb_left_camera_id()
        self._stream_view.set_multimodal_mode(multimodal, rgb_left_id)
        if multimodal:
            aux = self._stream_manager.auxiliary_camera
            aux_label = aux.camera_id if aux is not None else "未配置红外/辅助相机"
            self._statusbar.showMessage(
                f"融合标定预览：RGB 左目 {rgb_left_id or '未配置'} + {aux_label}"
            )

    @Slot()
    def _on_new_session(self):
        if self._background_tasks_block("新建会话"):
            return
        pairs = list(self._stream_manager.stereo_pairs.values())
        if not pairs:
            QMessageBox.warning(self, "提示", "请先配置相机")
            return
        logger.info("Create calibration session: pairs=%d", len(pairs))
        try:
            session = self._session_manager.create_session(
                self._board, pairs, auxiliary=self._stream_manager.auxiliary_camera, name="calib",
                camera_model=self._camera_model.value,
            )
            self._calib_panel.set_session(session)
        except Exception as exc:
            logger.exception("Create calibration session failed")
            self._calib_panel.append_log(f"创建会话失败，未采图：{exc}")
            self._statusbar.showMessage(f"创建会话失败：{exc}")
            QMessageBox.warning(self, "创建会话失败", str(exc).splitlines()[0])
            return
        self._current_session = session
        self._statusbar.showMessage(f"✓  新建会话: {self._current_session.name}")

    @Slot()
    def _on_load_session(self):
        if self._background_tasks_block("加载会话"):
            return
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择会话目录", str(sessions_dir())
        )
        if dir_path:
            if self._background_tasks_block("加载会话"):
                return
            logger.info("Load calibration session: %s", dir_path)
            try:
                session = CalibrationSession.load(Path(dir_path))
                self._calib_panel.set_session(session)
            except Exception as exc:
                text = f"加载会话失败，已保留当前会话：{exc}"
                self._statusbar.showMessage(text)
                self._calib_panel.append_log(text)
                logger.exception("Load calibration session failed: %s", dir_path)
                return
            self._current_session = session
            self._statusbar.showMessage(f"✓  加载会话: {self._current_session.name}")

    @Slot()
    def _on_export(self):
        if self._rig is None:
            QMessageBox.information(
                self,
                "导出标定结果",
                "当前没有可导出的标定结果。请先完成双目标定或多模态标定。",
            )
            return
        dialog = ExportDialog(self._rig, self)
        if not dialog.exec():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存所选标定结果", dialog.suggested_filename, dialog.file_filter,
        )
        if path:
            target = Path(path)
            if not target.suffix:
                target = target.with_suffix(dialog.file_suffix)
                if target.exists() and QMessageBox.question(
                    self, "确认覆盖文件", f"文件已存在：{target}\n是否替换？",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                ) != QMessageBox.Yes:
                    return
            try:
                exported = dialog.export_to(target)
            except Exception as exc:
                logger.exception("Manual calibration export failed: path=%s format=%s", path, dialog.format_key)
                message = f"导出失败：{str(exc).splitlines()[0]}。已有文件未被替换。"
                self._calib_panel.append_log(message)
                QMessageBox.warning(self, "导出失败", message)
                return
            rig = dialog.selected_rig()
            logger.info("Export calibration: path=%s format=%s pairs=%s aux=%s", exported,
                        dialog.format_key, list(rig.pairs), list(rig.aux_cameras))
            self._statusbar.showMessage(f"✓  已导出: {exported}")
            self._calib_panel.append_log(f"已导出 {dialog.format_key}：{exported}")

    @Slot(str)
    def _on_camera_connected(self, cid: str):
        logger.info("Camera connected: %s", cid)
        self._connection_connected.add(cid)
        self._update_connection_summary()
        total = len(self._connection_expected)
        connected = len(self._connection_connected)
        suffix = f"（{connected}/{total}）" if total else ""
        self._statusbar.showMessage(f"✓  相机 {cid} 已连接{suffix}")
        url = self._stream_manager.camera_url(cid)
        self._calib_panel.append_log(f"相机已连接: {cid}  {url}")
        self._calib_panel.refresh_pairs()

    @Slot(str, str)
    def _on_camera_disconnected(self, cid: str, reason: str):
        self._connection_connected.discard(cid)
        self._update_connection_summary()
        url = self._stream_manager.camera_url(cid)
        logger.warning("Camera disconnected: %s url=%s reason=%s", cid, url, reason)
        lifecycle_reasons = {"已断开", "已移除", "重新连接"}
        if reason in lifecycle_reasons:
            self._statusbar.showMessage(f"相机 {cid} 已断开")
            return

        short_reason = reason.splitlines()[0]
        self._statusbar.showMessage(f"✗  相机 {cid} 无法连接: {short_reason}")
        message = self._format_connection_failure(cid, url, short_reason)
        first_failure = cid not in self._connection_failures_shown
        if first_failure:
            self._connection_failures_shown.add(cid)
            self._calib_panel.append_log(message)
            QMessageBox.warning(self, "相机连接失败", message)

    @Slot()
    def _on_start_streams(self, *, preserve_trackers: bool = False):
        logger.info("User requested start streams")
        camera_urls = self._stream_manager.configured_camera_urls()
        if not camera_urls:
            QMessageBox.warning(self, "连接相机", "请先通过“相机配置”添加至少一路实时相机。")
            self._statusbar.showMessage("未配置实时相机")
            return

        self._connection_expected = set(camera_urls.keys())
        self._connection_connected.clear()
        self._connection_failures_shown.clear()
        self._stream_view.show_connecting(camera_urls, preserve_trackers=preserve_trackers)
        self._stream_manager.start_all()
        self._act_start_streams.setEnabled(False)
        self._act_stop_streams.setEnabled(True)
        self._update_connection_summary()
        self._calib_panel.refresh_pairs()
        self._statusbar.showMessage(f"相机连接中... 共 {len(camera_urls)} 路")
        self._calib_panel.append_log(
            "开始连接相机: " + ", ".join(f"{cid}={url}" for cid, url in camera_urls.items())
        )

    @Slot()
    def _on_stop_streams(self):
        logger.info("User requested stop streams")
        self._stream_manager.stop_all()
        self._stream_view.clear_all()
        self._act_start_streams.setEnabled(True)
        self._act_stop_streams.setEnabled(False)
        self._connection_expected.clear()
        self._connection_connected.clear()
        self._update_connection_summary()
        self._connection_failures_shown.clear()
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
            logger.info("Save board image: %s", path)
            self._board.save_image(path)
            self._statusbar.showMessage(f"✓  标定板已保存: {path}")

    @Slot()
    def _on_rgb_settings(self):
        if self._calib_panel._white_balance_guard.busy:
            QMessageBox.information(self, "RGB 参数", "请等待白平衡准备完成或停止连接后再修改参数。")
            return
        url = self._default_rgb_settings_url()
        urls = self._rgb_settings_urls()
        logger.info("Open RGB settings dialog: url=%s sync_urls=%s", url, ",".join(urls))
        dlg = RgbSettingsDialog(url, sync_urls=urls, parent=self)
        dlg.settings_applied.connect(self._on_rgb_settings_applied)
        dlg.exec()

    @Slot(list)
    def _on_rgb_settings_applied(self, urls: list):
        self._calib_panel._white_balance_guard.invalidate()
        matched: list[str] = []
        applied_urls = [str(url) for url in urls if str(url).strip()]
        for url in applied_urls:
            for camera_id in self._stream_manager.discard_frames_for_url(url, settle_seconds=0.8):
                if camera_id not in matched:
                    matched.append(camera_id)
        if matched:
            self._stream_view.clear_camera_frames(matched, "RGB 参数已应用，等待新画面...")
            text = "RGB 参数已同步应用，已丢弃旧帧，等待新画面后再采集: " + ", ".join(matched)
            logger.info(text)
            self._statusbar.showMessage(text)
            self._calib_panel.append_log(text)
        else:
            text = "RGB 参数已应用；未匹配当前实时相机缓存，拍照前请等待预览刷新"
            logger.info("%s urls=%s", text, ",".join(applied_urls))
            self._statusbar.showMessage(text)
            self._calib_panel.append_log(text)

    @Slot()
    def _on_open_logs(self):
        path = logs_dir()
        logger.info("Open logs directory: %s", path)
        ok = QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        if ok:
            self._statusbar.showMessage(f"日志目录: {path}")
        else:
            QMessageBox.information(self, "日志目录", str(path))

    @Slot(object)
    def _on_calib_finished(self, rig: MultiCameraRig):
        logger.info(
            "Calibration finished: pairs=%d cameras=%d aux=%d reference=%s",
            len(rig.pairs),
            len(rig.extrinsics),
            len(rig.aux_cameras),
            rig.reference_camera,
        )
        self._rig = rig
        self._calib_panel.set_result_model(rig)
        self._act_export.setEnabled(True)
        try:
            if self._current_session is None:
                self._current_session = self._session_manager.create_session(
                    self._board,
                    list(self._stream_manager.stereo_pairs.values()),
                    auxiliary=self._stream_manager.auxiliary_camera,
                    name="calib",
                    camera_model=self._camera_model.value,
                )
                self._calib_panel.set_session(self._current_session, preserve_model_assessment=True)
            saved, errors = self._current_session.save_calibration_results(rig)
        except Exception as exc:
            logger.exception("Automatic calibration result save failed")
            text = f"标定计算完成，但自动保存失败：{exc}。可通过“导出标定结果”另存。"
            self._statusbar.showMessage(text)
            self._calib_panel.set_result_storage(None, errors=[str(exc)])
            self._calib_panel.append_log(text)
            QMessageBox.warning(self, "标定文件保存失败", text)
            return
        directory = saved[0].parent.resolve()
        self._calib_panel.set_result_storage(directory, saved, errors)
        text = f"标定完成，已保存 {len(saved)} 个文件：{directory}"
        self._calib_panel.append_log(text)
        for path in saved:
            logger.info("Calibration result saved: %s", path.resolve())
        if errors:
            for error in errors:
                self._calib_panel.append_log(f"自动导出失败：{error}")
            text += "（部分格式导出失败，详见操作日志）"
        self._statusbar.showMessage(text)

    @Slot()
    def _on_calib_reset(self):
        logger.info("Calibration state reset")
        self._rig = None
        self._calib_panel.set_result_model(None)
        self._act_export.setEnabled(False)
        self._statusbar.showMessage("标定数据或策略已更新，请重新标定后再导出")

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

    def _background_tasks_block(self, action: str) -> bool:
        if self._calib_panel.has_failed_saves:
            text = f'{action}已拒绝：仍有原图保存失败，请重试保存或撤销对应采集，避免丢失原图。'
            self._statusbar.showMessage(text)
            self._calib_panel.append_log(text)
            return True
        if not self._calib_panel.has_active_background_tasks():
            return False
        text = (
            f"{action}已拒绝：后台任务尚未完成，请先停止自动采集，"
            "并等待标定、导入、保存或白平衡准备安全结束后重试。"
        )
        self._statusbar.showMessage(text)
        self._calib_panel.append_log(text)
        logger.warning(text)
        return True

    @Slot(object)
    def _on_model_choice(self, model):
        model = CameraModel(model)
        if self._background_tasks_block('应用推荐模型'):
            return
        title = '鱼眼' if model == CameraModel.FISHEYE else '针孔'
        if QMessageBox.question(
                self, '确认模型变更',
                f'将模型改为{title}。该设置影响全部相机组及辅助相机，已有标定结果会失效。\n'
                '原始图像和检测观测将保留，历史文件不删除；应用后需重新标定。是否继续？',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        session = self._current_session
        previous = deepcopy(session._metadata) if session is not None else None
        try:
            if session is not None:
                session.record_configuration(self._board, list(self._stream_manager.stereo_pairs.values()),
                                             self._stream_manager.auxiliary_camera, model.value)
        except Exception as exc:
            session._metadata = previous
            self._calib_panel.append_log(f'模型配置保存失败，未应用：{exc}')
            return
        self._camera_model = model
        self._calib_panel.apply_recommended_model(model)

    def closeEvent(self, event):
        if self._background_tasks_block("关闭窗口"):
            event.ignore()
            return
        if not self._calib_panel._white_balance_guard.shutdown():
            text = "关闭窗口已拒绝：白平衡资源尚未释放，请稍后重试。"
            self._statusbar.showMessage(text)
            self._calib_panel.append_log(text)
            logger.warning(text)
            event.ignore()
            return
        self._stream_manager.stop_all()
        for dialog in self._aux_panel_dialogs.values():
            dialog.hide()
        super().closeEvent(event)

    def _default_rgb_settings_url(self) -> str:
        urls = self._rgb_settings_urls()
        if urls:
            return urls[0]
        return DEFAULT_RGB_SETTINGS_URL

    def _rgb_settings_urls(self) -> list[str]:
        urls: list[str] = []
        for pair in self._stream_manager.stereo_pairs.values():
            if pair.left.stream_type == "rgb" and pair.left.url:
                urls.append(pair.left.url)
            if pair.right.stream_type == "rgb" and pair.right.url:
                urls.append(pair.right.url)
        return urls

    @staticmethod
    def _format_connection_failure(camera_id: str, url: str, reason: str) -> str:
        hints = [
            "确认设备已上电并与本机在同一网络",
            "确认相机服务地址、端口和 /stream 路径正确",
            "如果是 Windows 打包程序，请检查防火墙是否允许访问局域网",
        ]
        lower = reason.lower()
        if "http " in lower:
            hints.insert(1, "服务已有响应，但 HTTP 状态异常，请检查推流服务是否正常")
        elif "timeout" in lower or "timed out" in lower or "超时" in reason:
            hints.insert(1, "连接超时，优先检查 IP 是否可达和网络延迟")
        elif "cannot connect" in lower or "connect call failed" in lower or "refused" in lower:
            hints.insert(1, "主机或端口不可达，请检查 IP、端口和相机服务是否启动")

        lines = [
            f"相机无法连接: {camera_id}",
            f"视频流: {url or '未知'}",
            f"原因: {reason}",
            "建议:",
        ]
        lines.extend(f"- {hint}" for hint in hints)
        lines.append(f"详细日志: {logs_dir() / 'multicamera.log'}")
        return "\n".join(lines)
