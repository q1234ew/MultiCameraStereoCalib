"""Main application window — modern dark-themed layout."""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Slot, QSignalBlocker, QSize, QTimer, QObject, QEvent
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
    QInputDialog,
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
from .configuration import configuration_impact, continuation_impact
from .theme import TEXT, TEXT_DIM, SUCCESS, WARNING, BORDER, standard_icon, app_icon

logger = logging.getLogger(__name__)


class _WorkbenchVisibilityGuard(QObject):
    """A tentative central-widget swap must not reset the old live-guide samples."""

    def __init__(self, central):
        super().__init__()
        self.widgets = [central, *central.findChildren(QWidget)]
        for widget in self.widgets:
            widget.installEventFilter(self)

    def eventFilter(self, watched, event):
        return event.type() in (QEvent.Show, QEvent.Hide)

    def release(self):
        for widget in self.widgets:
            widget.removeEventFilter(self)


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
        self._creating_session = False
        self._empty_created_session = None
        self._empty_session_config = None
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

    def _init_widgets(self, *, publish=True):
        # Left: camera views  |  Right: control panel + 3D viewer (tabs)
        self._stream_view = MultiStreamView(self._stream_manager, self._board, parent=self)

        self._calib_panel = CalibrationPanel(
            self._stream_manager, self._board, parent=self
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
        workspace.setContentsMargins(12, 12, 12, 8)
        workspace.setSpacing(8)
        self._stereo_capture = StereoCaptureWorkspace(
            self._calib_panel, self._stream_view, self._board,
        )
        self._stereo_capture.layout().insertWidget(0, self._stereo_capture.banner)
        self._preview_heading = QWidget()
        self._preview_heading.setFixedHeight(32)
        heading = QHBoxLayout(self._preview_heading)
        heading.setContentsMargins(2, 0, 2, 0)
        heading.setSpacing(12)
        self._preview_title = QLabel('双路预览')
        self._preview_title.setObjectName('workbenchSectionTitle')
        heading.addWidget(self._preview_title)
        self._preview_note = QLabel('原图检测 · 软件配对')
        self._preview_note.setStyleSheet(f'color:{TEXT_DIM}; font-size:12px;')
        heading.addWidget(self._preview_note)
        heading.addStretch()
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
        heading.addWidget(self._review_bar, 1)
        self._review_bar.hide()
        self._preview_body = QWidget()
        preview_body = QVBoxLayout(self._preview_body)
        preview_body.setContentsMargins(0, 0, 0, 0)
        self._stream_view.setMinimumHeight(230)
        self._capture_preview_splitter = QSplitter(Qt.Horizontal)
        self._capture_preview_splitter.setChildrenCollapsible(False)
        self._capture_preview_splitter.setHandleWidth(5)
        self._preview_left = QWidget()
        self._preview_left_layout = QVBoxLayout(self._preview_left)
        self._preview_left_layout.setContentsMargins(0, 0, 6, 0)
        self._preview_left_layout.setSpacing(10)
        self._preview_left_layout.addWidget(self._preview_heading)
        self._preview_left_layout.addWidget(self._stream_view, 1)
        self._capture_preview_splitter.addWidget(self._preview_left)
        self._guide_scroll = QScrollArea()
        self._guide_scroll.setFrameShape(QFrame.NoFrame)
        self._guide_scroll.setWidgetResizable(True)
        self._guide_scroll.setMinimumWidth(320)
        self._guide_scroll.setMaximumWidth(480)
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
        self._capture_actions.setMinimumWidth(0)
        self._capture_actions.setMaximumWidth(400)
        self._capture_actions_layout = QVBoxLayout(self._capture_actions)
        self._capture_actions_layout.setContentsMargins(0, 0, 0, 0)
        self._capture_actions_layout.setSizeConstraint(QLayout.SetMinimumSize)
        self._gallery = PairGallery()
        self._gallery._toggle.toggled.connect(self._on_gallery_expansion_changed)
        self._preview_left_layout.addWidget(self._gallery)
        self._receipt_strip = QWidget()
        self._receipt_strip.setObjectName('captureReceiptStrip')
        receipt = QHBoxLayout(self._receipt_strip)
        receipt.setContentsMargins(8, 4, 4, 4)
        receipt.addWidget(self._stereo_capture.receipt, 1)
        self._btn_receipt_details = QToolButton()
        self._btn_receipt_details.setText('回看原图')
        self._btn_receipt_details.setAccessibleName('回看当前回执原图')
        self._btn_receipt_details.setIcon(app_icon('chevron-right'))
        self._btn_receipt_details.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._btn_receipt_details.setToolTip('回看当前回执对应原图及其保存、验收结果')
        self._btn_receipt_details.clicked.connect(self._show_latest_capture)
        receipt.addWidget(self._btn_receipt_details)
        self._preview_left_layout.addWidget(self._receipt_strip)
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
        self._capture_ui_timer = QTimer(self)
        self._capture_ui_timer.setInterval(250)
        self._capture_ui_timer.timeout.connect(self._refresh_capture_navigation)
        self._capture_ui_timer.start()

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

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_workbench_header())
        layout.addWidget(self._splitter, 1)
        if publish:
            self.setCentralWidget(central)
        self._set_stereo_capture_layout(True)
        return central

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
        self._receipt_strip.setVisible(active)
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
        density = (compact, self.height() < 900)
        previous = getattr(self, '_capture_density', None)
        if density == previous:
            return
        self._capture_density = density
        self._compact_capture = compact
        self._stereo_capture.set_compact(compact)
        self._calib_panel.set_capture_footer_compact(compact)
        self._stream_view.setMinimumHeight(170 if compact else 230)
        self._capture_actions.setMaximumWidth(16777215 if compact else 400)
        self._capture_bottom_layout.setDirection(
            QBoxLayout.TopToBottom if compact else QBoxLayout.LeftToRight,
        )
        self._gallery.setFixedHeight(38 if self.height() < 900 else 150)
        if density[1] != (previous[1] if previous else False):
            self._changing_gallery_density = True
            if density[1]:
                self._gallery_before_small = self._gallery._toggle.isChecked()
                self._gallery._toggle.setChecked(False)
            else:
                self._gallery._toggle.setChecked(getattr(self, '_gallery_before_small', True))
            self._changing_gallery_density = False
        self._capture_preview_splitter.setOrientation(Qt.Horizontal)
        self._capture_preview_splitter.setSizes([max(600, self.width() - 420), 400])

    def _on_gallery_expansion_changed(self, expanded):
        if getattr(self, '_compact_capture', False) and not getattr(self, '_changing_gallery_density', False):
            self._gallery_before_compact = expanded
        if self.height() < 900 and not getattr(self, '_changing_gallery_density', False):
            self._gallery_before_small = expanded
        if hasattr(self, '_aux_panel_dialogs'):
            self._sync_auxiliary_panels()

    def _sync_auxiliary_panels(self):
        """Keep low-height windows usable without compressing the video area."""
        panels = (
            ('gallery', '配对图像', self._gallery._body, self._gallery.layout(), 1,
             self._gallery._toggle, bool(self._gallery._pairs)),
            ('logs', '操作日志', self._logs_widget, self._workspace.layout(),
             self._logs_workspace_index, self._toggle_logs, True),
        )
        for key, title, widget, home, index, toggle, available in panels:
            floating = self.height() < 900 or (key == 'logs' and self._capture_layout_active)
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
        self._btn_import_capture = QToolButton()
        self._btn_import_capture.setText('导入图片')
        self._btn_import_capture.setIcon(app_icon('folder-input'))
        self._btn_import_capture.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._btn_import_capture.setAccessibleName('导入当前任务的配对图片')
        self._btn_import_capture.clicked.connect(self._import_capture_images)
        row.addWidget(self._btn_import_capture)
        return header

    def _import_capture_images(self):
        panel = self._calib_panel
        button = panel._btn_import_aux if panel._active_step == 2 else panel._btn_import
        if panel.workflow_phase == 'capture' and button.isEnabled():
            button.click()

    def _show_latest_capture(self):
        """Resolve the displayed receipt, never fall back to an unrelated group's image."""
        panel = self._calib_panel
        pairs = panel.preview_pairs()
        key = self._stereo_capture.receipt_key
        latest = next((p for p in pairs if p.key == key), None)
        if latest is not None:
            self._on_gallery_pair_selected(latest.key, {latest.left_id: latest.left,
                                                       latest.right_id: latest.right})

    def _refresh_capture_navigation(self):
        panel = self._calib_panel
        if not hasattr(self, '_btn_import_capture'):
            return
        capture = panel.workflow_phase == 'capture'
        self._btn_import_capture.setVisible(capture)
        source = panel._btn_import_aux if panel._active_step == 2 else panel._btn_import
        self._btn_import_capture.setEnabled(capture and source.isEnabled())
        self._btn_import_capture.setToolTip(source.toolTip() or '导入配对原图，保留原始文件')
        self._btn_import_capture.setText(source.text())
        self._stereo_capture._refresh_receipt(list(panel._selected_pairs().values()), panel.offline_selection)
        key = self._stereo_capture.receipt_key
        has_capture = key is not None and any(p.key == key for p in self._gallery._pairs)
        self._btn_receipt_details.setEnabled(has_capture)
        self._btn_receipt_details.setVisible(not bool(panel.offline_selection))
        self._preview_note.setText('离线原图 · 不修改源文件' if panel.offline_selection else
                                   '原图检测 · 软件配对')

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
        self._act_load_calibration = self._create_action('读取双目标定文件...')
        file_menu.addAction(self._act_load_calibration)

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
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        self._lbl_connections = QLabel("未连接")
        self._lbl_connections.setMinimumWidth(100)
        toolbar.addWidget(self._lbl_connections)

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
        toolbar.addSeparator()

        self._act_config_tb = self._create_action("相机设置")
        self._act_config_tb.setIcon(standard_icon(self, QStyle.SP_FileDialogDetailedView))
        self._act_config_tb.setToolTip('配置相机连接、融合相机和标定板')
        toolbar.addAction(self._act_config_tb)

        self._act_gen_board_tb = self._create_action("生成标定板")

        self._act_rgb_settings_tb = self._create_action("RGB 参数")
        self._act_rgb_settings_tb.setIcon(app_icon('sliders-horizontal'))
        self._act_rgb_settings_tb.setToolTip('同步调整左右 RGB 相机参数')
        toolbar.addAction(self._act_rgb_settings_tb)
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
        else:
            self._session_button.setText('未创建会话')
            self._session_button.setToolTip('选择“新建标定会话”，或采图时自动创建；创建会话不会自动拍照')

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
        self._act_load_session.triggered.connect(self._on_open_session)
        self._act_load_calibration.triggered.connect(self._on_load_calibration_file)
        self._act_export.triggered.connect(self._on_export)
        self._act_exit.triggered.connect(self.close)
        self._act_start_streams.triggered.connect(self._on_start_streams)
        self._act_stop_streams.triggered.connect(self._on_stop_streams)
        self._act_gen_board.triggered.connect(self._on_gen_board)
        self._act_gen_board_tb.triggered.connect(self._on_gen_board)
        self._act_rgb_settings.triggered.connect(self._on_rgb_settings)
        self._act_rgb_settings_tb.triggered.connect(self._on_rgb_settings)
        self._act_open_logs.triggered.connect(self._on_open_logs)
        self._act_restore_previews.triggered.connect(lambda: self._stream_view.restore_closed_previews())
        self._session_button.clicked.connect(self._open_session_directory)
        self._act_cloud_tools.triggered.connect(self._show_cloud_tools)
        self._connect_workspace_signals()
        self._connect_manager_signals()

    def _connect_manager_signals(self):
        self._stream_manager.camera_connected.connect(self._on_camera_connected)
        self._stream_manager.camera_disconnected.connect(self._on_camera_disconnected)

    def _connect_workspace_signals(self):
        self._stream_view.preview_visibility_changed.connect(
            lambda: self._act_restore_previews.setEnabled(self._stream_view.has_closed_previews)
        )

        self._calib_panel.calibration_finished.connect(self._on_calib_finished)
        self._calib_panel.calibration_reset.connect(self._on_calib_reset)
        self._calib_panel.pointcloud_ready.connect(self._on_pointcloud)
        self._calib_panel.frame_captured.connect(self._stream_view.notify_capture)
        self._calib_panel.aux_original_checked.connect(self._stream_view.show_aux_original_receipt)
        self._calib_panel.sequence_updated.connect(
            self._stream_view.set_capture_sequence
        )
        self._calib_panel._aux_guide.bind_preview(self._stream_view)
        self._calib_panel.images_imported.connect(self._stream_view.show_static_images)
        self._calib_panel.aux_images_imported.connect(self._stream_view.show_multimodal_images)
        self._calib_panel.preview_mode_changed.connect(self._on_preview_mode_changed)
        self._calib_panel.configure_aux_requested.connect(lambda: self._on_config(True))
        self._calib_panel.load_rgb_session_requested.connect(self._on_open_session)
        self._calib_panel.load_rgb_file_requested.connect(self._on_load_calibration_file)
        self._calib_panel.continue_aux_capture_requested.connect(self._on_continue_aux_capture)
        self._calib_panel.configure_stereo_requested.connect(lambda: self._on_config(False))
        self._calib_panel.model_choice_requested.connect(self._on_model_choice)
        self._calib_panel.data_source_changed.connect(self._update_connection_summary)
        self._calib_panel.data_source_changed.connect(self._refresh_capture_navigation)
        self._calib_panel._combo_pair.currentTextChanged.connect(self._refresh_capture_navigation)
        self._calib_panel.offline_imported.connect(self._gallery.show_first_page)
        self._calib_panel.diagnostic_preview_requested.connect(self._on_gallery_pair_selected)
        self._calib_panel.workflow_changed.connect(self._sync_workbench_header)
        self._calib_panel.preview_pairs_changed.connect(self._on_preview_pairs_changed)
        self._calib_panel.log_added.connect(self._on_log_added)
        self._calib_panel.export_requested.connect(self._on_export)
        self._calib_panel.open_results_requested.connect(self._open_result_directory)
        self._calib_panel.capture_started.connect(self._stream_view.end_review)
        self._calib_panel.create_session_requested.connect(self._ensure_session_for_capture)
        self._gallery.pair_selected.connect(self._on_gallery_pair_selected)
        self._stream_view.review_changed.connect(self._on_review_changed)
        self._stream_view.presentation_changed.connect(self._on_presentation_changed)
        self._stream_view.comparison_visible_changed.connect(self._comparison_host.setVisible)
        self._view_split.clicked.connect(lambda: self._stream_view.set_presentation("split"))
        self._view_aux.clicked.connect(lambda: self._stream_view.set_presentation("aux_focus"))
        self._combo_task.currentIndexChanged.connect(self._on_workflow_navigation)
        self._workflow_tabs.currentChanged.connect(self._on_workflow_navigation)

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
        self._refresh_capture_navigation()

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
        if self._capture_layout_active:
            self._stereo_capture.refresh()
        self._refresh_capture_navigation()
        self._sync_auxiliary_panels()

    @Slot(str, bool)
    def _on_review_changed(self, key, active):
        if not active:
            self._gallery.clear_selection()
        self._lbl_review.setText(f"回看 · {key}")
        self._lbl_review.setToolTip(key)
        self._review_bar.setVisible(active)
        self._stereo_capture.refresh()
        self._refresh_capture_navigation()

    @Slot(str, object)
    def _on_gallery_pair_selected(self, key, images):
        self._calib_panel.select_offline_preview(key)
        pair = next((p for p in self._calib_panel.preview_pairs() if p.key == key), None)
        if pair is not None:
            self._show_review_pair(pair)
        else:
            self._stream_view.show_review_images(key, images)
        self._calib_panel.show_review_rectification(key, images)

    def _show_review_pair(self, pair):
        recorded = pair.left_detection is not None and pair.right_detection is not None
        self._stream_view.show_review_images(
            pair.key, {pair.left_id: pair.left, pair.right_id: pair.right},
            detections={pair.left_id: pair.left_detection, pair.right_id: pair.right_detection} if recorded else None,
            detection_origin=pair.detection_origin if recorded else '预览重检',
            corner_total=pair.corner_total,
            save_state=' · '.join(filter(None, (pair.save_state, pair.display_quality_state))))

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
            self._session_button.showMenu()
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
    def _on_config(self, auxiliary: bool = False, *, resume_rgb: bool = False):
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
        if resume_rgb and self._current_session is not None:
            dlg.prefill_session_urls(self._current_session.metadata)

        def persist_resume(board, model, pairs, aux):
            if not resume_rgb:
                return True
            try:
                self._current_session.record_configuration(board, pairs, aux,
                    camera_model=CameraModel(model).value, continuation={
                        'confirmed_at': datetime.now().isoformat(),
                        'same_rgb_device_lens_settings': 'user_confirmed',
                        'rgb_result': deepcopy(self._calib_panel._rgb_result_provenance),
                        'bindings': {name: {'left_id': pair.left_id, 'right_id': pair.right_id,
                            'image_size': list(pair.left_intrinsics.image_size)}
                            for name, pair in self._calib_panel._pair_calibs.items()},
                    })
            except Exception as exc:
                text = f'续采配置记录失败，原配置和复核状态保留，未连接或采图：{exc}'
                self._calib_panel.append_log(text)
                QMessageBox.warning(self, '续采配置未应用', text)
                return False
            return True

        def confirm(board, model, pairs, aux):
            nonlocal impact
            if resume_rgb:
                try:
                    impact = continuation_impact(self._board, board, self._camera_model, model,
                        old_pairs, pairs, old_aux, aux, self._calib_panel._pair_calibs)
                    rgb_id = self._calib_panel._rgb_left_camera_id()
                    reference = next((p for p in pairs if p.left.camera_id == rgb_id), None)
                    if reference is None or not reference.left.url or aux is None or not aux.url:
                        raise ValueError('在线续采需填写原 RGB 左右流地址，并启用红外/辅助实时地址')
                    if self._current_session.multimodal_resume_state()['saved_pairs'] and old_aux and (
                            aux.camera_id, aux.modality, aux.pattern_board_config()) != (
                            old_aux.camera_id, old_aux.modality, old_aux.pattern_board_config()):
                        raise ValueError('会话已有融合原图，续采不能更换辅助相机或标定板；请新建独立会话')
                except ValueError as exc:
                    QMessageBox.warning(self, '续采配置不兼容', str(exc))
                    return False
            else:
                impact = configuration_impact(self._board, board, self._camera_model, model,
                                              old_pairs, pairs, old_aux, aux)
            if not self._calib_panel.has_affected_work(impact):
                return persist_resume(board, model, pairs, aux)
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
            return approved and persist_resume(board, model, pairs, aux)

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
            if (not resume_rgb and self._current_session is not None and
                    (impact.streams_changed or impact.reset_stereo)):
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
            if was_streaming and impact.streams_changed and not resume_rgb:
                self._on_start_streams(preserve_trackers=True)
            return True
        return False

    @Slot()
    def _on_continue_aux_capture(self):
        if self._background_tasks_block('继续融合采集'):
            return
        panel = self._calib_panel
        if self._current_session is None or panel._rgb_left_camera_id() not in panel._intrinsics:
            QMessageBox.information(self, '先加载双目结果', '请先加载已完成双目标定的会话，再继续融合采集。')
            return
        approved = QMessageBox.question(self, '确认沿用双目标定',
            '请确认仍是原 RGB 左右相机、原左右顺序，镜头、焦距、对焦和输出分辨率未改变。\n'
            '确认后可补充视频地址及红外配置，保留双目内参，并连接视频继续采图到当前会话。\n'
            '已有红外原图保留，辅助相机和板规格也须保持一致；需逐对复核后纳入联合标定。'
            '若设备或镜头改变，请取消并重新标定。',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if approved != QMessageBox.Yes or self._background_tasks_block('继续融合采集'):
            return
        if not self._on_config(True, resume_rgb=True):
            return
        panel.restore_saved_aux_progress()
        panel.show_workflow('capture', True)
        panel.append_log('已保留 RGB 双目标定，继续当前会话融合采集；'
                         '仅沿用已确认的设备/镜头设置，实时分辨率仍须匹配内参')
        self._on_start_streams()

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
        self._create_session(preserve_observations=False)

    @Slot()
    def _ensure_session_for_capture(self):
        if self._current_session is None:
            self._create_session(preserve_observations=True)
        elif self._calib_panel._session is None:
            self._calib_panel.set_session(self._current_session)
            self._refresh_session_button()

    def _session_configuration(self):
        return dict(board=self._board.to_dict(), model=self._camera_model.value,
            pairs=[(p.name, p.left.camera_id, p.left.url, p.right.camera_id, p.right.url)
                   for p in self._stream_manager.stereo_pairs.values()],
            auxiliary=self._stream_manager.auxiliary_camera.to_dict()
                      if self._stream_manager.auxiliary_camera is not None else None)

    def _create_session(self, *, preserve_observations):
        if self._creating_session:
            self._statusbar.showMessage('正在创建会话，请稍候；请勿重复操作')
            return
        if self._background_tasks_block("新建会话"):
            return
        pairs = list(self._stream_manager.stereo_pairs.values())
        if not pairs:
            QMessageBox.warning(self, "提示", "请先配置相机")
            return
        configuration = self._session_configuration()
        current = self._current_session
        if (current is not None and current is self._empty_created_session
                and self._empty_session_config == configuration
                and not self._calib_panel.has_session_data and current.is_pristine()):
            self._refresh_session_button()
            text = f'当前会话尚未采图，继续使用 {current.name}；未重复创建目录'
            self._statusbar.showMessage(text)
            self._calib_panel.append_log(text)
            return
        if (not preserve_observations and self._calib_panel.has_session_data
                and QMessageBox.question(self, '新建独立会话',
                    '新会话将清空当前内存中的采图、引导和标定结果。磁盘历史文件不会删除。\n'
                    '尚未落盘的导入数据仍保留在源目录。是否新建？',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes):
            return
        self._creating_session = True
        self._act_new_session.setEnabled(False)
        self._session_button.setEnabled(False)
        previous = self._calib_panel._session
        session = None
        logger.info("Create calibration session: pairs=%d", len(pairs))
        try:
            session = self._session_manager.create_session(
                self._board, pairs, auxiliary=self._stream_manager.auxiliary_camera, name="calib",
                camera_model=self._camera_model.value,
            )
            self._calib_panel.set_session(session)
        except Exception as exc:
            if session is not None and session is not previous:
                if self._calib_panel._session is session:
                    self._calib_panel._session = previous
                session.discard_unpublished()
            logger.exception("Create calibration session failed")
            self._calib_panel.append_log(f"创建会话失败，未采图：{exc}")
            self._statusbar.showMessage(f"创建会话失败：{exc}")
            QMessageBox.warning(self, "创建会话失败", str(exc).splitlines()[0])
            return
        finally:
            self._creating_session = False
            self._act_new_session.setEnabled(True)
            self._session_button.setEnabled(True)
        self._current_session = session
        if not preserve_observations:
            multimodal = self._calib_panel._active_step == 2
            self._calib_panel._reset_all()
            self._calib_panel.show_workflow('capture', multimodal)
            self._stream_view.begin_capture_session()
            self._stereo_capture.refresh()
        self._empty_created_session = session
        self._empty_session_config = deepcopy(configuration)
        self._refresh_session_button()
        self._statusbar.showMessage(f'新建会话成功：{session.name}；尚未采图')
        self._calib_panel.append_log(f'新建会话成功：{session.name}；点击采集后才保存原图')

    @Slot()
    def _on_load_session(self, *, choose_mode=False):
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
                from ..io.session_restore import inspect_session_files
                inventory = inspect_session_files(session)
                self._calib_panel.append_log(inventory['text'])
                rig = session.load_rig()
                open_mode = None
                if choose_mode:
                    from ..io.session_restore import prepare_session_restore
                    plan = prepare_session_restore(session, allow_uncomputed=rig is None)
                    open_mode = self._choose_session_open_mode(session, plan, inventory)
                    if open_mode is None:
                        return
                if rig is None:
                    if not choose_mode and (self._rig is not None or self._calib_panel._active_step == 2):
                        raise ValueError('会话缺少 RGB 标定结果，不能用于融合；已保留当前有效状态')
                    if session.metadata.get('board') and session.metadata.get('pairs'):
                        from ..io.session_restore import prepare_session_restore
                        plan = prepare_session_restore(session, allow_uncomputed=True)
                        if not self._load_session_result(session, plan,
                                confirm_workspace=not choose_mode, force_offline=choose_mode):
                            return
                    else:
                        raise ValueError('会话缺少标定板或相机组配置，无法恢复工作区')
                    self._calib_panel.append_log('已加载原图会话，尚无 RGB 标定结果；融合前需完成 RGB 标定')
                else:
                    from ..io.session_restore import prepare_session_restore
                    plan = prepare_session_restore(session)
                    if not self._load_session_result(session, plan,
                            confirm_workspace=not choose_mode, force_offline=choose_mode):
                        return
            except Exception as exc:
                text = f"加载会话失败，已保留当前会话：{exc}"
                self._statusbar.showMessage(text)
                self._calib_panel.append_log(text)
                logger.exception("Load calibration session failed: %s", dir_path)
                return
            self._current_session = session
            self._rig = self._calib_panel.rig
            self._act_export.setEnabled(self._rig is not None)
            self._empty_created_session = None
            self._empty_session_config = None
            self._refresh_session_button()
            self._session_button.setToolTip(inventory['text'] + '\n' + str(session.session_dir))
            self._statusbar.showMessage(f'加载会话：{session.name}；融合原图 {inventory["aux_pairs"]} 组；'
                                       '缺少项及下一步见会话文件检查日志')
            self._calib_panel.restore_session_images(auto_aux=inventory['auto_import'])
            if open_mode == 'online':
                self._schedule_session_continuation(session)

    @Slot()
    def _on_open_session(self):
        self._on_load_session(choose_mode=True)

    def _choose_session_open_mode(self, session, plan, inventory):
        from .widgets.session_open_dialog import SessionOpenDialog
        dialog = SessionOpenDialog(session.name, inventory['text'],
            self._session_workspace_differences(session, plan),
            can_continue=bool(plan.rig.pairs) and not plan.legacy_offline, parent=self)
        return dialog.mode if dialog.exec() == QDialog.Accepted else None

    def _schedule_session_continuation(self, session):
        previous = getattr(self, '_session_continuation_timer', None)
        if previous is not None:
            previous.stop()
            previous.deleteLater()
        timer = QTimer(self)
        self._session_continuation_timer = timer
        timer.setInterval(100)

        def resume():
            if self._current_session is not session:
                timer.stop()
                self._session_continuation_timer = None
                timer.deleteLater()
                return
            panel = self._calib_panel
            if (panel._session_restore_queue or panel._session_restore_aux or
                    panel._stereo_import_worker or panel._aux_import_worker):
                return
            timer.stop()
            self._session_continuation_timer = None
            timer.deleteLater()
            self._on_continue_aux_capture()

        timer.timeout.connect(resume)
        timer.start()

    def _session_workspace_differences(self, session, plan):
        """Inspect logical bindings without connecting devices or publishing a workspace."""
        differences = []
        manager = self._stream_manager
        current = manager.stereo_pairs
        recorded = {pair['name']: pair for pair in session.metadata.get('pairs', [])}
        for pair in plan.pairs:
            cfg = current.get(pair.name)
            if cfg is None:
                differences.append(f'相机组 {pair.name}：当前未配置；会话左 {pair.left.camera_id} / '
                                   f'右 {pair.right.camera_id}')
                continue
            binding = recorded.get(pair.name, {})
            for side, label in (('left', '左目'), ('right', '右目')):
                live, saved = getattr(cfg, side), getattr(pair, side)
                if live.camera_id != saved.camera_id:
                    differences.append(f'{pair.name} {label} ID：当前 {live.camera_id}；'
                                       f'会话 {saved.camera_id}')
                if live.stream_type != saved.stream_type:
                    differences.append(f'{pair.name} {label}视频类型：当前 {live.stream_type}；'
                                       f'会话 {saved.stream_type}')
                if not manager.is_virtual_pair(pair.name) and live.url != binding.get(side + '_url', ''):
                    # Identify the transport mismatch without exposing addresses or credentials.
                    differences.append(f'{pair.name} {label}视频地址不同（地址变化不等于已确认同一设备）')

        def compare_fields(label, actual, expected, fields, *, exact=False):
            for field, name in fields.items():
                if actual.get(field) != expected.get(field):
                    differences.append(f'{label} {name}：当前 {actual.get(field)}；会话 {expected.get(field)}')
            if exact:
                for field in sorted((actual.keys() | expected.keys()) - fields.keys()):
                    if field not in actual or field not in expected or actual[field] != expected[field]:
                        differences.append(f'{label} 扩展参数 {field} 不同')

        if self._camera_model != plan.model:
            differences.append(f'RGB 镜头模型：当前 {self._camera_model.value}；会话 {plan.model.value}')
        compare_fields('RGB 标定板', self._board.to_dict(), plan.board.to_dict(), {
            'cols': '列格数', 'rows': '行格数', 'square_length': '方格边长(m)',
            'marker_length': '标记边长(m)', 'dict_type': '字典',
        }, exact=True)
        aux, saved_aux = manager.auxiliary_camera, plan.auxiliary
        if aux is None and saved_aux is not None:
            differences.append(f'辅助相机：当前未配置；会话 {saved_aux.camera_id}（{saved_aux.modality}）')
        elif aux is not None and saved_aux is None:
            differences.append(f'辅助相机：当前 {aux.camera_id}（{aux.modality}）；会话未配置')
        elif aux is not None and saved_aux is not None:
            compare_fields('辅助相机', aux.to_dict(), saved_aux.to_dict(), {
                'camera_id': 'ID', 'modality': '模态',
            })
            compare_fields('辅助标定板', aux.pattern_board_config().to_dict(),
                           saved_aux.pattern_board_config().to_dict(), {
                'cols': '列点数', 'rows': '行点数', 'square_size': '点间距(m)',
                'pattern_type': '板型', 'asymmetric': '非对称网格',
            }, exact=True)
            if aux.url and aux.url != (session.metadata.get('auxiliary_camera') or {}).get('url'):
                differences.append('辅助相机视频地址不同（地址变化不等于已确认同一设备）')
        for name, pair in plan.rig.pairs.items():
            cfg = current.get(name)
            for camera_id, intr, camera in (
                    (pair.left_id, pair.left_intrinsics, cfg.left if cfg else None),
                    (pair.right_id, pair.right_intrinsics, cfg.right if cfg else None)):
                if camera is None or camera.camera_id != camera_id:
                    continue
                frame = manager.get_latest_frame(camera_id)
                if frame is not None and frame[0].shape[1::-1] != intr.image_size:
                    width, height = frame[0].shape[1::-1]
                    differences.append(f'{camera_id} 实时图像尺寸：当前 {width}×{height}；'
                                       f'会话内参 {intr.image_size[0]}×{intr.image_size[1]}')
        if plan.legacy_offline:
            differences.append('旧版离线结果未记录虚拟相机绑定；仅按 offline_L/R 恢复离线组，'
                               '不映射到真实相机')
        return differences

    def _load_session_result(self, session, plan, *, confirm_workspace=True, force_offline=False):
        from ..io.session_restore import prepare_session_result
        plan.verify_sources()
        prepared = prepare_session_result(session, plan.rig, plan.auxiliary) if plan.rig.pairs else None
        differences = self._session_workspace_differences(session, plan)
        offline = force_offline or bool(differences)
        if offline and confirm_workspace:
            details = '\n'.join(f'{pair.name}: 左 {pair.left.camera_id} / 右 {pair.right.camera_id}'
                                for pair in plan.pairs)
            reasons = '\n'.join('• ' + item for item in differences) or '本次导入要求离线打开。'
            continuation = ('继续 RGB + 声像仪 RGB 采集' if plan.auxiliary is not None and
                            plan.auxiliary.modality == 'acoustic_rgb' else '继续 RGB + 红外采集')
            answer = QMessageBox.question(self, '按会话信息离线打开',
                '是否按会话配置离线打开？具体隔离原因：\n'
                f'{reasons}\n\n会话绑定：\n{details}\n\n'
                '将断开当前视频流，替换内存中的采图和结果，恢复会话板规格及模型；'
                '不自动连接历史地址，不把旧内参应用到当前设备，磁盘历史文件不变。\n'
                '这是离线打开，不是在线续拍确认；相机组名或 ID 相同不能证明实物设备及镜头相同。\n'
                f'RGB+IR/辅助在线续采请在加载后使用结果页“{continuation}”，'
                '核对原左右设备、镜头/焦距/对焦及输出分辨率。\n'
                'RGB 双目补拍同样须核对原设备及镜头；普通“相机配置”修改地址会按配置变更处理，'
                '可能失效已恢复观测和结果。取消则保留当前工作区和视频流。',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                text = '已取消加载会话，当前会话和相机配置未改变'
                self._calib_panel.append_log(text)
                self._statusbar.showMessage(text)
                return False
            if self._background_tasks_block('加载会话'):
                return False
        self._stage_session_workspace(session, plan, prepared, offline)
        return True

    def _stage_session_workspace(self, session, plan, prepared, offline):
        """Build a detached workbench; keep the old QObject graph and streams untouched."""
        previous = dict(self.__dict__)
        old_central = self.centralWidget()
        old_visible = old_central.isVisible()
        old_children = set(self.children())
        old_objects = {id(obj) for obj in self.findChildren(QObject)}
        candidate = None
        visibility = _WorkbenchVisibilityGuard(old_central)
        try:
            self._board, self._camera_model = plan.board, plan.model
            self._current_session = session
            if offline:
                self._stream_manager = StreamManager(parent=self)
                for pair in plan.pairs:
                    self._stream_manager.add_virtual_stereo_pair(pair)
                if plan.auxiliary is not None:
                    self._stream_manager.set_auxiliary_camera(plan.auxiliary)
            # Density caching belongs to the widgets, not the top-level window.
            self._capture_density = None
            candidate = self._init_widgets(publish=False)
            self._calib_panel._camera_model = plan.model
            old_panel = previous['_calib_panel']
            for name in ('_combo_calib_strategy', '_combo_data_filter',
                         '_combo_capture_quality', '_combo_wb_mode'):
                combo = getattr(self._calib_panel, name)
                with QSignalBlocker(combo):
                    combo.setCurrentIndex(combo.findData(getattr(old_panel, name).currentData()))
            self._calib_panel._spin_min.setValue(old_panel._spin_min.value())
            self._calib_panel._check_wb.setChecked(old_panel._check_wb.isChecked())
            self._stereo_capture.guide_mode.setCurrentIndex(self._stereo_capture.guide_mode.findData(
                previous['_stereo_capture'].guide_mode.currentData()))
            self._calib_panel._log_text.setPlainText(old_panel._log_text.toPlainText())
            if plan.rig.pairs:
                self._calib_panel.restore_session_result(session, plan.rig,
                    board=plan.board, camera_model=plan.model, auxiliary=plan.auxiliary,
                    prepared=prepared)
            else:
                self._calib_panel.set_session(session)
            if offline:
                self._stream_view.update_board(plan.board)
                self._calib_panel.append_log('已按会话配置离线打开；未连接设备，历史原图和结果未改写')
                if plan.legacy_offline:
                    self._calib_panel.append_log('旧版 offline_L/R 已恢复为虚拟离线组；未绑定真实设备')
            else:
                self._stream_view.begin_capture_session()
            self._calib_panel.refresh_pairs()
            multimodal = plan.auxiliary is not None and bool(plan.rig.pairs)
            phase = 'results' if plan.rig.pairs else 'calibration'
            self._calib_panel.show_workflow(phase, multimodal)
            self._on_preview_mode_changed(multimodal)
            self._stream_view.set_presentation(previous['_stream_view'].presentation_mode)
            self._sync_workbench_header(phase, multimodal)
            self._connect_workspace_signals()
            if offline:
                self._connect_manager_signals()
            session.verify_loaded_metadata()
            plan.verify_sources()
            if prepared is not None:
                prepared.verify_sources()
            self._rig = self._calib_panel.rig
            # Only this pointer swap publishes the fully prepared workbench.
            self.takeCentralWidget()
            self.setCentralWidget(candidate)
        except Exception:
            abandoned = dict(self.__dict__)
            new_children = set(self.children()) - old_children
            self.__dict__.clear()
            self.__dict__.update(previous)
            if self.centralWidget() is not old_central:
                self.takeCentralWidget()
                QMainWindow.setCentralWidget(self, old_central)
                old_central.setVisible(old_visible)
            self._dispose_session_workspace(abandoned, candidate, new_children,
                                            keep_manager=previous['_stream_manager'],
                                            protected=old_objects)
            raise
        finally:
            visibility.release()
        # Cleanup happens after publication and cannot turn success into a false rollback report.
        self._dispose_session_workspace(previous, old_central, (),
                                        keep_manager=self._stream_manager)
        try:
            if offline:
                self._connection_expected = set()
                self._connection_connected = set()
                self._connection_failures_shown = set()
                self._act_start_streams.setEnabled(True)
                self._act_stop_streams.setEnabled(False)
            self._act_restore_previews.setEnabled(False)
            self._update_connection_summary()
        except Exception:
            logger.exception('Session committed; connection header refresh failed')

    def _dispose_session_workspace(self, state, central, children, *, keep_manager, protected=()):
        """Retire only workspace-owned resources, never shared menus/actions or a live manager."""
        roots = [central, state.get('_capture_ui_timer'), state.get('_view_group'),
                 *state.get('_aux_panel_dialogs', {}).values()]
        # Include partially constructed widgets if preparation failed before returning a root.
        shared = {id(v) for v in self.__dict__.values()}
        shared.update(protected)
        roots.extend(obj for obj in state.values() if isinstance(obj, QObject))
        roots.extend(obj for obj in children if id(obj) not in shared)
        manager = state.get('_stream_manager')
        if manager is not keep_manager:
            roots.append(manager)
        seen = set()
        for obj in roots:
            if (not isinstance(obj, QObject) or id(obj) in seen or obj is keep_manager
                    or id(obj) in shared):
                continue
            seen.add(id(obj))
            try:
                for child in [obj, *obj.findChildren(QObject)]:
                    child.blockSignals(True)
                    if isinstance(child, QTimer):
                        child.stop()
                    if isinstance(child, QDialog):
                        child.hide()
                if obj is manager:
                    manager.stop_all()
                    manager.clear_auxiliary_camera()
                    for name in list(manager.stereo_pairs):
                        manager.remove_stereo_pair(name)
                if isinstance(obj, QWidget):
                    obj.hide()
                obj.deleteLater()
            except Exception:
                logger.exception('Retired session workspace cleanup failed')

    @Slot()
    def _on_load_calibration_file(self):
        if self._background_tasks_block('读取双目标定文件'):
            return
        filename, _ = QFileDialog.getOpenFileName(self, '读取双目标定文件', '',
            '双目标定 (*.json *.yml *.yaml *.xml)')
        if not filename or self._background_tasks_block('读取双目标定文件'):
            return
        try:
            from ..io.calibration_import import load_stereo_calibration
            unit = None
            if Path(filename).suffix.lower() != '.json':
                text, approved = QInputDialog.getItem(self, '确认平移单位',
                    '请按文件中 T 的实际单位选择；不根据基线大小猜测。', ['米 (m)', '毫米 (mm)'], 0, False)
                if not approved:
                    return
                unit = 'mm' if text == '毫米 (mm)' else 'm'
            rig, provenance = load_stereo_calibration(filename, translation_unit=unit)
            pairs = [self._stream_manager.stereo_pairs.get(name) for name in rig.pairs]
            compatible = self._current_session is not None and all(
                cfg is not None and (cfg.left.camera_id, cfg.right.camera_id) ==
                (pair.left_id, pair.right_id) for cfg, pair in zip(pairs, rig.pairs.values()))
            compatible = compatible and all(p.stereo.model == self._camera_model for p in rig.pairs.values())
            if compatible:
                recorded = {p['name']: p for p in self._current_session.metadata.get('pairs', [])}
                compatible = all(name in recorded and
                    (recorded[name].get('left_id'), recorded[name].get('right_id')) ==
                    (p.left_id, p.right_id) for name, p in rig.pairs.items()) and (
                    self._current_session.metadata.get('camera_model', 'pinhole') == self._camera_model.value)
            details = '\n'.join(f'{name}: 左 {p.left_id} / 右 {p.right_id}；'
                f'{p.left_intrinsics.image_size[0]}×{p.left_intrinsics.image_size[1]}，{p.stereo.model.value}'
                for name, p in rig.pairs.items())
            message = ('结果将存入当前会话的新批次，原图和旧批次保留。' if compatible else
                       '将新建独立会话，原工作区的磁盘文件保留。')
            if QMessageBox.question(self, '确认读取双目参数', f'{details}\n\n{message}\n'
                '将清理内存中的旧检测/复核并重读当前会话原图，不自动连接设备。\n'
                '若关联当前会话，请确认其原图确实来自同一套设备、镜头及输出设置；不一致请取消并新建会话。\n'
                '请确认左/右对应；OpenCV M1/D1 为左，M2/D2 为右，R/T 表示左到右。\n'
                '文件有效不代表精度已验收；后续在线使用仍需确认同一设备、镜头与输出尺寸。',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
            if self._background_tasks_block('读取双目标定文件'):
                return
            provenance['physical_rgb_binding'] = 'user_confirmed_for_session' if compatible else 'unbound_offline'
            from ..streaming.stream_manager import CameraConfig, StereoPairConfig
            configs = [StereoPairConfig(name, CameraConfig(p.left_id, '', 'left', name),
                CameraConfig(p.right_id, '', 'right', name)) for name, p in rig.pairs.items()]
            model = next(iter(rig.pairs.values())).stereo.model
            session = self._current_session if compatible else self._session_manager.create_session(
                self._board, configs, auxiliary=self._stream_manager.auxiliary_camera,
                name='calib', camera_model=model.value)
            saved, errors = session.save_calibration_results(rig)
            if session.result_path.resolve() != saved[0].resolve():
                raise OSError('新标定批次索引未成功发布，未恢复旧参数冒充本次导入；'
                              '当前状态保留，新文件位于：' + str(saved[0].parent) + '；' + '；'.join(errors))
            from ..io.session_restore import prepare_session_restore, inspect_session_files
            plan = prepare_session_restore(session)
            if not self._load_session_result(session, plan, confirm_workspace=False, force_offline=True):
                return
            self._current_session, self._rig = session, self._calib_panel.rig
            self._calib_panel._rgb_result_provenance = provenance
            self._act_export.setEnabled(True)
            self._empty_created_session = self._empty_session_config = None
            self._refresh_session_button()
            inventory = inspect_session_files(session)
            self._calib_panel.append_log('已读取双目标定文件；参数通过有效性检查，精度待实际验证\n' + inventory['text'])
            self._calib_panel.restore_session_images(auto_aux=inventory['auto_import'])
            self._statusbar.showMessage('双目参数已读取：可继续 RGB+IR 采集或导入配对原图')
        except Exception as exc:
            logger.exception('Read stereo calibration failed: %s', filename)
            self._calib_panel.append_log(f'读取双目标定文件失败：{exc}')
            QMessageBox.warning(self, '标定文件不完整或不兼容', str(exc))

    @Slot()
    def _on_export(self):
        if self._rig is None:
            QMessageBox.information(
                self,
                "导出标定结果",
                "当前没有可导出的标定结果。请先完成双目标定或多模态标定。",
            )
            return
        dialog = ExportDialog(self._rig, self, excluded_aux={cid: entry['reason'] for cid, entry
            in self._calib_panel._historical_aux_results.items()})
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
        self._calib_panel.set_live_sources_active(True)
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
        self._calib_panel.set_live_sources_active(False)
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
                self._refresh_session_button()
            if 'offline' in rig.pairs and self._stream_manager.is_virtual_pair('offline'):
                self._current_session.record_configuration(
                    self._board, list(self._stream_manager.stereo_pairs.values()),
                    self._stream_manager.auxiliary_camera, camera_model=self._camera_model.value)
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
        from ..calibration.offline_aux import file_identity
        self._calib_panel._rgb_result_provenance = dict(kind='session_result',
            result=file_identity(saved[0]), session_name=self._current_session.name)
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
        timer = getattr(self, '_session_continuation_timer', None)
        if timer is not None:
            timer.stop()
            timer.deleteLater()
            self._session_continuation_timer = None
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
