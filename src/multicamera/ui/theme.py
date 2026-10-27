"""Unified dark theme for the application."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# ── Color palette ──────────────────────────────────────────────
BG_DARK = "#18181a"
BG_MID = "#202023"
BG_CARD = "#27272b"
BG_INPUT = "#303035"
BG_HOVER = "#38383e"
BORDER = "#46464e"
BORDER_FOCUS = "#6db7ff"
TEXT = "#f0f0f2"
TEXT_DIM = "#b8b8c1"
TEXT_HINT = "#92929f"
ACCENT = "#6db7ff"
PRIMARY = "#086ed3"
ACCENT_HOVER = "#075fb7"
SUCCESS = "#58d695"
WARNING = "#e7b65b"
DANGER = "#ff7b83"


def configure_application_style(app):
    from PySide6.QtGui import QFont, QFontDatabase

    families = set(QFontDatabase.families())
    family = next((name for name in ("PingFang SC", "Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans CJK SC")
                   if name in families), None)
    app.setFont(QFont(family) if family else QFontDatabase.systemFont(QFontDatabase.GeneralFont))
    stylesheet = app_stylesheet()
    if app.styleSheet() != stylesheet:
        app.setStyleSheet(stylesheet)
    app._style_reference = app.style()


@lru_cache(maxsize=128)
def app_icon(name, color=TEXT_DIM):
    """Render the bundled Lucide subset at native and high-DPI sizes."""
    from PySide6.QtCore import QByteArray, Qt
    from PySide6.QtGui import QIcon, QPainter, QPixmap
    from PySide6.QtSvg import QSvgRenderer

    source = (Path(__file__).with_name('icons') / f'{name}.svg').read_text(encoding='utf-8')
    icon = QIcon()
    for mode, stroke in ((QIcon.Normal, color), (QIcon.Active, TEXT),
                         (QIcon.Selected, TEXT), (QIcon.Disabled, TEXT_HINT)):
        renderer = QSvgRenderer(QByteArray(source.replace('currentColor', stroke).encode()))
        if not renderer.isValid():
            raise ValueError(f'Invalid icon: {name}')
        for scale in (1, 2, 3):
            pixmap = QPixmap(20 * scale, 20 * scale)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            pixmap.setDevicePixelRatio(scale)
            icon.addPixmap(pixmap, mode)
    return icon


def standard_icon(widget, standard, color=TEXT_DIM):
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QColor, QIcon, QPainter
    from PySide6.QtWidgets import QApplication, QStyle

    names = {
        QStyle.SP_MediaPlay: 'plug', QStyle.SP_MediaStop: 'unplug',
        QStyle.SP_FileDialogDetailedView: 'settings',
        QStyle.SP_DirOpenIcon: 'folder-open', QStyle.SP_ArrowLeft: 'chevron-left',
        QStyle.SP_ArrowRight: 'chevron-right', QStyle.SP_ArrowBack: 'undo-2',
        QStyle.SP_TitleBarCloseButton: 'x',
    }
    if standard in names:
        return app_icon(names[standard], color)

    application = QApplication.instance()
    style = application.style() if application is not None else widget.style()
    pixmap = style.standardIcon(standard).pixmap(QSize(20, 20))
    painter = QPainter(pixmap)
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), QColor(color))
    painter.end()
    return QIcon(pixmap)


def app_stylesheet() -> str:
    return f"""
    * {{
        font-family: "PingFang SC", "Helvetica Neue", "Microsoft YaHei";
    }}

    QMainWindow, QWidget {{
        background-color: {BG_DARK};
        color: {TEXT};
        font-size: 13px;
    }}

    /* ── Menu bar ───────────────────────────────────── */
    QMenuBar {{
        background: {BG_MID};
        color: {TEXT};
        border-bottom: 1px solid {BORDER};
        padding: 2px 0;
    }}
    QMenuBar::item {{
        padding: 5px 12px;
        border-radius: 4px;
    }}
    QMenuBar::item:selected {{
        background: {BG_HOVER};
    }}
    QMenu {{
        background: {BG_CARD};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 4px 0;
    }}
    QMenu::item {{
        padding: 6px 28px 6px 16px;
    }}
    QMenu::item:selected {{
        background: {ACCENT};
        color: white;
        border-radius: 4px;
    }}
    QMenu::separator {{
        height: 1px;
        background: {BORDER};
        margin: 4px 8px;
    }}

    /* ── Tool bar ───────────────────────────────────── */
    QToolBar {{
        background: {BG_MID};
        border-bottom: 1px solid {BORDER};
        spacing: 6px;
        padding: 4px 8px;
    }}
    QToolBar QToolButton {{
        background: {BG_CARD};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 5px 10px;
        font-size: 13px;
    }}
    QToolBar QToolButton:hover {{
        background: {BG_HOVER};
        border-color: {ACCENT};
    }}
    QToolBar QToolButton:disabled {{
        color: {TEXT_HINT};
        border-color: {BG_CARD};
    }}
    QToolBar::separator {{
        width: 1px;
        background: {BORDER};
        margin: 4px 6px;
    }}

    /* ── Status bar ─────────────────────────────────── */
    QStatusBar {{
        background: {BG_MID};
        color: {TEXT_DIM};
        border-top: 1px solid {BORDER};
        font-size: 12px;
        padding: 2px 8px;
    }}

    /* ── Scroll area ────────────────────────────────── */
    QScrollArea {{
        border: none;
        background: transparent;
    }}
    QScrollBar:vertical {{
        width: 8px;
        background: transparent;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER};
        border-radius: 4px;
        min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {TEXT_HINT};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        height: 8px;
        background: transparent;
    }}
    QScrollBar::handle:horizontal {{
        background: {BORDER};
        border-radius: 4px;
        min-width: 24px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
        background: transparent;
    }}

    /* ── Group box ──────────────────────────────────── */
    QGroupBox {{
        font-weight: bold;
        font-size: 13px;
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 8px;
        margin-top: 14px;
        padding: 16px 10px 10px 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 14px;
        padding: 0 6px;
    }}

    /* ── Buttons ────────────────────────────────────── */
    QPushButton {{
        background: {BG_CARD};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 7px 18px;
        font-size: 13px;
    }}
    QPushButton:hover {{
        background: {BG_HOVER};
        border-color: {ACCENT};
    }}
    QPushButton:pressed {{
        background: {ACCENT};
        color: white;
    }}
    QPushButton:disabled {{
        color: {TEXT_HINT};
        background: {BG_DARK};
        border-color: {BG_CARD};
    }}
    QPushButton:checked {{
        background: {ACCENT};
        color: white;
        border-color: {ACCENT};
    }}

    /* Primary action buttons */
    QPushButton[class="primary"] {{
        background: {PRIMARY};
        color: white;
        border: 1px solid transparent;
        font-weight: bold;
    }}
    QPushButton[class="primary"]:hover {{
        background: {ACCENT_HOVER};
    }}
    QPushButton[class="primary"]:focus {{ border-color: {BORDER_FOCUS}; }}
    QPushButton[class="primary"]:pressed {{ background: #07549f; }}
    QPushButton[class="primary"]:disabled {{
        background: {BG_INPUT};
        color: {TEXT_HINT};
        border: 1px solid {BORDER};
    }}
    QPushButton:focus, QToolButton:focus {{
        border-color: {BORDER_FOCUS};
    }}
    QToolButton {{
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 5px;
        color: {TEXT_DIM};
        background: transparent;
    }}
    QToolButton:hover {{ background: {BG_HOVER}; color: {TEXT}; }}
    QToolButton:checked {{ background: {BG_INPUT}; color: {ACCENT}; border-color: {BORDER}; }}
    QToolButton:disabled {{ color: {TEXT_HINT}; }}
    QFrame#actionFooter {{ border-top: 1px solid {BORDER}; background: {BG_MID}; }}
    QWidget#workbenchHeader {{ border-bottom: 1px solid {BORDER}; background: {BG_MID}; }}
    QFrame#inspectorHeading {{ border-bottom: 1px solid {BORDER}; }}

    /* ── Spin box / inputs ─────────────────────────── */
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{
        background: {BG_INPUT};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 5px 8px;
        font-size: 13px;
    }}
    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {{
        border-color: {ACCENT};
    }}
    QComboBox::drop-down {{
        border: none;
        padding-right: 8px;
    }}
    QComboBox QAbstractItemView {{
        background: {BG_CARD};
        color: {TEXT};
        border: 1px solid {BORDER};
        selection-background-color: {ACCENT};
    }}

    /* ── Progress bar ──────────────────────────────── */
    QProgressBar {{
        background: {BG_INPUT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        text-align: center;
        color: {TEXT};
        font-size: 12px;
        height: 22px;
    }}
    QProgressBar::chunk {{
        background: {PRIMARY};
        border-radius: 5px;
    }}

    /* ── Text edit (log) ───────────────────────────── */
    QTextEdit, QPlainTextEdit {{
        background: {BG_INPUT};
        color: {TEXT_DIM};
        border: 1px solid {BORDER};
        border-radius: 6px;
        font-family: "Menlo", "Consolas", "Courier New";
        font-size: 12px;
        padding: 4px;
    }}

    /* ── Table ─────────────────────────────────────── */
    QTableWidget {{
        background: {BG_INPUT};
        alternate-background-color: {BG_MID};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        gridline-color: {BORDER};
    }}
    QHeaderView::section {{
        background: {BG_CARD};
        color: {TEXT_DIM};
        border: none;
        border-bottom: 1px solid {BORDER};
        padding: 6px 8px;
        font-size: 12px;
    }}
    QTableWidget::item {{
        padding: 4px 8px;
    }}
    QTableWidget::item:selected {{
        background: {ACCENT};
        color: white;
    }}
    /* Tables embed scrollbars — slightly brighter track/handle than global bars */
    QTableWidget QScrollBar:vertical {{
        background: {BG_MID};
        width: 10px;
        margin: 0;
    }}
    QTableWidget QScrollBar::handle:vertical {{
        background: {TEXT_HINT};
        border-radius: 4px;
        min-height: 28px;
    }}
    QTableWidget QScrollBar::handle:vertical:hover {{
        background: {TEXT_DIM};
    }}
    QTableWidget QScrollBar:horizontal {{
        background: {BG_MID};
        height: 10px;
        margin: 0;
    }}
    QTableWidget QScrollBar::handle:horizontal {{
        background: {TEXT_HINT};
        border-radius: 4px;
        min-width: 28px;
    }}
    QTableWidget QScrollBar::handle:horizontal:hover {{
        background: {TEXT_DIM};
    }}

    /* ── Labels ────────────────────────────────────── */
    QLabel {{
        color: {TEXT};
    }}

    /* ── Splitter ──────────────────────────────────── */
    QSplitter::handle {{
        background: {BORDER};
    }}
    QSplitter::handle:horizontal {{
        width: 2px;
    }}
    QSplitter::handle:vertical {{
        height: 2px;
    }}

    /* ── Dock widget title ─────────────────────────── */
    QDockWidget {{
        color: {TEXT};
        titlebar-close-icon: none;
    }}
    QDockWidget::title {{
        background: {BG_MID};
        border-bottom: 1px solid {BORDER};
        padding: 6px 10px;
        font-weight: bold;
    }}

    /* ── Dialog ────────────────────────────────────── */
    QDialog {{
        background: {BG_DARK};
    }}
    QDialogButtonBox QPushButton {{
        min-width: 80px;
    }}

    /* ── Tab widget ────────────────────────────────── */
    QTabWidget::pane {{
        border: 1px solid {BORDER};
        border-radius: 6px;
        background: {BG_DARK};
    }}
    QTabBar::tab {{
        background: {BG_CARD};
        color: {TEXT_DIM};
        border: 1px solid {BORDER};
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        padding: 6px 16px;
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{
        background: {BG_DARK};
        color: {ACCENT};
        border-bottom: 2px solid {ACCENT};
    }}
    QTabBar#workflowTabs::tab {{
        min-width: 66px;
        padding: 10px 16px;
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        border-radius: 0;
    }}
    QTabBar#workflowTabs::tab:selected {{ color: {ACCENT}; border-bottom-color: {ACCENT}; }}
    QTabBar#workflowTabs::tab:focus {{ background: {BG_INPUT}; }}
    QListWidget#pairThumbnails {{ background: transparent; border: none; outline: none; }}
    QListWidget#pairThumbnails::item {{ border: 1px solid {BORDER}; border-radius: 4px; padding: 4px; }}
    QListWidget#pairThumbnails::item:selected {{ border-color: {ACCENT}; background: {BG_INPUT}; color: {TEXT}; }}
    """
