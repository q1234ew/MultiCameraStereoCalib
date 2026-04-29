"""Unified dark theme for the application."""

from __future__ import annotations

# ── Color palette ──────────────────────────────────────────────
BG_DARK = "#0f1923"
BG_MID = "#15202b"
BG_CARD = "#192734"
BG_INPUT = "#1e2d3d"
BG_HOVER = "#22303d"
BORDER = "#2f3e4e"
BORDER_FOCUS = "#1d9bf0"
TEXT = "#e7e9ea"
TEXT_DIM = "#8899a6"
TEXT_HINT = "#657786"
ACCENT = "#1d9bf0"
ACCENT_HOVER = "#1a8cd8"
SUCCESS = "#00ba7c"
WARNING = "#ffad1f"
DANGER = "#f4212e"


def app_stylesheet() -> str:
    return f"""
    * {{
        font-family: "PingFang SC", "Helvetica Neue", "Microsoft YaHei";
    }}

    QMainWindow, QWidget {{
        background-color: {BG_DARK};
        color: {TEXT};
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
        padding: 5px 14px;
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
        background: {ACCENT};
        color: white;
        border: none;
        font-weight: bold;
    }}
    QPushButton[class="primary"]:hover {{
        background: {ACCENT_HOVER};
    }}

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
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                     stop:0 {ACCENT}, stop:1 {SUCCESS});
        border-radius: 5px;
    }}

    /* ── Text edit (log) ───────────────────────────── */
    QTextEdit {{
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
    """
