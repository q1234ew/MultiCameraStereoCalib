import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from multicamera.logging_config import setup_logging
from multicamera.runtime_paths import logo_png_path
from multicamera.ui.main_window import MainWindow
from multicamera.ui.theme import app_stylesheet


def main():
    setup_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("MultiCamera Calibration")
    app.setOrganizationName("MultiCamera")
    app.setStyleSheet(app_stylesheet())

    logo = logo_png_path()
    if logo is not None:
        app.setWindowIcon(QIcon(str(logo)))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
