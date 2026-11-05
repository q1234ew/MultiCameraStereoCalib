"""Background analysis of a read-only calibration review snapshot."""

from PySide6.QtCore import QThread, Signal

from ...calibration.review import analyze_review


class ReviewWorker(QThread):
    completed = Signal(object, object, str)

    def __init__(self, key, board, rows, intrinsics, parent=None):
        super().__init__(parent)
        self.key, self.board, self.rows, self.intrinsics = key, board, rows, intrinsics

    def run(self):
        try:
            result = analyze_review(self.board, self.rows, self.intrinsics)
            self.completed.emit(self.key, result, '')
        except Exception as exc:
            self.completed.emit(self.key, None, str(exc))
