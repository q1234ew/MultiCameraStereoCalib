"""Run projection-model comparison on detached observation snapshots."""

from PySide6.QtCore import QThread, Signal

from ...calibration.model_selection import assess_camera_models


class ModelSelectionWorker(QThread):
    result_ready = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, left, right, stereo, parent=None):
        super().__init__(parent)
        self.left, self.right, self.stereo = left, right, stereo

    def run(self):
        try:
            self.result_ready.emit(assess_camera_models(
                self.left, self.right, self.stereo, cancelled=self.isInterruptionRequested,
                progress=self.progress.emit))
        except Exception as exc:
            self.failed.emit(str(exc))
