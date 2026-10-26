"""Snapshot-only capture analysis; never touches widgets or final results."""

import logging

from PySide6.QtCore import QObject, QRunnable, Signal

from ...calibration.adaptive_guidance import analyze_guidance

logger = logging.getLogger(__name__)


class GuidanceSignals(QObject):
    completed = Signal(object, object, str)


class GuidanceWorker(QRunnable):
    def __init__(self, key, board, model, observations, intrinsics):
        super().__init__()
        self.signals = GuidanceSignals()
        self.key, self.board, self.model = key, board, model
        self.observations, self.intrinsics = observations, intrinsics

    def run(self):
        try:
            result = analyze_guidance(
                self.board, self.model, self.observations, self.intrinsics,
            )
            self.signals.completed.emit(self.key, result, '')
        except Exception as exc:
            logger.exception('Adaptive capture analysis failed')
            self.signals.completed.emit(self.key, None, str(exc))
