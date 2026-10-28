"""Background worker for saving captured frame pairs."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class FrameSaveWorker(QThread):
    """Save one stereo frame pair in the background."""

    saved = Signal(str, str, int)
    failed = Signal(str, str, int, str)

    def __init__(
        self,
        token: str,
        session,
        pair_name: str,
        left,
        right,
        frame_idx: int,
        parent=None,
        quality=None,
    ):
        super().__init__(parent)
        self.token = token
        self.session = session
        self.pair_name = pair_name
        self.left = left
        self.right = right
        self.frame_idx = frame_idx
        self.quality = quality

    def run(self):
        try:
            if self.quality is not None:
                self.session.save_capture_review(self.pair_name, self.frame_idx, self.quality)
            saved_idx = self.session.save_frame_pair(
                self.pair_name,
                self.left,
                self.right,
                frame_idx=self.frame_idx,
            )
            self.saved.emit(self.token, self.pair_name, saved_idx)
        except Exception as exc:
            self.failed.emit(self.token, self.pair_name, self.frame_idx, str(exc))
