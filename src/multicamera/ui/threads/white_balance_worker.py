"""Cancellable, non-blocking first-capture white-balance preparation."""

import time

from PySide6.QtCore import QObject, QThread, QTimer, Signal

from ...streaming.white_balance import camera_base, stabilize_and_lock


class WhiteBalanceWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, urls, parent=None):
        super().__init__(parent)
        self.urls = urls

    def run(self):
        try:
            result = stabilize_and_lock(self.urls, cancelled=self.isInterruptionRequested, progress=self.progress.emit)
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class WhiteBalanceGuard(QObject):
    busy_changed = Signal(bool)
    message = Signal(str)
    completed = Signal(bool)

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.worker = None
        self.busy = False
        self.ready = set()
        self.stage = ''
        self.started_at = 0.
        self.cancel_requested = False
        self._generation = 0
        self._result = None
        self._error = None
        self._resume = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._settled)
        manager.camera_connected.connect(self.invalidate)
        manager.camera_disconnected.connect(self.invalidate)

    def invalidate(self, *_):
        self._generation += 1
        self.ready.clear()
        self._resume = None
        self._timer.stop()
        self.cancel_requested = False
        if self.worker is not None:
            self.worker.requestInterruption()
        elif self.busy:
            self.busy = False
            self.busy_changed.emit(False)
            self.completed.emit(False)

    def cancel(self):
        was_settling = self.busy and self.worker is None
        self.invalidate()
        self.cancel_requested = True
        self.stage = '正在取消准备'
        if was_settling:
            self.message.emit('采图已取消；相机已锁定的白平衡保持不变')

    def ensure(self, urls, resume):
        targets = list(dict.fromkeys(camera_base(url) for url in urls))
        if not targets or set(targets) <= self.ready:
            return True
        if self.busy:
            return False
        self.busy = True
        self.stage = '准备白平衡'
        self.started_at = time.monotonic()
        self.cancel_requested = False
        self._resume = resume
        self._result, self._error = None, None
        self._work_generation = self._generation
        self.worker = WhiteBalanceWorker(targets, self)
        self.worker.succeeded.connect(self._succeeded)
        self.worker.failed.connect(self._failed)
        self.worker.progress.connect(self._progress)
        self.worker.finished.connect(self._finished)
        self.busy_changed.emit(True)
        self.message.emit('RGB 白平衡稳定与锁定中，请保持场景不变；完成后自动采图')
        self.worker.start()
        return False

    def _succeeded(self, result):
        self._result = result

    def _failed(self, error):
        self._error = error

    def _progress(self, stage):
        if not self.cancel_requested:
            self.stage = stage

    def _finished(self):
        worker, self.worker = self.worker, None
        worker.deleteLater()
        if self._work_generation != self._generation:
            self._error = '用户已取消准备' if self.cancel_requested else '连接或设置已改变，白平衡准备结果已作废'
        if self._error or self._result is None:
            self._resume = None
            self.ready.clear()
            self.busy = False
            self.busy_changed.emit(False)
            self.message.emit('白平衡准备已取消，未采图' if self.cancel_requested else
                              '白平衡准备失败，未采图：' + (self._error or '无有效结果'))
            self.completed.emit(False)
            return
        self.ready.update(self._result)
        for url in self._result:
            self.manager.discard_frames_for_url(url, settle_seconds=.8)
        self.stage = '等待新画面'
        self._timer.start(1100)

    def _settled(self):
        resume, self._resume = self._resume, None
        self.busy = False
        self.busy_changed.emit(False)
        self.message.emit('RGB 白平衡已稳定并锁定；旧预览帧已丢弃')
        self.completed.emit(True)
        if resume is not None:
            resume()

    def shutdown(self):
        self.invalidate()
        return self.worker is None
