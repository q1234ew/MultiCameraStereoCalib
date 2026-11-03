"""One cancellable online capture request, gated by fresh paired observations."""

import time

from PySide6.QtCore import QObject, QTimer, Signal


class AuxLiveCaptureGate(QObject):
    changed = Signal(bool)
    TIMEOUT_SECONDS = 15.

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self.waiting = False
        self.started_at = 0.
        self.reason = ''
        self._clock = time.monotonic
        self._context = None
        self._timer = QTimer(self)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self.poll)

    def context(self):
        panel = self.panel
        return (panel._aux_guide.context_key(), panel._active_step, panel._workflow_phase,
                panel._white_balance_guard._generation, panel._check_wb.isChecked(),
                panel._combo_wb_mode.currentData())

    def start(self, *, reset=False):
        if self.waiting:
            return
        if reset:
            self.panel._aux_guide.invalidate_live()
        self._context = self.context()
        self.started_at = self._clock()
        self.waiting = True
        self.reason = '等待双路新帧和连续稳定检测'
        self._timer.start()
        self.changed.emit(True)
        self.poll()

    def poll(self):
        if not self.waiting:
            return
        panel = self.panel
        if self._context != self.context() or panel._aux_guide.reviewing:
            self.cancel('采图上下文已变化，请按当前配置重新采集')
            return
        if self._clock() - self.started_at >= self.TIMEOUT_SECONDS:
            self.cancel('等待稳定超时：' + self.reason)
            return
        frames, pairing = panel._aux_live_frames()
        if frames is None:
            self.reason = pairing
        else:
            self.reason = panel._aux_guide.capture_problem()
        if not self.reason:
            self._finish()
            panel._capture_aux_original()
            return
        panel._aux_guide._request_preview()

    def _finish(self):
        self._timer.stop()
        self.waiting = False
        self.changed.emit(False)

    def cancel(self, reason='已取消本次采图'):
        if not self.waiting:
            return
        self._finish()
        self.panel._aux_guide.show_receipt_message(reason + '\n本次未保存、未增加计数；调整后可重新采集')
        self.panel._log(reason + '；本次未保存、未增加计数')
