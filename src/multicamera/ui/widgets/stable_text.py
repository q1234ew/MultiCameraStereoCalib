"""Bounded status text that cannot resize neighbouring video previews."""

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QPainter, QTextLayout, QTextOption
from PySide6.QtWidgets import QLabel, QSizePolicy, QStyle, QStyleOption


class StableTextLabel(QLabel):
    """Reserve font-relative lines; elide overflow without losing the full text."""

    def __init__(self, text='', *, lines=2, parent=None):
        super().__init__(text, parent)
        self._lines = lines
        self.setTextFormat(Qt.PlainText)
        self.setWordWrap(True)
        self._reserve_lines()

    def _reserve_lines(self):
        policy = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        policy.setHeightForWidth(False)
        self.setSizePolicy(policy)
        height = self.fontMetrics().lineSpacing() * self._lines + 4
        self.setFixedHeight(height)

    def set_lines(self, lines):
        if self._lines != lines:
            self._lines = lines
            self._reserve_lines()
            self.update()

    def event(self, event):
        result = super().event(event)
        if hasattr(self, '_lines') and event.type() in (
                QEvent.FontChange, QEvent.StyleChange, QEvent.Polish):
            self._reserve_lines()
        return result

    def visible_text(self):
        width = max(1, self.contentsRect().width() - 8)
        rows = []
        for paragraph in self.text().split('\n'):
            if not paragraph:
                rows.append('')
                continue
            text = QTextLayout(paragraph, self.font())
            option = QTextOption()
            option.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
            text.setTextOption(option)
            text.beginLayout()
            while (line := text.createLine()).isValid():
                line.setLineWidth(width)
                rows.append(paragraph[line.textStart():line.textStart() + line.textLength()].strip())
            text.endLayout()
        if len(rows) > self._lines:
            rows[self._lines - 1] = self.fontMetrics().elidedText(
                ' '.join(rows[self._lines - 1:]), Qt.ElideRight, width)
        return '\n'.join(rows[:self._lines])

    def paintEvent(self, event):
        painter = QPainter(self)
        option = QStyleOption()
        option.initFrom(self)
        self.style().drawPrimitive(QStyle.PE_Widget, option, painter, self)
        painter.setFont(self.font())
        painter.setPen(self.palette().windowText().color())
        painter.drawText(self.contentsRect().adjusted(4, 2, -4, -2),
                         self.alignment(), self.visible_text())
