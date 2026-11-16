"""Building blocks of the interface: card, segmented control, readout, table, curve. Their look comes from the style
sheet of app/theme.py (objectName), except the curve, drawn with the colors it is given."""
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QButtonGroup, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)


def label(name: str, text: str = "", **properties) -> QLabel:
    """Label whose role (objectName) chooses the look: title, subtitle, section, caption..."""
    w = QLabel(text, **properties)
    w.setObjectName(name)
    return w


class Wrapped(QLabel):
    """A text on several lines if needed, as tall as its lines at its width. (A word-wrapped QLabel alone is given
    the height Qt guesses for it, which may be too short in the panel: measured, the last line was cut.)"""

    def __init__(self, name: str, text: str = ""):
        super().__init__(text, wordWrap=True)
        self.setObjectName(name)

    def _fit(self) -> None:
        height = self.heightForWidth(self.width())
        if height > 0 and height != self.minimumHeight():
            self.setMinimumHeight(height)

    def setText(self, text: str) -> None:
        super().setText(text)
        self._fit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit()


class Card(QFrame):
    """A titled group of the panel. self.body: its layout."""

    def __init__(self, spacing: int = 10):
        super().__init__()
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)     # its full height: the panel scrolls
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(14, 12, 14, 14)
        self.body.setSpacing(spacing)
        self.title = label("section")
        self.body.addWidget(self.title)

    def set_title(self, text: str) -> None:
        self.title.setText(text.upper())


class Segmented(QFrame):
    """Choice of one option among a few, side by side. changed(value) when the user picks another one."""
    changed = Signal(str)

    def __init__(self, values, current: str):
        super().__init__()
        self.setObjectName("segment")
        row = QHBoxLayout(self)
        row.setContentsMargins(3, 3, 3, 3)
        row.setSpacing(2)
        self.group = QButtonGroup(self)
        self.buttons = {}
        for v in values:
            b = QPushButton(checkable=True, checked=v == current)
            b.clicked.connect(lambda _=False, v=v: self.changed.emit(v))
            self.group.addButton(b)
            row.addWidget(b)
            self.buttons[v] = b

    def value(self) -> str:
        return next(v for v, b in self.buttons.items() if b.isChecked())

    def set_texts(self, texts: dict) -> None:
        for v, b in self.buttons.items():
            b.setText(texts[v])


class Readout(QWidget):
    """A measured quantity: its name, its value, its unit."""

    def __init__(self):
        super().__init__()
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        self.name = label("readoutName")
        column.addWidget(self.name)
        row = QHBoxLayout()
        row.setSpacing(4)
        self.value = label("readoutValue", "—")
        self.unit = label("readoutUnit")
        row.addWidget(self.value, 0, Qt.AlignBottom)
        row.addWidget(self.unit, 0, Qt.AlignBottom)
        row.addStretch(1)
        column.addLayout(row)

    def set_name(self, name: str, unit: str) -> None:
        self.name.setText(name)
        self.unit.setText(unit)


class Table(QWidget):
    """Values side by side: a name per row, a column per subject, with its title in its color."""

    def __init__(self, rows: int, columns: int):
        super().__init__()
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(0, 1)
        self.heads = [label("tableHead") for _ in range(columns)]
        self.names = [label("tableName") for _ in range(rows)]
        self.cells = [[label("tableValue", "—") for _ in range(columns)] for _ in range(rows)]
        for c, head in enumerate(self.heads):
            grid.addWidget(head, 0, c + 1, Qt.AlignRight)
        for r, name in enumerate(self.names):
            grid.addWidget(name, r + 1, 0)
            for c, cell in enumerate(self.cells[r]):
                grid.addWidget(cell, r + 1, c + 1, Qt.AlignRight)

    def set_heads(self, texts, colors) -> None:
        for head, text, color in zip(self.heads, texts, colors):
            head.setText(text)
            head.setStyleSheet(f"color: {color};")

    def set_names(self, texts) -> None:
        for name, text in zip(self.names, texts):
            name.setText(text)

    def set_value(self, row: int, column: int, text: str) -> None:
        self.cells[row][column].setText(text)


class Sparkline(QWidget):
    """Small curves of the last values, from 0 to the largest one, on the same scale; a single one is filled
    underneath."""

    def __init__(self):
        super().__init__()
        self.series = []                # [(values, QColor)]
        self.setMinimumHeight(48)

    def set_series(self, series) -> None:
        """series: [(values, color "#RRGGBB")]."""
        self.series = [(list(values), QColor(color)) for values, color in series]
        self.update()

    def paintEvent(self, event) -> None:
        series = [(values, color) for values, color in self.series if len(values) >= 2]
        if not series:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        top = max(max(values) for values, _ in series) * 1.1 or 1.0
        for values, color in series:
            n = len(values) - 1
            points = [QPointF(i * (w - 1) / n, h - 2 - (h - 6) * v / top) for i, v in enumerate(values)]
            line = QPainterPath(points[0])
            for q in points[1:]:
                line.lineTo(q)
            if len(series) == 1:
                area = QPainterPath(line)
                area.lineTo(points[-1].x(), h)
                area.lineTo(0, h)
                fill = QColor(color)
                fill.setAlpha(40)
                p.fillPath(area, fill)
            p.setPen(QPen(color, 1.6))
            p.drawPath(line)
        p.end()
