"""Building blocks of the interface: card, segmented control, readout, curve. Their look comes from the style sheet
of app/theme.py (objectName), except the curve, drawn with the colors it is given."""
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


def label(name: str, text: str = "", **properties) -> QLabel:
    """Label whose role (objectName) chooses the look: title, subtitle, section, caption..."""
    w = QLabel(text, **properties)
    w.setObjectName(name)
    return w


class Card(QFrame):
    """A titled group of the panel. self.body: its layout."""

    def __init__(self, spacing: int = 10):
        super().__init__()
        self.setObjectName("card")
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


class Sparkline(QWidget):
    """A small curve of the last values, from 0 to the largest one, filled underneath."""

    def __init__(self):
        super().__init__()
        self.values = []
        self.color = QColor("#EA580C")
        self.setMinimumHeight(48)

    def set_values(self, values, color: str = None) -> None:
        self.values = list(values)
        if color is not None:
            self.color = QColor(color)
        self.update()

    def paintEvent(self, event) -> None:
        if len(self.values) < 2:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        top = max(self.values) * 1.1 or 1.0
        n = len(self.values) - 1
        points = [QPointF(i * (w - 1) / n, h - 2 - (h - 6) * v / top) for i, v in enumerate(self.values)]
        line = QPainterPath(points[0])
        for q in points[1:]:
            line.lineTo(q)
        area = QPainterPath(line)
        area.lineTo(points[-1].x(), h)
        area.lineTo(0, h)
        fill = QColor(self.color)
        fill.setAlpha(40)
        p.fillPath(area, fill)
        p.setPen(QPen(self.color, 1.6))
        p.drawPath(line)
        p.end()
