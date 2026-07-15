# -*- coding: utf-8 -*-
"""
Macro COC v3.0 — GUI / Widgets

Widgets reutilisables : MacroCard, StatCard, RecordingIndicator, ActionButton, MacroTimeline.
"""

from typing import List, Optional, Tuple

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QProgressBar,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QFont

from qfluentwidgets import (
    CardWidget, PrimaryPushButton, PushButton, FluentIcon, StrongBodyLabel,
    BodyLabel, CaptionLabel, InfoBadge, ToolTipFilter,
)

from gui.theme import Theme
from utils.config import fmt_seconds, fmt_duration_for_list


class StatCard(CardWidget):
    """Carte miniature affichant une statistique (icone + valeur + label)."""

    def __init__(self, icon: FluentIcon, value: str, label: str, color: str = "#86efac", parent=None):
        super().__init__(parent)
        self.setMinimumWidth(140)
        self.setFixedHeight(90)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)

        # Icone + valeur
        top = QHBoxLayout()
        top.setSpacing(8)

        icon_widget = QLabel()
        icon_widget.setPixmap(icon.icon().pixmap(24, 24))
        top.addWidget(icon_widget)

        val_lbl = StrongBodyLabel(value)
        val_lbl.setStyleSheet(f"color: {color}; font-size: 18px;")
        top.addWidget(val_lbl)
        top.addStretch()
        layout.addLayout(top)

        # Label
        lbl = CaptionLabel(label)
        layout.addWidget(lbl)

    def set_value(self, value: str):
        """Met a jour la valeur affichee."""
        for child in self.findChildren(StrongBodyLabel):
            child.setText(value)
            break


class MacroCard(CardWidget):
    """Carte representant une macro dans le dashboard."""

    clicked = pyqtSignal(str)
    play_requested = pyqtSignal(str)

    def __init__(self, name: str, duration: float, event_count: int, parent=None):
        super().__init__(parent)
        self._name = name
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        # Titre
        title = StrongBodyLabel(name)
        title.setWordWrap(True)
        layout.addWidget(title)

        # Metadonnees
        meta_row = QHBoxLayout()
        meta_row.setSpacing(12)

        dur_lbl = CaptionLabel(fmt_duration_for_list(duration))
        meta_row.addWidget(dur_lbl)

        evt_badge = InfoBadge.custom(str(event_count), "#86efac", "#0b1220")
        meta_row.addWidget(evt_badge)

        evt_label = CaptionLabel("evenements")
        meta_row.addWidget(evt_label)
        meta_row.addStretch()
        layout.addLayout(meta_row)

        # Bouton Lire
        play_btn = PrimaryPushButton(FluentIcon.PLAY, "  Lire")
        play_btn.clicked.connect(lambda: self.play_requested.emit(self._name))
        layout.addWidget(play_btn)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._name)
        super().mousePressEvent(event)

    @property
    def name(self) -> str:
        return self._name


class RecordingIndicator(QWidget):
    """Indicateur d'enregistrement avec point rouge pulse et timer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self._opacity = 1.0
        self._elapsed_sec = 0.0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(8)

        # Point rouge
        self._dot = QLabel()
        self._dot.setFixedSize(12, 12)
        self._dot.setStyleSheet(
            "background-color: #ef4444; border-radius: 6px; min-width: 12px; min-height: 12px;"
        )
        layout.addWidget(self._dot)

        # Texte
        self._label = BodyLabel("Enregistrement... 00:00")
        self._label.setStyleSheet("color: #f87171;")
        layout.addWidget(self._label)

        layout.addStretch()

        # Animation de pulsation
        self._anim = QPropertyAnimation(self, b"opacity")
        self._anim.setDuration(800)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.3)
        self._anim.setLoopCount(-1)  # infinie
        self._anim.setEasingCurve(QEasingCurve.Type.InOutSine)

    def get_opacity(self) -> float:
        return self._opacity

    def set_opacity(self, value: float):
        self._opacity = value
        alpha = int(value * 255)
        self._dot.setStyleSheet(
            f"background-color: rgba(239, 68, 68, {alpha}); border-radius: 6px;"
        )
        self.update()

    opacity = pyqtProperty(float, get_opacity, set_opacity)

    def start(self):
        """Demarre l'indicateur et l'animation."""
        self._anim.start()
        self.show()
        self._elapsed_sec = 0.0
        self._update_label()

    def stop(self):
        """Arrete l'indicateur."""
        self._anim.stop()
        self.set_opacity(1.0)
        self.hide()

    def tick(self, elapsed: float):
        """Met a jour le timer."""
        self._elapsed_sec = elapsed
        self._update_label()

    def _update_label(self):
        self._label.setText(f"Enregistrement... {fmt_seconds(self._elapsed_sec)}")


def ActionButton(icon: FluentIcon, text: str, parent=None) -> PrimaryPushButton:
    """Bouton d'action principal avec icone, surdimensionne."""
    btn = PrimaryPushButton(icon, text, parent)
    btn.setMinimumHeight(48)
    btn.setMinimumWidth(180)
    font = btn.font()
    font.setPointSize(11)
    btn.setFont(font)
    return btn


def DangerButton(icon: FluentIcon, text: str, parent=None) -> PushButton:
    """Bouton danger (stop, delete) avec icone, style Fluent natif."""
    btn = PushButton(icon, text, parent)
    btn.setMinimumHeight(48)
    btn.setMinimumWidth(180)
    font = btn.font()
    font.setPointSize(11)
    btn.setFont(font)
    return btn


class MacroTimeline(QWidget):
    """
    Barre de timeline visuelle representant les evenements d'une macro.
    Chaque type d'evenement a une couleur differente.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self.setMinimumWidth(200)
        self._steps: List[dict] = []
        self._duration: float = 0.0

    def set_steps(self, steps: List[dict]):
        """Charge les steps pour affichage."""
        self._steps = steps
        self._duration = sum(max(0.0, float(s.get("t", 0.0))) for s in steps)
        self.update()

    def clear(self):
        """Vide la timeline."""
        self._steps = []
        self._duration = 0.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height() - 4
        y = 2

        # Fond
        painter.setBrush(QColor("#1e293b"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, y, w, h, 4, 4)

        if not self._steps or self._duration <= 0:
            painter.end()
            return

        # Barres colorees par type
        color_map = {
            "mouse_click": Theme.TIMELINE_CLICK,
            "mouse_move": Theme.TIMELINE_MOVE,
            "key_down": Theme.TIMELINE_KEY,
            "key_up": Theme.TIMELINE_KEY,
            "scroll": Theme.TIMELINE_SCROLL,
        }

        acc = 0.0
        for step in self._steps:
            dt = max(0.0, float(step.get("t", 0.0)))
            x = int((acc / self._duration) * w)
            width = max(2, int((dt / self._duration) * w))

            typ = step.get("type", "")
            color = color_map.get(typ, "#64748b")

            painter.setBrush(QColor(color))
            painter.drawRoundedRect(x, y, width, h, 3, 3)

            acc += dt

        painter.end()

    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        return self.size()


class StepListWidget(QWidget):
    """Liste des etapes d'une macro avec type, delta, et donnees."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)
        self._layout.addStretch()

    def set_steps(self, steps: List[dict]):
        """Affiche la liste des steps."""
        # Nettoyer
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        type_labels = {
            "mouse_move": ("Souris", "#818cf8"),
            "mouse_click": ("Clic", "#60a5fa"),
            "key_down": ("Touche ↓", "#f59e0b"),
            "key_up": ("Touche ↑", "#f59e0b"),
            "scroll": ("Scroll", "#34d399"),
        }
        default = ("?", "#64748b")

        for i, step in enumerate(steps):
            typ = step.get("type", "")
            dt_val = step.get("t", 0.0)
            data = step.get("data", {})
            label_text, color = type_labels.get(typ, default)

            row = QHBoxLayout()
            row.setSpacing(8)

            # Index
            idx = CaptionLabel(f"#{i + 1}")
            idx.setFixedWidth(30)
            row.addWidget(idx)

            # Type badge
            type_lbl = QLabel(label_text)
            type_lbl.setFixedWidth(65)
            type_lbl.setStyleSheet(
                f"color: {color}; font-size: 11px; font-weight: bold;"
            )
            row.addWidget(type_lbl)

            # Delta
            delta_lbl = CaptionLabel(f"+{dt_val:.2f}s")
            delta_lbl.setFixedWidth(55)
            row.addWidget(delta_lbl)

            # Data preview
            preview = self._data_preview(typ, data)
            data_lbl = CaptionLabel(preview)
            data_lbl.setStyleSheet("color: #94a3b8;")
            row.addWidget(data_lbl, stretch=1)

            self._layout.insertLayout(i, row)

        # Re-ajouter le stretch a la fin
        # Deja present (ajoute dans __init__)

    def _data_preview(self, typ: str, data: dict) -> str:
        """Genere un apercu textuel des donnees."""
        if typ in ("mouse_move", "mouse_click"):
            return f"({data.get('x', '?')}, {data.get('y', '?')})"
        elif typ in ("key_down", "key_up"):
            return data.get("key", "?")
        elif typ == "scroll":
            return f"dx={data.get('dx', 0)} dy={data.get('dy', 0)}"
        return ""

    def clear(self):
        """Vide la liste."""
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
