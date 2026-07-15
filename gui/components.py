# -*- coding: utf-8 -*-
"""
Macro COC v3.0 — GUI / Components

Widgets reutilisables pour la liste des macros.
Style adapte au theme Fluent sombre.
"""

from typing import Callable, Dict, List, Optional, Tuple

from PyQt6.QtWidgets import (
    QFrame, QLabel, QHBoxLayout, QVBoxLayout, QScrollArea, QWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from qfluentwidgets import SearchLineEdit, BodyLabel, CaptionLabel

from gui.theme import Theme
from utils.config import fmt_duration_for_list


# Couleurs adaptees au theme Fluent sombre
ROW_BG = "rgba(255, 255, 255, 0.04)"
ROW_HOVER = "rgba(255, 255, 255, 0.08)"
ROW_SELECTED = "rgba(34, 197, 94, 0.15)"
ROW_BORDER_SELECTED = "#22c55e"
TEXT_PRIMARY = "rgba(255, 255, 255, 0.93)"
TEXT_SECONDARY = "rgba(255, 255, 255, 0.60)"


class MacroRow(QFrame):
    """
    Widget representant une ligne de macro dans la liste.
    """

    clicked = pyqtSignal(str)

    def __init__(self, name: str, duration_txt: str, parent=None):
        super().__init__(parent)
        self._name = name
        self._selected = False

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(46)
        self._apply_style()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(8)

        # Nom
        self.lbl_name = BodyLabel(name)
        layout.addWidget(self.lbl_name, stretch=1)

        # Duree
        self.lbl_dur = CaptionLabel(duration_txt)
        layout.addWidget(self.lbl_dur)

    @property
    def name(self) -> str:
        return self._name

    def _apply_style(self):
        """Applique le style selon l'etat de selection."""
        if self._selected:
            bg = ROW_SELECTED
            border = f"border-left: 3px solid {ROW_BORDER_SELECTED};"
        else:
            bg = ROW_BG
            border = "border-left: 3px solid transparent;"

        self.setStyleSheet(f"""
            MacroRow {{
                background-color: {bg};
                border-radius: 6px;
                {border}
            }}
        """)

    def set_selected(self, selected: bool):
        self._selected = selected
        self._apply_style()

    def set_duration(self, dur_txt: str):
        self.lbl_dur.setText(dur_txt)

    # --- Events ---

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._name)
        super().mousePressEvent(event)

    def enterEvent(self, event):
        if not self._selected:
            self.setStyleSheet(f"""
                MacroRow {{
                    background-color: {ROW_HOVER};
                    border-radius: 6px;
                    border-left: 3px solid transparent;
                }}
            """)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._apply_style()
        super().leaveEvent(event)


class MacroList(QScrollArea):
    """
    Liste scrollable des macros avec recherche.
    API identique a la version precedente.
    """

    macro_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Container interne
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(4, 2, 4, 2)
        self._layout.setSpacing(2)
        self._layout.addStretch()
        self.setWidget(self._container)

        self._rows: Dict[str, MacroRow] = {}
        self._selected: Optional[str] = None
        self._meta: Dict[str, Tuple[int, float]] = {}

    def set_meta(self, meta: Dict[str, Tuple[int, float]]):
        self._meta = dict(meta)

    def refresh(
        self,
        names: List[str],
        selected: Optional[str],
        filter_term: Optional[str] = None
    ):
        # Purge
        for row in list(self._rows.values()):
            row.setParent(None)
            row.deleteLater()
        self._rows.clear()

        # Enlever le stretch
        if self._layout.count() > 0:
            item = self._layout.takeAt(self._layout.count() - 1)
            if item.spacerItem():
                del item

        filter_term = (filter_term or "").lower()

        for name in names:
            if filter_term and filter_term not in name.lower():
                continue

            _, d = self._meta.get(name, (0, 0.0))
            row = MacroRow(name, fmt_duration_for_list(d), self._container)
            row.clicked.connect(self._on_row_clicked)
            self._layout.addWidget(row)
            self._rows[name] = row

        self._layout.addStretch()

        if selected and selected in self._rows:
            self.select(selected, fire=False)

    def update_one(self, name: str):
        if name in self._rows:
            _, d = self._meta.get(name, (0, 0.0))
            self._rows[name].set_duration(fmt_duration_for_list(d))

    def select(self, name: str, fire: bool = True):
        if self._selected and self._selected in self._rows:
            self._rows[self._selected].set_selected(False)

        self._selected = name

        if name in self._rows:
            self._rows[name].set_selected(True)
            if fire:
                self.macro_selected.emit(name)

    def get_selected(self) -> Optional[str]:
        return self._selected

    def _on_row_clicked(self, name: str):
        self.select(name, fire=True)
