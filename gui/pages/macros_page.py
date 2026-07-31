# -*- coding: utf-8 -*-
"""Macro library and event preview page."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.components import MacroLibrary


class MacrosPage(QWidget):
    macro_selected = pyqtSignal(str)
    create_requested = pyqtSignal()
    rename_requested = pyqtSignal()
    delete_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)
        eyebrow = QLabel("MACRO LIBRARY")
        eyebrow.setObjectName("PageEyebrow")
        root.addWidget(eyebrow)
        title = QLabel("All macros")
        title.setObjectName("PageTitle")
        root.addWidget(title)
        subtitle = QLabel("Edit, rename or remove any macro.")
        subtitle.setObjectName("PageSubtitle")
        root.addWidget(subtitle)

        content = QHBoxLayout()
        content.setSpacing(16)
        self.library = MacroLibrary()
        self.library.setMinimumWidth(350)
        content.addWidget(self.library, 1)

        details = QFrame()
        details.setObjectName("Card")
        detail_layout = QVBoxLayout(details)
        detail_layout.setContentsMargins(18, 18, 18, 18)
        detail_layout.setSpacing(12)
        heading = QLabel("Event preview")
        heading.setObjectName("CardTitle")
        detail_layout.addWidget(heading)
        self.summary = QLabel("Select a macro to preview its events.")
        self.summary.setObjectName("CardCaption")
        detail_layout.addWidget(self.summary)
        self.events = QTableWidget(0, 4)
        self.events.setHorizontalHeaderLabels(["#", "Type", "Delay", "Data"])
        self.events.horizontalHeader().setStretchLastSection(True)
        self.events.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.events.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        detail_layout.addWidget(self.events, 1)
        content.addWidget(details, 2)
        root.addLayout(content, 1)

        self.library.macro_selected.connect(self.macro_selected)
        self.library.create_requested.connect(self.create_requested)
        self.library.rename_requested.connect(self.rename_requested)
        self.library.delete_requested.connect(self.delete_requested)

    def set_macro(self, macro) -> None:
        self.events.setRowCount(0)
        if not macro or not getattr(macro, "name", ""):
            self.summary.setText("Select a macro to preview its events.")
            return
        preview_steps = macro.steps[:120]
        preview_note = (
            f" · showing {len(preview_steps):,} of {macro.event_count():,}"
            if macro.event_count() > len(preview_steps)
            else ""
        )
        self.summary.setText(
            f"{macro.name} · {macro.event_count():,} events · {macro.duration():.2f}s{preview_note}"
        )
        self.events.setRowCount(len(preview_steps))
        for row, step in enumerate(preview_steps):
            values = [str(row + 1), step.step_type.to_string(), f"{step.time_delta:.3f}s", str(step.data)]
            for column, value in enumerate(values):
                self.events.setItem(row, column, QTableWidgetItem(value))
