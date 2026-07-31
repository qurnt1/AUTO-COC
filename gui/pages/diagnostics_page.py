# -*- coding: utf-8 -*-
"""Diagnostics page."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QFrame, QLabel, QPlainTextEdit, QVBoxLayout, QWidget


class DiagnosticsPage(QWidget):
    def __init__(self, *, log_path: Path, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)
        eyebrow = QLabel("SYSTEM")
        eyebrow.setObjectName("PageEyebrow")
        root.addWidget(eyebrow)
        title = QLabel("Diagnostics")
        title.setObjectName("PageTitle")
        root.addWidget(title)
        subtitle = QLabel("Recent application events.")
        subtitle.setObjectName("PageSubtitle")
        root.addWidget(subtitle)

        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        logs_title = QLabel("Activity log")
        logs_title.setObjectName("CardTitle")
        layout.addWidget(logs_title)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, 1)
        root.addWidget(card, 1)
        self.log_path = log_path
        self.refresh()

    def refresh(self) -> None:
        try:
            self.log_view.setPlainText("\n".join(self.log_path.read_text(encoding="utf-8").splitlines()[-30:]))
        except OSError as exc:
            self.log_view.setPlainText(f"Could not read log: {exc}")
