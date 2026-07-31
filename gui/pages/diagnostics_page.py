# -*- coding: utf-8 -*-
"""Diagnostics page."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QFrame, QGridLayout, QLabel, QPlainTextEdit, QVBoxLayout, QWidget


class DiagnosticsPage(QWidget):
    def __init__(self, *, app_version: str, python_version: str, log_path: Path, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)
        eyebrow = QLabel("SYSTEM HEALTH")
        eyebrow.setObjectName("PageEyebrow")
        root.addWidget(eyebrow)
        title = QLabel("Diagnostics et journal")
        title.setObjectName("PageTitle")
        root.addWidget(title)
        subtitle = QLabel("Les informations utiles quand une macro, une intégration ou une machine ne répond pas comme prévu.")
        subtitle.setObjectName("PageSubtitle")
        root.addWidget(subtitle)

        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        grid = QGridLayout()
        rows = [("Version application", app_version), ("Version Python", python_version), ("Fichier de log", str(log_path))]
        for row, (key, value) in enumerate(rows):
            key_label = QLabel(key)
            key_label.setObjectName("CardCaption")
            value_label = QLabel(value)
            value_label.setObjectName("Mono")
            grid.addWidget(key_label, row, 0)
            grid.addWidget(value_label, row, 1)
        layout.addLayout(grid)
        logs_title = QLabel("Derniers logs")
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
            self.log_view.setPlainText(f"Lecture impossible : {exc}")
