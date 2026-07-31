# -*- coding: utf-8 -*-
"""Telegram remote-control page."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget

from gui.components import StatusPill
from gui.theme import Theme


class RemotePage(QWidget):
    configure_requested = pyqtSignal()
    capture_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)
        eyebrow = QLabel("REMOTE CONTROL")
        eyebrow.setObjectName("PageEyebrow")
        root.addWidget(eyebrow)
        title = QLabel("Remote")
        title.setObjectName("PageTitle")
        root.addWidget(title)
        subtitle = QLabel("Control AUTO-COC from Telegram.")
        subtitle.setObjectName("PageSubtitle")
        root.addWidget(subtitle)

        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        top = QHBoxLayout()
        top.addWidget(StatusPill("Telegram", "Starting…", Theme.WARNING))
        top.addStretch()
        configure = QPushButton("Configure Telegram")
        configure.clicked.connect(self.configure_requested)
        top.addWidget(configure)
        layout.addLayout(top)
        info = QLabel("Remote commands use the same runtime state as the local console.")
        info.setObjectName("PageSubtitle")
        info.setWordWrap(True)
        layout.addWidget(info)
        commands = QListWidget()
        commands.addItems([
            "GO · run the selected macro",
            "STOP · stop immediately",
            "CAPTURE · send a screenshot",
            "LAUNCH_COC · open the configured launcher",
            "MENU · open more actions",
        ])
        commands.setMinimumHeight(210)
        layout.addWidget(commands)
        actions = QHBoxLayout()
        capture = QPushButton("Send screenshot")
        capture.clicked.connect(self.capture_requested)
        actions.addStretch()
        actions.addWidget(capture)
        layout.addLayout(actions)
        root.addWidget(card, 1)

    def set_status(self, text: str, color: str) -> None:
        pill = self.findChild(StatusPill)
        if pill:
            pill.set_status(text, color)
