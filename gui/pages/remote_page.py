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
        title = QLabel("Télécommande Telegram")
        title.setObjectName("PageTitle")
        root.addWidget(title)
        subtitle = QLabel("Supervise et contrôle AUTO-COC depuis ton téléphone, avec le même état que la console locale.")
        subtitle.setObjectName("PageSubtitle")
        root.addWidget(subtitle)

        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        top = QHBoxLayout()
        top.addWidget(StatusPill("Telegram", "Initialisation…", Theme.WARNING))
        top.addStretch()
        configure = QPushButton("Configurer Telegram…")
        configure.clicked.connect(self.configure_requested)
        top.addWidget(configure)
        layout.addLayout(top)
        info = QLabel("Les commandes distantes sont traitées dans le thread Telegram puis exécutées par le contrôleur Qt. L’interface locale reste utilisable même lorsque Telegram est hors ligne.")
        info.setObjectName("PageSubtitle")
        info.setWordWrap(True)
        layout.addWidget(info)
        commands = QListWidget()
        commands.addItems([
            "GO · lancer la macro sélectionnée",
            "STOP · arrêter immédiatement l’exécution",
            "CAPTURE · envoyer une capture d’écran",
            "LAUNCH_COC · lancer le chemin configuré",
            "MENU · afficher les actions secondaires",
        ])
        commands.setMinimumHeight(210)
        layout.addWidget(commands)
        actions = QHBoxLayout()
        capture = QPushButton("Envoyer une capture")
        capture.clicked.connect(self.capture_requested)
        actions.addStretch()
        actions.addWidget(capture)
        layout.addLayout(actions)
        root.addWidget(card, 1)

    def set_status(self, text: str, color: str) -> None:
        pill = self.findChild(StatusPill)
        if pill:
            pill.set_status(text, color)
