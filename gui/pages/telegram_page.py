# -*- coding: utf-8 -*-
"""Telegram configuration and remote-action page."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from gui.components import StatusPill
from gui.models import MacroSummary
from gui.theme import Theme
from utils.config import fmt_seconds


class TelegramPage(QWidget):
    configure_requested = pyqtSignal()
    capture_requested = pyqtSignal()
    create_action_requested = pyqtSignal()
    record_action_requested = pyqtSignal()
    rename_action_requested = pyqtSignal()
    remove_action_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._configured = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)
        eyebrow = QLabel("REMOTE CONTROL")
        eyebrow.setObjectName("PageEyebrow")
        root.addWidget(eyebrow)
        title = QLabel("Telegram settings")
        title.setObjectName("PageTitle")
        root.addWidget(title)
        subtitle = QLabel("Connection, screenshots and private routines used only by the bot.")
        subtitle.setObjectName("PageSubtitle")
        root.addWidget(subtitle)

        connection = QFrame()
        connection.setObjectName("Card")
        connection_layout = QHBoxLayout(connection)
        connection_layout.setContentsMargins(18, 16, 18, 16)
        connection_layout.setSpacing(12)
        self.status = StatusPill("Telegram", "Starting…", Theme.WARNING)
        self.status.setAccessibleName("Telegram connection status")
        connection_layout.addWidget(self.status)
        connection_layout.addStretch()
        self.capture_button = QPushButton("Send latest screenshot")
        self.capture_button.setToolTip("Replaces the previous bot screenshot to keep the chat compact.")
        self.capture_button.setAccessibleName("Send latest screenshot to Telegram")
        self.configure_button = QPushButton("Configure")
        self.configure_button.setObjectName("PrimaryButton")
        connection_layout.addWidget(self.capture_button)
        connection_layout.addWidget(self.configure_button)
        root.addWidget(connection)

        action = QFrame()
        action.setObjectName("CardRaised")
        action_layout = QVBoxLayout(action)
        action_layout.setContentsMargins(20, 18, 20, 18)
        action_layout.setSpacing(14)

        action_header = QHBoxLayout()
        action_title = QVBoxLayout()
        action_title.setSpacing(5)
        tag = QLabel("TELEGRAM ACTION")
        tag.setObjectName("Tag")
        tag.setAccessibleName("Telegram action")
        action_title.addWidget(tag, alignment=Qt.AlignmentFlag.AlignLeft)
        name = QLabel("Reload Clash of Clans")
        name.setObjectName("CardTitle")
        name.setStyleSheet("font-size: 18px;")
        action_title.addWidget(name)
        explanation = QLabel("This routine is called by Telegram. It never appears as a macro you can launch from Home.")
        explanation.setObjectName("PageSubtitle")
        explanation.setWordWrap(True)
        action_title.addWidget(explanation)
        action_header.addLayout(action_title, 1)
        self.action_state = StatusPill("Routine", "Not configured", Theme.TEXT_MUTED)
        action_header.addWidget(self.action_state, alignment=Qt.AlignmentFlag.AlignTop)
        action_layout.addLayout(action_header)

        self.action_name = QLabel("No routine assigned")
        self.action_name.setObjectName("SectionTitle")
        self.action_name.setWordWrap(True)
        self.action_meta = QLabel("Create it once, then record the steps that reopen or refresh CoC.")
        self.action_meta.setObjectName("CardCaption")
        self.action_meta.setWordWrap(True)
        action_layout.addWidget(self.action_name)
        action_layout.addWidget(self.action_meta)

        action_buttons = QHBoxLayout()
        action_buttons.setSpacing(8)
        self.create_button = QPushButton("Create routine")
        self.create_button.setObjectName("PrimaryButton")
        self.record_button = QPushButton("Record")
        self.rename_button = QPushButton("Rename")
        self.remove_button = QPushButton("Remove")
        self.remove_button.setObjectName("DangerButton")
        action_buttons.addWidget(self.create_button)
        action_buttons.addWidget(self.record_button)
        action_buttons.addWidget(self.rename_button)
        action_buttons.addStretch()
        action_buttons.addWidget(self.remove_button)
        action_layout.addLayout(action_buttons)
        root.addWidget(action)

        behavior = QFrame()
        behavior.setObjectName("Card")
        behavior_layout = QVBoxLayout(behavior)
        behavior_layout.setContentsMargins(18, 16, 18, 16)
        behavior_layout.setSpacing(6)
        behavior_title = QLabel("Compact conversation")
        behavior_title.setObjectName("CardTitle")
        behavior_layout.addWidget(behavior_title)
        behavior_text = QLabel(
            "AUTO-COC keeps one active control panel and one latest screenshot. Older bot panels and captures are removed when Telegram allows it."
        )
        behavior_text.setObjectName("PageSubtitle")
        behavior_text.setWordWrap(True)
        behavior_layout.addWidget(behavior_text)
        root.addWidget(behavior)
        root.addStretch()

        self.configure_button.clicked.connect(self.configure_requested)
        self.capture_button.clicked.connect(self.capture_requested)
        self.create_button.clicked.connect(self.create_action_requested)
        self.record_button.clicked.connect(self.record_action_requested)
        self.rename_button.clicked.connect(self.rename_action_requested)
        self.remove_button.clicked.connect(self.remove_action_requested)
        self.set_action(None)

    def set_status(self, text: str, color: str) -> None:
        self.status.set_status(text, color)
        self.capture_button.setEnabled(text.casefold() == "connected")

    def set_action(self, summary: MacroSummary | None) -> None:
        self._configured = summary is not None
        self.create_button.setVisible(not self._configured)
        self.record_button.setVisible(self._configured)
        self.rename_button.setVisible(self._configured)
        self.remove_button.setVisible(self._configured)
        if summary:
            self.action_state.set_status("Configured", Theme.ACCENT)
            self.action_name.setText(summary.name)
            self.action_meta.setText(f"{summary.events:,} events  ·  {fmt_seconds(summary.duration)}")
        else:
            self.action_state.set_status("Not configured", Theme.TEXT_MUTED)
            self.action_name.setText("No routine assigned")
            self.action_meta.setText("Create it once, then record the steps that reopen or refresh CoC.")

    def set_runtime_state(self, state_name: str, active_name: str | None) -> None:
        recording_this_action = state_name == "RECORDING" and self._configured and active_name == self.action_name.text()
        self.record_button.setText("Stop recording" if recording_this_action else "Record")
        self.record_button.setObjectName("DangerButton" if recording_this_action else "")
        busy_elsewhere = state_name in {"PLAYING", "STOPPING"} or (state_name == "RECORDING" and not recording_this_action)
        self.record_button.setEnabled(not busy_elsewhere)
        self.rename_button.setEnabled(state_name == "IDLE")
        self.remove_button.setEnabled(state_name == "IDLE")
        self.create_button.setEnabled(state_name == "IDLE")
        for button in (self.record_button, self.create_button):
            button.style().unpolish(button)
            button.style().polish(button)
