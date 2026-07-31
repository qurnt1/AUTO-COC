# -*- coding: utf-8 -*-
"""Modal PyQt6 dialogs used by the operator console."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from services.coc.models import CocLaunchProfile
from utils.config import as_bool


def _title_block(title: str, description: str) -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 12)
    layout.setSpacing(4)
    title_label = QLabel(title)
    title_label.setObjectName("PageTitle")
    title_label.setStyleSheet("font-size: 22px;")
    description_label = QLabel(description)
    description_label.setObjectName("PageSubtitle")
    description_label.setWordWrap(True)
    layout.addWidget(title_label)
    layout.addWidget(description_label)
    return widget


class TextInputDialog(QDialog):
    def __init__(self, title: str, prompt: str, initial: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(430)
        self.setModal(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 18)
        layout.addWidget(_title_block(title, prompt))
        self.entry = QLineEdit(initial)
        self.entry.selectAll()
        self.entry.setAccessibleName(prompt)
        layout.addWidget(self.entry)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.entry.returnPressed.connect(self.accept)
        self.entry.setFocus()

    @classmethod
    def get_text(cls, parent, title: str, prompt: str, initial: str = "") -> str | None:
        dialog = cls(title, prompt, initial, parent)
        return dialog.entry.text().strip() if dialog.exec() == QDialog.DialogCode.Accepted else None


class TelegramDialog(QDialog):
    def __init__(self, params: dict[str, str], guide_path: Path | None, on_save: Callable[[dict[str, str]], bool], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Telegram")
        self.setMinimumWidth(560)
        self.params = params
        self.guide_path = guide_path
        self.on_save = on_save
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.addWidget(_title_block("Telegram", "Remote control and screenshots."))

        form = QFormLayout()
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(14)
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.token.setPlaceholderText("Leave blank to keep the current token")
        self.token.setAccessibleName("Telegram bot token")
        self.chat_id = QLineEdit(params.get("telegram_chat_id", ""))
        self.chat_id.setPlaceholderText("Allowed numeric chat ID")
        self.chat_id.setAccessibleName("Telegram chat ID")
        form.addRow("Bot token", self.token)
        form.addRow("Chat ID", self.chat_id)
        layout.addLayout(form)

        hint = QLabel("The token is hidden. Leave it blank to keep the current value.")
        hint.setObjectName("CardCaption")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        actions = QHBoxLayout()
        guide = QPushButton("Open guide")
        guide.setObjectName("QuietButton")
        guide.clicked.connect(self._open_guide)
        actions.addWidget(guide)
        actions.addStretch()
        layout.addLayout(actions)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Save")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _open_guide(self) -> None:
        if self.guide_path and self.guide_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.guide_path.resolve())))
            return
        QMessageBox.warning(self, "Guide not found", "The local Telegram guide was not found.")

    def _save(self) -> None:
        chat_text = self.chat_id.text().strip()
        if chat_text:
            try:
                int(chat_text)
            except ValueError:
                QMessageBox.warning(self, "Invalid chat ID", "Chat ID must be an integer.")
                return
        if self.token.text().strip():
            self.params["telegram_bot_token"] = self.token.text().strip()
        self.params["telegram_chat_id"] = chat_text
        if self.on_save(self.params):
            self.accept()


class SettingsDialog(QDialog):
    def __init__(
        self,
        params: dict[str, str],
        on_save: Callable[[dict[str, str]], bool],
        on_shutdown: Callable[[], None],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(620)
        self.params = params
        self.on_save = on_save
        self.on_shutdown = on_shutdown

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.addWidget(_title_block("Settings", "Playback and Clash of Clans settings."))

        playback = QFrame()
        playback.setObjectName("Card")
        playback_layout = QVBoxLayout(playback)
        playback_layout.setContentsMargins(16, 14, 16, 14)
        playback_title = QLabel("Playback")
        playback_title.setObjectName("CardTitle")
        playback_layout.addWidget(playback_title)
        self.loop = QCheckBox("Loop playback")
        self.loop.setChecked(as_bool(params.get("auto_loop", "0")))
        self.loop.setAccessibleName("Loop playback")
        playback_layout.addWidget(self.loop)
        self.safeguard = QCheckBox("Stop if CoC disappears")
        self.safeguard.setChecked(as_bool(params.get("coc_safeguard", "0")))
        self.safeguard.setToolTip("Monitor the CoC process and window while a macro runs.")
        playback_layout.addWidget(self.safeguard)
        layout.addWidget(playback)

        system = QFrame()
        system.setObjectName("Card")
        system_layout = QVBoxLayout(system)
        system_layout.setContentsMargins(16, 14, 16, 14)
        system_title = QLabel("CoC")
        system_title.setObjectName("CardTitle")
        system_layout.addWidget(system_title)
        path_row = QHBoxLayout()
        path_label = QLabel("CoC launcher (optional)")
        path_label.setObjectName("CardCaption")
        self.coc_path = QLineEdit(params.get("coc_path", ""))
        self.coc_path.setPlaceholderText("Auto-detect, or choose an .exe / .lnk")
        self.coc_path.setAccessibleName("CoC launcher path")
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        path_row.addWidget(path_label)
        path_row.addWidget(self.coc_path, 1)
        path_row.addWidget(browse)
        system_layout.addLayout(path_row)
        detection = QGridLayout()
        detection.setHorizontalSpacing(10)
        coc_profile = CocLaunchProfile.from_params(params)
        self.process_names = QLineEdit(params.get("coc_process_names", ""))
        self.process_names.setPlaceholderText("ex. wsaClient.exe|ClashOfClans.exe")
        self.process_names.setToolTip("Process names separated by |.")
        self.process_names.setAccessibleName("Clash of Clans process names")
        self.window_titles = QLineEdit(params.get("coc_window_titles", "Clash of Clans"))
        self.window_titles.setPlaceholderText("ex. Clash of Clans|Google Play Games")
        self.window_titles.setToolTip("Window title fragments separated by |.")
        self.window_titles.setAccessibleName("Clash of Clans window titles")
        self.process_path_hint = QLineEdit(params.get("coc_process_path_hint", ""))
        self.process_path_hint.setPlaceholderText("Optional .exe path fragment")
        self.process_path_hint.setAccessibleName("CoC process path fragment")
        self.startup_timeout = QSpinBox()
        self.startup_timeout.setRange(5, 300)
        self.startup_timeout.setValue(int(coc_profile.startup_timeout))
        self.startup_timeout.setSuffix(" s")
        self.startup_timeout.setAccessibleName("CoC launch confirmation timeout")
        self.detection_interval = QDoubleSpinBox()
        self.detection_interval.setRange(0.25, 5.0)
        self.detection_interval.setDecimals(2)
        self.detection_interval.setSingleStep(0.25)
        self.detection_interval.setValue(coc_profile.detection_interval)
        self.detection_interval.setSuffix(" s")
        self.detection_interval.setAccessibleName("CoC detection interval")
        self.missing_tolerance = QSpinBox()
        self.missing_tolerance.setRange(1, 10)
        self.missing_tolerance.setValue(coc_profile.missing_tolerance)
        self.missing_tolerance.setSuffix(" checks")
        self.missing_tolerance.setAccessibleName("CoC missing tolerance")
        detection.addWidget(QLabel("Process names"), 0, 0)
        detection.addWidget(self.process_names, 0, 1)
        detection.addWidget(QLabel("Window titles"), 1, 0)
        detection.addWidget(self.window_titles, 1, 1)
        detection.addWidget(QLabel("Process path"), 2, 0)
        detection.addWidget(self.process_path_hint, 2, 1)
        detection.addWidget(QLabel("Launch timeout"), 3, 0)
        detection.addWidget(self.startup_timeout, 3, 1)
        detection.addWidget(QLabel("Missing checks"), 4, 0)
        detection.addWidget(self.missing_tolerance, 4, 1)
        detection.addWidget(QLabel("Detection interval"), 5, 0)
        detection.addWidget(self.detection_interval, 5, 1)
        system_layout.addLayout(detection)
        hint = QLabel("Leave the path empty to search Windows shortcuts automatically. A launch succeeds only after CoC is detected.")
        hint.setObjectName("CardCaption")
        hint.setWordWrap(True)
        system_layout.addWidget(hint)
        maintenance = QHBoxLayout()
        shutdown = QPushButton("Shut down PC")
        shutdown.setObjectName("DangerButton")
        shutdown.clicked.connect(self._confirm_shutdown)
        maintenance.addStretch()
        maintenance.addWidget(shutdown)
        system_layout.addLayout(maintenance)
        layout.addWidget(system)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Save")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose CoC launcher", "", "Applications (*.exe *.lnk);;All files (*)")
        if path:
            self.coc_path.setText(path)

    def _confirm_shutdown(self) -> None:
        answer = QMessageBox.question(self, "Shut down PC", "Shut down the computer?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.on_shutdown()

    def _save(self) -> None:
        self.params["auto_loop"] = "1" if self.loop.isChecked() else "0"
        self.params["coc_safeguard"] = "1" if self.safeguard.isChecked() else "0"
        self.params["coc_path"] = self.coc_path.text().strip()
        self.params["coc_process_names"] = self.process_names.text().strip()
        self.params["coc_window_titles"] = self.window_titles.text().strip() or "Clash of Clans"
        self.params["coc_process_path_hint"] = self.process_path_hint.text().strip()
        self.params["coc_startup_timeout"] = str(self.startup_timeout.value())
        self.params["coc_detection_interval"] = f"{self.detection_interval.value():g}"
        self.params["coc_missing_tolerance"] = str(self.missing_tolerance.value())
        if self.on_save(self.params):
            self.accept()
