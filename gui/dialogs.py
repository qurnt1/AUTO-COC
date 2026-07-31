# -*- coding: utf-8 -*-
"""Modal PyQt6 dialogs used by the operator console."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QUrl, Qt
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
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.theme import Theme
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
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Valider")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Annuler")
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
    def __init__(self, params: dict[str, str], guide_path: Path | None, on_save: Callable[[dict[str, str]], None], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Télécommande Telegram")
        self.setMinimumWidth(560)
        self.params = params
        self.guide_path = guide_path
        self.on_save = on_save
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.addWidget(_title_block("Télécommande Telegram", "Contrôle à distance, captures d’écran et commandes sécurisées."))

        form = QFormLayout()
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(14)
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.token.setPlaceholderText("Laisser vide pour conserver le token actuel")
        self.token.setAccessibleName("Token du bot Telegram")
        self.chat_id = QLineEdit(params.get("telegram_chat_id", ""))
        self.chat_id.setPlaceholderText("Identifiant numérique du chat autorisé")
        self.chat_id.setAccessibleName("Chat ID Telegram")
        form.addRow("Token du bot", self.token)
        form.addRow("Chat ID", self.chat_id)
        layout.addLayout(form)

        hint = QLabel("Le token est masqué dans l’interface. Le champ vide conserve la valeur existante.")
        hint.setObjectName("CardCaption")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        actions = QHBoxLayout()
        guide = QPushButton("Ouvrir le guide")
        guide.setObjectName("QuietButton")
        guide.clicked.connect(self._open_guide)
        actions.addWidget(guide)
        actions.addStretch()
        layout.addLayout(actions)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Enregistrer")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Annuler")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _open_guide(self) -> None:
        if self.guide_path and self.guide_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.guide_path.resolve())))
            return
        QMessageBox.warning(self, "Guide introuvable", "Le guide Telegram local n’a pas été trouvé.")

    def _save(self) -> None:
        chat_text = self.chat_id.text().strip()
        if chat_text:
            try:
                int(chat_text)
            except ValueError:
                QMessageBox.warning(self, "Chat ID invalide", "Le Chat ID doit être un nombre entier.")
                return
        if self.token.text().strip():
            self.params["telegram_bot_token"] = self.token.text().strip()
        self.params["telegram_chat_id"] = chat_text
        self.on_save(self.params)
        self.accept()


class SettingsDialog(QDialog):
    def __init__(
        self,
        params: dict[str, str],
        on_save: Callable[[dict[str, str]], None],
        on_telegram: Callable[[], None],
        on_diagnostics: Callable[[], None],
        on_shutdown: Callable[[], None],
        telegram_status: str,
        telegram_color: str,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Paramètres")
        self.setMinimumWidth(620)
        self.params = params
        self.on_save = on_save
        self.on_telegram = on_telegram
        self.on_diagnostics = on_diagnostics
        self.on_shutdown = on_shutdown

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.addWidget(_title_block("Paramètres", "Réglages de lecture, intégrations et maintenance de l’application."))

        playback = QFrame()
        playback.setObjectName("Card")
        playback_layout = QVBoxLayout(playback)
        playback_layout.setContentsMargins(16, 14, 16, 14)
        playback_title = QLabel("Lecture")
        playback_title.setObjectName("CardTitle")
        playback_layout.addWidget(playback_title)
        self.loop = QCheckBox("Relancer automatiquement la macro")
        self.loop.setChecked(as_bool(params.get("auto_loop", "0")))
        self.loop.setAccessibleName("Lecture en boucle")
        playback_layout.addWidget(self.loop)
        self.safeguard = QCheckBox("Safeguard CoC : arrêter la macro si CoC disparaît")
        self.safeguard.setChecked(as_bool(params.get("coc_safeguard", "0")))
        self.safeguard.setToolTip("Le contrôle vérifie le processus et la fenêtre CoC pendant la lecture.")
        playback_layout.addWidget(self.safeguard)
        layout.addWidget(playback)

        telegram = QFrame()
        telegram.setObjectName("Card")
        tg_layout = QGridLayout(telegram)
        tg_layout.setContentsMargins(16, 14, 16, 14)
        tg_title = QLabel("Télécommande")
        tg_title.setObjectName("CardTitle")
        tg_layout.addWidget(tg_title, 0, 0, 1, 2)
        status = QLabel(f"●  {telegram_status}")
        status.setStyleSheet(f"color: {telegram_color}; font-weight: 700;")
        tg_layout.addWidget(status, 1, 0)
        tg_button = QPushButton("Configurer Telegram…")
        tg_button.clicked.connect(self.on_telegram)
        tg_layout.addWidget(tg_button, 1, 1, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(telegram)

        system = QFrame()
        system.setObjectName("Card")
        system_layout = QVBoxLayout(system)
        system_layout.setContentsMargins(16, 14, 16, 14)
        system_title = QLabel("Lancement et maintenance")
        system_title.setObjectName("CardTitle")
        system_layout.addWidget(system_title)
        path_row = QHBoxLayout()
        path_label = QLabel("Lanceur CoC")
        path_label.setObjectName("CardCaption")
        self.coc_path = QLineEdit(params.get("coc_path", ""))
        self.coc_path.setPlaceholderText("Chemin vers le .exe ou .lnk de CoC")
        self.coc_path.setAccessibleName("Chemin du lanceur CoC")
        browse = QPushButton("Parcourir")
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
        self.process_names.setToolTip("Noms de processus séparés par |. Laisser vide si le titre de fenêtre suffit.")
        self.window_titles = QLineEdit(params.get("coc_window_titles", "Clash of Clans"))
        self.window_titles.setPlaceholderText("ex. Clash of Clans|Google Play Games")
        self.window_titles.setToolTip("Fragments de titres séparés par |.")
        self.process_path_hint = QLineEdit(params.get("coc_process_path_hint", ""))
        self.process_path_hint.setPlaceholderText("Fragment de chemin .exe (optionnel)")
        self.process_path_hint.setAccessibleName("Fragment de chemin du processus CoC")
        self.startup_timeout = QSpinBox()
        self.startup_timeout.setRange(5, 300)
        self.startup_timeout.setValue(int(coc_profile.startup_timeout))
        self.startup_timeout.setSuffix(" s")
        self.startup_timeout.setAccessibleName("Délai de confirmation CoC")
        self.detection_interval = QDoubleSpinBox()
        self.detection_interval.setRange(0.25, 5.0)
        self.detection_interval.setDecimals(2)
        self.detection_interval.setSingleStep(0.25)
        self.detection_interval.setValue(coc_profile.detection_interval)
        self.detection_interval.setSuffix(" s")
        self.detection_interval.setAccessibleName("Intervalle de détection CoC")
        self.missing_tolerance = QSpinBox()
        self.missing_tolerance.setRange(1, 10)
        self.missing_tolerance.setValue(coc_profile.missing_tolerance)
        self.missing_tolerance.setSuffix(" contrôles")
        self.missing_tolerance.setAccessibleName("Tolérance d’absence CoC")
        detection.addWidget(QLabel("Processus CoC"), 0, 0)
        detection.addWidget(self.process_names, 0, 1)
        detection.addWidget(QLabel("Titres de fenêtre"), 1, 0)
        detection.addWidget(self.window_titles, 1, 1)
        detection.addWidget(QLabel("Chemin processus"), 2, 0)
        detection.addWidget(self.process_path_hint, 2, 1)
        detection.addWidget(QLabel("Confirmation lancement"), 3, 0)
        detection.addWidget(self.startup_timeout, 3, 1)
        detection.addWidget(QLabel("Tolérance safeguard"), 4, 0)
        detection.addWidget(self.missing_tolerance, 4, 1)
        detection.addWidget(QLabel("Intervalle détection"), 5, 0)
        detection.addWidget(self.detection_interval, 5, 1)
        system_layout.addLayout(detection)
        hint = QLabel("AUTO-COC vérifie d’abord la présence réelle de CoC. Le bouton de lancement ouvre l’application configurée, puis la détection confirme son arrivée.")
        hint.setObjectName("CardCaption")
        hint.setWordWrap(True)
        system_layout.addWidget(hint)
        maintenance = QHBoxLayout()
        diagnostics = QPushButton("Diagnostics")
        diagnostics.clicked.connect(self.on_diagnostics)
        shutdown = QPushButton("Éteindre le PC")
        shutdown.setObjectName("DangerButton")
        shutdown.clicked.connect(self._confirm_shutdown)
        maintenance.addWidget(diagnostics)
        maintenance.addStretch()
        maintenance.addWidget(shutdown)
        system_layout.addLayout(maintenance)
        layout.addWidget(system)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Enregistrer")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Annuler")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choisir le lancement CoC", "", "Applications (*.exe *.lnk);;Tous les fichiers (*)")
        if path:
            self.coc_path.setText(path)

    def _confirm_shutdown(self) -> None:
        answer = QMessageBox.question(self, "Éteindre le PC", "Confirmer l’extinction du PC ?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
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
        self.on_save(self.params)
        self.accept()


class DiagnosticsDialog(QDialog):
    def __init__(self, *, app_version: str, python_version: str, telegram_status: str, pil_available: bool, mss_available: bool, log_path: Path, base_dir: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Diagnostics")
        self.resize(720, 560)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.addWidget(_title_block("Diagnostics", "État des dépendances, de l’environnement et des dernières opérations."))

        grid = QGridLayout()
        grid.setHorizontalSpacing(22)
        grid.setVerticalSpacing(10)
        rows = [
            ("Version", app_version, Theme.TEXT),
            ("Python", python_version, Theme.TEXT),
            ("Telegram", telegram_status, Theme.ACCENT if "Connecté" in telegram_status else Theme.WARNING),
            ("Pillow", "Disponible" if pil_available else "Indisponible", Theme.ACCENT if pil_available else Theme.DANGER),
            ("MSS", "Disponible" if mss_available else "Fallback actif", Theme.ACCENT if mss_available else Theme.WARNING),
            ("Espace disque", f"{shutil.disk_usage(base_dir).free / (1024 ** 3):.1f} Go libres", Theme.TEXT),
        ]
        for row, (label, value, color) in enumerate(rows):
            key = QLabel(label)
            key.setObjectName("CardCaption")
            value_label = QLabel(value)
            value_label.setStyleSheet(f"color: {color}; font-weight: 700;")
            grid.addWidget(key, row, 0)
            grid.addWidget(value_label, row, 1)
        layout.addLayout(grid)

        logs_title = QLabel("Derniers événements")
        logs_title.setObjectName("CardTitle")
        layout.addWidget(logs_title)
        log_view = QPlainTextEdit()
        log_view.setReadOnly(True)
        log_view.setObjectName("Mono")
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()[-12:]
            log_view.setPlainText("\n".join(lines) if lines else "Aucun log disponible.")
        except OSError as exc:
            log_view.setPlainText(f"Lecture des logs impossible : {exc}")
        layout.addWidget(log_view, 1)
        close = QPushButton("Fermer")
        close.clicked.connect(self.accept)
        layout.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)
