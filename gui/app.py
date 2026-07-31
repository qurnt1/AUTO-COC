# -*- coding: utf-8 -*-
"""Main PyQt6 window for AUTO-COC."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import keyboard
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui.components import NavButton, StatusPill
from gui.controller import RuntimeController, RuntimeState
from gui.dialogs import DiagnosticsDialog, SettingsDialog, TelegramDialog, TextInputDialog
from gui.pages.console_page import ConsolePage
from gui.pages.diagnostics_page import DiagnosticsPage
from gui.pages.macros_page import MacrosPage
from gui.pages.remote_page import RemotePage
from gui.theme import Theme, build_stylesheet
from utils.logger import clean_old_logs, get_logger


class MainWindow(QMainWindow):
    hotkey_signal = pyqtSignal(str)

    def __init__(
        self,
        params: dict[str, str],
        telegram,
        base_dir: Path,
        macros_dir: Path,
        params_path: Path,
        log_path: Path,
        icon_path: Path,
        guide_path: Path,
        app_version: str,
        python_version: str,
        protected_names: list[str],
    ):
        super().__init__()
        self.log = get_logger()
        self.app_version = app_version
        self.python_version = python_version
        self.base_dir = base_dir
        self.params_path = params_path
        self.log_path = log_path
        self.icon_path = icon_path
        self.guide_path = guide_path
        self.telegram = telegram
        self.controller = RuntimeController(
            params=params,
            telegram=telegram,
            macros_dir=macros_dir,
            params_path=params_path,
            protected_names=protected_names,
            app_version=app_version,
            python_version=python_version,
            parent=self,
        )
        self._closing = False
        self._hotkeys_registered = False
        self.setWindowTitle(f"AUTO-COC  /  {app_version}")
        self.setMinimumSize(1180, 760)
        self.resize(1480, 900)
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.setStyleSheet(build_stylesheet())
        self._set_windows_app_id()
        self._build_shell()
        self._connect_signals()
        self._register_shortcuts()
        self._register_hotkeys()

        self.metrics_timer = QTimer(self)
        self.metrics_timer.setInterval(200)
        self.metrics_timer.timeout.connect(self.controller.update_metrics)
        self.metrics_timer.start()
        self.telegram_timer = QTimer(self)
        self.telegram_timer.setInterval(250)
        self.telegram_timer.timeout.connect(self.controller.poll_telegram_status)
        self.telegram_timer.start()
        QTimer.singleShot(120, self.controller.bootstrap)

    def _build_shell(self) -> None:
        central = QWidget()
        central.setObjectName("WindowRoot")
        shell = QHBoxLayout(central)
        shell.setContentsMargins(14, 14, 14, 14)
        shell.setSpacing(14)

        sidebar = QFrame()
        sidebar.setObjectName("Card")
        sidebar.setFixedWidth(228)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(14, 16, 14, 14)
        sidebar_layout.setSpacing(6)

        brand = QVBoxLayout()
        brand.setContentsMargins(8, 4, 8, 18)
        brand.setSpacing(3)
        brand_name = QLabel("AUTO-COC")
        brand_name.setStyleSheet(f"font-size: 20px; font-weight: 800; color: {Theme.TEXT};")
        brand_tag = QLabel("OPERATOR CONSOLE")
        brand_tag.setStyleSheet(f"font-size: 10px; font-weight: 700; letter-spacing: 1px; color: {Theme.ACCENT};")
        brand.addWidget(brand_name)
        brand.addWidget(brand_tag)
        sidebar_layout.addLayout(brand)

        self.nav_buttons: list[NavButton] = []
        for label, tooltip in (("Console", "Vue d’ensemble et commandes"), ("Macros", "Bibliothèque et événements"), ("Télécommande", "Statut et commandes Telegram"), ("Diagnostics", "État système et journal")):
            button = NavButton(label, tooltip)
            self.nav_buttons.append(button)
            sidebar_layout.addWidget(button)
        self.nav_buttons[0].setChecked(True)
        sidebar_layout.addStretch()

        shortcuts = QLabel("F1  lancer / arrêter\nCtrl+Shift+1  lancer\nCtrl+Shift+0  stopper")
        shortcuts.setStyleSheet(f"color: {Theme.TEXT_SUBTLE}; font-family: 'Cascadia Mono'; font-size: 10px; padding: 8px;")
        sidebar_layout.addWidget(shortcuts)
        self.settings_button = QPushButton("Paramètres")
        self.settings_button.setObjectName("QuietButton")
        self.settings_button.setMinimumHeight(40)
        sidebar_layout.addWidget(self.settings_button)
        shell.addWidget(sidebar)

        content = QVBoxLayout()
        content.setSpacing(14)
        header = QFrame()
        header.setObjectName("Card")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 12, 18, 12)
        title_column = QVBoxLayout()
        title_column.setSpacing(2)
        self.header_title = QLabel("Console")
        self.header_title.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {Theme.TEXT};")
        header_subtitle = QLabel("AUTO-COC / contrôle local")
        header_subtitle.setStyleSheet(f"font-size: 11px; color: {Theme.TEXT_SUBTLE};")
        title_column.addWidget(self.header_title)
        title_column.addWidget(header_subtitle)
        header_layout.addLayout(title_column)
        header_layout.addStretch()
        self.state_pill = StatusPill("État", "Prêt", Theme.ACCENT)
        self.telegram_pill = StatusPill("Telegram", "Hors ligne", Theme.WARNING)
        self.coc_pill = StatusPill("CoC", "Non lancé", Theme.TEXT_MUTED)
        header_layout.addWidget(self.state_pill)
        header_layout.addWidget(self.telegram_pill)
        header_layout.addWidget(self.coc_pill)
        content.addWidget(header)

        self.pages = QStackedWidget()
        self.console_page = ConsolePage()
        self.macros_page = MacrosPage()
        self.remote_page = RemotePage()
        self.diagnostics_page = DiagnosticsPage(app_version=self.app_version, python_version=self.python_version, log_path=self.log_path)
        for page in (self.console_page, self.macros_page, self.remote_page, self.diagnostics_page):
            self.pages.addWidget(page)
        content.addWidget(self.pages, 1)
        shell.addLayout(content, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Prêt")

    def _connect_signals(self) -> None:
        for index, button in enumerate(self.nav_buttons):
            button.clicked.connect(lambda checked, idx=index: self._navigate(idx))
        self.settings_button.clicked.connect(self.open_settings)

        pages = (self.console_page, self.macros_page)
        for page in pages:
            page.macro_selected.connect(self.controller.select_macro)
            page.create_requested.connect(self.create_macro)
            page.rename_requested.connect(self.rename_macro)
            page.delete_requested.connect(self.delete_macro)
        self.console_page.record_requested.connect(self.toggle_recording)
        self.console_page.play_requested.connect(self.controller.start_playback)
        self.console_page.stop_requested.connect(self.controller.stop_all)
        self.console_page.coc_requested.connect(self.controller.launch_coc)
        self.console_page.loop_toggled.connect(self.controller.set_auto_loop)
        self.console_page.safeguard_toggled.connect(self.controller.set_safeguard)
        self.remote_page.configure_requested.connect(self.open_telegram)
        self.remote_page.capture_requested.connect(self.controller.capture_screen)

        signals = self.controller.signals
        signals.state_changed.connect(self._on_state_changed)
        signals.macros_changed.connect(self._on_macros_changed)
        signals.macro_selected.connect(self._on_macro_selected)
        signals.metrics_changed.connect(self._on_metrics_changed)
        signals.activity.connect(self._on_activity)
        signals.error.connect(self._on_error)
        signals.telegram_status.connect(self._on_telegram_status)
        signals.coc_presence_changed.connect(self._on_coc_presence)
        signals.safeguard_triggered.connect(self.controller.handle_safeguard_loss)
        self.hotkey_signal.connect(self._on_hotkey)

    def _register_shortcuts(self) -> None:
        self.shortcut_new = QShortcut(QKeySequence("Ctrl+N"), self)
        self.shortcut_new.activated.connect(self.create_macro)
        self.shortcut_search = QShortcut(QKeySequence("Ctrl+F"), self)
        self.shortcut_search.activated.connect(self.console_page.library.search.setFocus)
        self.shortcut_stop = QShortcut(QKeySequence("Escape"), self)
        self.shortcut_stop.activated.connect(self.controller.stop_all)
        self.shortcut_play = QShortcut(QKeySequence("Ctrl+Return"), self)
        self.shortcut_play.activated.connect(self.controller.start_playback)

    def _register_hotkeys(self) -> None:
        try:
            keyboard.add_hotkey("f1", lambda: self.hotkey_signal.emit("toggle"))
            keyboard.add_hotkey("ctrl+shift+1", lambda: self.hotkey_signal.emit("play"))
            keyboard.add_hotkey("ctrl+shift+0", lambda: self.hotkey_signal.emit("stop"))
            self._hotkeys_registered = True
            self._on_activity("Raccourcis globaux actifs", "success")
        except Exception as exc:
            self._on_activity(f"Raccourcis globaux indisponibles · {exc}", "warning")

    def _navigate(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        self.header_title.setText(self.nav_buttons[index].text())
        self.nav_buttons[index].setChecked(True)
        if index == 3:
            self.diagnostics_page.refresh()

    def _on_macros_changed(self, items, selected: str) -> None:
        self.console_page.library.set_items(items, selected)
        self.macros_page.library.set_items(items, selected)

    def _refresh_macro_libraries(self) -> None:
        self._on_macros_changed(self.controller.summaries(), self.controller.current_macro_name or "")

    def _on_macro_selected(self, macro) -> None:
        self.console_page.set_macro(macro)
        self.macros_page.set_macro(macro)

    def _on_state_changed(self, state_name: str, label: str) -> None:
        colors = {
            "IDLE": Theme.ACCENT,
            "RECORDING": Theme.DANGER,
            "PLAYING": Theme.INFO,
            "STOPPING": Theme.WARNING,
            "ERROR": Theme.DANGER,
        }
        messages = {
            "IDLE": "Prêt à exécuter",
            "RECORDING": "Capture en cours · effectue tes actions dans CoC",
            "PLAYING": "La macro est en cours d’exécution",
            "STOPPING": "Arrêt sécurisé en cours",
            "ERROR": "Une opération nécessite ton attention",
        }
        self.state_pill.set_status(label, colors.get(state_name, Theme.TEXT_MUTED))
        self.console_page.set_state(label, colors.get(state_name, Theme.TEXT_MUTED), messages.get(state_name, label), state_name)
        self.statusBar().showMessage(label)

    def _on_metrics_changed(self, elapsed: float, events: int, duration: float, cycles: int) -> None:
        self.console_page.set_metrics(elapsed, events, duration, cycles, self.controller.state.name, self.controller.auto_loop)

    def _on_activity(self, message: str, level: str) -> None:
        self.console_page.activity.add(message, level)
        self.statusBar().showMessage(message, 5000)

    def _on_error(self, title: str, detail: str) -> None:
        message = f"{title}\n\n{detail}" if detail else title
        QMessageBox.warning(self, title, message)

    def _on_telegram_status(self, status: str, color: str) -> None:
        self.telegram_pill.set_status(status, color)
        self.remote_page.set_status(status, color)

    def _on_coc_presence(self, snapshot) -> None:
        self.console_page.set_coc_presence(snapshot)
        if snapshot.error:
            self.coc_pill.set_status("Indisponible", Theme.WARNING)
        elif snapshot.present:
            self.coc_pill.set_status("Détecté", Theme.ACCENT)
        else:
            self.coc_pill.set_status("Absent", Theme.WARNING)
        self.console_page.set_safeguard(self.controller.safeguard_enabled)

    def _on_hotkey(self, action: str) -> None:
        if action == "toggle":
            if self.controller.state == RuntimeState.RECORDING:
                self.controller.stop_recording()
            elif self.controller.state == RuntimeState.PLAYING:
                self.controller.stop_all()
            else:
                self.controller.start_playback()
        elif action == "play":
            self.controller.start_playback()
        elif action == "stop":
            self.controller.stop_all()

    def toggle_recording(self) -> None:
        if self.controller.state == RuntimeState.RECORDING:
            self.controller.stop_recording()
        else:
            self.controller.start_recording()

    def create_macro(self) -> None:
        name = TextInputDialog.get_text(self, "Nouvelle macro", "Nom de la macro", "Nouvelle macro")
        if name:
            self.controller.create_macro(name)

    def rename_macro(self) -> None:
        if not self.controller.current_macro_name:
            return
        name = TextInputDialog.get_text(self, "Renommer la macro", "Nouveau nom", self.controller.current_macro_name)
        if name:
            self.controller.rename_macro(name)

    def delete_macro(self) -> None:
        name = self.controller.current_macro_name
        if not name:
            return
        answer = QMessageBox.question(self, "Supprimer la macro", f"Supprimer « {name} » ?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.controller.delete_macro()

    def open_settings(self) -> None:
        status, color = self.telegram.get_status()
        dialog = SettingsDialog(
            self.controller.params,
            self._save_params,
            self.open_telegram,
            self.open_diagnostics,
            self._confirm_shutdown,
            status,
            color,
            self,
        )
        dialog.exec()

    def _save_params(self, params: dict[str, str]) -> None:
        self.controller.params.update(params)
        self.controller.refresh_coc_profile()
        self.controller.refresh_coc_presence()
        self.controller.save_params()
        self.controller.sync_safeguard_state()
        self._on_activity("Paramètres sauvegardés", "success")

    def open_telegram(self) -> None:
        dialog = TelegramDialog(self.controller.params, self.guide_path, self._save_telegram, self)
        dialog.exec()

    def _save_telegram(self, params: dict[str, str]) -> None:
        self.controller.params.update(params)
        self.controller.save_params()
        chat_id = None
        if params.get("telegram_chat_id", "").strip():
            chat_id = int(params["telegram_chat_id"].strip())
        self.telegram.configure(params.get("telegram_bot_token", ""), chat_id)
        if self.telegram.is_configured and not self.telegram.is_running:
            self.telegram.start()
        self.controller.poll_telegram_status()
        self._on_activity("Configuration Telegram mise à jour", "success")

    def open_diagnostics(self) -> None:
        pil_available = False
        mss_available = False
        try:
            import PIL  # noqa: F401
            pil_available = True
        except ImportError:
            pass
        try:
            import mss  # noqa: F401
            mss_available = True
        except ImportError:
            pass
        dialog = DiagnosticsDialog(
            app_version=self.app_version,
            python_version=self.python_version,
            telegram_status=self.telegram.get_status()[0],
            pil_available=pil_available,
            mss_available=mss_available,
            log_path=self.log_path,
            base_dir=self.base_dir,
            parent=self,
        )
        dialog.exec()

    def _confirm_shutdown(self) -> None:
        answer = QMessageBox.question(self, "Éteindre le PC", "Confirmer l’extinction du PC ?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.controller.request_shutdown()

    def _set_windows_app_id(self) -> None:
        if os.name == "nt":
            try:
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AutoCoc.OperatorConsole.v4")
            except Exception:
                pass

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._closing:
            event.accept()
            return
        if self.controller.is_busy:
            answer = QMessageBox.question(self, "Fermer AUTO-COC", "Une opération est en cours. Arrêter et fermer ?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self._closing = True
        self.metrics_timer.stop()
        self.telegram_timer.stop()
        clean_old_logs(self.log_path.parent, "app.log*", 24)
        self.controller.shutdown()
        event.accept()
