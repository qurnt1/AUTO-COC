# -*- coding: utf-8 -*-
"""Main PyQt6 window for AUTO-COC."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import keyboard
from PyQt6.QtCore import QTimer, pyqtSignal
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

from gui.components import NavButton
from gui.controller import RuntimeController, RuntimeState
from gui.dialogs import SettingsDialog, TelegramDialog, TextInputDialog
from gui.pages.console_page import ConsolePage
from gui.pages.diagnostics_page import DiagnosticsPage
from gui.pages.macros_page import MacrosPage
from gui.pages.telegram_page import TelegramPage
from gui.theme import Theme, build_stylesheet
from utils.logger import clean_old_logs, get_logger


class MainWindow(QMainWindow):
    hotkey_signal = pyqtSignal(str)

    def __init__(
        self,
        params: dict[str, str],
        telegram,
        macros_dir: Path,
        params_path: Path,
        log_path: Path,
        icon_path: Path,
        guide_path: Path,
        system_names: list[str],
    ):
        super().__init__()
        self.log = get_logger()
        self.log_path = log_path
        self.guide_path = guide_path
        self.telegram = telegram
        self.controller = RuntimeController(
            params=params,
            telegram=telegram,
            macros_dir=macros_dir,
            params_path=params_path,
            system_names=system_names,
            parent=self,
        )
        self._closing = False
        self._editing_system_role: str | None = None
        self._return_macro_name: str | None = None
        self.setWindowTitle("AUTO-COC")
        self.setMinimumSize(1000, 680)
        self.resize(1280, 800)
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
        sidebar.setFixedWidth(198)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(14, 16, 14, 14)
        sidebar_layout.setSpacing(6)

        brand = QVBoxLayout()
        brand.setContentsMargins(8, 4, 8, 18)
        brand_name = QLabel("AUTO-COC")
        brand_name.setStyleSheet(f"font-size: 20px; font-weight: 800; color: {Theme.TEXT};")
        brand.addWidget(brand_name)
        sidebar_layout.addLayout(brand)

        self.nav_buttons: list[NavButton] = []
        for label, tooltip in (("Home", "Run and monitor macros"), ("Macros", "Browse and edit macros"), ("Telegram", "Telegram settings and actions"), ("Diagnostics", "System log")):
            button = NavButton(label, tooltip)
            self.nav_buttons.append(button)
            sidebar_layout.addWidget(button)
        self.nav_buttons[0].setChecked(True)
        sidebar_layout.addStretch()

        self.settings_button = QPushButton("Settings")
        self.settings_button.setObjectName("QuietButton")
        self.settings_button.setMinimumHeight(40)
        sidebar_layout.addWidget(self.settings_button)
        shell.addWidget(sidebar)

        content = QVBoxLayout()
        content.setSpacing(0)

        self.pages = QStackedWidget()
        self.console_page = ConsolePage()
        self.macros_page = MacrosPage()
        self.telegram_page = TelegramPage()
        self.diagnostics_page = DiagnosticsPage(log_path=self.log_path)
        for page in (self.console_page, self.macros_page, self.telegram_page, self.diagnostics_page):
            self.pages.addWidget(page)
        content.addWidget(self.pages, 1)
        shell.addLayout(content, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _connect_signals(self) -> None:
        for index, button in enumerate(self.nav_buttons):
            button.clicked.connect(lambda checked, idx=index: self._navigate(idx))
        self.settings_button.clicked.connect(self.open_settings)

        self.console_page.macro_selected.connect(self.controller.select_macro)
        self.macros_page.macro_selected.connect(self.controller.select_macro)
        self.macros_page.create_requested.connect(self.create_macro)
        self.macros_page.rename_requested.connect(self.rename_macro)
        self.macros_page.delete_requested.connect(self.delete_macro)
        self.console_page.record_requested.connect(self.toggle_recording)
        self.console_page.run_requested.connect(self.toggle_playback)
        self.console_page.coc_requested.connect(self.controller.launch_coc)
        self.console_page.loop_toggled.connect(self.controller.set_auto_loop)
        self.console_page.safeguard_toggled.connect(self.controller.set_safeguard)
        self.telegram_page.configure_requested.connect(self.open_telegram)
        self.telegram_page.capture_requested.connect(self.controller.capture_screen)
        self.telegram_page.create_action_requested.connect(self.create_telegram_action)
        self.telegram_page.record_action_requested.connect(self.toggle_telegram_action_recording)
        self.telegram_page.rename_action_requested.connect(self.rename_telegram_action)
        self.telegram_page.remove_action_requested.connect(self.remove_telegram_action)

        signals = self.controller.signals
        signals.state_changed.connect(self._on_state_changed)
        signals.macros_changed.connect(self._on_macros_changed)
        signals.macro_selected.connect(self._on_macro_selected)
        signals.metrics_changed.connect(self._on_metrics_changed)
        signals.playback_options_changed.connect(self._on_playback_options)
        signals.activity.connect(self._on_activity)
        signals.error.connect(self._on_error)
        signals.telegram_status.connect(self._on_telegram_status)
        signals.coc_presence_changed.connect(self._on_coc_presence)
        signals.coc_launching_changed.connect(self.console_page.set_coc_launching)
        signals.safeguard_triggered.connect(self.controller.handle_safeguard_loss)
        self.hotkey_signal.connect(self._on_hotkey)

    def _register_shortcuts(self) -> None:
        self.shortcut_new = QShortcut(QKeySequence("Ctrl+N"), self)
        self.shortcut_new.activated.connect(self.create_macro)
        self.shortcut_search = QShortcut(QKeySequence("Ctrl+F"), self)
        self.shortcut_search.activated.connect(self._focus_macro_search)
        self.shortcut_stop = QShortcut(QKeySequence("Escape"), self)
        self.shortcut_stop.activated.connect(self.controller.stop_all)
        self.shortcut_play = QShortcut(QKeySequence("Ctrl+Return"), self)
        self.shortcut_play.activated.connect(self.toggle_playback)

    def _register_hotkeys(self) -> None:
        f1_registered = False
        try:
            keyboard.add_hotkey("f1", lambda: self.hotkey_signal.emit("toggle"))
            f1_registered = True
            keyboard.add_hotkey("ctrl+shift+1", lambda: self.hotkey_signal.emit("play"))
            keyboard.add_hotkey("ctrl+shift+0", lambda: self.hotkey_signal.emit("stop"))
            self._on_activity("Global hotkeys ready", "success")
        except Exception as exc:
            self._on_activity(f"Global hotkeys unavailable · {exc}", "warning")
        if not f1_registered:
            self.shortcut_f1 = QShortcut(QKeySequence("F1"), self)
            self.shortcut_f1.activated.connect(self.toggle_playback)

    def _navigate(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        self.nav_buttons[index].setChecked(True)
        if index == 3:
            self.diagnostics_page.refresh()

    def _on_macros_changed(self, items, selected: str) -> None:
        user_items = tuple(item for item in items if not item.is_telegram_action)
        user_names = {item.name for item in user_items}
        selected_user = selected if selected in user_names else (self._return_macro_name or "")
        self.console_page.set_macros(user_items, selected_user)
        self.macros_page.library.set_items(user_items, selected_user)
        self.telegram_page.set_action(next((item for item in items if item.role == "reload_coc"), None))

    def _on_macro_selected(self, macro) -> None:
        summary = next((item for item in self.controller.summaries() if item.name == getattr(macro, "name", "")), None)
        if summary and summary.is_telegram_action:
            return
        self.console_page.set_macro(macro)
        self.macros_page.set_macro(macro)

    def _on_state_changed(self, state_name: str, label: str) -> None:
        self.console_page.set_state(label, state_name)
        self.telegram_page.set_runtime_state(state_name, self.controller.current_macro_name)
        self.statusBar().showMessage(label)
        if state_name == "IDLE" and self._editing_system_role:
            QTimer.singleShot(0, self._finish_system_action_edit)

    def _on_metrics_changed(self, elapsed: float, events: int, duration: float, cycles: int) -> None:
        self.console_page.set_metrics(elapsed, events, duration, cycles, self.controller.state.name, self.controller.auto_loop)

    def _on_playback_options(self, loop: bool, safeguard: bool) -> None:
        self.console_page.set_loop(loop)
        self.console_page.set_safeguard(safeguard)

    def _on_activity(self, message: str, level: str) -> None:
        self.statusBar().showMessage(message, 5000)

    def _on_error(self, title: str, detail: str) -> None:
        message = f"{title}\n\n{detail}" if detail else title
        QMessageBox.warning(self, title, message)

    def _on_telegram_status(self, status: str, color: str) -> None:
        self.telegram_page.set_status(status, color)

    def _on_coc_presence(self, snapshot) -> None:
        self.console_page.set_coc_presence(snapshot)

    def _on_hotkey(self, action: str) -> None:
        if action == "toggle":
            self.toggle_playback()
        elif action == "play":
            self.controller.start_playback()
        elif action == "stop":
            self.controller.stop_all()

    def toggle_recording(self) -> None:
        if self.controller.state == RuntimeState.RECORDING:
            self.controller.stop_recording()
        else:
            self.controller.start_recording()

    def toggle_playback(self) -> None:
        if self.controller.state == RuntimeState.PLAYING:
            self.controller.stop_all()
        elif self.controller.state == RuntimeState.IDLE:
            self.controller.start_playback()

    def _focus_macro_search(self) -> None:
        self._navigate(1)
        self.macros_page.library.search.setFocus()

    def create_macro(self) -> None:
        name = TextInputDialog.get_text(self, "New macro", "Macro name", "New macro")
        if name:
            self.controller.create_macro(name)

    def rename_macro(self) -> None:
        if not self.controller.current_macro_name:
            return
        name = TextInputDialog.get_text(self, "Rename macro", "New name", self.controller.current_macro_name)
        if name:
            self.controller.rename_macro(name)

    def delete_macro(self) -> None:
        name = self.controller.current_macro_name
        if not name:
            return
        answer = QMessageBox.question(self, "Delete macro", f"Delete “{name}”?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.controller.delete_macro()

    def create_telegram_action(self) -> None:
        name = TextInputDialog.get_text(self, "Create Telegram routine", "Routine name", "Reload CoC")
        if not name:
            return
        self._begin_system_action_edit("reload_coc")
        if not self.controller.create_system_macro("reload_coc", name):
            self._finish_system_action_edit()
            return
        self._finish_system_action_edit()

    def toggle_telegram_action_recording(self) -> None:
        summary = self.controller.system_summary("reload_coc")
        if not summary:
            return
        if self.controller.state == RuntimeState.RECORDING and self.controller.current_macro_name == summary.name:
            self.controller.stop_recording()
            return
        if self.controller.state != RuntimeState.IDLE:
            return
        self._begin_system_action_edit("reload_coc")
        if not self.controller.select_system_macro("reload_coc") or not self.controller.start_recording():
            self._finish_system_action_edit()

    def rename_telegram_action(self) -> None:
        summary = self.controller.system_summary("reload_coc")
        if not summary or self.controller.state != RuntimeState.IDLE:
            return
        name = TextInputDialog.get_text(self, "Rename Telegram routine", "New name", summary.name)
        if not name:
            return
        self._begin_system_action_edit("reload_coc")
        if self.controller.select_system_macro("reload_coc"):
            self.controller.rename_macro(name)
        self._finish_system_action_edit()

    def remove_telegram_action(self) -> None:
        summary = self.controller.system_summary("reload_coc")
        if not summary or self.controller.state != RuntimeState.IDLE:
            return
        answer = QMessageBox.question(
            self,
            "Remove Telegram routine",
            f"Remove “{summary.name}” from Telegram and delete its recording?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._begin_system_action_edit("reload_coc")
        if self.controller.select_system_macro("reload_coc"):
            self.controller.delete_macro()
        self._finish_system_action_edit()

    def _begin_system_action_edit(self, role: str) -> None:
        self._editing_system_role = role
        current_name = self.controller.current_macro_name
        user_names = {item.name for item in self.controller.user_summaries()}
        self._return_macro_name = current_name if current_name in user_names else self._return_macro_name

    def _finish_system_action_edit(self) -> None:
        if not self._editing_system_role:
            return
        self._editing_system_role = None
        self.controller.refresh_macros()
        user_summaries = self.controller.user_summaries()
        user_names = {item.name for item in user_summaries}
        target = self._return_macro_name if self._return_macro_name in user_names else (user_summaries[0].name if user_summaries else None)
        self._return_macro_name = target
        if target:
            self.controller.select_macro(target)

    def open_settings(self) -> None:
        dialog = SettingsDialog(
            dict(self.controller.params),
            self._save_params,
            self._confirm_shutdown,
            self,
        )
        dialog.exec()

    def _save_params(self, params: dict[str, str]) -> bool:
        previous = dict(self.controller.params)
        self.controller.params.clear()
        self.controller.params.update(params)
        if not self.controller.save_params():
            self.controller.params.clear()
            self.controller.params.update(previous)
            self._on_error("Settings not saved", "The settings file is not writable.")
            return False
        self.controller.refresh_coc_profile()
        self.controller.refresh_coc_presence()
        self.controller.sync_safeguard_state()
        self.controller.sync_playback_options()
        self._on_activity("Settings saved", "success")
        return True

    def open_telegram(self) -> None:
        dialog = TelegramDialog(dict(self.controller.params), self.guide_path, self._save_telegram, self)
        dialog.exec()

    def _save_telegram(self, params: dict[str, str]) -> bool:
        previous = dict(self.controller.params)
        self.controller.params.clear()
        self.controller.params.update(params)
        if not self.controller.save_params():
            self.controller.params.clear()
            self.controller.params.update(previous)
            self._on_error("Telegram settings not saved", "The settings file is not writable.")
            return False
        chat_id = None
        if params.get("telegram_chat_id", "").strip():
            chat_id = int(params["telegram_chat_id"].strip())
        self.telegram.configure(params.get("telegram_bot_token", ""), chat_id)
        if self.telegram.is_configured and not self.telegram.is_running:
            self.telegram.start()
        self.controller.poll_telegram_status()
        self._on_activity("Telegram settings saved", "success")
        return True

    def _confirm_shutdown(self) -> None:
        answer = QMessageBox.question(self, "Shut down PC", "Shut down the computer?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.controller.request_shutdown()

    def _set_windows_app_id(self) -> None:
        if os.name == "nt":
            try:
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AutoCoc.OperatorConsole")
            except Exception:
                pass

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._closing:
            event.accept()
            return
        if self.controller.is_busy:
            answer = QMessageBox.question(self, "Close AUTO-COC", "An operation is running. Stop and close?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self._closing = True
        self.metrics_timer.stop()
        self.telegram_timer.stop()
        clean_old_logs(self.log_path.parent, "app.log*", 24)
        self.controller.shutdown()
        event.accept()
