# -*- coding: utf-8 -*-
"""
Macro COC v3.0 — GUI / App

Fenetre principale FluentWindow avec navigation sidebar.
"""

import os
import queue
import threading
import time
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional

import keyboard
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QApplication,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QFont

from qfluentwidgets import (
    FluentWindow, NavigationItemPosition,
    FluentIcon, PrimaryPushButton, PushButton, SwitchButton, LineEdit,
    StrongBodyLabel, BodyLabel, CaptionLabel, CardWidget, InfoBar, InfoBarPosition,
)

from gui.theme import Theme
from gui.widgets import StatCard
from gui.dialogs import (
    TextInputDialog, TelegramAutomationDialog, DiagnosticsDialog, show_toast,
)
from gui.pages.home_page import HomePage
from gui.pages.macros_page import MacrosPage
from models.macro import Macro, is_protected_macro
from services.recorder_service import Recorder, Player, trim_tail
from services.telegram_service import TelegramBotService, TelegramCommand
from utils.config import (
    read_params_csv, write_params_csv, read_macro_file, write_macro_file,
    read_macro_meta, list_macros, macro_path_from_name, sanitize_macro_name,
    as_bool, fmt_seconds, fmt_duration_for_list,
)
from utils.logger import get_logger, clean_old_logs
from utils.system import (
    grab_screenshot_png_bytes, resolve_exe_from_path,
    perform_shutdown,
)

# Optional imports
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False


class State(Enum):
    IDLE = auto()
    RECORDING = auto()
    PLAYING = auto()


class App(FluentWindow):
    """Application principale Macro COC avec navigation Fluent."""

    cycle_completed = pyqtSignal()
    player_stopped = pyqtSignal()

    def __init__(
        self,
        params: Dict[str, str],
        tg_service: TelegramBotService,
        base_dir: Path,
        config_dir: Path,
        macros_dir: Path,
        params_path: Path,
        log_path: Path,
        icon_path: Path,
        icon_png_path: Path,
        guide_html_path: Path,
        legacy_macro_path: Path,
        app_version: str,
        python_version: str,
        protected_macro_names: List[str],
    ):
        super().__init__()
        self._log = get_logger()
        self._log.info(f"=== Demarrage GUI Macro COC v{app_version} ===")
        print(f"[APP] Demarrage GUI Macro COC v{app_version}", flush=True)

        # Config
        self.params = params
        self.tg = tg_service
        self.BASE_DIR = base_dir
        self.CONFIG_DIR = config_dir
        self.MACROS_DIR = macros_dir
        self.PARAMS_PATH = params_path
        self.LOG_PATH = log_path
        self.ICON_PATH = icon_path
        self.ICON_PNG_PATH = icon_png_path
        self.GUIDE_HTML_PATH = guide_html_path
        self.LEGACY_MACRO = legacy_macro_path
        self.APP_VERSION = app_version
        self.PYTHON_VERSION = python_version
        self.PROTECTED_NAMES = protected_macro_names

        # Fenetre
        print("[APP] Configuration de la fenetre...", flush=True)
        self.setWindowTitle("Macro COC v3.0")
        self.resize(1280, 800)
        self.setMinimumSize(1024, 640)
        if self.ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(self.ICON_PATH)))
        print("[APP] Fenetre configuree.", flush=True)

        # Etat
        self.current_state = State.IDLE
        self.current_macro_name: Optional[str] = None
        self.current_macro_path: Optional[Path] = None
        self.all_macro_names: List[str] = []
        self._coc_launched_once = False
        self._coc_is_running_tg = False

        # Modele
        self.macro = Macro()
        self.rec = Recorder()
        self.player = Player(self.rec)
        print("[APP] Modele et services initialises.", flush=True)

        # Variables UI
        self.cycle_count = 0
        self.session_seconds = 0.0
        self.tg_status_text = "Initialisation..."
        self.tg_status_color = Theme.STATUS_WARN

        # Fenetres ouvertes
        self.open_windows: Dict[str, object] = {}

        # Timers
        self._play_timer: Optional[QTimer] = None
        self._last_tick: Optional[float] = None
        self._recording_timer: Optional[QTimer] = None
        self._recording_elapsed: float = 0.0

        # Connecter les signaux player (thread-safe)
        self.cycle_completed.connect(self._on_cycle_completed)
        self.player_stopped.connect(self._on_player_stopped)

        # Setup UI
        print("[APP] Construction de la navigation...", flush=True)
        try:
            self._setup_navigation()
            print("[APP] Navigation construite.", flush=True)
        except Exception as e:
            print(f"[APP] ERREUR navigation: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

        print("[APP] Enregistrement des hotkeys...", flush=True)
        self._register_hotkeys()

        # Polling Telegram
        self._tg_timer = QTimer(self)
        self._tg_timer.timeout.connect(self._poll_telegram_queue)
        self._tg_timer.start(100)
        print("[APP] Timer Telegram demarre.", flush=True)

        # Bootstrap
        print("[APP] Planification du bootstrap...", flush=True)
        QTimer.singleShot(200, self._bootstrap)

        # Hook pour les exceptions Qt non capturees
        self._setup_exception_hook()

    def _setup_exception_hook(self):
        """Installe un hook pour afficher les exceptions Qt dans une boite de dialogue."""
        import sys as _sys
        from PyQt6.QtWidgets import QMessageBox

        def _qt_excepthook(exc_type, exc_value, exc_tb):
            import traceback
            tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
            print(f"[APP] EXCEPTION QT: {tb_str}", flush=True)
            self._log.error(f"EXCEPTION QT: {tb_str}")
            QMessageBox.critical(
                self, "Erreur",
                f"{exc_type.__name__}: {exc_value}\n\n"
                f"Consultez config/app.log pour les details.\n\n"
                f"{tb_str[-500:]}"
            )

        _sys.excepthook = _qt_excepthook

    # =========================
    #     Navigation
    # =========================

    def _setup_navigation(self):
        """Configure les pages et la navigation."""
        self.home_page = HomePage(self)
        self.macros_page = MacrosPage(self)
        self.settings_widget = self._create_settings_page()

        self.addSubInterface(self.home_page, FluentIcon.HOME, "Accueil")
        self.addSubInterface(self.macros_page, FluentIcon.APPLICATION, "Mes Macros")
        self.addSubInterface(
            self.settings_widget, FluentIcon.SETTING,
            "Parametres", NavigationItemPosition.BOTTOM
        )

        # Demarrer sur la page macros
        self.switchTo(self.macros_page)

    def _create_settings_page(self) -> QWidget:
        """Cree la page de parametres."""
        page = QWidget()
        page.setObjectName("settingsPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        title = StrongBodyLabel("Parametres")
        title.setStyleSheet("font-size: 22px;")
        layout.addWidget(title)

        # --- General ---
        gen_card = CardWidget()
        gen_layout = QVBoxLayout(gen_card)
        gen_layout.setContentsMargins(20, 16, 20, 16)
        gen_layout.setSpacing(10)

        gen_title = BodyLabel("General")
        gen_title.setStyleSheet("font-weight: bold;")
        gen_layout.addWidget(gen_title)

        # Auto-loop
        loop_row = QHBoxLayout()
        loop_row.addWidget(BodyLabel("Lecture en boucle (auto_loop)"))
        loop_row.addStretch()
        self._settings_loop_switch = SwitchButton()
        self._settings_loop_switch.setChecked(as_bool(self.params.get("auto_loop", "0"), False))
        loop_row.addWidget(self._settings_loop_switch)
        gen_layout.addLayout(loop_row)

        # CoC path
        coc_row = QHBoxLayout()
        coc_row.addWidget(BodyLabel("Chemin CoC (.lnk ou .exe)"))
        gen_layout.addLayout(coc_row)
        self._settings_coc_path = LineEdit()
        self._settings_coc_path.setText(self.params.get("coc_path", ""))
        self._settings_coc_path.setClearButtonEnabled(True)
        gen_layout.addWidget(self._settings_coc_path)

        layout.addWidget(gen_card)

        # --- Telegram ---
        tg_card = CardWidget()
        tg_layout = QVBoxLayout(tg_card)
        tg_layout.setContentsMargins(20, 16, 20, 16)
        tg_layout.setSpacing(10)

        tg_title = BodyLabel("Telegram")
        tg_title.setStyleSheet("font-weight: bold;")
        tg_layout.addWidget(tg_title)

        self._settings_tg_status = BodyLabel(f"Statut: {self.tg_status_text}")
        self._settings_tg_status.setStyleSheet(f"color: {self.tg_status_color};")
        tg_layout.addWidget(self._settings_tg_status)

        tg_btns = QHBoxLayout()
        config_btn = PushButton(FluentIcon.SEND, "  Configurer Telegram")
        config_btn.clicked.connect(self._open_tg_config)
        tg_btns.addWidget(config_btn)
        tg_btns.addStretch()
        tg_layout.addLayout(tg_btns)

        layout.addWidget(tg_card)

        # --- Systeme ---
        sys_card = CardWidget()
        sys_layout = QVBoxLayout(sys_card)
        sys_layout.setContentsMargins(20, 16, 20, 16)
        sys_layout.setSpacing(10)

        sys_title = BodyLabel("Systeme")
        sys_title.setStyleSheet("font-weight: bold;")
        sys_layout.addWidget(sys_title)

        sys_btns = QHBoxLayout()
        diag_btn = PushButton(FluentIcon.INFO, "  Diagnostic")
        diag_btn.clicked.connect(self._open_diagnostics)
        sys_btns.addWidget(diag_btn)

        shutdown_btn = PushButton(FluentIcon.POWER_BUTTON, "  Eteindre le PC")
        shutdown_btn.clicked.connect(self._request_shutdown)
        sys_btns.addWidget(shutdown_btn)
        sys_btns.addStretch()
        sys_layout.addLayout(sys_btns)

        layout.addWidget(sys_card)

        # Save button
        save_row = QHBoxLayout()
        save_row.addStretch()
        save_btn = PrimaryPushButton(FluentIcon.SAVE, "  Enregistrer les parametres")
        save_btn.clicked.connect(self._save_settings)
        save_row.addWidget(save_btn)
        layout.addLayout(save_row)

        layout.addStretch()
        return page

    # =========================
    #     Bootstrap
    # =========================

    def _bootstrap(self):
        """Initialisation post-UI."""
        print("[APP] Bootstrap demarre...", flush=True)
        try:
            print("[APP] Bootstrap: _ensure_protected_macros...", flush=True)
            self._ensure_protected_macros()
            print("[APP] Bootstrap: Macros protegees OK.", flush=True)

            print("[APP] Bootstrap: _refresh_macro_sidebar...", flush=True)
            self._refresh_macro_sidebar()
            print(f"[APP] Bootstrap: Sidebar OK ({len(self.all_macro_names)} macros).", flush=True)

            wanted = self.params.get("last_macro", "").strip()
            print(f"[APP] Bootstrap: last_macro='{wanted}'", flush=True)
            target = wanted if wanted in self.all_macro_names else (
                self.all_macro_names[0] if self.all_macro_names else None
            )
            print(f"[APP] Bootstrap: target='{target}'", flush=True)

            if target:
                print(f"[APP] Bootstrap: select({target})...", flush=True)
                self.macros_page.macro_list.select(target)
                print(f"[APP] Bootstrap: select OK.", flush=True)

            print(f"[APP] Bootstrap: tg.is_ready={self.tg.is_ready}, tg.is_running={self.tg.is_running}", flush=True)
            if self.tg.is_ready and self.tg.is_running:
                print("[APP] Bootstrap: tg.send_message...", flush=True)
                self.tg.send_message(f"Macro COC v{self.APP_VERSION} lancee.")
                print("[APP] Bootstrap: tg.send_message OK.", flush=True)
                print("[APP] Bootstrap: tg.replace_controls...", flush=True)
                self.tg.replace_controls(self._tg_status_text("Pret"), coc_launched=self._coc_is_running_tg)
                print("[APP] Bootstrap: tg.replace_controls OK.", flush=True)
            else:
                print(f"[APP] Bootstrap: Telegram pas pret (ready={self.tg.is_ready}, running={self.tg.is_running}), skip.", flush=True)

            print("[APP] Bootstrap: home_page.refresh...", flush=True)
            self.home_page.refresh()
            print("[APP] Bootstrap: home_page.refresh OK.", flush=True)
            print("[APP] Bootstrap termine.", flush=True)
        except Exception as e:
            print(f"[APP] ERREUR bootstrap: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

    # =========================
    #     Public API (pour les pages)
    # =========================

    def on_play(self, notify_tg: bool = True):
        if self.current_state != State.IDLE:
            return
        steps = [s.to_dict() for s in self.macro.steps]
        if not steps:
            show_toast(self, "Attention", "Aucune macro a lire.", "warning")
            return

        self.current_state = State.PLAYING
        self._update_all_ui()
        self.cycle_count = 0
        self.session_seconds = 0.0
        self.macros_page.update_session_stats(0, 0.0)

        self.player.on_cycle = self.cycle_completed.emit
        self.player.on_stopped = self.player_stopped.emit
        self.player.play(steps, loop=self._auto_loop())

        self._start_timer()
        self.macros_page._status_label.setText("Lecture...")
        self.macros_page._status_progress.setValue(0)
        self.macros_page._status_progress.show()

        if notify_tg:
            self.tg.replace_controls(self._tg_status_text("Lecture"), coc_launched=self._coc_is_running_tg)

    def toggle_record(self):
        if self.current_state == State.RECORDING:
            # Stop recording
            self.rec.stop()
            self._recording_elapsed = 0.0
            if self._recording_timer:
                self._recording_timer.stop()
                self._recording_timer = None
            self.macros_page._recording_indicator.stop()

            steps = trim_tail(self.rec.get_steps_as_dicts(), 3.0)
            self.macro.set_steps_from_dicts(steps)
            write_macro_file(self.current_macro_path, self.current_macro_name, steps)
            self._update_macro_display()
            self.current_state = State.IDLE
            self._update_all_ui()
            self.macros_page._status_label.setText("Pret")
            show_toast(self, "Succes", f"Macro '{self.current_macro_name}' sauvegardee.", "success")

        elif self.current_state == State.IDLE and self.current_macro_path:
            self.current_state = State.RECORDING
            self._update_all_ui()
            self.macro.clear()
            self.rec.start()
            self.macros_page._status_label.setText("Enregistrement...")

            # Demarrer l'indicateur
            self.macros_page._recording_indicator.start()
            self._recording_elapsed = 0.0
            self._recording_timer = QTimer(self)
            self._recording_timer.timeout.connect(self._tick_recording)
            self._recording_timer.start(200)

    def force_stop_all(self, notify_tg: bool = True):
        if self.current_state == State.RECORDING:
            self.rec.stop()
            self.macros_page._recording_indicator.stop()
            if self._recording_timer:
                self._recording_timer.stop()
                self._recording_timer = None
        if self.current_state == State.PLAYING:
            self.player.stop()
            self.macros_page._status_progress.hide()
        self._stop_timer()
        self.current_state = State.IDLE
        self._update_all_ui()
        self.macros_page._status_label.setText("Pret")
        if notify_tg:
            self.tg.replace_controls(self._tg_status_text("Arrete"), coc_launched=self._coc_is_running_tg)

    def macro_new(self):
        dlg = TextInputDialog(self, "Nouvelle macro", "Nom:", "Nouvelle Macro")
        name = dlg.show()
        if not name or is_protected_macro(name):
            return
        name = sanitize_macro_name(name)
        path = macro_path_from_name(self.MACROS_DIR, name)
        write_macro_file(path, name, [])
        self._refresh_macro_sidebar(keep_selection=False)
        self.macros_page.macro_list.select(name, fire=True)
        show_toast(self, "Creee", f"Macro '{name}' creee.", "success")

    def macro_rename(self):
        if not self.current_macro_name or is_protected_macro(self.current_macro_name):
            return
        dlg = TextInputDialog(self, "Renommer", "Nouveau nom:", self.current_macro_name)
        new_name = dlg.show()
        if not new_name or is_protected_macro(new_name):
            return
        new_name = sanitize_macro_name(new_name)
        new_path = macro_path_from_name(self.MACROS_DIR, new_name)
        self.macro.name = new_name
        write_macro_file(new_path, new_name, [s.to_dict() for s in self.macro.steps])
        if self.current_macro_path and self.current_macro_path != new_path:
            try:
                self.current_macro_path.unlink()
            except Exception:
                pass
        self.current_macro_name = new_name
        self.current_macro_path = new_path
        self._refresh_macro_sidebar(keep_selection=False)
        self.macros_page.macro_list.select(new_name, fire=False)
        show_toast(self, "Renommee", f"Macro renommee en '{new_name}'.", "success")

    def macro_delete(self):
        if not self.current_macro_path or is_protected_macro(self.current_macro_name):
            return
        from qfluentwidgets import MessageBox
        result = MessageBox(
            "Supprimer",
            f"Supprimer << {self.current_macro_name} >> ?",
            self
        ).exec()
        if not result:
            return
        try:
            self.current_macro_path.unlink()
        except Exception:
            pass
        name = self.current_macro_name
        self.current_macro_name = None
        self.current_macro_path = None
        self.macro = Macro()
        self._refresh_macro_sidebar(keep_selection=False)
        self.macros_page.clear_details()
        show_toast(self, "Supprimee", f"Macro '{name}' supprimee.", "warning")

    def launch_coc_once(self):
        path = self.params.get("coc_path", "").strip()
        if not path:
            show_toast(self, "Info", "Aucun chemin CoC configure.", "info")
            return
        if self._coc_launched_once:
            show_toast(self, "Info", "CoC deja lance.", "info")
            return
        try:
            os.startfile(path) if os.name == 'nt' else None
            self._coc_launched_once = True
            self._coc_is_running_tg = True
            self.tg.replace_controls(self._tg_status_text("CoC lance"), coc_launched=True)
            show_toast(self, "Lance", "Clash of Clans lance.", "success")
        except Exception as e:
            self._log.error(f"Launch error: {e}")
            show_toast(self, "Erreur", f"Impossible de lancer CoC: {e}", "error")

    def send_capture_to_tg(self):
        threading.Thread(target=self._send_capture, daemon=True).start()

    # =========================
    #     Methodes internes
    # =========================

    def _select_macro_by_name(self, name: str):
        if self.current_state != State.IDLE:
            return
        path = self.LEGACY_MACRO if name == self.LEGACY_MACRO.stem else macro_path_from_name(self.MACROS_DIR, name)
        if not path.exists():
            return
        n, steps, sha1, _ = read_macro_file(path)
        self.macro = Macro(name=n)
        self.macro.set_steps_from_dicts(steps)
        self.macro.sha1 = sha1
        self.current_macro_name = name
        self.current_macro_path = path
        self._update_macro_display()
        self.params["last_macro"] = name
        self._save_params()
        self.macros_page._status_macro_label.setText(f"Macro: {name}")

    def _update_macro_display(self):
        n = self.macro.event_count()
        d = self.macro.duration()
        self.macros_page.update_macro_info(self.current_macro_name or "—", n, d)
        self.macros_page.update_timeline([s.to_dict() for s in self.macro.steps])
        self.macros_page._status_macro_label.setText(f"Macro: {self.current_macro_name or '—'}")

    def _refresh_macro_sidebar(self, keep_selection: bool = True):
        items = list_macros(self.MACROS_DIR, self.LEGACY_MACRO, self.PROTECTED_NAMES)
        self.all_macro_names = [n for n, _ in items]
        meta = {n: read_macro_meta(p) for n, p in items}
        self.macros_page.macro_list.set_meta(meta)
        sel = self.current_macro_name if keep_selection else None
        term = self.macros_page.search.text() if keep_selection else ""
        self.macros_page.macro_list.refresh(self.all_macro_names, selected=sel, filter_term=term)
        self.home_page.refresh()

    def _update_all_ui(self):
        self.macros_page.update_ui_state()
        self._update_settings_page()

    def _update_settings_page(self):
        """Rafraichit la page de parametres."""
        self._settings_tg_status.setText(f"Statut: {self.tg_status_text}")
        self._settings_tg_status.setStyleSheet(f"color: {self.tg_status_color};")

    def _save_settings(self):
        self.params["auto_loop"] = "1" if self._settings_loop_switch.isChecked() else "0"
        self.params["coc_path"] = self._settings_coc_path.text().strip()
        self._save_params()
        show_toast(self, "Sauvegarde", "Parametres sauvegardes.", "success")

    def _ensure_protected_macros(self):
        for name in self.PROTECTED_NAMES:
            path = macro_path_from_name(self.MACROS_DIR, name)
            if not path.exists():
                write_macro_file(path, name, [])

    def _play_protected_macro(self, name: str):
        path = macro_path_from_name(self.MACROS_DIR, name)
        if not path.exists():
            return
        _, steps, _, _ = read_macro_file(path)
        if steps and self.current_state == State.IDLE:
            self.current_state = State.PLAYING
            self.player.play(steps, loop=False)
            self.tg.replace_controls(self._tg_status_text(f"Lecture {name}"), coc_launched=self._coc_is_running_tg)

    def _save_params(self):
        write_params_csv(self.PARAMS_PATH, self.params)

    def _auto_loop(self) -> bool:
        return as_bool(self.params.get("auto_loop", "0"))

    def _tg_status_text(self, prefix: str) -> str:
        loop = "ON" if self._auto_loop() else "OFF"
        return f"{prefix} | Macro: {self.current_macro_name or '—'} | Loop: {loop}"

    def _send_capture(self):
        png = grab_screenshot_png_bytes()
        if png:
            self.tg.send_photo(png, caption=f"📸 {self.current_macro_name or 'Capture'}")

    # =========================
    #     Timer callbacks
    # =========================

    def _tick_recording(self):
        """Timer pendant l'enregistrement."""
        self._recording_elapsed += 0.2
        self.macros_page._recording_indicator.tick(self._recording_elapsed)

    def _start_timer(self):
        self._last_tick = time.perf_counter()
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._tick_timer)
        self._play_timer.start(200)

    def _stop_timer(self):
        if self._play_timer:
            self._play_timer.stop()
            self._play_timer = None

    def _tick_timer(self):
        now = time.perf_counter()
        self.session_seconds += now - (self._last_tick or now)
        self._last_tick = now
        self.macros_page.update_session_stats(self.cycle_count, self.session_seconds)

    def _on_cycle_completed(self):
        self.cycle_count += 1
        self.macros_page.update_session_stats(self.cycle_count, self.session_seconds)

    def _on_player_stopped(self):
        self._stop_timer()
        self.macros_page._status_progress.hide()
        self.current_state = State.IDLE
        self._update_all_ui()
        self.macros_page._status_label.setText("Pret")

    # =========================
    #     Telegram
    # =========================

    def _poll_telegram_queue(self):
        try:
            while not self.tg.command_queue.empty():
                cmd: TelegramCommand = self.tg.command_queue.get_nowait()
                self._handle_tg_command(cmd.command, cmd.meta)
        except Exception as e:
            self._log.error(f"Erreur polling TG queue: {e}")

    def _handle_tg_command(self, cmd: str, meta: dict):
        if cmd == "STOP":
            self.force_stop_all(notify_tg=False)
            self.tg.replace_controls(self._tg_status_text("Arrete"), coc_launched=self._coc_is_running_tg)
        elif cmd == "GO":
            self.on_play(notify_tg=False)
            self.tg.replace_controls(self._tg_status_text("Lecture"), coc_launched=self._coc_is_running_tg)
        elif cmd == "MENU":
            self.tg.replace_menu("Parametres", loop_state=self._auto_loop())
        elif cmd == "BACK":
            if self.tg.get_last_menu_id():
                self.tg.delete_message(self.tg.get_last_menu_id())
                self.tg.clear_last_menu_id()
            self.tg.replace_controls(self._tg_status_text("Pret"), coc_launched=self._coc_is_running_tg)
        elif cmd == "CAPTURE":
            threading.Thread(target=self._send_capture, daemon=True).start()
        elif cmd == "TOGGLE_LOOP":
            self.params["auto_loop"] = "0" if self._auto_loop() else "1"
            self._save_params()
            self.tg.send_message(f"Loop {'ON' if self._auto_loop() else 'OFF'}.")
            self.tg.replace_menu("Parametres", loop_state=self._auto_loop())
        elif cmd == "SHUTDOWN_ASK":
            self.tg.push_shutdown_confirm()
        elif cmd == "SHUTDOWN_CONFIRM":
            self.tg.send_message("Extinction...")
            threading.Thread(target=perform_shutdown, daemon=True).start()
        elif cmd == "SHUTDOWN_CANCEL":
            self.tg.send_message("Extinction annulee.")
            self.tg.replace_controls(self._tg_status_text("Pret"), coc_launched=self._coc_is_running_tg)
        elif cmd == "LAUNCH_COC":
            self.launch_coc_once()
        elif cmd.startswith("SELECT_MACRO:"):
            name = cmd[13:]
            if name in self.all_macro_names:
                self.macros_page.macro_list.select(name, fire=True)
                self.tg.replace_controls(self._tg_status_text(f"Macro: {name}"), coc_launched=self._coc_is_running_tg)
        elif cmd == "SELECT_MACRO_LIST":
            self.tg.push_macro_selection(self.all_macro_names)
        elif cmd == "RELOAD_COC":
            self._play_protected_macro("Recharger COC")
        elif cmd == "VALIDATE_ARRIVAL":
            self._play_protected_macro("Valider arrivee")

    def _update_tg_status(self):
        status, color = self.tg.get_status()
        self.tg_status_text = status
        self.tg_status_color = color

    # =========================
    #     Settings actions
    # =========================

    def _open_tg_config(self):
        def on_save(p):
            self.params.update(p)
            self._save_params()

        def on_close(name):
            if name in self.open_windows:
                self.open_windows.pop(name, None)

        win = TelegramAutomationDialog(
            self, self.params, on_save, self.GUIDE_HTML_PATH,
            icon_path=self.ICON_PATH, name="tg_config", on_close_cb=on_close,
        )
        self.open_windows["tg_config"] = win
        win.show()

    def _open_diagnostics(self):
        def on_close(name):
            if name in self.open_windows:
                self.open_windows.pop(name, None)

        win = DiagnosticsDialog(
            self, self.tg_status_text, self.APP_VERSION, self.PYTHON_VERSION,
            PIL_AVAILABLE, MSS_AVAILABLE, self.LOG_PATH, self.BASE_DIR,
            icon_path=self.ICON_PATH, name="diagnostics", on_close_cb=on_close,
        )
        self.open_windows["diagnostics"] = win
        win.show()

    def _request_shutdown(self):
        from qfluentwidgets import MessageBox
        if not MessageBox("Extinction", "Eteindre le PC ?", self).exec():
            return
        if not MessageBox("Confirmer", "Confirmer l'extinction ?", self).exec():
            return
        threading.Thread(target=perform_shutdown, daemon=True).start()

    # =========================
    #     Hotkeys
    # =========================

    def _register_hotkeys(self):
        try:
            keyboard.add_hotkey("f1", self._hk_toggle)
            keyboard.add_hotkey("ctrl+shift+1", lambda: self.on_play(notify_tg=True))
            keyboard.add_hotkey("ctrl+shift+0", lambda: self.force_stop_all(notify_tg=True))
        except Exception as e:
            self._log.error(f"Hotkeys error: {e}")

    def _hk_toggle(self):
        if self.current_state == State.PLAYING:
            self.force_stop_all(notify_tg=True)
        elif self.current_state == State.IDLE:
            self.on_play(notify_tg=True)

    # =========================
    #     Close
    # =========================

    def closeEvent(self, event):
        self._log.info("Fermeture...")
        self.force_stop_all(notify_tg=False)
        if self.tg.is_ready:
            self.tg.send_message("Application fermee.")
        try:
            keyboard.remove_all_hotkeys()
        except Exception:
            pass
        clean_old_logs(self.LOG_PATH.parent, "app.log*", 24)
        event.accept()
