# -*- coding: utf-8 -*-
"""
Macro COC v3.0 — GUI / App

Fenêtre principale de l'application.
"""

import ctypes
import os
import queue
import shutil
import threading
import time
from enum import Enum, auto
from pathlib import Path
from tkinter import messagebox, filedialog
from typing import Dict, List, Optional

import customtkinter as ctk
import keyboard

from gui.theme import Theme
from gui.components import MacroList
from gui.dialogs import (
    TextInputDialog, SettingsDialog, TelegramAutomationDialog, DiagnosticsDialog
)
from models.macro import Macro, is_protected_macro
from services.recorder_service import Recorder, Player, trim_tail
from services.telegram_service import TelegramBotService, TelegramCommand
from utils.config import (
    read_params_csv, write_params_csv, read_macro_file, write_macro_file,
    read_macro_meta, list_macros, macro_path_from_name, sanitize_macro_name,
    as_bool, fmt_seconds, fmt_duration_for_list
)
from utils.logger import get_logger, clean_old_logs
from utils.system import (
    grab_screenshot_png_bytes, resolve_exe_from_path, kill_process_by_name,
    is_process_running, perform_shutdown, ensure_directories
)

# Pillow check
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


class App(ctk.CTk):
    """Application principale Macro COC."""

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
        self._log.info(f"=== Démarrage GUI Macro COC v{app_version} ===")
        
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
        
        # CustomTkinter setup
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        # Windows App ID
        if os.name == 'nt':
            try:
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MacroCOC.App.v3")
            except Exception:
                pass
        
        # Fenêtre
        self.title("Macro COC v3.0")
        self.geometry("1200x720")
        self.minsize(1024, 640)
        self._apply_icon()
        
        # État
        self.current_state = State.IDLE
        self.current_macro_name: Optional[str] = None
        self.current_macro_path: Optional[Path] = None
        self.all_macro_names: List[str] = []
        self._coc_launched_once = False
        self._coc_is_running_tg = False
        
        # Modèle
        self.macro = Macro()
        self.rec = Recorder()
        self.player = Player(self.rec)
        
        # Variables UI
        self.status_var = ctk.StringVar(value="Prêt")
        self.cycle_count = ctk.IntVar(value=0)
        self.session_seconds = 0.0
        self.session_elapsed = ctk.StringVar(value="00:00")
        self.macro_duration = ctk.StringVar(value="00:00")
        self.macro_events = ctk.StringVar(value="0")
        self.tg_status_var = ctk.StringVar(value="Initialisation...")
        self.tg_status_color_var = ctk.StringVar(value=Theme.STATUS_WARN)
        
        # Windows ouvertes
        self.open_windows: Dict[str, ctk.CTkToplevel] = {}
        
        # Timer
        self._play_timer_running = False
        self._last_tick: Optional[float] = None
        
        # Build UI
        self._build_ui()
        self._register_hotkeys()
        self.protocol("WM_DELETE_WINDOW", self.safe_quit)
        
        # Bootstrap
        self.after(150, self._bootstrap)
        
        # Polling queue Telegram
        self.after(100, self._poll_telegram_queue)
    
    def _apply_icon(self):
        if self.ICON_PATH.exists():
            try:
                self.iconbitmap(str(self.ICON_PATH))
            except Exception:
                pass

    def _build_ui(self):
        """Construit l'interface utilisateur."""
        root = ctk.CTkFrame(self, corner_radius=0, fg_color=Theme.APP_BG)
        root.pack(fill="both", expand=True)
        
        # Header
        header = ctk.CTkFrame(root, corner_radius=0, fg_color=Theme.HEADER_BG, height=Theme.HEADER_HEIGHT)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        self._title_lbl = ctk.CTkLabel(header, text="Macro COC v3.0", font=ctk.CTkFont(size=24, weight="bold"))
        self._title_lbl.pack(side="left", padx=20)
        
        ctk.CTkButton(header, text="Paramètres", width=120, command=self.open_settings).pack(side="right", padx=20)
        
        # Body
        body = ctk.CTkFrame(root, fg_color=Theme.APP_BG)
        body.pack(fill="both", expand=True, padx=12, pady=12)
        body.grid_columnconfigure(0, weight=0)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)
        
        # Left panel
        left = ctk.CTkFrame(body, corner_radius=12, fg_color=Theme.LEFT_CONTAINER_BG, width=320)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        left.pack_propagate(False)
        
        ctk.CTkLabel(left, text="Macros", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=12, pady=(12, 4))
        
        self.search_entry = ctk.CTkEntry(left, placeholder_text="Rechercher...")
        self.search_entry.pack(fill="x", padx=8, pady=(0, 8))
        self.search_entry.bind("<KeyRelease>", self._filter_macro_list)
        
        self.macro_list = MacroList(left, on_select=self._select_macro_by_name, on_rclick=self._show_macro_menu)
        self.macro_list.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        
        # Actions
        actions = ctk.CTkFrame(left, fg_color=Theme.LEFT_ACTIONS_BG)
        actions.pack(fill="x", padx=8, pady=(0, 12))
        
        self.btn_new = ctk.CTkButton(actions, text="Nouveau", width=80, command=self.macro_new,
                                     fg_color=Theme.BTN_PRIMARY_BG, hover_color=Theme.BTN_PRIMARY_HOVER)
        self.btn_new.pack(side="left", padx=4, pady=8)
        
        self.btn_rename = ctk.CTkButton(actions, text="Renommer", width=90, command=self.macro_rename)
        self.btn_rename.pack(side="left", padx=4, pady=8)
        
        self.btn_delete = ctk.CTkButton(actions, text="Supprimer", width=90, command=self.macro_delete,
                                        fg_color=Theme.BTN_STOP_BG, hover_color=Theme.BTN_STOP_HOVER)
        self.btn_delete.pack(side="left", padx=4, pady=8)
        
        # Center panel
        main = ctk.CTkFrame(body, corner_radius=12, fg_color=Theme.CENTER_BG)
        main.grid(row=0, column=1, sticky="nsew")
        
        # Info bar
        info = ctk.CTkFrame(main, fg_color=Theme.INFO_BG)
        info.pack(fill="x", padx=16, pady=(16, 8))
        
        self.lbl_macro_name = ctk.CTkLabel(info, text="Macro : —", font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_macro_name.pack(side="left", padx=6, pady=10)
        
        self.lbl_macro_meta = ctk.CTkLabel(info, text="Non enregistrée", text_color=Theme.INFO_TEXT_MUTED)
        self.lbl_macro_meta.pack(side="right", padx=6)
        
        # Controls
        controls = ctk.CTkFrame(main, fg_color="transparent")
        controls.pack(fill="x", padx=16, pady=(6, 10))
        
        self.btn_rec = ctk.CTkButton(controls, text="Enregistrer", height=44, width=200,
                                     fg_color=Theme.BTN_PRIMARY_BG, hover_color=Theme.BTN_PRIMARY_HOVER,
                                     command=self.toggle_record)
        self.btn_rec.pack(side="left", padx=6)
        
        self.btn_play = ctk.CTkButton(controls, text="Lire", height=44, width=170,
                                      fg_color=Theme.BTN_PRIMARY_BG, hover_color=Theme.BTN_PRIMARY_HOVER,
                                      command=lambda: self.on_play(notify_tg=True))
        self.btn_play.pack(side="left", padx=6)
        
        self.btn_stop = ctk.CTkButton(controls, text="Stopper", height=44, width=210,
                                      fg_color=Theme.BTN_STOP_BG, hover_color=Theme.BTN_STOP_HOVER,
                                      command=lambda: self.force_stop_all(notify_tg=True))
        self.btn_stop.pack(side="left", padx=6)
        
        # Stats
        stats = ctk.CTkFrame(main, corner_radius=12, fg_color=Theme.CARD_BG)
        stats.pack(fill="x", padx=16, pady=(6, 12))
        
        for label, var in [("Durée macro", self.macro_duration), ("Événements", self.macro_events),
                           ("Cycles", self.cycle_count), ("Temps total", self.session_elapsed)]:
            f = ctk.CTkFrame(stats, fg_color="transparent")
            f.pack(fill="x", padx=12, pady=4)
            ctk.CTkLabel(f, text=label).pack(side="left")
            ctk.CTkLabel(f, textvariable=var, text_color="#86efac").pack(side="right")
        
        # Launch CoC
        ctk.CTkButton(main, text="Lancer CoC", height=44, width=180,
                      fg_color=Theme.BTN_LAUNCH_BG, hover_color=Theme.BTN_LAUNCH_HOVER,
                      command=self.launch_coc_once).pack(padx=16, pady=(0, 12), anchor="w")
        
        # Footer
        footer = ctk.CTkFrame(root, fg_color=Theme.STATUS_BG, height=28)
        footer.pack(fill="x", side="bottom")
        ctk.CTkLabel(footer, textvariable=self.status_var, anchor="w").pack(side="left", padx=10)

    def _bootstrap(self):
        """Initialisation post-UI."""
        self._ensure_protected_macros()
        self._refresh_macro_sidebar()
        self._update_tg_status()
        
        # Sélectionner la dernière macro
        wanted = self.params.get("last_macro", "").strip()
        target = wanted if wanted in self.all_macro_names else (self.all_macro_names[0] if self.all_macro_names else None)
        
        if target:
            self.macro_list.select(target)
        
        # Message TG
        if self.tg.is_ready:
            self.tg.send_message(f"Macro COC v{self.APP_VERSION} lancée.")
            self.tg.replace_controls(self._tg_status_text("Prêt"), coc_launched=self._coc_is_running_tg)

    def _poll_telegram_queue(self):
        """Polling de la queue Telegram."""
        try:
            while not self.tg.command_queue.empty():
                cmd: TelegramCommand = self.tg.command_queue.get_nowait()
                self._handle_tg_command(cmd.command, cmd.meta)
        except Exception as e:
            self._log.error(f"Erreur polling TG queue: {e}")
        self.after(100, self._poll_telegram_queue)

    def _handle_tg_command(self, cmd: str, meta: dict):
        """Gère une commande Telegram."""
        if cmd == "STOP":
            self.force_stop_all(notify_tg=False)
            self.tg.replace_controls(self._tg_status_text("Arrêté"), coc_launched=self._coc_is_running_tg)
        elif cmd == "GO":
            self.on_play(notify_tg=False)
            self.tg.replace_controls(self._tg_status_text("Lecture"), coc_launched=self._coc_is_running_tg)
        elif cmd == "MENU":
            self.tg.replace_menu("Paramètres", loop_state=self._auto_loop())
        elif cmd == "BACK":
            if self.tg.get_last_menu_id():
                self.tg.delete_message(self.tg.get_last_menu_id())
                self.tg.clear_last_menu_id()
            self.tg.replace_controls(self._tg_status_text("Prêt"), coc_launched=self._coc_is_running_tg)
        elif cmd == "CAPTURE":
            threading.Thread(target=self._send_capture, daemon=True).start()
        elif cmd == "TOGGLE_LOOP":
            self.params["auto_loop"] = "0" if self._auto_loop() else "1"
            self._save_params()
            self.tg.send_message(f"Loop {'ON' if self._auto_loop() else 'OFF'}.")
            self.tg.replace_menu("Paramètres", loop_state=self._auto_loop())
        elif cmd == "SHUTDOWN_ASK":
            self.tg.push_shutdown_confirm()
        elif cmd == "SHUTDOWN_CONFIRM":
            self.tg.send_message("🛑 Extinction...")
            threading.Thread(target=perform_shutdown, daemon=True).start()
        elif cmd == "SHUTDOWN_CANCEL":
            self.tg.send_message("Extinction annulée.")
            self.tg.replace_controls(self._tg_status_text("Prêt"), coc_launched=self._coc_is_running_tg)
        elif cmd == "LAUNCH_COC":
            self.launch_coc_once()
        elif cmd.startswith("SELECT_MACRO:"):
            name = cmd[13:]
            if name in self.all_macro_names:
                self.macro_list.select(name, fire=True)
                self.tg.replace_controls(self._tg_status_text(f"Macro: {name}"), coc_launched=self._coc_is_running_tg)
        elif cmd == "SELECT_MACRO_LIST":
            self.tg.push_macro_selection(self.all_macro_names)
        elif cmd == "RELOAD_COC":
            self._play_protected_macro("Recharger COC")
        elif cmd == "VALIDATE_ARRIVAL":
            self._play_protected_macro("Valider arrivée")

    def _tg_status_text(self, prefix: str) -> str:
        loop = "ON" if self._auto_loop() else "OFF"
        return f"{prefix} | Macro: {self.current_macro_name or '—'} | Loop: {loop}"

    def _auto_loop(self) -> bool:
        return as_bool(self.params.get("auto_loop", "0"))

    def _update_tg_status(self):
        status, color = self.tg.get_status()
        self.tg_status_var.set(status)
        self.tg_status_color_var.set(color)

    def _send_capture(self):
        png = grab_screenshot_png_bytes()
        if png:
            self.tg.send_photo(png, caption=f"📸 {self.current_macro_name or 'Capture'}")

    def _refresh_macro_sidebar(self, keep_selection: bool = True):
        items = list_macros(self.MACROS_DIR, self.LEGACY_MACRO, self.PROTECTED_NAMES)
        self.all_macro_names = [n for n, _ in items]
        meta = {n: read_macro_meta(p) for n, p in items}
        self.macro_list.set_meta(meta)
        sel = self.current_macro_name if keep_selection else None
        term = self.search_entry.get() if keep_selection else ""
        self.macro_list.refresh(self.all_macro_names, selected=sel, filter_term=term)

    def _filter_macro_list(self, event=None):
        self._refresh_macro_sidebar(keep_selection=True)

    def _show_macro_menu(self, event, name: str):
        pass  # Context menu - simplified for now

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
        self._update_macro_info()
        self.params["last_macro"] = name
        self._save_params()

    def _update_macro_info(self):
        n = self.macro.event_count()
        d = self.macro.duration()
        self.macro_events.set(str(n))
        self.macro_duration.set(fmt_seconds(d))
        name = self.current_macro_name or "—"
        meta = "Non enregistrée" if n == 0 else f"{n} évts | {fmt_seconds(d)}"
        self.lbl_macro_name.configure(text=f"Macro : {name}")
        self.lbl_macro_meta.configure(text=meta)

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

    # === Macro CRUD ===
    def macro_new(self):
        dlg = TextInputDialog(self, "Nouvelle macro", "Nom:", "Nouvelle Macro")
        name = dlg.show()
        if not name or is_protected_macro(name):
            return
        name = sanitize_macro_name(name)
        path = macro_path_from_name(self.MACROS_DIR, name)
        write_macro_file(path, name, [])
        self._refresh_macro_sidebar(keep_selection=False)
        self.macro_list.select(name, fire=True)

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
        self.macro_list.select(new_name, fire=False)

    def macro_delete(self):
        if not self.current_macro_path or is_protected_macro(self.current_macro_name):
            return
        if not messagebox.askyesno("Supprimer", f"Supprimer « {self.current_macro_name} » ?"):
            return
        try:
            self.current_macro_path.unlink()
        except Exception:
            pass
        self.current_macro_name = None
        self.current_macro_path = None
        self.macro = Macro()
        self._refresh_macro_sidebar(keep_selection=False)
        self._update_macro_info()

    # === Recording/Playback ===
    def toggle_record(self):
        if self.current_state == State.RECORDING:
            self.rec.stop()
            steps = trim_tail(self.rec.get_steps_as_dicts(), 3.0)
            self.macro.set_steps_from_dicts(steps)
            write_macro_file(self.current_macro_path, self.current_macro_name, steps)
            self._update_macro_info()
            self.current_state = State.IDLE
            self._update_ui_state()
        elif self.current_state == State.IDLE and self.current_macro_path:
            self.current_state = State.RECORDING
            self._update_ui_state()
            self.macro.clear()
            self.rec.start()

    def on_play(self, notify_tg: bool = True):
        if self.current_state != State.IDLE:
            return
        steps = [s.to_dict() for s in self.macro.steps]
        if not steps:
            return
        self.current_state = State.PLAYING
        self._update_ui_state()
        self.cycle_count.set(0)
        self.session_seconds = 0.0
        self.player.on_cycle = lambda: self.after(0, lambda: self.cycle_count.set(self.cycle_count.get() + 1))
        self.player.on_stopped = lambda: self.after(0, self._on_player_stopped)
        self.player.play(steps, loop=self._auto_loop())
        self._start_timer()
        if notify_tg:
            self.tg.replace_controls(self._tg_status_text("Lecture"), coc_launched=self._coc_is_running_tg)

    def force_stop_all(self, notify_tg: bool = True):
        if self.current_state == State.RECORDING:
            self.rec.stop()
        if self.current_state == State.PLAYING:
            self.player.stop()
        self._stop_timer()
        self.current_state = State.IDLE
        self._update_ui_state()
        if notify_tg:
            self.tg.replace_controls(self._tg_status_text("Arrêté"), coc_launched=self._coc_is_running_tg)

    def _on_player_stopped(self):
        self._stop_timer()
        self.current_state = State.IDLE
        self._update_ui_state()

    def _update_ui_state(self):
        idle = self.current_state == State.IDLE
        self.btn_new.configure(state="normal" if idle else "disabled")
        self.btn_rename.configure(state="normal" if idle else "disabled")
        self.btn_delete.configure(state="normal" if idle else "disabled")
        self.btn_play.configure(state="normal" if idle else "disabled")
        rec_text = "Arrêter" if self.current_state == State.RECORDING else "Enregistrer"
        self.btn_rec.configure(text=rec_text)

    def _start_timer(self):
        self._play_timer_running = True
        self._last_tick = time.perf_counter()
        self.after(200, self._tick_timer)

    def _stop_timer(self):
        self._play_timer_running = False

    def _tick_timer(self):
        if not self._play_timer_running:
            return
        now = time.perf_counter()
        self.session_seconds += now - (self._last_tick or now)
        self._last_tick = now
        self.session_elapsed.set(fmt_seconds(self.session_seconds))
        self.after(200, self._tick_timer)

    # === Hotkeys ===
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

    # === CoC Launch ===
    def launch_coc_once(self):
        path = self.params.get("coc_path", "").strip()
        if not path:
            return
        if self._coc_launched_once:
            return
        try:
            os.startfile(path) if os.name == 'nt' else None
            self._coc_launched_once = True
            self._coc_is_running_tg = True
            self.tg.replace_controls(self._tg_status_text("CoC lancé"), coc_launched=True)
        except Exception as e:
            self._log.error(f"Launch error: {e}")

    # === Settings ===
    def open_settings(self):
        if "settings" in self.open_windows:
            return
        def on_save(p):
            self.params.update(p)
            self._save_params()
        def on_close(name):
            self.open_windows.pop(name, None)
        def open_tg():
            TelegramAutomationDialog(self, self.params, on_save, self.GUIDE_HTML_PATH)
        def open_diag():
            DiagnosticsDialog(self, self.tg_status_var.get(), self.APP_VERSION, self.PYTHON_VERSION,
                            PIL_AVAILABLE, MSS_AVAILABLE, self.LOG_PATH, self.BASE_DIR)
        win = SettingsDialog(self, self.params, on_save, open_tg, open_diag,
                            self.tg_status_var, self.tg_status_color_var,
                            lambda: None, self._request_shutdown, name="settings", on_close_cb=on_close)
        self.open_windows["settings"] = win

    def _request_shutdown(self):
        if messagebox.askyesno("Extinction", "Éteindre le PC ?"):
            if messagebox.askyesno("Confirmer", "Confirmer l'extinction ?"):
                threading.Thread(target=perform_shutdown, daemon=True).start()

    def safe_quit(self):
        self._log.info("Fermeture...")
        self.force_stop_all(notify_tg=False)
        if self.tg.is_ready:
            self.tg.send_message("Application fermée.")
        try:
            keyboard.remove_all_hotkeys()
        except Exception:
            pass
        clean_old_logs(self.LOG_PATH.parent, "app.log*", 24)
        self.destroy()
