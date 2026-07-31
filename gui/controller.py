# -*- coding: utf-8 -*-
"""Application controller and Qt signal bridge."""

from __future__ import annotations

import queue
import threading
import time
from enum import Enum, auto
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QTimer, QObject, Qt, pyqtSignal

from gui.models import MacroSummary
from models.macro import Macro, is_protected_macro
from services.recorder_service import Player, Recorder, trim_tail
from services.coc.detector import CocDetector
from services.coc.launcher import CocLauncher
from services.coc.models import CocLaunchProfile, CocPresence
from services.coc.safeguard import CocPresenceMonitor
from services.telegram_service import TelegramBotService, TelegramCommand
from utils.config import (
    as_bool,
    list_macros,
    macro_path_from_name,
    read_macro_file,
    read_macro_meta,
    sanitize_macro_name,
    write_macro_file,
    write_params_csv,
)
from utils.logger import get_logger
from utils.system import grab_screenshot_png_bytes, perform_shutdown


class RuntimeState(Enum):
    IDLE = auto()
    RECORDING = auto()
    PLAYING = auto()
    STOPPING = auto()
    ERROR = auto()


class RuntimeSignals(QObject):
    state_changed = pyqtSignal(str, str)
    macros_changed = pyqtSignal(object, str)
    macro_selected = pyqtSignal(object)
    metrics_changed = pyqtSignal(float, int, float, int)
    cycle_changed = pyqtSignal(int)
    activity = pyqtSignal(str, str)
    error = pyqtSignal(str, str)
    telegram_status = pyqtSignal(str, str)
    coc_presence_changed = pyqtSignal(object)
    safeguard_triggered = pyqtSignal(str)
    coc_snapshot_observed = pyqtSignal(object)
    coc_lost_observed = pyqtSignal(object)
    coc_detection_error_observed = pyqtSignal(str)
    player_cycle_observed = pyqtSignal()
    player_stopped_observed = pyqtSignal()


class RuntimeController(QObject):
    """Owns runtime state and keeps the interface thread-safe."""

    def __init__(
        self,
        params: dict[str, str],
        telegram: TelegramBotService,
        macros_dir: Path,
        params_path: Path,
        protected_names: list[str],
        app_version: str,
        python_version: str,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.signals = RuntimeSignals()
        self.log = get_logger()
        self.params = params
        self.telegram = telegram
        self.macros_dir = macros_dir
        self.params_path = params_path
        self.protected_names = protected_names
        self.app_version = app_version
        self.python_version = python_version

        self.state = RuntimeState.IDLE
        self.current_macro = Macro()
        self.current_macro_name: str | None = None
        self.current_macro_path: Path | None = None
        self._summaries = []
        self._record_started_at: float | None = None
        self._playback_started_at: float | None = None
        self._playback_duration = 0.0
        self._cycles = 0
        self._coc_launched = False
        self._coc_running_for_tg = False
        self.coc_profile = CocLaunchProfile.from_params(params)
        self.coc_detector = CocDetector(self.coc_profile)
        self.coc_launcher = CocLauncher()
        self.coc_presence = CocPresence(False, False, False, reason="CoC non vérifié")
        self._coc_last_present = False
        self._safeguard_triggered = False
        self.coc_monitor = CocPresenceMonitor(
            self.coc_detector,
            self.coc_profile.detection_interval,
            self.coc_profile.missing_tolerance,
            lambda snapshot: self.signals.coc_snapshot_observed.emit(snapshot),
            lambda snapshot: self.signals.coc_lost_observed.emit(snapshot),
            lambda detail: self.signals.coc_detection_error_observed.emit(detail),
        )
        self._coc_launch_timer = QTimer(self)
        self._coc_launch_timer.setSingleShot(True)
        self._coc_launch_timer.timeout.connect(self._on_coc_launch_timeout)
        self._closing = False
        self._bootstrapped = False

        self.recorder = Recorder()
        self.player = Player(self.recorder)
        self.player.on_cycle = lambda: self.signals.player_cycle_observed.emit()
        self.player.on_stopped = lambda: self.signals.player_stopped_observed.emit()
        self.signals.coc_snapshot_observed.connect(self._apply_coc_snapshot, Qt.ConnectionType.QueuedConnection)
        self.signals.coc_lost_observed.connect(self._handle_coc_lost, Qt.ConnectionType.QueuedConnection)
        self.signals.coc_detection_error_observed.connect(self._handle_coc_detection_error, Qt.ConnectionType.QueuedConnection)
        self.signals.player_cycle_observed.connect(self._player_cycle, Qt.ConnectionType.QueuedConnection)
        self.signals.player_stopped_observed.connect(self._player_stopped, Qt.ConnectionType.QueuedConnection)

    @property
    def is_busy(self) -> bool:
        return self.state in {RuntimeState.RECORDING, RuntimeState.PLAYING, RuntimeState.STOPPING}

    @property
    def auto_loop(self) -> bool:
        return as_bool(self.params.get("auto_loop", "0"))

    @property
    def coc_launched(self) -> bool:
        return self._coc_launched

    @property
    def safeguard_enabled(self) -> bool:
        return as_bool(self.params.get("coc_safeguard", "0"))

    def bootstrap(self) -> None:
        if self._bootstrapped:
            return
        self._bootstrapped = True
        self.coc_monitor.start()
        self._ensure_protected_macros()
        self.refresh_macros()
        self._emit_telegram_status()
        last = self.params.get("last_macro", "").strip()
        editable_items = [item for item in self._summaries if item.editable]
        if last and any(item.name == last for item in editable_items):
            self.select_macro(last)
        elif editable_items:
            self.select_macro(editable_items[0].name)
        self._activity("Console prête", "success")
        if self.telegram.is_ready:
            self.telegram.send_message(f"Macro COC v{self.app_version} lancée.")
            self._replace_tg_controls("Prêt")

    def refresh_macros(self, selected_name: str | None = None) -> None:
        items = list_macros(self.macros_dir, self.protected_names)
        self._summaries = []
        for name, path in items:
            events, duration = read_macro_meta(path)
            self._summaries.append(self._summary(name, events, duration))
        chosen = selected_name or self.current_macro_name or (self._summaries[0].name if self._summaries else "")
        self.signals.macros_changed.emit(tuple(self._summaries), chosen)

    def summaries(self) -> tuple:
        return tuple(self._summaries)

    def _summary(self, name: str, events: int, duration: float):
        if is_protected_macro(name):
            descriptions = {
                "recharger coc": "Réinitialise l’environnement de jeu avant une nouvelle session.",
                "valider arrivée": "Effectue la séquence de validation de début de session.",
            }
            return MacroSummary(name, events, duration, "system", descriptions.get(name.casefold(), "Routine intégrée à AUTO-COC."))
        return MacroSummary(name, events, duration, "user", "Macro créée par toi.")

    def select_macro(self, name: str) -> None:
        if self.is_busy:
            return
        if is_protected_macro(name):
            self._error("Cette routine système n’est pas une macro utilisateur.", "Elle est pilotée par les commandes système dédiées.")
            return
        path = macro_path_from_name(self.macros_dir, name)
        if not path.exists():
            self._error(f"La macro « {name} » est introuvable.", "Vérifie le dossier config/macros.")
            return
        macro_name, steps, sha1, updated_at = read_macro_file(path)
        self.current_macro = Macro(name=macro_name)
        self.current_macro.set_steps_from_dicts(steps)
        self.current_macro.sha1 = sha1
        self.current_macro.updated_at = updated_at
        self.current_macro_name = name
        self.current_macro_path = path
        if self.params.get("last_macro") != name:
            self.params["last_macro"] = name
            self.save_params()
        self.signals.macro_selected.emit(self.current_macro)
        self.signals.macros_changed.emit(tuple(self._summaries), name)
        self._activity(f"Macro sélectionnée · {name}", "info")

    def create_macro(self, name: str) -> bool:
        if self.is_busy:
            return False
        clean_name = sanitize_macro_name(name)
        if not clean_name or is_protected_macro(clean_name):
            self._error("Ce nom de macro n’est pas disponible.", "Choisis un nom différent des macros système.")
            return False
        path = macro_path_from_name(self.macros_dir, clean_name)
        if path.exists():
            self._error("Cette macro existe déjà.", "Utilise Renommer ou choisis un autre nom.")
            return False
        if not write_macro_file(path, clean_name, []):
            self._error("La macro n’a pas pu être créée.", "Vérifie les droits d’écriture du dossier config.")
            return False
        self.refresh_macros(clean_name)
        self.select_macro(clean_name)
        self._activity(f"Macro créée · {clean_name}", "success")
        return True

    def rename_macro(self, new_name: str) -> bool:
        if not self.current_macro_name or not self.current_macro_path or self.is_busy:
            return False
        if is_protected_macro(self.current_macro_name):
            self._error("Cette macro système est protégée.", "Elle ne peut pas être renommée.")
            return False
        clean_name = sanitize_macro_name(new_name)
        new_path = macro_path_from_name(self.macros_dir, clean_name)
        if not clean_name or is_protected_macro(clean_name) or (new_path.exists() and new_path != self.current_macro_path):
            self._error("Ce nom de macro n’est pas disponible.", "Choisis un nom différent.")
            return False
        if not write_macro_file(new_path, clean_name, [step.to_dict() for step in self.current_macro.steps]):
            self._error("La macro n’a pas pu être renommée.", "Vérifie les droits d’écriture du dossier config.")
            return False
        if new_path != self.current_macro_path:
            self.current_macro_path.unlink(missing_ok=True)
        self.current_macro.name = clean_name
        self.current_macro_name = clean_name
        self.current_macro_path = new_path
        self.refresh_macros(clean_name)
        self.signals.macro_selected.emit(self.current_macro)
        self._activity(f"Macro renommée · {clean_name}", "success")
        return True

    def delete_macro(self) -> bool:
        if not self.current_macro_name or not self.current_macro_path or self.is_busy:
            return False
        if is_protected_macro(self.current_macro_name):
            self._error("Cette macro système est protégée.", "Elle ne peut pas être supprimée.")
            return False
        name = self.current_macro_name
        try:
            self.current_macro_path.unlink()
        except OSError as exc:
            self._error("La macro n’a pas pu être supprimée.", str(exc))
            return False
        self.current_macro = Macro()
        self.current_macro_name = None
        self.current_macro_path = None
        self.refresh_macros()
        self.signals.macro_selected.emit(self.current_macro)
        self._activity(f"Macro supprimée · {name}", "success")
        return True

    def start_recording(self) -> bool:
        if self.is_busy or not self.current_macro_path:
            return False
        try:
            self.recorder.start()
        except Exception as exc:
            self._error("Impossible de démarrer l’enregistrement.", str(exc))
            self._set_state(RuntimeState.IDLE, "Prêt")
            return False
        self._record_started_at = time.perf_counter()
        self.current_macro.clear()
        self._set_state(RuntimeState.RECORDING, "Enregistrement")
        self._activity(f"Enregistrement démarré · {self.current_macro_name}", "warning")
        return True

    def stop_recording(self) -> bool:
        if self.state != RuntimeState.RECORDING:
            return False
        self.recorder.stop()
        steps = trim_tail(self.recorder.get_steps_as_dicts(), 3.0)
        self.current_macro.set_steps_from_dicts(steps)
        if not self.current_macro_path or not write_macro_file(self.current_macro_path, self.current_macro_name or "Macro", steps):
            self._error("L’enregistrement n’a pas pu être sauvegardé.", "Vérifie les droits d’écriture du dossier config.")
            self._set_state(RuntimeState.IDLE, "Prêt")
            return False
        self._record_started_at = None
        self.refresh_macros(self.current_macro_name)
        self.signals.macro_selected.emit(self.current_macro)
        self._set_state(RuntimeState.IDLE, "Macro sauvegardée")
        self._activity(f"Macro sauvegardée · {len(steps):,} événements", "success")
        return True

    def start_playback(self) -> bool:
        if self.is_busy or self.current_macro.is_empty():
            if self.current_macro.is_empty():
                self._error("Cette macro est vide.", "Enregistre des événements avant de la lire.")
            return False
        self.refresh_coc_presence()
        if self.safeguard_enabled and not self.coc_presence.present:
            detail = "La détection CoC est indisponible."
            if not self.coc_presence.error:
                detail = "Le safeguard bloque la macro tant que CoC n’est pas présent."
            self._error("CoC non vérifié.", detail)
            return False
        try:
            self._cycles = 0
            self._playback_duration = self.current_macro.duration()
            self._playback_started_at = time.perf_counter()
            self._safeguard_triggered = False
            self._set_state(RuntimeState.PLAYING, "Lecture")
            if self.safeguard_enabled:
                self.coc_monitor.arm()
            self.player.play([step.to_dict() for step in self.current_macro.steps], loop=self.auto_loop)
        except Exception as exc:
            self.coc_monitor.disarm()
            self._playback_started_at = None
            self._set_state(RuntimeState.IDLE, "Prêt")
            self._error("Impossible de démarrer la lecture.", str(exc))
            return False
        self._activity(f"Lecture démarrée · {self.current_macro_name}", "success")
        self._replace_tg_controls("Lecture")
        return True

    def stop_all(self) -> None:
        if self.state == RuntimeState.RECORDING:
            self.stop_recording()
            return
        if self.state == RuntimeState.PLAYING:
            self._set_state(RuntimeState.STOPPING, "Arrêt en cours")
            self.coc_monitor.disarm()
            self.player.stop()
            self._playback_started_at = None

    def set_safeguard(self, enabled: bool) -> None:
        self.params["coc_safeguard"] = "1" if enabled else "0"
        self.save_params()
        self._apply_safeguard_state(enabled)
        self._activity(f"Safeguard CoC · {'activé' if enabled else 'désactivé'}", "info")

    def sync_safeguard_state(self) -> None:
        """Apply a value loaded from the settings dialog without rewriting it."""
        self._apply_safeguard_state(self.safeguard_enabled)

    def _apply_safeguard_state(self, enabled: bool) -> None:
        if not enabled:
            self.coc_monitor.disarm()
            return
        if self.state != RuntimeState.PLAYING:
            return
        self.refresh_coc_presence()
        if self.coc_presence.error:
            self._activity("Safeguard en attente · détection CoC indisponible", "warning")
        elif not self.coc_presence.present:
            self.handle_safeguard_loss("CoC n’est plus détecté.")
        else:
            self.coc_monitor.arm()

    def refresh_coc_profile(self) -> None:
        self.coc_profile = CocLaunchProfile.from_params(self.params)
        self.coc_detector.profile = self.coc_profile
        self.coc_monitor.interval = self.coc_profile.detection_interval
        self.coc_monitor.missing_tolerance = self.coc_profile.missing_tolerance

    def refresh_coc_presence(self) -> None:
        self._apply_coc_snapshot(self.coc_detector.snapshot())

    def update_metrics(self) -> None:
        now = time.perf_counter()
        elapsed = 0.0
        events = len(self.recorder.steps) if self.state == RuntimeState.RECORDING else self.current_macro.event_count()
        duration = self.current_macro.duration()
        if self.state == RuntimeState.RECORDING and self._record_started_at:
            elapsed = max(0.0, now - self._record_started_at)
        elif self.state == RuntimeState.PLAYING and self._playback_started_at:
            elapsed = max(0.0, now - self._playback_started_at)
        self.signals.metrics_changed.emit(elapsed, events, duration, self._cycles)

    def set_auto_loop(self, enabled: bool) -> None:
        self.params["auto_loop"] = "1" if enabled else "0"
        self.save_params()
        self._activity(f"Lecture en boucle · {'activée' if enabled else 'désactivée'}", "info")

    def save_params(self) -> None:
        write_params_csv(self.params_path, self.params)

    def launch_coc(self) -> bool:
        self.refresh_coc_profile()
        result = self.coc_launcher.launch(self.coc_profile, self.coc_detector)
        if not result.started:
            self._error("CoC n’a pas pu être lancé.", result.message)
            return False
        self._coc_launch_timer.stop()
        self._coc_launched = result.already_present
        if result.already_present:
            self._activity("CoC déjà détecté", "success")
        else:
            self._coc_launch_timer.start(int(self.coc_profile.startup_timeout * 1000))
            self._activity("Lancement CoC demandé · vérification en cours", "success")
        return True

    def _on_coc_launch_timeout(self) -> None:
        if self.coc_presence.present:
            return
        self._coc_launched = False
        detail = "Le lanceur a démarré, mais CoC n’a pas été détecté dans le délai configuré."
        if self.coc_presence.error:
            detail = f"La détection CoC est indisponible : {self.coc_presence.error}"
        self._error("CoC non confirmé", detail)

    def capture_screen(self) -> None:
        if not self.telegram.is_ready:
            self._error("Telegram n’est pas connecté.", "Configure le bot et le Chat ID avant d’envoyer une capture.")
            return
        threading.Thread(target=self._capture_worker, daemon=True, name="ScreenCapture").start()
        self._activity("Capture d’écran en préparation…", "info")

    def _capture_worker(self) -> None:
        png = grab_screenshot_png_bytes()
        if png:
            self.telegram.send_photo(png, caption=f"Capture · {self.current_macro_name or 'AUTO-COC'}")
            self.signals.activity.emit("Capture envoyée via Telegram", "success")
        else:
            self.signals.error.emit("Capture impossible", "Aucune méthode de capture n’a fonctionné.")

    def request_shutdown(self) -> None:
        threading.Thread(target=perform_shutdown, daemon=True, name="SystemShutdown").start()
        self._activity("Extinction du système demandée", "warning")

    def drain_telegram_commands(self) -> None:
        while True:
            try:
                command: TelegramCommand = self.telegram.command_queue.get_nowait()
            except queue.Empty:
                break
            self.handle_telegram_command(command.command, command.meta)

    def handle_telegram_command(self, command: str, meta: dict[str, Any]) -> None:
        self._activity(f"Commande Telegram · {command}", "info")
        if command == "STOP":
            self.stop_all()
        elif command == "GO":
            self.start_playback()
        elif command == "CAPTURE":
            self.capture_screen()
        elif command == "LAUNCH_COC":
            self.launch_coc()
        elif command == "TOGGLE_LOOP":
            self.set_auto_loop(not self.auto_loop)
            self._replace_tg_controls("Boucle mise à jour")
        elif command == "SELECT_MACRO_LIST":
            self.telegram.push_macro_selection([item.name for item in self._summaries if item.editable])
        elif command.startswith("SELECT_MACRO:"):
            self.select_macro(command.split(":", 1)[1])
        elif command == "RELOAD_COC":
            self._play_protected_macro("Recharger COC")
        elif command == "VALIDATE_ARRIVAL":
            self._play_protected_macro("Valider arrivée")
        elif command == "MENU":
            self.telegram.replace_menu("Paramètres", loop_state=self.auto_loop)
        elif command == "BACK":
            self.telegram.replace_controls(self._tg_status_text("Prêt"), coc_launched=self._coc_running_for_tg)
        elif command == "SHUTDOWN_ASK":
            self.telegram.push_shutdown_confirm()
        elif command == "SHUTDOWN_CONFIRM":
            self.request_shutdown()
        elif command == "SHUTDOWN_CANCEL":
            self.telegram.replace_controls(self._tg_status_text("Prêt"), coc_launched=self._coc_running_for_tg)

    def _play_protected_macro(self, name: str) -> None:
        if self.is_busy:
            self._error("Une autre opération est déjà en cours.", "Arrête la lecture avant de lancer cette macro.")
            return
        if self.safeguard_enabled:
            self.refresh_coc_presence()
            if not self.coc_presence.present:
                detail = "La détection CoC est indisponible."
                if not self.coc_presence.error:
                    detail = "Le safeguard bloque la routine tant que CoC n’est pas présent."
                self._error("CoC non vérifié.", detail)
                return
        path = macro_path_from_name(self.macros_dir, name)
        if not path.exists():
            self._error(f"La macro système « {name} » est introuvable.", "Réinstalle la macro protégée.")
            return
        _, steps, _, _ = read_macro_file(path)
        if not steps:
            self._error(f"La macro système « {name} » est vide.", "")
            return
        self._playback_duration = sum(max(0.0, float(step.get("t", 0.0))) for step in steps)
        self._playback_started_at = time.perf_counter()
        self._safeguard_triggered = False
        self._set_state(RuntimeState.PLAYING, f"Lecture · {name}")
        if self.safeguard_enabled:
            self.coc_monitor.arm()
        try:
            self.player.play(steps, loop=False)
        except Exception as exc:
            self.coc_monitor.disarm()
            self._playback_started_at = None
            self._set_state(RuntimeState.IDLE, "Prêt")
            self._error("Impossible de lancer la routine système.", str(exc))
            return
        self._replace_tg_controls(f"Lecture · {name}")

    def _ensure_protected_macros(self) -> None:
        for name in self.protected_names:
            path = macro_path_from_name(self.macros_dir, name)
            if not path.exists():
                write_macro_file(path, name, [])

    def _player_cycle(self) -> None:
        self._cycles += 1
        self.signals.cycle_changed.emit(self._cycles)

    def _player_stopped(self) -> None:
        if self._closing:
            return
        self.coc_monitor.disarm()
        self._playback_started_at = None
        if self._safeguard_triggered:
            self._safeguard_triggered = False
            self._set_state(RuntimeState.IDLE, "Arrêt safeguard · CoC perdu")
            self._activity("Lecture arrêtée par le safeguard · CoC perdu", "warning")
            self._replace_tg_controls("Safeguard")
        else:
            self._set_state(RuntimeState.IDLE, "Lecture terminée")
            self._activity("Lecture terminée", "success")
            self._replace_tg_controls("Terminé")

    def _apply_coc_snapshot(self, snapshot: CocPresence) -> None:
        self.coc_presence = snapshot
        if snapshot.present and self._coc_launch_timer.isActive():
            self._coc_launch_timer.stop()
            self._activity("CoC détecté · lancement confirmé", "success")
        self._coc_launched = snapshot.present
        self._coc_running_for_tg = snapshot.present
        if self._bootstrapped and snapshot.present != self._coc_last_present:
            level = "success" if snapshot.present else "warning"
            if snapshot.error:
                level = "warning"
            self._activity(snapshot.reason, level)
        self._coc_last_present = snapshot.present
        self.signals.coc_presence_changed.emit(snapshot)

    def _handle_coc_lost(self, snapshot: CocPresence) -> None:
        if self.state == RuntimeState.PLAYING and self.safeguard_enabled:
            self.log.warning("Safeguard CoC déclenché : %s", snapshot.reason)
            self.signals.safeguard_triggered.emit("CoC n’est plus détecté.")

    def _handle_coc_detection_error(self, detail: str) -> None:
        self.log.warning("Détection CoC indisponible : %s", detail)
        self.signals.activity.emit("Détection CoC indisponible · safeguard en attente", "warning")

    def handle_safeguard_loss(self, reason: str) -> None:
        if self.state != RuntimeState.PLAYING or not self.safeguard_enabled:
            return
        self._safeguard_triggered = True
        self._set_state(RuntimeState.STOPPING, "Safeguard · arrêt en cours")
        self._activity(f"Safeguard déclenché · {reason}", "warning")
        self.player.stop()

    def _set_state(self, state: RuntimeState, label: str) -> None:
        self.state = state
        self.signals.state_changed.emit(state.name, label)

    def _activity(self, message: str, level: str = "info") -> None:
        self.log.info(message)
        self.signals.activity.emit(message, level)

    def _error(self, title: str, detail: str) -> None:
        self.log.error(f"{title} {detail}".strip())
        self.signals.error.emit(title, detail)
        self.signals.activity.emit(title, "error")

    def _emit_telegram_status(self) -> None:
        status, color = self.telegram.get_status()
        self.signals.telegram_status.emit(status, color)

    def poll_telegram_status(self) -> None:
        self._emit_telegram_status()
        self.drain_telegram_commands()

    def _tg_status_text(self, prefix: str) -> str:
        loop = "ON" if self.auto_loop else "OFF"
        return f"{prefix} | Macro: {self.current_macro_name or '—'} | Loop: {loop}"

    def _replace_tg_controls(self, prefix: str) -> None:
        if self.telegram.is_ready:
            self.telegram.replace_controls(self._tg_status_text(prefix), coc_launched=self._coc_running_for_tg)

    def shutdown(self) -> None:
        self._closing = True
        self.coc_monitor.stop()
        try:
            if self.state == RuntimeState.RECORDING:
                self.recorder.stop()
            if self.player.is_playing():
                self.player.stop()
        finally:
            try:
                import keyboard
                keyboard.remove_all_hotkeys()
            except Exception:
                pass
            if self.telegram.is_ready:
                self.telegram.send_message("Application fermée.")
            self.telegram.stop()
