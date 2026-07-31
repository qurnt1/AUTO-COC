# -*- coding: utf-8 -*-
"""Application controller and Qt signal bridge."""

from __future__ import annotations

import os
import queue
import threading
import time
from enum import Enum, auto
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QTimer, QObject, Qt, pyqtSignal

from gui.models import MacroSummary
from models.macro import Macro
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


SYSTEM_ROLE_DEFAULTS = {
    "reload_coc": "Reload CoC",
}


class RuntimeSignals(QObject):
    state_changed = pyqtSignal(str, str)
    macros_changed = pyqtSignal(object, str)
    macro_selected = pyqtSignal(object)
    metrics_changed = pyqtSignal(float, int, float, int)
    playback_options_changed = pyqtSignal(bool, bool)
    activity = pyqtSignal(str, str)
    error = pyqtSignal(str, str)
    telegram_status = pyqtSignal(str, str)
    coc_presence_changed = pyqtSignal(object)
    coc_launching_changed = pyqtSignal(bool)
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
        system_names: list[str],
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.signals = RuntimeSignals()
        self.log = get_logger()
        self.params = params
        self.telegram = telegram
        self.macros_dir = macros_dir
        self.params_path = params_path
        defaults = tuple(system_names) or tuple(SYSTEM_ROLE_DEFAULTS.values())
        self.system_defaults = dict(zip(SYSTEM_ROLE_DEFAULTS, defaults))

        self.state = RuntimeState.IDLE
        self.current_macro = Macro()
        self.current_macro_name: str | None = None
        self.current_macro_path: Path | None = None
        self._summaries = []
        self._record_started_at: float | None = None
        self._playback_started_at: float | None = None
        self._playback_duration = 0.0
        self._playback_event_count = 0
        self._cycles = 0
        self._manual_stop_requested = False
        self._capture_in_progress = threading.Event()
        self._coc_running_for_tg = False
        self.coc_profile = CocLaunchProfile.from_params(params)
        self.coc_detector = CocDetector(self.coc_profile)
        self.coc_launcher = CocLauncher()
        self.coc_presence = CocPresence(False, False, False, reason="CoC not verified")
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
        self._shutdown_requested = False
        self._telegram_ready_seen = False
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
    def safeguard_enabled(self) -> bool:
        return as_bool(self.params.get("coc_safeguard", "0"))

    def bootstrap(self) -> None:
        if self._bootstrapped:
            return
        self._bootstrapped = True
        self.coc_monitor.start()
        self._ensure_system_macros()
        self.refresh_macros()
        self.sync_playback_options()
        self._emit_telegram_status()
        last = self.params.get("last_macro", "").strip()
        user_macros = self.user_summaries()
        if last and any(item.name == last for item in user_macros):
            self.select_macro(last)
        elif user_macros:
            self.select_macro(user_macros[0].name)
        self._sync_telegram_ready()

    def refresh_macros(self, selected_name: str | None = None) -> None:
        items = list_macros(self.macros_dir)
        self._summaries = []
        for name, path in items:
            events, duration = read_macro_meta(path)
            self._summaries.append(self._summary(name, events, duration))
        user_macros = self.user_summaries()
        chosen = selected_name or self.current_macro_name or (user_macros[0].name if user_macros else "")
        self.signals.macros_changed.emit(tuple(self._summaries), chosen)

    def summaries(self) -> tuple[MacroSummary, ...]:
        return tuple(self._summaries)

    def user_summaries(self) -> tuple[MacroSummary, ...]:
        return tuple(item for item in self._summaries if not item.is_telegram_action)

    def system_summary(self, role: str) -> MacroSummary | None:
        return next((item for item in self._summaries if item.role == role), None)

    def _summary(self, name: str, events: int, duration: float) -> MacroSummary:
        role = next(
            (role for role in self.system_defaults if self._system_macro_name(role).casefold() == name.casefold()),
            "",
        )
        return MacroSummary(name, events, duration, role=role)

    def select_macro(self, name: str, *, remember: bool = True) -> None:
        if self.is_busy:
            return
        path = macro_path_from_name(self.macros_dir, name)
        if not path.exists():
            self._error(f"Macro “{name}” not found.", "Check the config/macros folder.")
            return
        macro_name, steps, sha1, updated_at = read_macro_file(path)
        self.current_macro = Macro(name=macro_name)
        self.current_macro.set_steps_from_dicts(steps)
        self.current_macro.sha1 = sha1
        self.current_macro.updated_at = updated_at
        self.current_macro_name = name
        self.current_macro_path = path
        if remember and self.params.get("last_macro") != name:
            self.params["last_macro"] = name
            self.save_params()
        self.signals.macro_selected.emit(self.current_macro)
        self.signals.macros_changed.emit(tuple(self._summaries), name)
        self._activity(f"Macro selected · {name}", "info")

    def create_macro(self, name: str) -> bool:
        if self.is_busy:
            return False
        clean_name = sanitize_macro_name(name)
        if not clean_name:
            self._error("This macro name is not available.", "Choose a different name.")
            return False
        path = macro_path_from_name(self.macros_dir, clean_name)
        if path.exists():
            self._error("This macro already exists.", "Rename it or choose another name.")
            return False
        if not write_macro_file(path, clean_name, []):
            self._error("The macro could not be created.", "Check write access to the config folder.")
            return False
        self.refresh_macros(clean_name)
        self.select_macro(clean_name)
        self._activity(f"Macro created · {clean_name}", "success")
        return True

    def create_system_macro(self, role: str, name: str) -> bool:
        if role not in self.system_defaults or self.is_busy:
            return False
        clean_name = sanitize_macro_name(name)
        if not clean_name:
            self._error("This routine name is not available.", "Choose a different name.")
            return False
        path = macro_path_from_name(self.macros_dir, clean_name)
        if path.exists():
            self._error("This routine already exists.", "Choose a different name.")
            return False
        if not write_macro_file(path, clean_name, []):
            self._error("The routine could not be created.", "Check write access to the config folder.")
            return False
        if not self._assign_system_role(role, clean_name):
            path.unlink(missing_ok=True)
            return False
        self.refresh_macros(clean_name)
        self.select_macro(clean_name, remember=False)
        self._activity(f"Telegram routine created · {clean_name}", "success")
        return True

    def select_system_macro(self, role: str) -> bool:
        summary = self.system_summary(role)
        if not summary:
            return False
        self.select_macro(summary.name, remember=False)
        return self.current_macro_name == summary.name

    def rename_macro(self, new_name: str) -> bool:
        if not self.current_macro_name or not self.current_macro_path or self.is_busy:
            return False
        old_name = self.current_macro_name
        clean_name = sanitize_macro_name(new_name)
        new_path = macro_path_from_name(self.macros_dir, clean_name)
        if not clean_name or (new_path.exists() and new_path != self.current_macro_path):
            self._error("This macro name is not available.", "Choose a different name.")
            return False
        if not write_macro_file(new_path, clean_name, [step.to_dict() for step in self.current_macro.steps]):
            self._error("The macro could not be renamed.", "Check write access to the config folder.")
            return False
        next_params = self._params_with_renamed_macro(old_name, clean_name)
        if next_params != self.params and not write_params_csv(self.params_path, next_params):
            if new_path != self.current_macro_path:
                new_path.unlink(missing_ok=True)
            self._error("The macro could not be renamed.", "The settings file could not be updated.")
            return False
        self._replace_params(next_params)
        if new_path != self.current_macro_path:
            self.current_macro_path.unlink(missing_ok=True)
        self.current_macro.name = clean_name
        self.current_macro_name = clean_name
        self.current_macro_path = new_path
        self.refresh_macros(clean_name)
        self.signals.macro_selected.emit(self.current_macro)
        self._activity(f"Macro renamed · {clean_name}", "success")
        return True

    def delete_macro(self) -> bool:
        if not self.current_macro_name or not self.current_macro_path or self.is_busy:
            return False
        name = self.current_macro_name
        next_params = self._params_without_macro(name)
        staged_path = self.current_macro_path.with_suffix(
            f"{self.current_macro_path.suffix}.delete.{os.getpid()}.{time.time_ns()}"
        )
        try:
            self.current_macro_path.replace(staged_path)
        except OSError as exc:
            self._error("The macro could not be deleted.", str(exc))
            return False
        if next_params != self.params and not write_params_csv(self.params_path, next_params):
            try:
                staged_path.replace(self.current_macro_path)
            except OSError as exc:
                self.log.error("Could not roll back macro deletion: %s", exc)
            self._error("The macro could not be deleted.", "The settings file could not be updated.")
            return False
        self._replace_params(next_params)
        try:
            staged_path.unlink()
        except OSError as exc:
            self.log.warning("Deleted macro cleanup is pending for %s: %s", staged_path.name, exc)
        self.current_macro = Macro()
        self.current_macro_name = None
        self.current_macro_path = None
        self.refresh_macros()
        user_macros = self.user_summaries()
        if user_macros:
            self.select_macro(user_macros[0].name)
        else:
            self.signals.macro_selected.emit(self.current_macro)
        self._activity(f"Macro deleted · {name}", "success")
        return True

    def start_recording(self) -> bool:
        if self.is_busy or not self.current_macro_path:
            return False
        try:
            self.recorder.start()
        except Exception as exc:
            self._error("Could not start recording.", str(exc))
            self._set_state(RuntimeState.IDLE, "Ready")
            return False
        self._record_started_at = time.perf_counter()
        self._playback_started_at = None
        self._playback_duration = 0.0
        self._playback_event_count = 0
        self._cycles = 0
        self.current_macro.clear()
        self._set_state(RuntimeState.RECORDING, "Recording")
        self._activity(f"Recording started · {self.current_macro_name}", "warning")
        return True

    def stop_recording(self) -> bool:
        if self.state != RuntimeState.RECORDING:
            return False
        self.recorder.stop()
        steps = trim_tail(self.recorder.get_steps_as_dicts(), 3.0)
        self.current_macro.set_steps_from_dicts(steps)
        if not self.current_macro_path or not write_macro_file(self.current_macro_path, self.current_macro_name or "Macro", steps):
            self._error("The recording could not be saved.", "Check write access to the config folder.")
            self._set_state(RuntimeState.IDLE, "Ready")
            return False
        self._record_started_at = None
        self.refresh_macros(self.current_macro_name)
        self.signals.macro_selected.emit(self.current_macro)
        self._set_state(RuntimeState.IDLE, "Macro saved")
        self._activity(f"Macro saved · {len(steps):,} events", "success")
        return True

    def start_playback(self) -> bool:
        if self.is_busy or self.current_macro.is_empty():
            if self.current_macro.is_empty():
                self._error("This macro is empty.", "Record some events before playing it.")
            return False
        self.refresh_coc_presence()
        if self.safeguard_enabled and not self.coc_presence.present:
            detail = "CoC detection is unavailable."
            if not self.coc_presence.error:
                detail = "Safeguard blocks playback until CoC is detected."
            self._error("CoC not verified.", detail)
            return False
        try:
            self._cycles = 0
            self._playback_duration = self.current_macro.duration()
            self._playback_event_count = self.current_macro.event_count()
            self._playback_started_at = time.perf_counter()
            self._safeguard_triggered = False
            self._manual_stop_requested = False
            started = self.player.play(
                [step.to_dict() for step in self.current_macro.steps],
                loop=self.auto_loop,
            )
            if not started:
                raise RuntimeError("A previous playback worker is still stopping.")
            self._set_state(RuntimeState.PLAYING, "Playing")
            if self.safeguard_enabled:
                self.coc_monitor.arm()
        except Exception as exc:
            self.coc_monitor.disarm()
            self._playback_started_at = None
            self._set_state(RuntimeState.IDLE, "Ready")
            self._error("Could not start playback.", str(exc))
            return False
        self._activity(f"Playback started · {self.current_macro_name}", "success")
        self._replace_tg_controls("Playing")
        return True

    def stop_all(self) -> None:
        if self.state == RuntimeState.RECORDING:
            self.stop_recording()
            return
        if self.state == RuntimeState.PLAYING:
            self._manual_stop_requested = True
            self._set_state(RuntimeState.STOPPING, "Stopping")
            self.coc_monitor.disarm()
            self.player.stop()
            self._playback_started_at = None

    def set_safeguard(self, enabled: bool) -> None:
        previous = self.params.get("coc_safeguard")
        self.params["coc_safeguard"] = "1" if enabled else "0"
        if not self.save_params():
            if previous is None:
                self.params.pop("coc_safeguard", None)
            else:
                self.params["coc_safeguard"] = previous
            self._error("Safeguard setting was not saved.", "The settings file is not writable.")
            self.sync_playback_options()
            return
        self._apply_safeguard_state(enabled)
        self.sync_playback_options()
        self._activity(f"CoC safeguard · {'enabled' if enabled else 'disabled'}", "info")

    def sync_safeguard_state(self) -> None:
        """Apply a value loaded from the settings dialog without rewriting it."""
        self._apply_safeguard_state(self.safeguard_enabled)

    def sync_playback_options(self) -> None:
        self.signals.playback_options_changed.emit(self.auto_loop, self.safeguard_enabled)

    def _apply_safeguard_state(self, enabled: bool) -> None:
        if not enabled:
            self.coc_monitor.disarm()
            return
        if self.state != RuntimeState.PLAYING:
            return
        self.refresh_coc_presence()
        if self.coc_presence.error:
            self._activity("Safeguard waiting · CoC detection unavailable", "warning")
        elif not self.coc_presence.present:
            self.handle_safeguard_loss("CoC is no longer detected.")
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
        events = self.current_macro.event_count()
        duration = self.current_macro.duration()
        if self.state == RuntimeState.RECORDING and self._record_started_at:
            events = len(self.recorder.steps)
            elapsed = max(0.0, now - self._record_started_at)
        elif self.state == RuntimeState.PLAYING and self._playback_started_at:
            events = self._playback_event_count
            duration = self._playback_duration
            elapsed = max(0.0, now - self._playback_started_at)
        self.signals.metrics_changed.emit(elapsed, events, duration, self._cycles)

    def set_auto_loop(self, enabled: bool) -> None:
        previous = self.params.get("auto_loop")
        self.params["auto_loop"] = "1" if enabled else "0"
        if not self.save_params():
            if previous is None:
                self.params.pop("auto_loop", None)
            else:
                self.params["auto_loop"] = previous
            self._error("Loop setting was not saved.", "The settings file is not writable.")
            self.sync_playback_options()
            return
        self.sync_playback_options()
        self._activity(f"Loop playback · {'enabled' if enabled else 'disabled'}", "info")

    def save_params(self) -> bool:
        return write_params_csv(self.params_path, self.params)

    def launch_coc(self) -> bool:
        if self.is_busy or self._coc_launch_timer.isActive():
            return False
        self.refresh_coc_profile()
        self.signals.coc_launching_changed.emit(True)
        result = self.coc_launcher.launch(self.coc_profile, self.coc_detector)
        if not result.started:
            self.signals.coc_launching_changed.emit(False)
            self._error("Could not launch CoC.", result.message)
            return False
        self._coc_launch_timer.stop()
        if result.already_present:
            self.signals.coc_launching_changed.emit(False)
            self._activity("CoC already detected", "success")
        else:
            self._coc_launch_timer.start(int(self.coc_profile.startup_timeout * 1000))
            self._activity("CoC launch requested · checking", "success")
        return True

    def _on_coc_launch_timeout(self) -> None:
        self.signals.coc_launching_changed.emit(False)
        if self.coc_presence.present:
            return
        detail = "The launcher started, but CoC was not detected before the timeout."
        if self.coc_presence.error:
            detail = f"CoC detection is unavailable: {self.coc_presence.error}"
        self._error("CoC was not confirmed", detail)

    def capture_screen(self) -> None:
        if not self.telegram.is_ready:
            self._error("Telegram is not connected.", "Configure the bot and chat ID before sending a screenshot.")
            return
        if self._capture_in_progress.is_set():
            self._activity("A Telegram screenshot is already being prepared", "info")
            return
        self._capture_in_progress.set()
        threading.Thread(target=self._capture_worker, daemon=True, name="ScreenCapture").start()
        self._activity("Preparing screenshot…", "info")

    def _capture_worker(self) -> None:
        try:
            png = grab_screenshot_png_bytes()
        except Exception as exc:
            self._capture_in_progress.clear()
            self.signals.error.emit("Screenshot unavailable", str(exc))
            return
        if not png:
            self._capture_in_progress.clear()
            self.signals.error.emit("Screenshot unavailable", "No capture method succeeded.")
            return
        future = self.telegram.send_latest_screenshot(
            png,
            caption=f"Capture · {self.current_macro_name or 'AUTO-COC'}",
            controls_text=self._tg_status_text("Ready"),
            coc_launched=self._coc_running_for_tg,
        )
        if future is None:
            self._capture_in_progress.clear()
            self.signals.error.emit("Screenshot not sent", "Telegram is no longer ready.")
            return
        future.add_done_callback(self._capture_completed)

    def _capture_completed(self, future) -> None:
        try:
            message_id = future.result()
        except Exception as exc:
            self.signals.error.emit("Screenshot not sent", str(exc))
        else:
            if message_id:
                self.signals.activity.emit("Latest screenshot sent via Telegram", "success")
            else:
                self.signals.error.emit("Screenshot not sent", "Telegram rejected the screenshot or control panel.")
        finally:
            self._capture_in_progress.clear()

    def request_shutdown(self) -> None:
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        if self.state == RuntimeState.RECORDING:
            if not self.stop_recording():
                self._shutdown_requested = False
                return
            self._start_system_shutdown()
            return
        if self.state in {RuntimeState.PLAYING, RuntimeState.STOPPING}:
            if self.state == RuntimeState.PLAYING:
                self.stop_all()
            self._activity("System shutdown waiting for playback to stop", "warning")
            return
        self._start_system_shutdown()

    def _start_system_shutdown(self) -> None:
        self.coc_monitor.stop()
        threading.Thread(target=perform_shutdown, daemon=True, name="SystemShutdown").start()
        self._activity("System shutdown requested", "warning")

    def drain_telegram_commands(self) -> None:
        while True:
            try:
                command: TelegramCommand = self.telegram.command_queue.get_nowait()
            except queue.Empty:
                break
            self.handle_telegram_command(command.command, command.meta)

    def handle_telegram_command(self, command: str, meta: dict[str, Any]) -> None:
        self._activity(f"Telegram command · {command}", "info")
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
            self._replace_tg_controls("Loop updated")
        elif command == "SELECT_MACRO_LIST":
            self.telegram.push_macro_selection([item.name for item in self.user_summaries()])
        elif command.startswith("SELECT_MACRO:"):
            self.select_macro(command.split(":", 1)[1])
        elif command == "RELOAD_COC":
            self._play_system_macro("reload_coc")
        elif command == "MENU":
            self.telegram.replace_menu("Settings", loop_state=self.auto_loop)
        elif command == "BACK":
            self.telegram.replace_controls(self._tg_status_text("Ready"), coc_launched=self._coc_running_for_tg)
        elif command in {"CANCEL_SELECTION", "DUMMY_COC_STATUS"}:
            self.telegram.replace_controls(self._tg_status_text("Ready"), coc_launched=self._coc_running_for_tg)
        elif command == "SHUTDOWN_ASK":
            self.telegram.push_shutdown_confirm()
        elif command == "SHUTDOWN_CONFIRM":
            self.request_shutdown()
        elif command == "SHUTDOWN_CANCEL":
            self.telegram.replace_controls(self._tg_status_text("Ready"), coc_launched=self._coc_running_for_tg)

    def _play_system_macro(self, role: str) -> None:
        if self.is_busy:
            self._error("Another operation is already running.", "Stop playback before running this macro.")
            return
        if self.safeguard_enabled:
            self.refresh_coc_presence()
            if not self.coc_presence.present:
                detail = "CoC detection is unavailable."
                if not self.coc_presence.error:
                    detail = "Safeguard blocks this routine until CoC is detected."
                self._error("CoC not verified.", detail)
                return
        name = self._system_macro_name(role)
        if not name:
            self._error("Reload CoC is not configured.", "Create and record the routine from the Telegram page.")
            return
        path = macro_path_from_name(self.macros_dir, name)
        if not path.exists():
            self._error(f"The routine “{name}” could not be found.", "Recreate it from the Telegram page.")
            return
        _, steps, _, _ = read_macro_file(path)
        if not steps:
            self._error(f"The routine “{name}” is empty.", "Record it from the Telegram page before running it.")
            return
        self._playback_duration = sum(max(0.0, float(step.get("t", 0.0))) for step in steps)
        self._playback_event_count = len(steps)
        self._cycles = 0
        self._playback_started_at = time.perf_counter()
        self._safeguard_triggered = False
        self._manual_stop_requested = False
        try:
            if not self.player.play(steps, loop=False):
                raise RuntimeError("A previous playback worker is still stopping.")
            self._set_state(RuntimeState.PLAYING, f"Playing · {name}")
            if self.safeguard_enabled:
                self.coc_monitor.arm()
        except Exception as exc:
            self.coc_monitor.disarm()
            self._playback_started_at = None
            self._set_state(RuntimeState.IDLE, "Ready")
            self._error("Could not run the routine.", str(exc))
            return
        self._replace_tg_controls(f"Playing · {name}")

    def _ensure_system_macros(self) -> None:
        next_params = dict(self.params)
        next_params.pop("system_validate_arrival_macro", None)
        for role, default_name in self.system_defaults.items():
            key = f"system_{role}_macro"
            assigned_name = next_params.get(key, "").strip()
            if role == "reload_coc" and assigned_name.casefold() == "recharger coc":
                legacy_target = macro_path_from_name(self.macros_dir, default_name)
                if legacy_target.exists():
                    assigned_name = default_name
            if not assigned_name and key not in next_params:
                default_path = macro_path_from_name(self.macros_dir, default_name)
                assigned_name = default_name if default_path.exists() else ""
            if assigned_name and not macro_path_from_name(self.macros_dir, assigned_name).exists():
                assigned_name = ""
            next_params[key] = assigned_name
        if next_params != self.params:
            if write_params_csv(self.params_path, next_params):
                self._replace_params(next_params)
            else:
                self._error("Telegram routines could not be migrated.", "The settings file is not writable.")

    def _system_macro_name(self, role: str) -> str:
        return self.params.get(f"system_{role}_macro", self.system_defaults.get(role, "")).strip()

    def _params_with_renamed_macro(self, old_name: str, new_name: str) -> dict[str, str]:
        next_params = dict(self.params)
        for role in self.system_defaults:
            key = f"system_{role}_macro"
            if next_params.get(key, "").strip().casefold() == old_name.strip().casefold():
                next_params[key] = new_name
        if next_params.get("last_macro", "").strip().casefold() == old_name.strip().casefold():
            next_params["last_macro"] = new_name
        return next_params

    def _params_without_macro(self, name: str) -> dict[str, str]:
        next_params = dict(self.params)
        for role in self.system_defaults:
            key = f"system_{role}_macro"
            if next_params.get(key, "").strip().casefold() == name.strip().casefold():
                next_params[key] = ""
        if next_params.get("last_macro", "").strip().casefold() == name.strip().casefold():
            next_params["last_macro"] = ""
        return next_params

    def _assign_system_role(self, role: str, name: str) -> bool:
        next_params = dict(self.params)
        next_params[f"system_{role}_macro"] = name
        if not write_params_csv(self.params_path, next_params):
            self._error("The Telegram routine could not be assigned.", "The settings file is not writable.")
            return False
        self._replace_params(next_params)
        return True

    def _replace_params(self, values: dict[str, str]) -> None:
        self.params.clear()
        self.params.update(values)

    def _player_cycle(self) -> None:
        self._cycles += 1

    def _player_stopped(self) -> None:
        if self._closing:
            return
        self.coc_monitor.disarm()
        self._playback_started_at = None
        if self._safeguard_triggered:
            self._safeguard_triggered = False
            self._manual_stop_requested = False
            self._set_state(RuntimeState.IDLE, "Safeguard stopped · CoC lost")
            self._activity("Playback stopped by safeguard · CoC lost", "warning")
            self._replace_tg_controls("Safeguard")
        elif self._manual_stop_requested:
            self._manual_stop_requested = False
            self._set_state(RuntimeState.IDLE, "Stopped")
            self._activity("Playback stopped", "info")
            self._replace_tg_controls("Stopped")
        else:
            self._set_state(RuntimeState.IDLE, "Playback complete")
            self._activity("Playback complete", "success")
            self._replace_tg_controls("Complete")
        if self._shutdown_requested:
            self._start_system_shutdown()

    def _apply_coc_snapshot(self, snapshot: CocPresence) -> None:
        self.coc_presence = snapshot
        if snapshot.present and self._coc_launch_timer.isActive():
            self._coc_launch_timer.stop()
            self.signals.coc_launching_changed.emit(False)
            self._activity("CoC detected · launch confirmed", "success")
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
            self.log.warning("CoC safeguard triggered: %s", snapshot.reason)
            self.signals.safeguard_triggered.emit("CoC is no longer detected.")

    def _handle_coc_detection_error(self, detail: str) -> None:
        self.log.warning("CoC detection unavailable: %s", detail)
        self.signals.activity.emit("CoC detection unavailable · safeguard waiting", "warning")

    def handle_safeguard_loss(self, reason: str) -> None:
        if self.state != RuntimeState.PLAYING or not self.safeguard_enabled:
            return
        self._safeguard_triggered = True
        self._set_state(RuntimeState.STOPPING, "Safeguard · stopping")
        self._activity(f"Safeguard triggered · {reason}", "warning")
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
        self._sync_telegram_ready()
        self._emit_telegram_status()
        self.drain_telegram_commands()

    def _sync_telegram_ready(self) -> None:
        ready = self.telegram.is_ready
        if ready and not self._telegram_ready_seen:
            self._telegram_ready_seen = True
            self.telegram.send_message("AUTO-COC is ready.")
            self._replace_tg_controls("Ready")
        elif not ready:
            self._telegram_ready_seen = False

    def _tg_status_text(self, prefix: str) -> str:
        loop = "ON" if self.auto_loop else "OFF"
        return f"{prefix} | Macro: {self.current_macro_name or '—'} | Loop: {loop}"

    def _replace_tg_controls(self, prefix: str) -> None:
        if self.telegram.is_ready:
            self.telegram.replace_controls(self._tg_status_text(prefix), coc_launched=self._coc_running_for_tg)

    def shutdown(self) -> None:
        self._coc_launch_timer.stop()
        if self.state == RuntimeState.RECORDING:
            self.stop_recording()
        self._closing = True
        self.coc_monitor.stop()
        try:
            if self.player.is_playing():
                self.player.stop()
        finally:
            try:
                import keyboard
                keyboard.remove_all_hotkeys()
            except Exception:
                pass
            if self.telegram.is_ready:
                self.telegram.send_message("AUTO-COC closed.")
            self.telegram.stop()
