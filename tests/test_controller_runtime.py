# -*- coding: utf-8 -*-
"""Controller regressions for user-visible runtime state."""

from __future__ import annotations

from concurrent.futures import Future
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from gui.controller import RuntimeController, RuntimeState


class _TelegramStub:
    is_ready = True
    command_queue = None

    def __init__(self) -> None:
        self.future: Future[int | None] = Future()
        self.screenshot_args = None
        self.messages = []
        self.control_updates = 0

    def send_latest_screenshot(self, *args, **kwargs):
        self.screenshot_args = (args, kwargs)
        return self.future

    def replace_controls(self, *_args, **_kwargs) -> None:
        self.control_updates += 1

    def send_message(self, message, **_kwargs) -> None:
        self.messages.append(message)

    def stop(self) -> None:
        return None


class ControllerRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _controller(self, root: Path, telegram: _TelegramStub) -> RuntimeController:
        return RuntimeController(
            params={"coc_safeguard": "0"},
            telegram=telegram,
            macros_dir=root / "macros",
            params_path=root / "data.csv",
            system_names=["Reload CoC"],
        )

    def test_manual_stop_is_not_reported_as_natural_completion(self) -> None:
        with TemporaryDirectory() as temp_dir:
            controller = self._controller(Path(temp_dir), _TelegramStub())
            states = []
            controller.signals.state_changed.connect(lambda _state, label: states.append(label))
            controller.state = RuntimeState.STOPPING
            controller._manual_stop_requested = True

            controller._player_stopped()

            self.assertEqual(controller.state, RuntimeState.IDLE)
            self.assertEqual(states[-1], "Stopped")
            controller.shutdown()

    def test_screenshot_success_is_emitted_only_after_telegram_ack(self) -> None:
        with TemporaryDirectory() as temp_dir:
            telegram = _TelegramStub()
            controller = self._controller(Path(temp_dir), telegram)
            activities = []
            controller.signals.activity.connect(lambda message, _level: activities.append(message))
            controller._capture_in_progress.set()

            with patch("gui.controller.grab_screenshot_png_bytes", return_value=b"png"):
                controller._capture_worker()

            self.assertTrue(controller._capture_in_progress.is_set())
            self.assertNotIn("Latest screenshot sent via Telegram", activities)
            telegram.future.set_result(101)
            self.assertFalse(controller._capture_in_progress.is_set())
            self.assertIn("Latest screenshot sent via Telegram", activities)
            controller.shutdown()

    def test_playback_does_not_enter_playing_when_worker_rejects_start(self) -> None:
        with TemporaryDirectory() as temp_dir:
            controller = self._controller(Path(temp_dir), _TelegramStub())
            controller.current_macro.set_steps_from_dicts([{"t": 0.0, "type": "nop", "data": {}}])

            with patch.object(controller, "refresh_coc_presence"), patch.object(
                controller.player,
                "play",
                return_value=False,
            ):
                self.assertFalse(controller.start_playback())

            self.assertEqual(controller.state, RuntimeState.IDLE)
            controller.shutdown()

    def test_shutdown_waits_for_playback_stop_confirmation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            controller = self._controller(Path(temp_dir), _TelegramStub())
            controller.state = RuntimeState.PLAYING

            with patch.object(controller, "stop_all", side_effect=lambda: setattr(controller, "state", RuntimeState.STOPPING)), patch.object(
                controller,
                "_start_system_shutdown",
            ) as start_shutdown:
                controller.request_shutdown()
                start_shutdown.assert_not_called()
                controller._manual_stop_requested = True
                controller._player_stopped()
                start_shutdown.assert_called_once_with()

            controller.shutdown()

    def test_shutdown_is_cancelled_when_recording_cannot_be_saved(self) -> None:
        with TemporaryDirectory() as temp_dir:
            controller = self._controller(Path(temp_dir), _TelegramStub())
            controller.state = RuntimeState.RECORDING

            with patch.object(controller, "stop_recording", return_value=False), patch.object(
                controller,
                "_start_system_shutdown",
            ) as start_shutdown:
                controller.request_shutdown()

            start_shutdown.assert_not_called()
            self.assertFalse(controller._shutdown_requested)
            controller.state = RuntimeState.IDLE
            controller.shutdown()

    def test_telegram_controls_are_sent_on_first_ready_transition(self) -> None:
        with TemporaryDirectory() as temp_dir:
            telegram = _TelegramStub()
            telegram.is_ready = False
            controller = self._controller(Path(temp_dir), telegram)

            controller._sync_telegram_ready()
            telegram.is_ready = True
            controller._sync_telegram_ready()
            controller._sync_telegram_ready()

            self.assertEqual(telegram.messages, ["AUTO-COC is ready."])
            self.assertEqual(telegram.control_updates, 1)
            controller.shutdown()


if __name__ == "__main__":
    unittest.main()
