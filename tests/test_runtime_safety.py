# -*- coding: utf-8 -*-
"""Deterministic lifecycle checks for playback and CoC monitoring workers."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from services.coc.models import CocPresence
from services.coc.safeguard import CocPresenceMonitor
from services.recorder_service import Player


class _RecorderStub:
    def mark_self_play(self, _active: bool) -> None:
        return None


class _BlockingDetector:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def snapshot(self) -> CocPresence:
        self.entered.set()
        self.release.wait()
        return CocPresence(True, True, False, pids=(1,), process_names=("CoC.exe",))


def _build_player() -> Player:
    with patch("services.recorder_service.MouseController", return_value=object()), patch(
        "services.recorder_service.KeyController", return_value=object()
    ):
        return Player(_RecorderStub())


class RuntimeSafetyTests(unittest.TestCase):
    def test_player_keeps_live_worker_and_stop_flag_after_join_timeout(self) -> None:
        player = _build_player()
        entered = threading.Event()
        release = threading.Event()
        stopped = threading.Event()
        callbacks = 0

        def blocking_cycle(_steps) -> bool:
            entered.set()
            release.wait()
            return False

        def on_stopped() -> None:
            nonlocal callbacks
            callbacks += 1
            stopped.set()

        player._run_one_cycle = blocking_cycle
        player.on_stopped = on_stopped
        self.assertTrue(player.play([{"t": 0.0, "type": "nop", "data": {}}]))
        self.assertTrue(entered.wait(1.0))
        worker = player._thread
        self.assertIsNotNone(worker)

        with patch.object(worker, "join", return_value=None):
            player.stop()

        self.assertTrue(worker.is_alive())
        self.assertIs(player._thread, worker)
        self.assertTrue(player._stop_flag.is_set())
        self.assertEqual(callbacks, 0)

        self.assertFalse(player.play([{"t": 0.0, "type": "nop", "data": {}}]))
        self.assertIs(player._thread, worker)

        release.set()
        worker.join(1.0)
        self.assertFalse(worker.is_alive())
        self.assertTrue(stopped.wait(1.0))
        self.assertEqual(callbacks, 1)

        player.stop()
        self.assertEqual(callbacks, 1)

    def test_player_callback_is_once_after_natural_completion(self) -> None:
        player = _build_player()
        stopped = threading.Event()
        callbacks = 0

        def on_stopped() -> None:
            nonlocal callbacks
            callbacks += 1
            stopped.set()

        player.on_stopped = on_stopped
        player.play([{"t": 0.0, "type": "nop", "data": {}}])
        self.assertTrue(stopped.wait(1.0))
        player.stop()
        self.assertEqual(callbacks, 1)

    def test_presence_monitor_keeps_live_worker_after_join_timeout(self) -> None:
        detector = _BlockingDetector()
        monitor = CocPresenceMonitor(
            detector,
            interval=0.01,
            missing_tolerance=2,
            on_snapshot=lambda _snapshot: None,
            on_lost=lambda _snapshot: self.fail("present CoC must not trigger safeguard loss"),
        )
        monitor.start()
        self.assertTrue(detector.entered.wait(1.0))
        worker = monitor._thread
        self.assertIsNotNone(worker)

        with patch.object(worker, "join", return_value=None):
            monitor.stop()

        self.assertTrue(worker.is_alive())
        self.assertIs(monitor._thread, worker)

        monitor.start()
        self.assertIs(monitor._thread, worker)

        detector.release.set()
        worker.join(1.0)
        self.assertFalse(worker.is_alive())
        monitor.stop()
        self.assertIsNone(monitor._thread)


if __name__ == "__main__":
    unittest.main()
