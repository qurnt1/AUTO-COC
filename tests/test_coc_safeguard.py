# -*- coding: utf-8 -*-
"""Focused checks for CoC presence monitoring."""

from __future__ import annotations

import threading
import unittest

from services.coc.models import CocPresence
from services.coc.safeguard import CocPresenceMonitor


class _MissingDetector:
    def snapshot(self) -> CocPresence:
        return CocPresence(False, False, False, reason="CoC introuvable")


class CocSafeguardTests(unittest.TestCase):
    def test_lost_callback_requires_configured_tolerance(self) -> None:
        lost = threading.Event()
        snapshots: list[CocPresence] = []
        monitor = CocPresenceMonitor(
            _MissingDetector(),
            interval=0.01,
            missing_tolerance=2,
            on_snapshot=snapshots.append,
            on_lost=lambda _snapshot: lost.set(),
        )
        monitor.arm()
        monitor.start()
        try:
            self.assertTrue(lost.wait(0.5))
            self.assertGreaterEqual(len(snapshots), 2)
            self.assertFalse(monitor.armed)
        finally:
            monitor.stop()


if __name__ == "__main__":
    unittest.main()
