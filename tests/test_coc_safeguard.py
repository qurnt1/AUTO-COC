# -*- coding: utf-8 -*-
"""Focused checks for CoC presence monitoring."""

from __future__ import annotations

import threading
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from services.coc.launcher import CocLauncher
from services.coc.models import CocPresence
from services.coc.safeguard import CocPresenceMonitor


class _MissingDetector:
    def snapshot(self) -> CocPresence:
        return CocPresence(False, False, False, reason="CoC introuvable")


class CocSafeguardTests(unittest.TestCase):
    def test_launcher_resolves_extension_hidden_shortcut(self) -> None:
        with TemporaryDirectory() as temp_dir:
            configured = Path(temp_dir) / "Clash of Clans"
            shortcut = Path(f"{configured}.lnk")
            shortcut.touch()
            self.assertEqual(CocLauncher._resolve_target(configured), shortcut)

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
