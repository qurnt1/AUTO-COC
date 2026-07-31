# -*- coding: utf-8 -*-
"""Background safeguard that stops playback when CoC disappears."""

from __future__ import annotations

import threading
from collections.abc import Callable

from services.coc.detector import CocDetector
from services.coc.models import CocPresence


class CocPresenceMonitor:
    """Poll CoC without blocking the Qt event loop."""

    def __init__(
        self,
        detector: CocDetector,
        interval: float,
        missing_tolerance: int,
        on_snapshot: Callable[[CocPresence], None],
        on_lost: Callable[[CocPresence], None],
    ):
        self.detector = detector
        self.interval = max(0.25, interval)
        self.missing_tolerance = max(1, missing_tolerance)
        self.on_snapshot = on_snapshot
        self.on_lost = on_lost
        self._stop = threading.Event()
        self._armed = False
        self._missing_count = 0
        self._thread: threading.Thread | None = None

    @property
    def armed(self) -> bool:
        return self._armed

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="CocPresenceMonitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    def arm(self) -> None:
        self._missing_count = 0
        self._armed = True

    def disarm(self) -> None:
        self._missing_count = 0
        self._armed = False

    def _run(self) -> None:
        while not self._stop.is_set():
            snapshot = self.detector.snapshot()
            self.on_snapshot(snapshot)
            if self._armed and not snapshot.present:
                self._missing_count += 1
                if self._missing_count >= self.missing_tolerance:
                    self._armed = False
                    self._missing_count = 0
                    self.on_lost(snapshot)
            elif snapshot.present:
                self._missing_count = 0
            self._stop.wait(self.interval)
