# -*- coding: utf-8 -*-
"""Background safeguard that stops playback when CoC disappears."""

from __future__ import annotations

import threading
from collections.abc import Callable

from services.coc.detector import CocDetector
from services.coc.models import CocPresence
from utils.logger import get_logger


class CocPresenceMonitor:
    """Poll CoC without blocking the Qt event loop."""

    def __init__(
        self,
        detector: CocDetector,
        interval: float,
        missing_tolerance: int,
        on_snapshot: Callable[[CocPresence], None],
        on_lost: Callable[[CocPresence], None],
        on_error: Callable[[str], None] | None = None,
    ):
        self.detector = detector
        self.interval = max(0.25, interval)
        self.missing_tolerance = max(1, missing_tolerance)
        self.on_snapshot = on_snapshot
        self.on_lost = on_lost
        self.on_error = on_error
        self.log = get_logger()
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._armed = False
        self._missing_count = 0
        self._thread: threading.Thread | None = None

    @property
    def armed(self) -> bool:
        with self._state_lock:
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
        with self._state_lock:
            self._missing_count = 0
            self._armed = True

    def disarm(self) -> None:
        with self._state_lock:
            self._missing_count = 0
            self._armed = False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                snapshot = self.detector.snapshot()
            except Exception as exc:
                self._notify_error(f"snapshot : {exc}")
                self._stop.wait(self.interval)
                continue

            self._notify_snapshot(snapshot)
            if snapshot.error:
                self._notify_error(snapshot.error)
            else:
                self._update_presence(snapshot)
            self._stop.wait(self.interval)

    def _update_presence(self, snapshot: CocPresence) -> None:
        lost = False
        with self._state_lock:
            if self._armed and not snapshot.present:
                self._missing_count += 1
                if self._missing_count >= self.missing_tolerance:
                    self._armed = False
                    self._missing_count = 0
                    lost = True
            elif snapshot.present:
                self._missing_count = 0
        if lost:
            self._notify_lost(snapshot)

    def _notify_snapshot(self, snapshot: CocPresence) -> None:
        try:
            self.on_snapshot(snapshot)
        except Exception as exc:
            self._notify_error(f"callback présence : {exc}")

    def _notify_lost(self, snapshot: CocPresence) -> None:
        try:
            self.on_lost(snapshot)
        except Exception as exc:
            self._notify_error(f"callback safeguard : {exc}")

    def _notify_error(self, detail: str) -> None:
        self.log.error("Monitoring CoC indisponible : %s", detail)
        if self.on_error:
            try:
                self.on_error(detail)
            except Exception as exc:
                self.log.error("Callback erreur monitoring CoC impossible : %s", exc)
