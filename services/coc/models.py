# -*- coding: utf-8 -*-
"""Domain models for the CoC runtime integration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def _split_values(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split("|") if part.strip())


@dataclass
class CocLaunchProfile:
    """Persisted launch and detection rules for the local CoC setup."""

    launcher_path: str = ""
    process_names: tuple[str, ...] = field(default_factory=tuple)
    window_titles: tuple[str, ...] = ("Clash of Clans",)
    process_path_hint: str = ""
    startup_timeout: float = 60.0
    detection_interval: float = 1.0
    missing_tolerance: int = 3

    @property
    def is_configured(self) -> bool:
        return bool(self.launcher_path or self.process_names or self.window_titles)

    @property
    def launcher(self) -> Path | None:
        path = self.launcher_path.strip()
        return Path(path) if path else None

    @classmethod
    def from_params(cls, params: dict[str, str]) -> "CocLaunchProfile":
        try:
            startup_timeout = max(5.0, float(params.get("coc_startup_timeout", "60")))
        except ValueError:
            startup_timeout = 60.0
        try:
            interval = min(5.0, max(0.25, float(params.get("coc_detection_interval", "1"))))
        except ValueError:
            interval = 1.0
        try:
            tolerance = min(10, max(1, int(params.get("coc_missing_tolerance", "3"))))
        except ValueError:
            tolerance = 3
        return cls(
            launcher_path=params.get("coc_path", "").strip(),
            process_names=_split_values(params.get("coc_process_names", "")),
            window_titles=_split_values(params.get("coc_window_titles", "Clash of Clans")) or ("Clash of Clans",),
            process_path_hint=params.get("coc_process_path_hint", "").strip(),
            startup_timeout=startup_timeout,
            detection_interval=interval,
            missing_tolerance=tolerance,
        )

    def apply_to_params(self, params: dict[str, str]) -> None:
        params["coc_path"] = self.launcher_path.strip()
        params["coc_process_names"] = "|".join(self.process_names)
        params["coc_window_titles"] = "|".join(self.window_titles)
        params["coc_process_path_hint"] = self.process_path_hint.strip()
        params["coc_startup_timeout"] = str(int(self.startup_timeout))
        params["coc_detection_interval"] = f"{self.detection_interval:g}"
        params["coc_missing_tolerance"] = str(self.missing_tolerance)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class CocPresence:
    """One point-in-time observation of the CoC process/window state."""

    present: bool
    process_found: bool
    window_found: bool
    pids: tuple[int, ...] = field(default_factory=tuple)
    process_names: tuple[str, ...] = field(default_factory=tuple)
    window_titles: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""
    error: str = ""
