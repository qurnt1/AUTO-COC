# -*- coding: utf-8 -*-
"""Launch strategies for the configured CoC target."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from services.coc.detector import CocDetector
from services.coc.models import CocLaunchProfile, CocPresence


@dataclass(frozen=True)
class CocLaunchResult:
    started: bool
    already_present: bool = False
    pid: int | None = None
    message: str = ""


class CocLauncher:
    """Start CoC without claiming success before detection confirms it."""

    def launch(self, profile: CocLaunchProfile, detector: CocDetector) -> CocLaunchResult:
        current: CocPresence = detector.snapshot()
        if current.present:
            return CocLaunchResult(True, already_present=True, message="CoC is already detected.")
        configured_target = profile.launcher
        discovered = configured_target is None
        configured_target = configured_target or self.discover_target()
        if configured_target is None:
            return CocLaunchResult(
                False,
                message="No Clash of Clans shortcut was found. Configure a launcher in Settings.",
            )
        target = self._resolve_target(configured_target)
        if not target.exists() or not target.is_file():
            return CocLaunchResult(False, message=f"Le lanceur CoC est introuvable : {configured_target}")
        try:
            if target.suffix.casefold() == ".exe":
                process = subprocess.Popen([str(target)], cwd=str(target.parent), close_fds=True)
                return CocLaunchResult(True, pid=process.pid, message="CoC launch requested.")
            if target.suffix.casefold() == ".lnk" and os.name == "nt":
                os.startfile(str(target))
                source = "auto-detected Windows shortcut" if discovered else "Windows shortcut"
                return CocLaunchResult(True, message=f"CoC launch requested through the {source}.")
            return CocLaunchResult(False, message="The CoC launcher must be an .exe or Windows .lnk file.")
        except OSError as exc:
            return CocLaunchResult(False, message=str(exc))

    @staticmethod
    def _resolve_target(configured_target: Path) -> Path:
        """Accept Explorer's extension-hidden Start Menu shortcut paths."""
        if configured_target.exists() or configured_target.suffix.casefold() == ".lnk":
            return configured_target
        candidates = [Path(f"{configured_target}.lnk")]
        if configured_target.suffix:
            candidates.append(configured_target.with_suffix(".lnk"))
        return next((candidate for candidate in candidates if candidate.exists()), configured_target)

    @staticmethod
    def discover_target(roots: Iterable[Path] | None = None) -> Path | None:
        """Find a CoC shortcut in standard Windows shortcut locations."""
        if roots is None:
            candidates = []
            for variable, suffix in (
                ("APPDATA", Path("Microsoft/Windows/Start Menu/Programs")),
                ("PROGRAMDATA", Path("Microsoft/Windows/Start Menu/Programs")),
                ("PUBLIC", Path("Desktop")),
            ):
                base = os.environ.get(variable, "").strip()
                if base:
                    candidates.append(Path(base) / suffix)
            candidates.append(Path.home() / "Desktop")
            roots = candidates

        matches: list[Path] = []
        for root in roots:
            if not root.exists():
                continue
            try:
                matches.extend(
                    path
                    for path in root.rglob("*.lnk")
                    if "clash of clans" in path.stem.casefold()
                )
            except OSError:
                continue
        return min(matches, key=lambda path: (len(path.parts), len(str(path)))) if matches else None
