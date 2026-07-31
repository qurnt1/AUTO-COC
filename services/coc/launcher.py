# -*- coding: utf-8 -*-
"""Launch strategies for the configured CoC target."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

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
            return CocLaunchResult(True, already_present=True, message="CoC est déjà détecté.")
        configured_target = profile.launcher
        if configured_target is None:
            return CocLaunchResult(False, message="Aucun lanceur CoC configuré.")
        target = self._resolve_target(configured_target)
        if not target.exists():
            return CocLaunchResult(False, message=f"Le lanceur CoC est introuvable : {configured_target}")
        try:
            if target.suffix.casefold() == ".exe":
                process = subprocess.Popen([str(target)], cwd=str(target.parent), close_fds=True)
                return CocLaunchResult(True, pid=process.pid, message="Lancement CoC demandé.")
            if os.name == "nt":
                os.startfile(str(target))
                return CocLaunchResult(True, message="Lancement CoC demandé via le raccourci Windows.")
            return CocLaunchResult(False, message="Le lancement CoC est disponible sur Windows uniquement.")
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
