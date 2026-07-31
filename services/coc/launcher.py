# -*- coding: utf-8 -*-
"""Launch strategies for the configured CoC target."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

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
        target = profile.launcher
        if target is None:
            return CocLaunchResult(False, message="Aucun lanceur CoC configuré.")
        if not target.exists():
            return CocLaunchResult(False, message=f"Le lanceur CoC est introuvable : {target}")
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
