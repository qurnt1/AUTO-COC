# -*- coding: utf-8 -*-
"""Presentation models shared by the PyQt6 pages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MacroSummary:
    name: str
    events: int
    duration: float
    kind: str = "user"
    description: str = ""
    updated_at: str = ""

    @property
    def protected(self) -> bool:
        return self.kind == "system"

    @property
    def editable(self) -> bool:
        return self.kind == "user"
