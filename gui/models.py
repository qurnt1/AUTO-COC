# -*- coding: utf-8 -*-
"""Presentation models shared by the PyQt6 pages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MacroSummary:
    name: str
    events: int
    duration: float
    updated_at: str = ""
