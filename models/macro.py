# -*- coding: utf-8 -*-
"""
AUTO-COC macro models

Définition des structures de données pour les macros utilisant dataclasses.
Remplace les dictionnaires utilisés dans v2.1.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class StepType(Enum):
    """Types d'événements dans une macro."""
    MOUSE_MOVE = auto()
    MOUSE_CLICK = auto()
    SCROLL = auto()
    KEY_DOWN = auto()
    KEY_UP = auto()
    NOP = auto()  # No-operation (pour les tests)

    @classmethod
    def from_string(cls, s: str) -> "StepType":
        """Convertit une chaîne en StepType."""
        mapping = {
            "mouse_move": cls.MOUSE_MOVE,
            "mouse_click": cls.MOUSE_CLICK,
            "scroll": cls.SCROLL,
            "key_down": cls.KEY_DOWN,
            "key_up": cls.KEY_UP,
            "nop": cls.NOP,
        }
        return mapping.get(s.lower(), cls.NOP)

    def to_string(self) -> str:
        """Convertit le StepType en chaîne."""
        return self.name.lower()


@dataclass
class Step:
    """
    Un événement individuel dans une macro.
    
    Attributes:
        time_delta: Temps d'attente avant cet événement (secondes)
        step_type: Type d'événement
        data: Données spécifiques à l'événement (coordonnées, touches, etc.)
    """
    time_delta: float
    step_type: StepType
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire (compatible v2.1)."""
        return {
            "t": self.time_delta,
            "type": self.step_type.to_string(),
            "data": dict(self.data)
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Step":
        """Crée un Step depuis un dictionnaire (compatible v2.1)."""
        return cls(
            time_delta=float(d.get("t", 0.0)),
            step_type=StepType.from_string(d.get("type", "nop")),
            data=dict(d.get("data", {}))
        )


@dataclass
class Macro:
    """
    Représentation complète d'une macro.
    
    Attributes:
        name: Nom de la macro
        steps: Liste des événements
        sha1: Hash SHA1 du contenu (pour vérification d'écriture)
        updated_at: Date de dernière modification (ISO 8601)
    """
    name: str = "Nouvelle Macro"
    steps: List[Step] = field(default_factory=list)
    sha1: str = ""
    updated_at: str = ""

    def __post_init__(self):
        """Calcule le hash et la date si non fournis."""
        if not self.sha1:
            self.sha1 = self.compute_hash()
        if not self.updated_at:
            self.updated_at = self._get_iso_utc_now()

    @staticmethod
    def _get_iso_utc_now() -> str:
        """Retourne l'heure actuelle en ISO 8601 UTC."""
        return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    def compute_hash(self) -> str:
        """Calcule le SHA1 stable du contenu."""
        try:
            steps_dicts = [s.to_dict() for s in self.steps]
            data = json.dumps(steps_dicts, sort_keys=True, ensure_ascii=False).encode('utf-8')
            return hashlib.sha1(data).hexdigest()
        except Exception:
            return ""

    def duration(self) -> float:
        """Durée totale de la macro en secondes."""
        return sum(max(0.0, step.time_delta) for step in self.steps)

    def event_count(self) -> int:
        """Nombre d'événements."""
        return len(self.steps)

    def is_empty(self) -> bool:
        """Vérifie si la macro est vide."""
        return len(self.steps) == 0

    def clear(self) -> None:
        """Vide la macro."""
        self.steps = []
        self.sha1 = self.compute_hash()
        self.updated_at = self._get_iso_utc_now()

    def set_steps(self, steps: List[Step]) -> None:
        """Remplace les steps et recalcule le hash."""
        self.steps = list(steps)
        self.sha1 = self.compute_hash()
        self.updated_at = self._get_iso_utc_now()

    def set_steps_from_dicts(self, dicts: List[Dict[str, Any]]) -> None:
        """Remplace les steps depuis une liste de dictionnaires (v2.1 compat)."""
        self.steps = [Step.from_dict(d) for d in dicts]
        self.sha1 = self.compute_hash()
        self.updated_at = self._get_iso_utc_now()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit en dictionnaire pour sérialisation JSON.
        Compatible avec le format v2.1.
        """
        return {
            "name": self.name,
            "updated_at": self.updated_at,
            "sha1": self.sha1,
            "steps": [step.to_dict() for step in self.steps]
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Macro":
        """Crée une Macro depuis un dictionnaire (format v2.1)."""
        steps = [Step.from_dict(s) for s in d.get("steps", [])]
        macro = cls(
            name=d.get("name", "Macro"),
            steps=steps,
            sha1=d.get("sha1", ""),
            updated_at=d.get("updated_at", "")
        )
        # Recalcule le hash si non fourni
        if not macro.sha1:
            macro.sha1 = macro.compute_hash()
        return macro

    def to_json(self, indent: int = 2) -> str:
        """Sérialise en JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "Macro":
        """Désérialise depuis JSON."""
        return cls.from_dict(json.loads(json_str))

    def has_changed(self, other_hash: str) -> bool:
        """Vérifie si la macro a changé par rapport à un hash de référence."""
        return self.sha1 != other_hash
