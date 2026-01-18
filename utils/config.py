# -*- coding: utf-8 -*-
"""
Macro COC v3.0 — Utils / Config

Gestion des fichiers de configuration CSV et JSON avec écriture atomique.
"""

import csv
import json
import os
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger


def get_iso_utc_now() -> str:
    """Retourne l'heure actuelle en ISO 8601 UTC avec 'Z'."""
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


# =========================
#     CSV Configuration
# =========================

def read_params_csv(path: Path) -> Dict[str, str]:
    """
    Lecture stricte du CSV {parameter_name;parameter_value}.
    
    Format attendu:
    parameter_name;parameter_value
    key1;value1
    key2;value2
    
    Args:
        path: Chemin vers le fichier CSV
        
    Returns:
        Dictionnaire des paramètres
    """
    log = get_logger()
    params: Dict[str, str] = {}
    EXPECTED_HEADERS = {"parameter_name", "parameter_value"}
    
    if not path.exists():
        return params
    
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter=";")
            
            # Lire l'en-tête
            try:
                header_row = next(reader)
                headers = {h.strip().lower() for h in header_row}
            except StopIteration:
                log.warning(f"CSV {path.name} est vide.")
                return params
            
            # Valider l'en-tête
            if not EXPECTED_HEADERS.issubset(headers):
                log.warning(f"CSV {path.name} a un en-tête invalide. Attendu: {EXPECTED_HEADERS}")
                f.seek(0)  # Réinitialiser pour lecture legacy
            
            # Lire les données
            for i, row in enumerate(reader, 1):
                if not row or len(row) < 2:
                    continue
                k, v = row[0].strip(), row[1].strip()
                
                # Ignorer l'en-tête si présente
                if i == 1 and k.lower() == "parameter_name":
                    continue
                
                if not k:
                    continue
                
                params[k] = v
                
    except Exception as e:
        log.error(f"Erreur de lecture de {path}: {e}")
    
    return params


def write_params_csv(path: Path, params: Dict[str, str]) -> bool:
    """
    Écriture atomique du CSV.
    
    Args:
        path: Chemin vers le fichier CSV
        params: Dictionnaire des paramètres
        
    Returns:
        True si l'écriture a réussi
    """
    log = get_logger()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".csv.tmp.{os.getpid()}")
    
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["parameter_name", "parameter_value"])
            for k in sorted(params.keys()):
                w.writerow([k, str(params[k])])
        
        # Opération atomique
        os.replace(tmp_path, path)
        log.info(f"Paramètres sauvegardés (atomique) dans {path.name}")
        return True
        
    except Exception as e:
        log.error(f"Échec de l'écriture atomique CSV: {e}")
        if tmp_path.exists():
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return False


# =========================
#     JSON Macros
# =========================

def get_macro_hash(steps: List[dict]) -> str:
    """Calcule le SHA-1 stable du contenu d'une macro."""
    try:
        data = json.dumps(steps, sort_keys=True, ensure_ascii=False).encode('utf-8')
        return hashlib.sha1(data).hexdigest()
    except Exception:
        return ""


def read_macro_file(path: Path) -> Tuple[str, List[dict], str, str]:
    """
    Lit le JSON, retourne (name, steps, sha1, updated_at).
    
    Args:
        path: Chemin vers le fichier JSON
        
    Returns:
        Tuple (nom, steps, sha1, updated_at)
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        name = data.get("name", path.stem)
        steps = data.get("steps", [])
        sha1 = data.get("sha1", get_macro_hash(steps))
        updated_at = data.get("updated_at", get_iso_utc_now())
        return name, steps, sha1, updated_at
    except Exception:
        return path.stem, [], "", ""


def write_macro_file(
    path: Path, 
    name: str, 
    steps: List[dict],
    current_hash: Optional[str] = None
) -> bool:
    """
    Écriture atomique du JSON macro, avec vérification de hash.
    
    Args:
        path: Chemin vers le fichier JSON
        name: Nom de la macro
        steps: Liste des événements
        current_hash: Hash actuel pour comparaison (skip si identique)
        
    Returns:
        True si une écriture a eu lieu, False sinon
    """
    log = get_logger()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    new_hash = get_macro_hash(steps)
    
    # Ne pas réécrire si le hash n'a pas changé
    if current_hash and new_hash == current_hash:
        log.info(f"Macro '{name}' non modifiée (hash identique), écriture annulée.")
        return False
    
    data = {
        "name": name,
        "updated_at": get_iso_utc_now(),
        "sha1": new_hash,
        "steps": steps
    }
    
    tmp_path = path.with_suffix(f".json.tmp.{os.getpid()}")
    try:
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        os.replace(tmp_path, path)
        log.info(f"Macro '{name}' sauvegardée (atomique) dans {path.name}")
        return True
    except Exception as e:
        log.error(f"Échec de l'écriture atomique JSON: {e}")
        if tmp_path.exists():
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return False


def read_macro_meta(path: Path) -> Tuple[int, float]:
    """Retourne (nb_evts, duree_sec) depuis le fichier."""
    try:
        _, steps, _, _ = read_macro_file(path)
        duration = sum(max(0.0, float(ev.get("t", 0.0))) for ev in steps)
        return len(steps), duration
    except Exception:
        return 0, 0.0


# =========================
#     Macro Listing
# =========================

_SAFE_NAME_RE = re.compile(r"[^0-9a-zA-ZàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ _\-\.\(\)]")


def natural_sort_key(s: str) -> List[Any]:
    """Clé de tri pour le tri 'naturel'."""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r'([0-9]+)', s)
    ]


def sanitize_macro_name(name: str) -> str:
    """Nettoie un nom de macro pour le système de fichiers."""
    name = name.strip()
    name = _SAFE_NAME_RE.sub("_", name)
    return name or "Macro"


def list_macros(
    macros_dir: Path,
    legacy_macro: Optional[Path] = None,
    protected_names: Optional[List[str]] = None
) -> List[Tuple[str, Path]]:
    """
    Liste les macros, triées par nom naturel.
    Les macros protégées sont toujours en premier.
    
    Args:
        macros_dir: Dossier contenant les macros JSON
        legacy_macro: Chemin vers l'ancienne macro (compat)
        protected_names: Liste des noms de macros protégées
        
    Returns:
        Liste de tuples (nom, chemin)
    """
    macros_dir.mkdir(parents=True, exist_ok=True)
    items: List[Tuple[str, Path]] = []
    
    if legacy_macro and legacy_macro.exists():
        items.append((legacy_macro.stem, legacy_macro))
    
    for p in macros_dir.glob("*.json"):
        items.append((p.stem, p))
    
    # Séparer les macros protégées
    protected = protected_names or []
    protected_lower = [n.strip().lower() for n in protected]
    
    protected_items = []
    other_items = []
    
    for n, p in items:
        n_lower = n.strip().lower()
        if n_lower in protected_lower:
            protected_items.append((n, p))
        else:
            other_items.append((n, p))
    
    # Tri naturel seulement pour les autres items
    other_items.sort(key=lambda item: natural_sort_key(item[0]))
    
    # Reconstruire la liste (protégées en premier)
    final_items = protected_items + other_items
    
    # Déduplications
    seen, out = set(), []
    for n, p in final_items:
        if n not in seen:
            out.append((n, p))
            seen.add(n)
    
    return out


def macro_path_from_name(macros_dir: Path, name: str) -> Path:
    """Génère le chemin de fichier pour un nom de macro."""
    return macros_dir / f"{sanitize_macro_name(name)}.json"


# =========================
#     Helpers
# =========================

def as_bool(v: Optional[str], default: bool = False) -> bool:
    """Convertit une valeur en booléen."""
    if v is None:
        return default
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "on", "t", "vrai"}


def fmt_seconds(sec: float) -> str:
    """Formate des secondes en mm:ss."""
    sec = max(0, int(round(sec)))
    m, s = divmod(sec, 60)
    return f"{m:02d}:{s:02d}"


def fmt_duration_for_list(d: float) -> str:
    """Durée affichée dans la liste ; 0 -> 'Non enregistrée'."""
    if d <= 0.0:
        return "Non enregistrée"
    return fmt_seconds(d)
