# -*- coding: utf-8 -*-
"""
Macro COC — v2.1 (Refactoring Senior)

Application de macro-recording et playback avec interface customtkinter, 
contrôle à distance via Telegram (polling), et gestion robuste.

"""

# =========================
#         Imports
# =========================
import argparse
import csv
import json
import os
import re
import threading
import time
import subprocess
import sys
import webbrowser
import logging
import logging.handlers
import traceback
import hashlib
import tempfile
import shutil
import ctypes
import queue
from io import BytesIO
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any, Set
from enum import Enum, auto
from datetime import datetime, timezone
from tkinter import messagebox, filedialog

# Libs tierces (incluses dans le code utilisateur)
import customtkinter as ctk
import requests
from requests.adapters import HTTPAdapter, Retry
from pynput import mouse, keyboard as pyn_keyboard
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyController, Key
import keyboard  # hotkeys globaux

# Libs optionnelles (vérifiées à l'exécution)
try:
    from PIL import Image, ImageTk, ImageGrab
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image, ImageTk, ImageGrab = None, None, None

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    mss = None

# Pour les tests unitaires
try:
    from unittest.mock import Mock, patch
except ImportError:
    Mock, patch = None, None

# =========================
#      Configuration
# =========================
APP_VERSION = "2.1.0"
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
RECHARGER_MACRO_NAME = "Recharger COC"  # Macro protégée
VALIDER_MACRO_NAME = "Valider arrivée" # NOUVELLE macro protégée

# --- Chemins ---
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
ICON_PATH = CONFIG_DIR / "icon.ico"
ICON_PNG_PATH = CONFIG_DIR / "image.png"
PARAMS_PATH = CONFIG_DIR / "data.csv"
MACROS_DIR = CONFIG_DIR / "macros"
LEGACY_MACRO = CONFIG_DIR / "macro.json"  # compat éventuelle
GUIDE_HTML_PATH = CONFIG_DIR / "guide_telegram.html"
LOG_PATH = CONFIG_DIR / "app.log"

# --- Machine à états (Req 1) ---
class State(Enum):
    IDLE = auto()
    RECORDING = auto()
    PLAYING = auto()

# --- Verrou global (Req 1) ---
# Verrou pour les transitions d'état, l'accès au modèle,
# et les opérations de l'API Telegram.
BASE_LOCK = threading.RLock()

# =========================
#    Logging (Req 3)
# =========================
log = logging.getLogger("MacroApp")

def setup_logging():
    """Configure le logger principal avec rotation."""
    # TÂCHE 1: Nettoyage des anciens logs au démarrage
    try:
        # Assure que LOG_PATH.parent existe avant de lister
        ensure_dirs()
        # Utilise print car le logger n'est pas encore prêt
        print(f"Nettoyage des logs > {24}h dans {LOG_PATH.parent}...")
        clean_old_logs(LOG_PATH.parent, "app.log*", 24)
    except Exception as e:
        print(f"Erreur (pré-logging) clean_old_logs: {e}")
    # ---
    ensure_dirs()
    log.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Handler 1: Fichier rotatif (1Mo * 3 backups)
    try:
        handler_file = logging.handlers.RotatingFileHandler(
            LOG_PATH,
            maxBytes=1_048_576,  # 1 Mo
            backupCount=3,
            encoding="utf-8"
        )
        handler_file.setFormatter(formatter)
        log.addHandler(handler_file)
    except Exception as e:
        print(f"Erreur: Impossible de créer le logger de fichier à {LOG_PATH}: {e}")

    # Handler 2: Console (pour debug)
    handler_console = logging.StreamHandler(sys.stdout)
    handler_console.setFormatter(formatter)
    log.addHandler(handler_console)

def clean_old_logs(log_dir: Path, pattern: str, max_age_hours: int = 24):
    """Supprime les fichiers logs plus anciens que max_age_hours."""
    try:
        cutoff = time.time() - (max_age_hours * 3600)
        # Utilise glob pour gérer les wildcards (ex: app.log*)
        search_path = log_dir / pattern
        # Note: glob.glob requiert un string, surtout pour les wildcards
        for log_file in glob(str(search_path)):
            try:
                p = Path(log_file)
                if not p.is_file():
                    continue
                
                file_time = p.stat().st_mtime
                if file_time < cutoff:
                    # Log ici, car setup_logging a déjà été appelé
                    log.info(f"Nettoyage ancien log: {p.name}")
                    os.remove(p)
            except Exception as e:
                log.warning(f"Échec suppression ancien log {log_file}: {e}")
    except Exception as e:
        log.error(f"Erreur lors du nettoyage des logs: {e}")
def log_uncaught_exception(exc_type, exc_value, exc_traceback):
    """Callback pour sys.excepthook pour logger les crashs."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    log.error("EXCEPTION NON CAPTURÉE", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = log_uncaught_exception

# =========================
#    Utils (Req 4, 5, 7, 10)
# =========================
def ensure_dirs():
    """Crée les dossiers de configuration requis."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    MACROS_DIR.mkdir(parents=True, exist_ok=True)

def apply_window_icon(win):
    """Applique icon.ico à n'importe quelle fenêtre (root ou toplevel)."""
    if not ICON_PATH.exists():
        log.warning(f"Icône non trouvée à {ICON_PATH}")
        return
    p = str(ICON_PATH.resolve())
    try:
        win.iconbitmap(p)
    except Exception as e:
        log.error(f"Erreur (iconbitmap) : {e}")
        try:
            win.wm_iconbitmap(p)
        except Exception as e2:
            log.error(f"Erreur (wm_iconbitmap) : {e2}")

def get_iso_utc_now() -> str:
    """Retourne l'heure actuelle en ISO 8601 UTC avec 'Z'."""
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

def natural_sort_key(s: str) -> List[Any]:
    """Clé de tri pour le tri "naturel" (Req 4)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'([0-9]+)', s)]

_SAFE_NAME_RE = re.compile(r"[^0-9a-zA-ZàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ _\-\.\(\)]")
def sanitize_macro_name(name: str) -> str:
    name = name.strip()
    name = _SAFE_NAME_RE.sub("_", name)
    return name or "Macro"

def macro_path_from_name(name: str) -> Path:
    return MACROS_DIR / f"{sanitize_macro_name(name)}.json"

def as_bool(v: Optional[str], default=False) -> bool:
    if v is None:
        return default
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "on", "t", "vrai"}

def play_start_tone():
    try:
        import winsound
        winsound.Beep(1000, 200)
    except Exception:
        pass # Échoue silencieusement si winsound n'est pas dispo

def fmt_seconds(sec: float) -> str:
    sec = max(0, int(round(sec)))
    m, s = divmod(sec, 60)
    return f"{m:02d}:{s:02d}"

def fmt_dur_for_list(d: float) -> str:
    """Durée affichée dans la liste de gauche ; 0 -> 'Non enregistrée'."""
    if d <= 0.0:
        return "Non enregistrée"
    return fmt_seconds(d)

def grab_screenshot_png_bytes() -> Optional[bytes]:
    """Retourne une capture d'écran en PNG (bytes) ou None si impossible."""
    # 1) Pillow ImageGrab si dispo
    if PIL_AVAILABLE and ImageGrab is not None:
        try:
            img = ImageGrab.grab()  # écran principal
            bio = BytesIO()
            img.save(bio, format="PNG")
            return bio.getvalue()
        except Exception as e:
            log.warning(f"Échec ImageGrab (Pillow): {e}")
            pass
    # 2) Fallback mss si installé
    if MSS_AVAILABLE and mss is not None:
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[0]  # tout l'écran virtuel
                raw = sct.grab(monitor)
                if Image is None: # On a besoin de PIL pour convertir
                    log.warning("MSS a capturé, mais PIL manque pour convertir.")
                    return None
                img = Image.frombytes("RGB", (raw.width, raw.height), raw.rgb)
                bio = BytesIO()
                img.save(bio, format="PNG")
                return bio.getvalue()
        except Exception as e:
            log.warning(f"Échec capture MSS: {e}")
            pass
    log.error("Aucune méthode de capture d'écran n'a fonctionné.")
    return None

def resolve_exe_from_path(path_str: str) -> Optional[str]:
    """
    À partir d'un .exe ou d'un .lnk, tente de déduire le nom d'exe à fermer.
    (Req 5)
    """
    if not path_str:
        return None
    p = Path(path_str)
    s = str(p)
    if s.lower().endswith(".exe"):
        return p.name
    if os.name == "nt" and s.lower().endswith(".lnk"):
        try:
            # Échappement correct des quotes pour PowerShell (Req 5)
            escaped = s.replace("'", "''")
            ps = f"(New-Object -ComObject WScript.Shell).CreateShortcut('{escaped}').TargetPath"
            cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps]
            
            # Utilise subprocess.run avec timeout (Req 5)
            # 5s est raisonnable pour une query COM locale
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                encoding='utf-8', 
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            out = result.stdout.strip()
            if result.returncode == 0 and out and out.lower().endswith(".exe"):
                return Path(out).name
            elif result.returncode != 0:
                log.warning(f"PowerShell .lnk resolve a échoué: {result.stderr}")
        except subprocess.TimeoutExpired:
            log.error("Timeout lors de la résolution du .lnk via PowerShell.")
        except Exception as e:
            log.error(f"Erreur résolution .lnk: {e}")
    return None

def kill_process_by_name(exe_name: str):
    """Tente de tuer un processus par son nom (Req 5)."""
    if os.name != 'nt' or not exe_name:
        return
    
    log.info(f"Tentative de fermeture de '{exe_name}'...")
    killed = False
    
    # Méthode 1: taskkill (préférée)
    try:
        cmd_taskkill = ["taskkill", "/F", "/IM", exe_name]
        result = subprocess.run(
            cmd_taskkill, 
            capture_output=True, 
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        # taskkill retourne 0 ou 128 (non trouvé) si succès
        if result.returncode in (0, 128):
            log.info(f"taskkill pour '{exe_name}' exécuté (code {result.returncode}).")
            killed = True # On suppose que c'est bon
        else:
            log.warning(f"taskkill a échoué (code {result.returncode}): {result.stderr.decode('cp850', errors='ignore')}")
    except Exception as e:
        log.error(f"Erreur taskkill: {e}")

    # Méthode 2: wmic (fallback demandé, Req 5)
    # On le tente même si taskkill a réussi, pour être sûr (ou si taskkill a échoué)
    try:
        cmd_wmic = ["wmic", "process", "where", f"name='{exe_name}'", "call", "terminate"]
        result = subprocess.run(
            cmd_wmic, 
            capture_output=True, 
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        if result.returncode == 0:
            log.info(f"WMIC terminate pour '{exe_name}' exécuté.")
        else:
            log.warning(f"WMIC a échoué (code {result.returncode}): {result.stderr.decode('cp850', errors='ignore')}")
    except Exception as e:
        log.error(f"Erreur WMIC: {e}")

# --- Accessibilité (Req 4) ---
def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    """Convertit #RRGGBB en (R, G, B)."""
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def get_luminance(r_8b, g_8b, b_8b) -> float:
    """Calcule la luminance relative (WCAG)."""
    vals = []
    for v in (r_8b, g_8b, b_8b):
        v = v / 255.0
        v = v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
        vals.append(v)
    return 0.2126 * vals[0] + 0.7152 * vals[1] + 0.0722 * vals[2]

def get_contrast_ratio(hex_fg: str, hex_bg: str) -> float:
    """Calcule le ratio de contraste (WCAG)."""
    try:
        lum1 = get_luminance(*hex_to_rgb(hex_fg))
        lum2 = get_luminance(*hex_to_rgb(hex_bg))
        if lum1 > lum2:
            return (lum1 + 0.05) / (lum2 + 0.05)
        return (lum2 + 0.05) / (lum1 + 0.05)
    except Exception:
        return 1.0

def ensure_contrast(fg: str, bg: str, target_ratio: float = 4.5) -> str:
    """
    Vérifie le contraste. Si insuffisant, retourne un fallback (blanc/noir).
    NOTE: C'est un exemple de l'implémentation. Le Thème est déjà compliant.
    """
    ratio = get_contrast_ratio(fg, bg)
    if ratio >= target_ratio:
        return fg
    
    # Fallback basique
    bg_lum = get_luminance(*hex_to_rgb(bg))
    if bg_lum > 0.5:
        return "#000000" # Fond clair -> texte noir
    else:
        return "#FFFFFF" # Fond sombre -> texte blanc


# =========================
#      Thème (Req 4)
# =========================
class Theme:
    HEADER_HEIGHT = 88
    ICON_PNG_SIZE = (140, 88)

    APP_BG = "#0b1220"
    HEADER_BG = "#0b1220"
    STATUS_BG = "#0b0f19"
    LEFT_CONTAINER_BG = "#0e1624"
    LEFT_HEADER_TEXT = "#e5e7eb"
    LEFT_ACTIONS_BG = "#132033"
    
    ROW_BG = "#152235"
    ROW_HOVER = "#1d2d45"
    ROW_SELECTED = "#2a4365"
    ROW_NAME_COLOR = "#e5e7eb"
    ROW_DUR_COLOR = "#93c5fd"
    
    CENTER_BG = "#111827"
    INFO_BG = "#0e1624"
    INFO_TEXT_MUTED = "#cbd5e1"
    
    BTN_PRIMARY_BG = "#22c55e"
    BTN_PRIMARY_HOVER = "#16a34a"
    BTN_STOP_BG = "#ef4444"
    BTN_STOP_HOVER = "#b91c1c"
    
    BTN_LAUNCH_BG = "#0ea5e9"
    BTN_LAUNCH_HOVER = "#0284c7"
    
    BTN_DISABLED_BG = "#374151"
    BTN_DISABLED_TEXT = "#9ca3af"

    CARD_BG = "#0b1220"
    DIVIDER = "#334155"
    
    # Couleurs de statut (Req 4)
    STATUS_OK = "#22c55e"
    STATUS_WARN = "#f59e0b"
    STATUS_ERROR = "#ef4444"
    
    # Vérification de conformité (Req 4)
    # Ratio (e5e7eb / 152235) = 13.1 (OK > 4.5)
    # Ratio (93c5fd / 152235) = 7.1 (OK > 4.5)
    TEXT_COMPLIANT = ensure_contrast(ROW_NAME_COLOR, ROW_BG)
    TEXT_MUTED_COMPLIANT = ensure_contrast(INFO_TEXT_MUTED, INFO_BG)


# =========================
#   I/O Atomique (Req 3)
# =========================

def read_params_csv(path: Path) -> Dict[str, str]:
    """Lecture stricte du CSV {parameter_name;parameter_value} (Req 3)."""
    params: Dict[str, str] = {}
    EXPECTED_HEADERS = {"parameter_name", "parameter_value"}
    
    if not path.exists():
        return params
        
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter=";")
            
            # 1. Lire l'en-tête
            try:
                header_row = next(reader)
                headers = {h.strip().lower() for h in header_row}
            except StopIteration:
                log.warning(f"CSV {path.name} est vide.")
                return params
            
            # 2. Valider l'en-tête
            if not EXPECTED_HEADERS.issubset(headers):
                log.warning(f"CSV {path.name} a un en-tête invalide. Attendu : {EXPECTED_HEADERS}. Trouvé : {headers}")
                # Tentative de lecture "legacy" (sans en-tête)
                f.seek(0)
                
            # 3. Lire les données
            for i, row in enumerate(reader, 1):
                if not row or len(row) < 2:
                    continue
                k, v = row[0].strip(), row[1].strip()
                
                # Ignorer l'en-tête si on l'a trouvée
                if i == 1 and EXPECTED_HEADERS.issubset(headers) and k.lower() == "parameter_name":
                    continue
                    
                if not k:
                    continue
                
                # Req 3: Logger les clés inconnues (ici, on charge tout,
                # mais la *validation* se fait au chargement dans l'App)
                params[k] = v
                
    except Exception as e:
        log.error(f"Erreur de lecture de {path}: {e}\n{traceback.format_exc()}")
    return params

def write_params_csv(path: Path, params: Dict[str, str]):
    """Écriture atomique du CSV (Req 3)."""
    ensure_dirs()
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
        
    except Exception as e:
        log.error(f"Échec de l'écriture atomique CSV: {e}\n{traceback.format_exc()}")
        if tmp_path.exists():
            try: os.remove(tmp_path)
            except Exception: pass

def get_macro_hash(steps: List[dict]) -> str:
    """Calcule le SHA-1 stable du contenu d'une macro (Req 3)."""
    try:
        # json.dumps avec sort_keys=True garantit un hash stable
        data = json.dumps(steps, sort_keys=True, ensure_ascii=False).encode('utf-8')
        return hashlib.sha1(data).hexdigest()
    except Exception:
        log.error("Échec du hachage de la macro.")
        return ""

def read_macro_file(path: Path) -> Tuple[str, List[dict], str, str]:
    """Lit le JSON, retourne (name, steps, sha1, updated_at)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        name = data.get("name", path.stem)
        steps = data.get("steps", [])
        sha1 = data.get("sha1", get_macro_hash(steps)) # Calcule si manque
        updated_at = data.get("updated_at", get_iso_utc_now())
        return name, steps, sha1, updated_at
    except Exception:
        return path.stem, [], "", ""

def write_macro_file(path: Path, name: str, steps: List[dict], 
                     current_hash: Optional[str] = None) -> bool:
    """
    Écriture atomique du JSON macro, avec vérification de hash (Req 3).
    Retourne True si une écriture a eu lieu, False sinon.
    """
    ensure_dirs()
    
    new_hash = get_macro_hash(steps)
    
    # Req 3: Ne pas réécrire si le hash n'a pas changé
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
        log.error(f"Échec de l'écriture atomique JSON: {e}\n{traceback.format_exc()}")
        if tmp_path.exists():
            try: os.remove(tmp_path)
            except Exception: pass
        return False

def list_macros() -> List[Tuple[str, Path]]:
    """Liste les macros, triées par nom naturel (Req 4).
    Les macros protégées (Recharger, Valider) sont toujours en premier."""
    ensure_dirs()
    items: List[Tuple[str, Path]] = []
    
    if LEGACY_MACRO.exists():
        items.append((LEGACY_MACRO.stem, LEGACY_MACRO))
        
    for p in MACROS_DIR.glob("*.json"):
        items.append((p.stem, p))

    # --- MODIFICATION: Séparer les macros protégées ---
    recharger_item = None
    valider_item = None
    other_items = []
    
    protected_recharger_lower = RECHARGER_MACRO_NAME.strip().lower()
    protected_valider_lower = VALIDER_MACRO_NAME.strip().lower()

    for n, p in items:
        n_lower = n.strip().lower()
        if n_lower == protected_recharger_lower:
            recharger_item = (n, p)
        elif n_lower == protected_valider_lower:
            valider_item = (n, p)
        else:
            other_items.append((n, p))

    # Tri naturel (Req 4) seulement pour les autres items
    other_items.sort(key=lambda item: natural_sort_key(item[0]))
    
    # Reconstruire la liste finale
    final_items = []
    if recharger_item:
        final_items.append(recharger_item)
    if valider_item:
        final_items.append(valider_item)
    final_items.extend(other_items)
    # --- FIN MODIFICATION ---
    
    seen, out = set(), []
    for n, p in final_items:
        if n not in seen:
            out.append((n, p))
            seen.add(n)
    return out

def read_macro_meta(path: Path) -> Tuple[int, float]:
    """Retourne (nb_evts, duree_sec) depuis le fichier."""
    try:
        _, steps, _, _ = read_macro_file(path)
        duration = sum(max(0.0, float(ev.get("t", 0.0))) for ev in steps)
        return len(steps), duration
    except Exception:
        return 0, 0.0

# =========================
#   Telegram Bridge (Req 2)
# =========================
class TelegramBridge:
    """
    Bridge Telegram robuste avec polling, rate-limiter, et backoff.
    """
    def __init__(self, token: str, chat_id: str, on_command: Callable):
        self.token = (token or "").strip()
        self.chat_id = None
        try:
            self.chat_id = int(str(chat_id).strip()) if chat_id else None
        except Exception:
            self.chat_id = None
            
        self._on_command = on_command
        self._base = f"https://api.telegram.org/bot{self.token}"
        self._offset = None
        
        # État du poller (Req 2)
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._running_flag = threading.Event() # Flag atomique

        # Session HTTP avec backoff (Req 2)
        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5, # backoff: 0.5s, 1s, 2s, 4s, 8s (capé à 5s plus bas)
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        # Cap du backoff (Req 2)
        Retry.BACKOFF_MAX = 5.0
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

        # Rate-limiter (Req 2: 1 jeton / 60ms)
        self._rate_limiter_queue = queue.Queue()
        self._rate_limiter_thread = threading.Thread(
            target=self._token_bucket, 
            daemon=True,
            name="TG-RateLimiter"
        )
        
        # Gestion d'UI
        self._last_controls_id: Optional[int] = None
        self._last_menu_id: Optional[int] = None
        # TÂCHE 3: Suppression de last_photo_message_id
        
        # Map pour callbacks > 64 octets (Req 7)
        self._callback_data_map: Dict[str, str] = {}

        if self.token:
            self._rate_limiter_thread.start()

    def _token_bucket(self):
        """Alimente le seau de jetons pour le rate-limiter."""
        while True:
            self._rate_limiter_queue.put(True)
            time.sleep(0.06) # 1 jeton / 60ms

    def is_configured(self) -> bool:
        return bool(self.token)

    def ready(self) -> bool:
        return bool(self.token and self.chat_id)

    def get_status(self) -> Tuple[str, str]:
        """Retourne (status_text, color_hex) pour l'UI (Req 4)."""
        if not self.is_configured():
            return "Token manquant", Theme.STATUS_ERROR
        if not self.chat_id:
            return "Chat ID manquant", Theme.STATUS_WARN
        if not self._running_flag.is_set():
            return "Poller arrêté", Theme.STATUS_WARN
        return "Connecté", Theme.STATUS_OK

    def start(self):
        """Démarre le poller (protégé par flag atomique, Req 2)."""
        
        # --- AJOUT ---
        # Bloquer ici jusqu'à ce que l'ancien thread (s'il existe) soit mort.
        # Ceci est critique pour éviter un 409 Conflict.
        # C'est sans danger car resume_telegram_poller()
        # appelle start() dans un thread séparé.
        if self._poll_thread and self._poll_thread.is_alive():
            current_thread_id = threading.current_thread().ident
            old_thread_id = self._poll_thread.ident
            if current_thread_id != old_thread_id:
                log.info(f"TG Bridge: Attente de la fin de l'ancien poller (TID: {old_thread_id})...")
                # Doit attendre la fin potentielle du long poll (max 25s + 5s buffer)
                self._poll_thread.join(timeout=30.0) 
                if self._poll_thread.is_alive():
                    log.warning("TG Bridge: L'ancien poller n'a pas pu être arrêté. Un conflit 409 est possible.")
        # --- FIN AJOUT ---

        if self._running_flag.is_set():
            log.warning("TG Bridge: Démarrage demandé mais déjà en cours.")
            return
        
        self._running_flag.set()
        self._stop_evt.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop, 
            daemon=True, 
            name="TG-Poller"
        )
        self._poll_thread.start()
        
        # Log plus robuste
        try:
            log.info(f"TG Bridge: Poller démarré (Nouveau TID: {self._poll_thread.ident}).")
        except Exception:
            log.info("TG Bridge: Poller démarré.")

    def stop(self):
        """Arrête le poller proprement (en signalant, sans joindre)."""
        if not self._running_flag.is_set():
            return
            
        try:
            self._running_flag.clear()
            self._stop_evt.set()
            log.info("TG Bridge: Signal d'arrêt envoyé au poller.")
        except Exception as e:
            log.error(f"Erreur lors de l'arrêt du poller TG: {e}")

    # ---------- API (Req 1, 2) ----------
    def _request(self, method: str, verb: str = "POST", timeout: int = 15, **kwargs) -> dict:
        """
        Wrapper centralisé pour les requêtes API.
        Gère le rate-limiter, le verrou global, et le backoff (via self.session).
        """
        # 1. Attendre le rate-limiter (Req 2)
        self._rate_limiter_queue.get()
        
        # 2. Verrouiller *uniquement* la vérification du flag d'arrêt
        with BASE_LOCK:
            if self._stop_evt.is_set(): # Ne pas envoyer si on s'arrête
                return {}
        
        # 3. Effectuer l'appel réseau HORS DU VERROU
        #    (requests.Session est thread-safe, ce n'est pas un problème)
        url = f"{self._base}/{method}"
        try:
            r = self.session.request(verb, url, timeout=timeout, **kwargs)
            r.raise_for_status() # Lève une exception pour 4xx/5xx
            return r.json()
        
        except requests.exceptions.HTTPError as e:
            # Géré par le backoff de self.session, mais on log l'échec final
            log.error(f"TG API Échec HTTP (après retries) pour {method}: {e}")
            try:
                log.error(f"Détail réponse: {e.response.text}")
            except Exception: pass
            
        except requests.exceptions.RequestException as e:
            # Erreurs réseau, timeouts, etc.
            log.error(f"TG API Erreur Réseau pour {method}: {e}")
            
        except Exception as e:
            log.error(f"TG API Erreur Inconnue pour {method}: {e}\n{traceback.format_exc()}")
            
        return {} # Échec

    def delete_message(self, message_id: int):
        self._request("deleteMessage", json={"chat_id": self.chat_id, "message_id": message_id})

    def discard_backlog(self):
        """Confirme tous les anciens updates et supprime les messages du chat."""
        if not self.is_configured(): return
        log.info("TG Bridge: Purge du backlog...")
        data = self._request("getUpdates", verb="GET", params={"timeout": 0, "limit": 100}, timeout=10)
        res = data.get("result", [])
        
        deleted_count = 0
        if not res:
            log.info("TG Bridge: Backlog vide.")
            return

        for up in res:
            self._offset = up.get("update_id", 0) + 1
            
            # --- TÂCHE 6: Purge des messages au démarrage ---
            msg = up.get("message") or up.get("edited_message") or up.get("callback_query", {}).get("message")
            # On vérifie le chat_id SEULEMENT s'il est déjà configuré
            if msg and self.chat_id and msg.get("chat", {}).get("id") == self.chat_id:
                try:
                    msg_id = msg.get("message_id")
                    if msg_id:
                        self.delete_message(msg_id)
                        deleted_count += 1
                except Exception as e:
                    log.warning(f"Erreur purge backlog (delete): {e}")
            # --- FIN TÂCHE 6 ---

        log.info(f"TG Bridge: Backlog purgé ({deleted_count} messages supprimés), offset réglé à {self._offset}")

    def send(self, text: str):
        if not self.ready(): return
        self._request("sendMessage", json={"chat_id": self.chat_id, "text": text})

    # TÂCHE 3: Remplacement de send_or_edit_photo
    def send_photo(self, png_bytes: bytes, caption: str = ""):
        """Envoie une nouvelle photo (remplace send_or_edit_photo)."""
        if not self.ready() or not png_bytes:
            return

        files = {"photo": ("screenshot.png", png_bytes, "image/png")}
        
        resp = self._request(
            "sendPhoto",
            data={"chat_id": str(self.chat_id), "caption": caption},
            files=files,
            timeout=30
        )
        
        new_msg_id = ((resp or {}).get("result") or {}).get("message_id")
        if new_msg_id:
            log.info(f"TG: sendPhoto réussi (Nouvel ID: {new_msg_id})")
        else:
            log.error(f"TG: sendPhoto a échoué. Réponse: {resp}")

    def send_animation(self, gif_bytes: bytes, caption: str = ""):
        """Envoie une animation (GIF) (Req 6)."""
        if not self.ready() or not gif_bytes:
            return
            
        resp = self._request(
            "sendAnimation",
            data={"chat_id": str(self.chat_id), "caption": caption},
            files={"animation": ("capture.gif", gif_bytes, "image/gif")},
            timeout=45
        )
        if not resp.get("ok"):
            log.error(f"TG: sendAnimation a échoué. Réponse: {resp}")

    def edit_message_reply_markup(self, message_id: int, reply_markup: Optional[dict] = None):
        """Modifie un clavier (ou le supprime si reply_markup=None) (Req 2)."""
        if not self.ready() or not message_id:
            return
        self._request("editMessageReplyMarkup", json={
            "chat_id": self.chat_id,
            "message_id": message_id,
            "reply_markup": reply_markup or {} # {} vide pour supprimer
        })

    # ---------- Claviers (Req 7) ----------
    def _controls_markup(self, coc_launched: bool = False):
        # TÂCHE 4: Bouton Lancer CoC dynamique
        coc_button_row = []
        if coc_launched:
            # Libellé "COC lancé ✅"
            coc_button_row = [{"text": "COC lancé ✅", "callback_data": "DUMMY_COC_STATUS"}]
        else:
            coc_button_row = [{"text": "Lancer CoC", "callback_data": "LAUNCH_COC"}]

        return {
            "inline_keyboard": [
                [{"text": "Paramètres ⚙️", "callback_data": "MENU"},
                 {"text": "Capture 📸", "callback_data": "CAPTURE"}],
                coc_button_row, # Ligne dynamique
                [{"text": "Lancer ✅", "callback_data": "GO"},
                 {"text": "Stop ❌", "callback_data": "STOP"}],
            ]
        }

    def replace_controls(self, text: str = "Commandes :", coc_launched: bool = False):
        if not self.ready(): return
        with BASE_LOCK:
            if self._last_controls_id:
                self.delete_message(self._last_controls_id)
            resp = self._request("sendMessage", json={"chat_id": self.chat_id, "text": text,
                                                               "reply_markup": self._controls_markup(coc_launched=coc_launched)})
            self._last_controls_id = (resp or {}).get("result", {}).get("message_id")

    def _menu_markup(self, loop_state: bool):
        # TÂCHE 2: Bouton loop unique
        if loop_state:
            loop_text = "Désactiver loop"
            loop_cb = "TOGGLE_LOOP"
        else:
            loop_text = "Activer loop"
            loop_cb = "TOGGLE_LOOP"

        return {
            "inline_keyboard": [
                [{"text": "⬅️ Retour", "callback_data": "BACK"}],
                # TÂCHE 5: Emoji
                [{"text": "📴 Éteindre PC", "callback_data": "SHUTDOWN_ASK"}],
                [{"text": "Choisir macro", "callback_data": "SELECT_MACRO_LIST"}],
                # TÂCHE 5: Emoji
                [{"text": "🔃 Recharger COC", "callback_data": "RELOAD_COC"}],
                
                # --- NOUVEAU BOUTON ---
                [{"text": "Valider arrivée 👌", "callback_data": "VALIDATE_ARRIVAL"}],
                # --- FIN NOUVEAU BOUTON ---
                
                # TÂCHE 2: Bouton loop unique
                [{"text": loop_text, "callback_data": loop_cb}]
                # TÂCHE 3: Suppression AutoCap
            ]
        }
    
    def replace_menu(self, title: str = "Paramètres", loop_state: bool = False):
        if not self.ready(): return
        with BASE_LOCK:
            if self._last_menu_id:
                self.delete_message(self._last_menu_id)
            resp = self._request("sendMessage", json={"chat_id": self.chat_id, "text": title,
                                                               "reply_markup": self._menu_markup(loop_state=loop_state)})
            self._last_menu_id = (resp or {}).get("result", {}).get("message_id")

    def _hash_callback_data(self, data: str) -> str:
        """Tronque et hashe le callback_data s'il dépasse 64 octets (Req 7)."""
        data_bytes = data.encode('utf-8')
        if len(data_bytes) <= 64:
            return data
        
        # Tronquer (en gardant de la marge pour le hash)
        # On ne peut pas juste tronquer les bytes, on doit tronquer la string
        prefix = data[:58] # 58 chars + '_' + 5 hash = 64
        while len(prefix.encode('utf-8')) > 58:
            prefix = prefix[:-1]
            
        hash_suffix = hashlib.sha1(data_bytes).hexdigest()[:5]
        hashed_data = f"{prefix}_{hash_suffix}"
        
        # Stocker le mapping
        self._callback_data_map[hashed_data] = data
        log.warning(f"Callback data > 64 octets, tronqué : '{data}' -> '{hashed_data}'")
        return hashed_data

    def push_macro_selection(self, macro_names: List[str]):
        if not self.ready(): return
        buttons = []
        row = []
        for name in macro_names:
            callback_data = self._hash_callback_data(f"SELECT_MACRO:{name}")
            row.append({"text": name, "callback_data": callback_data})
            if len(row) >= 2:
                buttons.append(row); row = []
        if row:
            buttons.append(row)
        buttons.append([{"text": "Annuler ↩️", "callback_data": "CANCEL_SELECTION"}])
        
        self._request("sendMessage", json={
            "chat_id": self.chat_id,
            "text": "🗂️ Quelle macro lancer ?",
            "reply_markup": {"inline_keyboard": buttons}
        })

    def push_shutdown_confirm(self, origin_msg_id: Optional[int] = None):
        """Demande de confirmation d'extinction (Req 5)."""
        if not self.ready(): return
        
        # Si possible, on nettoie le clavier du message qui a demandé l'extinction
        if origin_msg_id:
            self.edit_message_reply_markup(origin_msg_id, None)

        self._request("sendMessage", json={
            "chat_id": self.chat_id,
            "text": "⚠️ Confirmer l’extinction du PC ?",
            "reply_markup": {
                "inline_keyboard": [
                    [{"text": "Annuler", "callback_data": "SHUTDOWN_CANCEL"}],
                    [{"text": "✅ Confirmer l’extinction", "callback_data": "SHUTDOWN_CONFIRM"}]
                ]
            }
        })

    # ---------- Boucle polling (Req 2) ----------
    def _poll_loop(self):
        while not self._stop_evt.is_set():
            try:
                params = {"timeout": 25}
                if self._offset is not None:
                    params["offset"] = self._offset
                
                # Le _request est bloquant (rate-limiter + lock)
                data = self._request("getUpdates", verb="GET", params=params, timeout=30)
                
                results = data.get("result", [])
                for up in results:
                    if self._stop_evt.is_set(): break
                    self._offset = up.get("update_id", 0) + 1
                    
                    # --- Callback buttons ---
                    cq = up.get("callback_query")
                    if cq:
                        from_id = cq.get("from", {}).get("id")
                        data_cb = (cq.get("data") or "").strip()
                        msg = cq.get("message", {})
                        msg_id = msg.get("message_id")

                        # 1. Valider l'utilisateur
                        if self.chat_id and from_id != self.chat_id:
                            log.warning(f"Callback ignoré (chat_id invalide): {from_id}")
                            continue
                        if not self.chat_id:
                            self.chat_id = from_id
                            log.info(f"Auto-assignation du chat_id: {from_id}")

                        # 2. Répondre immédiatement (Req 2)
                        self._answer_callback(cq.get("id"))

                        # 3. Parser la commande
                        cmd = self._parse_command(data_cb)
                        if not cmd:
                            log.warning(f"Commande callback inconnue: {data_cb}")
                            continue
                            
                        # 4. Nettoyer les messages éphémères
                        
                        # Si la commande vient du menu principal, on supprime le clavier
                        # (le helper dans App s'occupera de supprimer le message)
                        if msg_id == self._last_menu_id:
                            if cmd not in {"BACK", "SELECT_MACRO_LIST", "SHUTDOWN_ASK"}:
                                # Supprime le clavier en attendant que App supprime le msg
                                self.edit_message_reply_markup(msg_id, None)
                        
                        # Si la commande vient du menu de sélection de macro
                        elif cmd.startswith("SELECT_MACRO:") or cmd == "CANCEL_SELECTION":
                             # Supprime le message "Quelle macro lancer ?"
                             self.delete_message(msg_id)
                        
                        # Si la commande vient de la confirmation de shutdown
                        elif cmd in {"SHUTDOWN_CONFIRM", "SHUTDOWN_CANCEL"}:
                            # Supprime le message "Confirmer l'extinction ?"
                            self.delete_message(msg_id)

                        # 5. Émettre la commande
                        meta = {"from": from_id, "message_id": msg_id}
                        self._emit(cmd, meta)
                        continue
                        
                    # --- Messages texte ---
                    msg = up.get("message") or up.get("edited_message")
                    if not msg: continue
                    
                    from_id = msg.get("chat", {}).get("id")
                    text = (msg.get("text") or "").strip()
                    
                    # 1. Valider l'utilisateur
                    if self.chat_id and from_id != self.chat_id:
                        continue
                    if not self.chat_id:
                        self.chat_id = from_id
                        log.info(f"Auto-assignation du chat_id: {from_id}")

                    # 2. Parser la commande (Whitelist)
                    cmd = self._parse_command_text(text)
                    if cmd:
                        self._emit(cmd, {"from": from_id, "message_id": msg.get("message_id")})
                        
                if not results and not self._stop_evt.is_set():
                    time.sleep(0.5) # Pause si pas d'updates
                    
            except Exception as e:
                if not self._stop_evt.is_set():
                    log.error(f"Erreur majeure dans poll_loop: {e}\n{traceback.format_exc()}")
                    time.sleep(5) # Pause avant de reprendre
    
    def _parse_command(self, s: str) -> Optional[str]:
        """Parse un callback_data (déjà mappé si > 64 octets)."""
        # 1. Vérifier la map de hash (Req 7)
        if s in self._callback_data_map:
            s = self._callback_data_map[s]
            
        # 2. Commandes connues (Whitelist implicite)
        if s.startswith("SELECT_MACRO:"):
            return s # "SELECT_MACRO:Nom"
        
        # Commandes fixes
        KNOWN_CALLBACKS = {
            "STOP", "GO", "MENU", "BACK", "SHUTDOWN_ASK", "SHUTDOWN_CONFIRM",
            "SHUTDOWN_CANCEL", "CAPTURE", "LAUNCH_COC", "RELOAD_COC",
            # TÂCHE 3: Suppression AutoCapture
            # "AUTO_CAP_ON", "AUTO_CAP_OFF", 
            # TÂCHE 2: Remplacement loop
            # "LOOP_ON", "LOOP_OFF",
            "TOGGLE_LOOP",
            # TÂCHE 4: Bouton CoC
            "DUMMY_COC_STATUS",
            "SELECT_MACRO_LIST", "CANCEL_SELECTION",
            
            # --- NOUVELLE COMMANDE ---
            "VALIDATE_ARRIVAL"
            # --- FIN NOUVELLE COMMANDE ---
        }
        if s in KNOWN_CALLBACKS:
            return s
            
        return None
    
    def _parse_command_text(self, s: str) -> Optional[str]:
        """Parse un message texte (Whitelist stricte, Req 7)."""
        t = (s or "").strip().lower()
        
        # Whitelist de commandes texte (Req 7)
        TEXT_COMMAND_MAP = {
            "stop": "STOP", "/stop": "STOP", "arreter": "STOP", "arrêter": "STOP", "pause": "STOP",
            "go": "GO", "/go": "GO", "start": "GO", "lancer": "GO", "reprendre": "GO",
            "shutdown": "SHUTDOWN_ASK", "/shutdown": "SHUTDOWN_ASK", "eteindre": "SHUTDOWN_ASK", "éteindre": "SHUTDOWN_ASK", "poweroff": "SHUTDOWN_ASK",
            "capture": "CAPTURE", "/capture": "CAPTURE", "screenshot": "CAPTURE", "screen": "CAPTURE",
            "gif": "CAPTURE_GIF",
            "menu": "MENU", "/menu": "MENU",
            "relancer": "RELOAD_COC", "/relancer": "RELOAD_COC",
            "launch": "LAUNCH_COC", "/launch": "LAUNCH_COC",
        }
        return TEXT_COMMAND_MAP.get(t)

    def _emit(self, cmd: str, meta: dict):
        try:
            if callable(self._on_command):
                log.info(f"TG: Commande reçue: {cmd} (meta: {meta})")
                self._on_command(cmd, meta or {})
        except Exception as e:
            log.error(f"Erreur lors de l'émission (on_command) de '{cmd}': {e}\n{traceback.format_exc()}")

    def _answer_callback(self, callback_id: str):
        try:
            if not callback_id: return
            # On n'utilise pas _request ici car c'est une réponse rapide
            # qui ne doit pas être bloquée par le rate-limiter principal.
            # C'est "fire-and-forget".
            requests.post(f"{self._base}/answerCallbackQuery",
                          json={"callback_query_id": callback_id},
                          timeout=5)
        except Exception:
            pass # Échec silencieux


# =========================
#       Modèle Macro (Req 1, 3)
# =========================
class MacroModel:
    def __init__(self):
        self.name = "Macro"
        self.steps: List[dict] = []
        self.current_hash: Optional[str] = None
        self._lock = threading.Lock() # Verrou spécifique au modèle

    def clear(self):
        with self._lock:
            self.steps = []
            self.current_hash = get_macro_hash(self.steps)

    def set_steps(self, steps: List[dict]):
        with self._lock:
            self.steps = list(steps)
            self.current_hash = get_macro_hash(self.steps)

    def get_steps(self) -> List[dict]:
        with self._lock:
            return list(self.steps)

    def duration(self) -> float:
        with self._lock:
            return sum(max(0.0, ev.get("t", 0.0)) for ev in self.steps)

    def save(self, path: Path, force_new_hash: bool = False) -> bool:
        """
        Sauvegarde via écriture atomique/hash (Req 3).
        
        :param force_new_hash: Si True, ignore le hash actuel et force
                               l'écriture. (Utile pour 'Renommer')
        """
        with self._lock:
            # Si force_new_hash, on passe None à write_macro_file
            # pour sauter la vérification d'hash.
            hash_to_check = None if force_new_hash else self.current_hash
            
            # Passe le hash actuel pour éviter l'écriture si inchangé
            return write_macro_file(path, self.name, self.steps, hash_to_check)

    def load(self, path: Path) -> bool:
        """Charge depuis le fichier (Req 3)."""
        try:
            name, steps, sha1, _ = read_macro_file(path)
            with self._lock:
                self.name = name
                self.steps = steps
                self.current_hash = sha1
            return True
        except Exception as e:
            log.error(f"Échec chargement macro {path.name}: {e}")
            return False


# =========================
#     Recorder / Player (Req 1)
# =========================
class Recorder:
    def __init__(self):
        self.recording = False
        self.steps: List[dict] = []
        self._mouse_listener = None
        self._keyboard_listener = None
        self._t0 = None
        self._last_t = None
        self._self_playing_flag = False
        self.grace_seconds = 3.0
        self._grace_until = None
        self._grace_started = False

    def _now(self): return time.perf_counter()

    def start(self):
        if self.recording: return
        self.steps = []
        self.recording = True
        self._t0 = self._now()
        self._last_t = self._t0
        self._grace_until = self._t0 + self.grace_seconds
        self._grace_started = True
        try:
            self._mouse_listener = mouse.Listener(
                on_move=self._on_move, on_click=self._on_click, on_scroll=self._on_scroll
            ); self._mouse_listener.start()
            self._keyboard_listener = pyn_keyboard.Listener(
                on_press=self._on_key_down, on_release=self._on_key_up
            ); self._keyboard_listener.start()
        except Exception as e:
            log.error(f"Échec démarrage listeners pynput: {e}\n{traceback.format_exc()}")
            self.stop()
            raise

    def stop(self):
        self.recording = False
        self._grace_started = False
        try:
            if self._mouse_listener: self._mouse_listener.stop()
        finally:
            self._mouse_listener = None
        try:
            if self._keyboard_listener: self._keyboard_listener.stop()
        finally:
            self._keyboard_listener = None

    def mark_self_play(self, flag: bool):
        self._self_playing_flag = flag

    def _push(self, typ, data):
        if not self.recording or self._self_playing_flag:
            return
        t = self._now()
        if self._grace_until is not None and t < self._grace_until:
            return
        if self._grace_started and t >= self._grace_until:
            self._last_t = t # Le premier step a dt=0
            self._grace_started = False
            play_start_tone()
        dt = t - self._last_t
        self._last_t = t
        self.steps.append({"t": dt, "type": typ, "data": data})

    # callbacks
    def _on_move(self, x, y): self._push("mouse_move", {"x": int(x), "y": int(y)})
    def _on_click(self, x, y, button, pressed):
        self._push("mouse_click", {"x": int(x), "y": int(y), "button": button_to_str(button), "action": "down" if pressed else "up"})
    def _on_scroll(self, x, y, dx, dy): self._push("scroll", {"x": int(x), "y": int(y), "dx": int(dx), "dy": int(dy)})
    def _on_key_down(self, key): self._push("key_down", {"key": key_to_str(key)})
    def _on_key_up(self, key): self._push("key_up", {"key": key_to_str(key)})

# --- Helpers pynput ---
def button_to_str(btn):
    if btn == Button.left: return "left"
    if btn == Button.right: return "right"
    if btn == Button.middle: return "middle"
    return "left"

def str_to_button(s):
    return {"left": Button.left, "right": Button.right, "middle": Button.middle}.get(s, Button.left)

def key_to_str(k):
    try:
        if hasattr(k, 'char') and k.char is not None:
            return k.char
        else:
            return str(k).split('.')[-1]
    except Exception:
        return str(k)

def str_to_key(s):
    if not s: return s
    if len(s) == 1: return s
    aliases = {"control": "ctrl", "win": "cmd", "altgr": "alt_gr"}
    s = aliases.get(s, s)
    try:
        return getattr(Key, s)
    except AttributeError:
        pass
    specials = {
        "enter": Key.enter, "shift": Key.shift, "ctrl": Key.ctrl, "alt": Key.alt, "esc": Key.esc,
        "tab": Key.tab, "space": Key.space, "backspace": Key.backspace, "delete": Key.delete,
        "up": Key.up, "down": Key.down, "left": Key.left, "right": Key.right,
        "cmd": getattr(Key, "cmd", None), "alt_gr": getattr(Key, "alt_gr", Key.alt),
        "caps_lock": getattr(Key, "caps_lock", None), "num_lock": getattr(Key, "num_lock", None),
        "scroll_lock": getattr(Key, "scroll_lock", None),
        "f1": Key.f1, "f2": Key.f2, "f3": Key.f3, "f4": Key.f4, "f5": Key.f5, "f6": Key.f6,
        "f7": Key.f7, "f8": Key.f8, "f9": Key.f9, "f10": Key.f10, "f11": Key.f11, "f12": Key.f12,
    }
    k = specials.get(s)
    if k is not None:
        return k
    
    # Gérer les KeyCode (ex: <65>)
    if s.startswith('<') and s.endswith('>'):
        try:
            vk = int(s[1:-1])
            return pyn_keyboard.KeyCode(vk=vk)
        except ValueError:
            pass
            
    log.warning(f"Touche spéciale non reconnue: '{s}'")
    return s

class Player:
    """
    Player haute-précision.
    Utilise une boucle de timing absolue pour éviter la dérive (Req 1).
    """
    def __init__(self, recorder: Recorder):
        self._mouse = MouseController()
        self._keys = KeyController()
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._loop = False
        self._recorder = recorder
        self.on_cycle: Optional[Callable] = None
        self.on_stopped: Optional[Callable] = None
        self._pressed_keys: Set[Any] = set()
        self._pressed_buttons: Set[Button] = set()

    def is_playing(self):
        return self._thread is not None and self._thread.is_alive()

    def stop(self):
        """Arrête la lecture et relâche toutes les touches/boutons."""
        self._stop_flag.set()
        if self._thread:
            try:
                self._thread.join(timeout=2.0)
            except Exception: pass
        self._thread = None
        self._recorder.mark_self_play(False)
        
        # Relâcher les inputs (failsafe)
        try:
            for k in list(self._pressed_keys):
                try: self._keys.release(k)
                except Exception: pass
            self._pressed_keys.clear()
            
            for b in list(self._pressed_buttons):
                try: self._mouse.release(b)
                except Exception: pass
            self._pressed_buttons.clear()
        except Exception as e:
            log.error(f"Erreur release inputs: {e}")
            
        self._stop_flag.clear()
        
        if self.on_stopped:
            try: self.on_stopped()
            except Exception as e:
                log.error(f"Erreur callback on_stopped: {e}")

    def play(self, steps, loop=False):
        if self.is_playing():
            return
        self._loop = loop
        self._thread = threading.Thread(
            target=self._run, 
            args=(steps,), 
            daemon=True,
            name="MacroPlayer"
        )
        self._thread.start()

    def _wait(self, seconds: float):
        """Attente interruptible."""
        self._stop_flag.wait(timeout=max(0.0, seconds))

    def _run_one_cycle(self, steps: List[dict]) -> bool:
        """
        Exécute un cycle de macro avec timing absolu pour
        compenser la dérive (Req 1).
        """
        if not steps:
            return False
            
        # 1. Calculer les timestamps absolus
        abs_times, acc = [], 0.0
        for ev in steps:
            dt = float(ev.get("t", 0.0)); acc += dt; abs_times.append(acc)
            
        start = time.perf_counter()
        
        # 2. Boucle d'exécution
        for i, ev in enumerate(steps):
            if self._stop_flag.is_set(): return False
            
            # 3. Calculer le temps d'attente
            target_time_abs = abs_times[i]
            now = time.perf_counter()
            time_elapsed = now - start
            
            remain = target_time_abs - time_elapsed
            
            if remain > 0.001: # Seuil minimal
                self._wait(remain)
                if self._stop_flag.is_set(): return False

            # 4. Appliquer l'événement
            self._apply(ev)
            
        return True # Cycle complet

    def _run(self, steps):
        self._recorder.mark_self_play(True)
        try:
            while not self._stop_flag.is_set():
                cycle_completed = self._run_one_cycle(steps)
                
                if self._stop_flag.is_set(): break
                
                if cycle_completed and self.on_cycle:
                    try: self.on_cycle()
                    except Exception as e:
                        log.error(f"Erreur callback on_cycle: {e}")
                        
                if not self._loop:
                    break
        finally:
            self._recorder.mark_self_play(False)
            if not self._stop_flag.is_set():
                # S'est terminé naturellement, appeler on_stopped
                if self.on_stopped:
                    try: self.on_stopped()
                    except Exception as e:
                        log.error(f"Erreur callback on_stopped (fin naturelle): {e}")

    def _apply(self, ev):
        typ, data = ev.get("type"), ev.get("data")
        if not typ or not data:
            return
        try:
            if typ == "mouse_move":
                self._mouse.position = (data["x"], data["y"])
            elif typ == "mouse_click":
                btn = str_to_button(data["button"])
                if data["action"] == "down":
                    self._mouse.press(btn); self._pressed_buttons.add(btn)
                else:
                    self._mouse.release(btn); self._pressed_buttons.discard(btn)
            elif typ == "scroll":
                self._mouse.scroll(data["dx"], data["dy"])
            elif typ == "key_down":
                k = str_to_key(data["key"]); self._keys.press(k); self._pressed_keys.add(k)
            elif typ == "key_up":
                k = str_to_key(data["key"]); self._keys.release(k); self._pressed_keys.discard(k)
        except Exception as e:
            # Ignorer les erreurs (ex: touche invalide, etc.)
            log.warning(f"Erreur application step {typ}: {e}")
            pass


# =========================
#     Popups (Req 4, 10)
# =========================
class BaseToplevel(ctk.CTkToplevel):
    """Fenêtre popup de base avec gestion d'icône et de fermeture."""
    def __init__(self, master, *a, **k):
        self._window_name = k.pop("name", None)
        self._on_close_cb = k.pop("on_close_cb", None)
        
        super().__init__(master, *a, **k)
        apply_window_icon(self)
        self.attributes("-topmost", True)
        self.protocol("WM_DELETE_WINDOW", self._on_wm_close)
        
        # Centrer sur le parent
        self.transient(master)
        self.grab_set()

    def _on_wm_close(self):
        """Appelé par le 'X' de la fenêtre."""
        self.grab_release()
        if callable(self._on_close_cb):
            try:
                self._on_close_cb(self._window_name)
            except Exception as e:
                log.error(f"Erreur on_close_cb ({self._window_name}): {e}")
            self._on_close_cb = None
        super().destroy()

    def destroy(self):
        """Appelé par les boutons (Sauver, Annuler, Fermer)."""
        self.grab_release()
        if callable(self._on_close_cb):
            try:
                self._on_close_cb(self._window_name)
            except Exception as e:
                log.error(f"Erreur on_close_cb ({self._window_name}): {e}")
            self._on_close_cb = None
        super().destroy()

class TextInputDialog(BaseToplevel):
    """Popup pour demander un nom (macro, etc.)."""
    def __init__(self, master, title: str, prompt: str, initial: str = ""):
        super().__init__(master, name="TextInput")
        self.title(title)
        self.geometry("380x160")
        self.resizable(False, False)
        
        wrap = ctk.CTkFrame(self, corner_radius=12); wrap.pack(fill="both", expand=True, padx=12, pady=12)
        ctk.CTkLabel(wrap, text=prompt, text_color=Theme.TEXT_COMPLIANT).pack(anchor="w", pady=(4,6))
        
        self._var = ctk.StringVar(value=initial)
        self._entry = ctk.CTkEntry(wrap, textvariable=self._var); self._entry.pack(fill="x")
        
        btns = ctk.CTkFrame(wrap, fg_color="transparent"); btns.pack(fill="x", pady=(12,0))
        ctk.CTkButton(btns, text="Annuler", width=100, fg_color="#374151", hover_color="#4b5563",
                        command=self._cancel).pack(side="right")
        ctk.CTkButton(btns, text="Valider", width=110, command=self._ok).pack(side="right", padx=(0,8))
        
        self._entry.focus_set()
        self._result: Optional[str] = None
        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self._cancel())

    def _ok(self):
        self._result = self._var.get().strip() or None
        self.destroy()

    def _cancel(self):
        self._result = None
        self.destroy()

    def show(self) -> Optional[str]:
        self.wait_window(self)
        return self._result

class TelegramAutomationDialog(BaseToplevel):
    """
    Fenêtre de config Telegram (Req 2).
    Gère la pause/reprise du poller lors de l'ouverture du guide.
    """
    def __init__(self, master, params: Dict[str, str], on_save: Callable, **kwargs):
        super().__init__(master, **kwargs)
        self.title("Automatisation Telegram")
        self.geometry("640x260")
        self.resizable(False, False)
        self._params = params
        self._on_save = on_save
        self._master_app: 'App' = master # Pour appeler pause/resume

        root = ctk.CTkFrame(self, corner_radius=12); root.pack(fill="both", expand=True, padx=12, pady=12)
        root.grid_columnconfigure(0, weight=1)

        # Champs
        grid = ctk.CTkFrame(root, fg_color="transparent"); grid.grid(row=0, column=0, sticky="ew")
        grid.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(grid, text="Bot token").grid(row=0, column=0, sticky="w", padx=(0,10), pady=(2,6))
        self._tg_token = ctk.StringVar(value=self._params.get("telegram_bot_token", ""))
        ctk.CTkEntry(grid, textvariable=self._tg_token).grid(row=0, column=1, sticky="ew", pady=(2,6))
        ctk.CTkLabel(grid, text="Chat ID").grid(row=1, column=0, sticky="w", padx=(0,10), pady=(2,6))
        self._tg_chat = ctk.StringVar(value=self._params.get("telegram_chat_id", ""))
        ctk.CTkEntry(grid, textvariable=self._tg_chat).grid(row=1, column=1, sticky="ew", pady=(2,6))

        # Bouton guide HTML (Req 2: pause poller)
        btns2 = ctk.CTkFrame(root, fg_color="transparent"); btns2.grid(row=1, column=0, sticky="ew", pady=(10,8))
        ctk.CTkButton(btns2, text="Ouvrir le guide (page HTML locale)", width=260,
                        command=self._open_local_guide).pack(side="left", padx=(0,8))

        # Bas
        bottom = ctk.CTkFrame(root, fg_color="transparent"); bottom.grid(row=2, column=0, sticky="ew")
        ctk.CTkButton(bottom, text="Fermer", width=120, fg_color="#374151", hover_color="#4b5563",
                        command=self.destroy).pack(side="right")
        ctk.CTkButton(bottom, text="Enregistrer", width=130, command=self._save).pack(side="right", padx=(0,8))

    def _open_local_guide(self):
        """Ouvre le guide HTML et met le poller en pause (Req 2)."""
        log.info("Ouverture du guide HTML local.")
        # L'ancien code quittait l'app. Le nouveau met en pause le poller.
        self._master_app.pause_telegram_poller()
        
        try:
            ensure_dirs()
            # (Le contenu HTML est identique à celui fourni)
            html_content = """<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Guide — Connexion Telegram (Macro COC)</title><meta name="viewport" content="width=device-width, initial-scale=1"><style>:root{--bg:#0b1220;--card:#0e1624;--muted:#94a3b8;--text:#e5e7eb;--accent:#80b7f7;--ok:#22c55e;--warn:#eab308;--danger:#ef4444;}*{box-sizing:border-box}body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif;background:var(--bg);color:var(--text);line-height:1.6;margin:0;padding:24px}h1,h2,h3{color:#fff;margin:0 0 12px}p,li{color:var(--text)}a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}.card{background:var(--card);border-radius:12px;padding:16px;margin:16px 0}.badge{display:inline-block;font-size:12px;padding:2px 8px;border-radius:999px;background:#1f2937;color:#cbd5e1;margin-left:8px}.note{background:#132033;padding:12px;border-radius:10px}kbd{background:#1f2937;padding:2px 6px;border-radius:4px;border:1px solid #374151}pre{background:#111827;color:#e5e7eb;padding:12px;border-radius:10px;overflow:auto}code{background:#111827;color:#e5e7eb;padding:2px 6px;border-radius:6px}ul,ol{margin:8px 0 8px 22px}.small{color:var(--muted);font-size:14px}.hero{display:flex;gap:14px;align-items:center;margin-bottom:10px}.hero .dot{width:10px;height:10px;border-radius:50%;background:var(--ok)}</style></head><body><div class="hero"><div class="dot"></div><h1>Connexion Telegram — méthode 100% manuelle</h1><span class="badge">Macro COC</span></div><div class="note"><b>Important :</b> un seul programme doit interroger <code>getUpdates</code> à la fois. L'application Macro COC a mis son service d'écoute (poller) <b>en pause</b> pendant que vous consultez ce guide. <b>Fermez cette fenêtre de paramètres</b> (via "Enregistrer" ou "Fermer") pour le réactiver.</div><div class="card"><h2>Résumé rapide (TL;DR)</h2><ol><li>Crée un bot dans Telegram avec <a href="https://t.me/BotFather" target="_blank" rel="noopener">@BotFather</a> → <code>/newbot</code>. Copie le <b>token</b>.</li><li>Parle à ton bot (clique le lien que BotFather te donne) et envoie <code>/start</code>.</li><li>Ouvre l'URL <code>https://api.telegram.org/bot&lt;TOKEN&gt;/getUpdates</code> dans ton navigateur.</li><li>Récupère ton <b>chat_id</b> dans la réponse JSON (<code>result[0].message.chat.id</code>).</li><li>Colle <b>token</b> et <b>chat_id</b> dans cette fenêtre → Enregistrer.</li></ol></div><p class="small">© Macro COC — Guide local</p></body></html>"""
            GUIDE_HTML_PATH.write_text(html_content, encoding="utf-8")
            webbrowser.open(GUIDE_HTML_PATH.resolve().as_uri())
            messagebox.showinfo(
                "Poller en pause",
                "Le service Telegram est en pause.\n"
                "Enregistrez ou fermez cette fenêtre pour le réactiver.",
                parent=self
            )
        except Exception as e:
            log.error(f"Erreur ouverture guide: {e}")
            messagebox.showerror("Guide", f"Erreur: {e}", parent=self)
            # On réactive le poller si le guide n'a pas pu s'ouvrir
            self._master_app.resume_telegram_poller()

    def _save(self):
        self._params["telegram_bot_token"] = self._tg_token.get().strip()
        self._params["telegram_chat_id"] = self._tg_chat.get().strip()
        if callable(self._on_save):
            self._on_save(self._params)
        self.destroy() # Déclenche _on_wm_close via BaseToplevel, qui appelle on_close_cb

class SettingsDialog(BaseToplevel):
    """Fenêtre de paramètres généraux (Req 4, 7)."""
    def __init__(self, master, params: Dict[str, str], on_save: Callable,
                 open_tg_dialog_cb: Callable, open_diag_dialog_cb: Callable,
                 tg_status_var: ctk.StringVar, tg_status_color_var: ctk.StringVar,
                 purge_tg_backlog_cb: Callable,
                 **kwargs):
        super().__init__(master, **kwargs)
        self.title("Paramètres")
        self.geometry("540x520")
        self.resizable(False, False)
        self._params = params
        self._on_save = on_save
        self._open_tg = open_tg_dialog_cb
        self._open_diag = open_diag_dialog_cb
        self._purge_tg = purge_tg_backlog_cb

        root = ctk.CTkFrame(self, corner_radius=12); root.pack(fill="both", expand=True, padx=12, pady=12)

        # Lecture en boucle
        row1 = ctk.CTkFrame(root, fg_color="transparent"); row1.pack(fill="x", pady=(4,8))
        ctk.CTkLabel(row1, text="Lecture en boucle (auto_loop)", text_color=Theme.LEFT_HEADER_TEXT).pack(side="left")
        self.auto_loop_var = ctk.BooleanVar(value=as_bool(self._params.get("auto_loop", "0"), False))
        ctk.CTkSwitch(row1, text="", variable=self.auto_loop_var).pack(side="right")
        
        sep = ctk.CTkFrame(root, height=1, fg_color=Theme.DIVIDER); sep.pack(fill="x", pady=8)

        # Section Telegram (Req 4)
        row2 = ctk.CTkFrame(root, fg_color="transparent"); row2.pack(fill="x", pady=(4,8))
        ctk.CTkLabel(row2, text="Automatisation Telegram", text_color=Theme.LEFT_HEADER_TEXT,
                       font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w")
        
        # Statut TG (Req 4)
        status_frame = ctk.CTkFrame(row2, fg_color="transparent"); status_frame.pack(fill="x", pady=(4,0))
        ctk.CTkLabel(status_frame, text="Statut:").pack(side="left", padx=(0, 6))
        self.lbl_tg_status = ctk.CTkLabel(status_frame, textvariable=tg_status_var, text_color=Theme.STATUS_OK)
        self.lbl_tg_status.pack(side="left")
        
        # Garder une référence pour pouvoir désenregistrer le trace
        self.tg_status_color_var = tg_status_color_var
        self._tg_trace_id = self.tg_status_color_var.trace_add("write", self._update_tg_color)
        self._update_tg_color(None, None, None) # Init
        
        btn_frame_tg = ctk.CTkFrame(row2, fg_color="transparent"); btn_frame_tg.pack(fill="x", pady=(8,0))
        ctk.CTkButton(btn_frame_tg, text="Configurer Telegram…", width=200,
                        command=self._open_tg).pack(side="left")
        ctk.CTkButton(btn_frame_tg, text="Purger backlog", width=140,
                        fg_color="#374151", hover_color="#4b5563",
                        command=self._purge_tg).pack(side="left", padx=(8,0))

        # Chemin de lancement CoC
        sep2 = ctk.CTkFrame(root, height=1, fg_color=Theme.DIVIDER); sep2.pack(fill="x", pady=12)
        row3 = ctk.CTkFrame(root, fg_color="transparent"); row3.pack(fill="x", pady=(4,0))
        ctk.CTkLabel(row3, text="Chemin de lancement CoC", text_color=Theme.LEFT_HEADER_TEXT,
                       font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(row3, text="Raccourci (.lnk) ou .exe", text_color="#94a3b8").pack(anchor="w")
        self.coc_path_var = ctk.StringVar(value=self._params.get("coc_path", ""))
        ctk.CTkEntry(row3, textvariable=self.coc_path_var).pack(fill="x", pady=(4,0))
        
        # Diag / Shutdown
        sep3 = ctk.CTkFrame(root, height=1, fg_color=Theme.DIVIDER); sep3.pack(fill="x", pady=12)
        row4 = ctk.CTkFrame(root, fg_color="transparent"); row4.pack(fill="x", pady=(4,0))
        
        # Bouton Diag (Req 10)
        ctk.CTkButton(row4, text="État du système…", width=180,
                        fg_color="#1f2937", hover_color="#374151",
                        command=self._open_diag).pack(side="left")
                        
        # Bouton Shutdown (Req 5)
        ctk.CTkButton(row4, text="📴 Éteindre le PC", width=160, # TÂCHE 5: Emoji
                        fg_color=Theme.BTN_STOP_BG, hover_color=Theme.BTN_STOP_HOVER,
                        command=self.master.request_local_shutdown).pack(side="right", padx=(8,0))

        # Bas
        btns = ctk.CTkFrame(root, fg_color="transparent"); btns.pack(fill="x", side="bottom", pady=(10,0))
        ctk.CTkButton(btns, text="Fermer", width=110, fg_color="#374151", hover_color="#4b5563",
                        command=self.destroy).pack(side="right")
        ctk.CTkButton(btns, text="Enregistrer", width=120, command=self._save).pack(side="right", padx=(0,8))

    def _save(self):
        self._params["auto_loop"] = "1" if self.auto_loop_var.get() else "0"
        self._params["coc_path"] = self.coc_path_var.get().strip()
            
        if callable(self._on_save):
            self._on_save(self._params)
        self.destroy()

    def _update_tg_color(self, var, idx, mode):
        """Met à jour la couleur du label, en vérifiant s'il existe."""
        try:
            # Vérifie si le widget existe avant de le configurer
            if self.lbl_tg_status.winfo_exists():
                color = self.tg_status_color_var.get()
                self.lbl_tg_status.configure(text_color=color)
            else:
                # Le widget n'existe plus, on se désenregistre
                self._remove_tg_trace()
        except Exception:
            # Le widget est probablement en cours de destruction
            self._remove_tg_trace()

    def _remove_tg_trace(self):
        """Désenregistre le callback de trace."""
        if hasattr(self, '_tg_trace_id') and self._tg_trace_id:
            try:
                self.tg_status_color_var.trace_remove("write", self._tg_trace_id)
                log.info("Trace de couleur TG (Settings) désenregistrée.")
            except Exception as e:
                log.warning(f"Échec de la suppression du trace TG: {e}")
            self._tg_trace_id = None # Évite double suppression

    def destroy(self):
        """Surcharge de destroy pour nettoyer le trace."""
        self._remove_tg_trace()
        super().destroy() # Appelle BaseToplevel.destroy()

class DiagnosticsDialog(BaseToplevel):
    """Fenêtre de diagnostic système (Req 10)."""
    def __init__(self, master, tg_status: str, **kwargs):
        super().__init__(master, **kwargs)
        self.title("État du système")
        self.geometry("700x500")
        
        root = ctk.CTkFrame(self, corner_radius=12); root.pack(fill="both", expand=True, padx=12, pady=12)
        
        scroll = ctk.CTkScrollableFrame(root, fg_color=Theme.CENTER_BG)
        scroll.pack(fill="both", expand=True)

        def add_row(key: str, value: str, color: str = Theme.TEXT_COMPLIANT):
            f = ctk.CTkFrame(scroll, fg_color="transparent")
            f.pack(fill="x")
            ctk.CTkLabel(f, text=key, text_color=Theme.TEXT_MUTED_COMPLIANT, width=180, anchor="e").pack(side="left", padx=(0, 10))
            ctk.CTkLabel(f, text=value, text_color=color, anchor="w").pack(side="left", expand=True, fill="x")

        # --- Infos (Req 10) ---
        add_row("Version App:", APP_VERSION)
        add_row("Version Python:", PYTHON_VERSION)
        
        pil_txt, pil_col = ("Oui", Theme.STATUS_OK) if PIL_AVAILABLE else ("Non (CAPTURE D'ÉCRAN INDISPONIBLE)", Theme.STATUS_ERROR)
        add_row("Librairie Pillow:", pil_txt, pil_col)
        
        mss_txt, mss_col = ("Oui", Theme.STATUS_OK) if MSS_AVAILABLE else ("Non (fallback capture)", Theme.STATUS_WARN)
        add_row("Librairie MSS:", mss_txt, mss_col)
        
        add_row("Statut Telegram:", tg_status)

        try:
            disk = shutil.disk_usage(BASE_DIR)
            free_gb = disk.free / (1024**3)
            add_row("Espace disque (app):", f"{free_gb:.2f} Go libres")
        except Exception as e:
            add_row("Espace disque (app):", f"Erreur: {e}", Theme.STATUS_ERROR)

        ctk.CTkLabel(scroll, text="Derniers logs (5)", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 5), anchor="w")
        
        log_text = ctk.CTkTextbox(scroll, height=200, fg_color=Theme.APP_BG, text_color=Theme.TEXT_MUTED_COMPLIANT,
                                  font=ctk.CTkFont(family="Courier New", size=12))
        log_text.pack(fill="x", expand=True)
        
        try:
            if LOG_PATH.exists():
                with LOG_PATH.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
                    log_text.insert("1.0", "".join(lines[-5:]))
            else:
                log_text.insert("1.0", "Fichier log non encore créé.")
        except Exception as e:
            log_text.insert("1.0", f"Erreur lecture log: {e}")
        log_text.configure(state="disabled")

        ctk.CTkButton(root, text="Fermer", command=self.destroy).pack(side="bottom", anchor="e", pady=(10,0))


# =========================
#   Liste macros (Req 4, 9)
# =========================
class MacroRow:
    """Une ligne cliquable : Nom à gauche, Durée à droite."""
    def __init__(self, parent, name: str, duration_txt: str, on_click, on_rclick):
        self.name = name
        self.frame = ctk.CTkFrame(parent, corner_radius=8, fg_color=Theme.ROW_BG)
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=0)
        
        self.lbl_name = ctk.CTkLabel(self.frame, text=name, anchor="w", text_color=Theme.TEXT_COMPLIANT)
        self.lbl_name.grid(row=0, column=0, sticky="ew", padx=(10,6), pady=8)

        self.lbl_dur = ctk.CTkLabel(self.frame, text=duration_txt, anchor="e", text_color=Theme.ROW_DUR_COLOR)
        self.lbl_dur.grid(row=0, column=1, sticky="e", padx=(6,10))

        # Tooltip "Non enregistrée" (Req 4)
        if duration_txt == "Non enregistrée":
            # TODO: Implémenter un vrai tooltip (CTK n'en a pas nativement)
            # Faute de mieux, on se contente de l'indicateur visuel
            pass

        # Événements
        def bind_all(widget):
            widget.bind("<Button-1>", lambda e: on_click(self.name))
            widget.bind("<Enter>", lambda e: self._hover(True))
            widget.bind("<Leave>", lambda e: self._hover(False))
            # Clic droit pour menu contextuel (Req 9: Dupliquer, etc.)
            widget.bind("<Button-3>", lambda e: on_rclick(e, self.name))
            
        bind_all(self.frame); bind_all(self.lbl_name); bind_all(self.lbl_dur)
        self._selected = False

    def pack(self, **kw): self.frame.pack(**kw)
    def destroy(self): self.frame.destroy()

    def set_selected(self, sel: bool):
        self._selected = sel
        self.frame.configure(fg_color=Theme.ROW_SELECTED if sel else Theme.ROW_BG)

    def _hover(self, enter: bool):
        if self._selected: return
        self.frame.configure(fg_color=Theme.ROW_HOVER if enter else Theme.ROW_BG)

    def set_duration(self, dur_txt: str):
        self.lbl_dur.configure(text=dur_txt)

class MacroList(ctk.CTkScrollableFrame):
    """Liste scrollable des macros (Req 4: Recherche)."""
    def __init__(self, master, on_select, on_rclick):
        super().__init__(master, corner_radius=12, fg_color=Theme.LEFT_CONTAINER_BG)
        self._on_select = on_select
        self._on_rclick = on_rclick
        self._rows: Dict[str, MacroRow] = {}
        self._selected: Optional[str] = None
        self._meta: Dict[str, Tuple[int, float]] = {}

    def set_meta(self, meta: Dict[str, Tuple[int, float]]):
        self._meta = dict(meta)

    def refresh(self, names: List[str], selected: Optional[str], filter_term: Optional[str] = None):
        """Rafraîchit la liste, en appliquant un filtre (Req 4)."""
        # Purge
        for r in self._rows.values():
            r.destroy()
        self._rows.clear()
        
        filter_term = filter_term.lower() if filter_term else None
        
        # Rebuild
        for name in names:
            # Filtre de recherche (Req 4)
            if filter_term and filter_term not in name.lower():
                continue
                
            _, d = self._meta.get(name, (0, 0.0))
            row = MacroRow(self, name, fmt_dur_for_list(d), 
                           on_click=self.select, on_rclick=self._on_rclick)
            row.pack(fill="x", padx=6, pady=4)
            self._rows[name] = row
            
        if selected and selected in self._rows:
            self.select(selected, fire=False)

    def update_one(self, name: str):
        """Met à jour la durée d'une seule ligne."""
        if name in self._rows:
            _, d = self._meta.get(name, (0, 0.0))
            self._rows[name].set_duration(fmt_dur_for_list(d))

    def select(self, name: str, fire: bool = True):
        """Sélectionne une macro dans la liste."""
        if self._selected and self._selected in self._rows:
            self._rows[self._selected].set_selected(False)
            
        self._selected = name
        
        if name in self._rows:
            self._rows[name].set_selected(True)
            if fire and callable(self._on_select):
                try:
                    self._on_select(name)
                except Exception as e:
                    log.error(f"Erreur on_select({name}): {e}\n{traceback.format_exc()}")


# =========================
#     Application (Req 1-10)
# =========================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        log.info(f"=== Démarrage Macro COC v{APP_VERSION} ===")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        ensure_dirs()

        try:
            if os.name == 'nt':
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MacroCOC.App")
        except Exception:
            pass

        # Fenêtre
        self.title("Macro COC")
        self.geometry("1200x720")
        self.minsize(1024, 640)
        apply_window_icon(self)

        # --- État (Req 1) ---
        self.current_state = State.IDLE
        
        self.params = read_params_csv(PARAMS_PATH)

        self.model = MacroModel()
        self.rec = Recorder()
        self.player = Player(self.rec)
        
        self.open_windows: Dict[str, ctk.CTkToplevel] = {}

        # --- Telegram (Req 2) ---
        self.tg_status_var = ctk.StringVar(value="Initialisation...")
        self.tg_status_color_var = ctk.StringVar(value=Theme.STATUS_WARN)
        self.tg = TelegramBridge(
            token=self.params.get("telegram_bot_token", ""),
            chat_id=self.params.get("telegram_chat_id", ""),
            on_command=self._on_tg_command
        )
        self._update_tg_status_vars()

        # Flags
        self._coc_launched_once = False
        self._coc_is_running_tg = False


        # Timer de lecture UI (Req 1)
        self._play_timer_running = False
        self._last_tick: Optional[float] = None

        # variables UI
        self.status_var = ctk.StringVar(value="Prêt")
        self.cycle_count = ctk.IntVar(value=0)
        self.session_seconds = 0.0
        self.session_elapsed = ctk.StringVar(value="00:00")
        self.macro_duration = ctk.StringVar(value="00:00")
        self.macro_events = ctk.StringVar(value="0")

        self.current_macro_name: Optional[str] = None
        self.current_macro_path: Optional[Path] = None
        self.all_macro_names: List[str] = [] # Cache pour la recherche

        # ================= Layout principal =================
        root = ctk.CTkFrame(self, corner_radius=0, fg_color=Theme.APP_BG); root.pack(fill="both", expand=True)

        # --- Header ---
        header = ctk.CTkFrame(root, corner_radius=0, fg_color=Theme.HEADER_BG, height=Theme.HEADER_HEIGHT)
        header.pack(fill="x")
        header.grid_propagate(False)
        header.grid_columnconfigure(0, weight=0); header.grid_columnconfigure(1, weight=1); header.grid_columnconfigure(2, weight=0)
        header.grid_rowconfigure(0, weight=1)

        self._hdr_icon = None
        if PIL_AVAILABLE and ICON_PNG_PATH.exists():
            try:
                self._hdr_icon = ctk.CTkImage(Image.open(ICON_PNG_PATH), size=Theme.ICON_PNG_SIZE)
                ctk.CTkLabel(header, image=self._hdr_icon, text="").grid(row=0, column=0, padx=(16, 12), sticky="nsw")
            except Exception as e:
                log.warning(f"Impossible de charger image.png: {e}")

        self._title_lbl = ctk.CTkLabel(header, text="Macro COC", anchor="center", font=ctk.CTkFont(size=24, weight="bold"))
        self._title_lbl.grid(row=0, column=1, sticky="nsew")

        btn_params = ctk.CTkButton(header, text="Paramètres", width=120, height=36, command=self.open_settings)
        btn_params.grid(row=0, column=2, padx=(12, 16), sticky="e")

        # --- Corps ---
        body = ctk.CTkFrame(root, fg_color=Theme.APP_BG); body.pack(fill="both", expand=True, padx=12, pady=12)
        body.grid_columnconfigure(0, weight=0); body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # --- Panneau gauche (Req 4: Recherche, Req 9: Menu) ---
        left = ctk.CTkFrame(body, corner_radius=12, fg_color=Theme.LEFT_CONTAINER_BG, width=320)
        left.grid(row=0, column=0, sticky="nsw", padx=(0,12))
        left.pack_propagate(False) # Empêche le panneau de gauche de rétrécir
        
        ctk.CTkLabel(left, text="Macros", font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=Theme.LEFT_HEADER_TEXT).pack(anchor="w", padx=12, pady=(12,4))
                     
        # Champ de recherche (Req 4)
        self.search_entry = ctk.CTkEntry(left, placeholder_text="Rechercher...")
        self.search_entry.pack(fill="x", padx=8, pady=(0, 8))
        self.search_entry.bind("<KeyRelease>", self._filter_macro_list)

        self.macro_list = MacroList(left, on_select=self._select_macro_by_name, on_rclick=self._show_macro_context_menu)
        self.macro_list.pack(fill="both", expand=True, padx=8, pady=(0,8))

        # Barre d’actions
        actions = ctk.CTkFrame(left, fg_color=Theme.LEFT_ACTIONS_BG)
        actions.pack(fill="x", padx=8, pady=(0,12))
        
        # --- MODIFICATION: Bouton "Nouveau" en vert ---
        self.btn_macro_new = ctk.CTkButton(
            actions, text="Nouveau", width=80, command=self.macro_new,
            fg_color=Theme.BTN_PRIMARY_BG, hover_color=Theme.BTN_PRIMARY_HOVER
        )
        # --- FIN MODIFICATION ---
        
        self.btn_macro_new.pack(side="left", padx=4, pady=8)
        self.btn_macro_rename = ctk.CTkButton(actions, text="Renommer", width=90, command=self.macro_rename)
        self.btn_macro_rename.pack(side="left", padx=4, pady=8)
        self.btn_macro_delete = ctk.CTkButton(actions, text="Supprimer", width=90, fg_color=Theme.BTN_STOP_BG, hover_color=Theme.BTN_STOP_HOVER,
                                command=self.macro_delete)
        self.btn_macro_delete.pack(side="left", padx=4, pady=8)
        
        # Menu contextuel pour macros (Req 9)
        self.macro_menu = ctk.CTkFrame(self, fg_color=Theme.LEFT_ACTIONS_BG, border_width=1, border_color=Theme.DIVIDER)
        ctk.CTkButton(self.macro_menu, text="Dupliquer", anchor="w", fg_color="transparent", hover_color=Theme.ROW_HOVER,
                        command=self.macro_duplicate).pack(fill="x", padx=5, pady=2)
        ctk.CTkButton(self.macro_menu, text="Exporter...", anchor="w", fg_color="transparent", hover_color=Theme.ROW_HOVER,
                        command=self.macro_export).pack(fill="x", padx=5, pady=2)
        ctk.CTkButton(self.macro_menu, text="Importer...", anchor="w", fg_color="transparent", hover_color=Theme.ROW_HOVER,
                        command=self.macro_import).pack(fill="x", padx=5, pady=2)
        self.bind("<Button-1>", lambda e: self.macro_menu.place_forget()) # Cacher si on clique ailleurs


        # --- Zone centrale ---
        main = ctk.CTkFrame(body, corner_radius=12, fg_color=Theme.CENTER_BG)
        main.grid(row=0, column=1, sticky="nsew")

        # Bandeau macro sélectionnée (Req 4)
        info = ctk.CTkFrame(main, fg_color=Theme.INFO_BG)
        info.pack(fill="x", padx=16, pady=(16,8))
        self.lbl_macro_name = ctk.CTkLabel(info, text="Macro sélectionnée : —", font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_macro_name.pack(side="left", padx=6, pady=10)
        self.lbl_macro_meta = ctk.CTkLabel(info, text="Non enregistrée", text_color=Theme.INFO_TEXT_MUTED)
        self.lbl_macro_meta.pack(side="right", padx=6)

        # Contrôles principaux (Req 4: États visuels)
        controls = ctk.CTkFrame(main, fg_color="transparent")
        controls.pack(fill="x", padx=16, pady=(6,10))

        self.btn_rec_toggle = ctk.CTkButton(
            controls, text="Enregistrer", height=44, width=200,
            fg_color=Theme.BTN_PRIMARY_BG, hover_color=Theme.BTN_PRIMARY_HOVER, command=self.toggle_record
        ); self.btn_rec_toggle.pack(side="left", padx=6)

        self.btn_play = ctk.CTkButton(
            controls, text="Lire la macro", height=44, width=170,
            fg_color=Theme.BTN_PRIMARY_BG, hover_color=Theme.BTN_PRIMARY_HOVER, command=lambda: self.on_play(notify_tg=True)
        ); self.btn_play.pack(side="left", padx=6)

        self.btn_stop_all = ctk.CTkButton(
            controls, text="Stopper la macro", height=44, width=210,
            fg_color=Theme.BTN_STOP_BG, hover_color=Theme.BTN_STOP_HOVER, command=lambda: self.force_stop_all(notify_tg=True)
        ); self.btn_stop_all.pack(side="left", padx=6)

        # Statistiques
        stats = ctk.CTkFrame(main, corner_radius=12, fg_color=Theme.CARD_BG)
        stats.pack(fill="x", padx=16, pady=(6,12))
        def add_stat(parent, text: str, var: ctk.Variable):
            f = ctk.CTkFrame(parent, fg_color="transparent"); f.pack(fill="x", padx=12, pady=(4,4))
            ctk.CTkLabel(f, text=text).pack(side="left")
            ctk.CTkLabel(f, textvariable=var, text_color="#86efac").pack(side="right")
        add_stat(stats, "Durée de la macro", self.macro_duration)
        add_stat(stats, "Événements", self.macro_events)
        add_stat(stats, "Cycles effectués (session)", self.cycle_count)
        add_stat(stats, "Temps total de lecture", self.session_elapsed)

        # Bouton Lancer CoC
        launch_wrap = ctk.CTkFrame(main, fg_color="transparent")
        launch_wrap.pack(fill="x", padx=16, pady=(0,12))
        self.btn_launch_coc = ctk.CTkButton(
            launch_wrap, text="Lancer CoC", height=44, width=180,
            fg_color=Theme.BTN_LAUNCH_BG, hover_color=Theme.BTN_LAUNCH_HOVER,
            command=self.launch_coc_once
        ); self.btn_launch_coc.pack(side="left", padx=6)

        # --- Pied ---
        footer = ctk.CTkFrame(root, fg_color=Theme.APP_BG); footer.pack(fill="x", side="bottom")
        footer.grid_columnconfigure(0, weight=1)
        hint = ctk.CTkLabel(footer, text="F1 = Basculer lecture | Ctrl+Shift+1 = Play | Ctrl+Shift+0 = Stop",
                            anchor="center", text_color="#94a3b8")
        hint.grid(row=0, column=0, sticky="ew", pady=(6, 0))
        status = ctk.CTkFrame(footer, height=28, fg_color=Theme.STATUS_BG)
        status.grid(row=1, column=0, sticky="ew")
        ctk.CTkLabel(status, textvariable=self.status_var, anchor="w").pack(side="left", padx=10)

        # Hotkeys + callbacks
        self._register_hotkeys()
        self.protocol("WM_DELETE_WINDOW", self.safe_quit)
        self.update_ui_for_state() # Init état UI

        # Bootstrap
        self.after(150, self._bootstrap)

    # ---------- Gestion état (Req 1) ----------

    def transition_to(self, new_state: State) -> bool:
        """
        Tente une transition d'état, la bloque si illégale (Req 1).
        Utilise le verrou global.
        """
        with BASE_LOCK:
            current = self.current_state
            if current == new_state:
                return True # Déjà dans cet état
                
            log.info(f"Demande transition: {current.name} -> {new_state.name}")
            
            # Transitions illégales (Req 1)
            if current == State.PLAYING and new_state == State.RECORDING:
                msg = "Impossible d'enregistrer pendant la lecture."
                log.warning(msg)
                self.toast(msg)
                return False
                
            if current == State.RECORDING and new_state == State.PLAYING:
                msg = "Impossible de lire pendant l'enregistrement."
                log.warning(msg)
                self.toast(msg)
                return False

            # Transition légale
            self.current_state = new_state
            log.info(f"Nouvel état: {self.current_state.name}")
            
            # Mettre à jour l'UI depuis le thread principal
            self.after(0, self.update_ui_for_state)
            
            return True

    def update_ui_for_state(self):
        """Met à jour l'UI (boutons) en fonction de l'état (Req 4)."""
        state = self.current_state
        
        # État des boutons de macro (gauche)
        macro_btn_state = "normal" if state == State.IDLE else "disabled"
        self.btn_macro_new.configure(state=macro_btn_state)
        self.btn_macro_rename.configure(state=macro_btn_state)
        self.btn_macro_delete.configure(state=macro_btn_state)

        # Bouton Enregistrer
        if state == State.RECORDING:
            self.btn_rec_toggle.configure(
                text="Enregistrement... (Stop)", 
                fg_color=Theme.BTN_STOP_BG, 
                hover_color=Theme.BTN_STOP_HOVER,
                state="normal"
            )
        else:
            rec_state = "normal"
            if state == State.PLAYING: 
                rec_state = "disabled"
                
            self.btn_rec_toggle.configure(
                text="Enregistrer",
                fg_color=Theme.BTN_PRIMARY_BG, 
                hover_color=Theme.BTN_PRIMARY_HOVER,
                state=rec_state
            )

        # Bouton Lire
        play_state = "disabled" if (state == State.RECORDING or state == State.PLAYING) else "normal"
        self.btn_play.configure(state=play_state)

        # Bouton Stop
        stop_state = "normal" if (state == State.RECORDING or state == State.PLAYING) else "disabled"
        self.btn_stop_all.configure(state=stop_state)
        
        # Mettre à jour le statut global
        self._update_status_bar()

    # ---------- Helpers ----------
    def _macro_label(self) -> str:
        return self.current_macro_name or "—"

    def _update_status_bar(self):
        """Met à jour la barre de statut principale."""
        loop_state = "ON" if self._auto_loop_enabled() else "OFF"
        prefix = self.current_state.name.capitalize()
        
        if self.current_state == State.IDLE:
            prefix = "Prêt"
        elif self.current_state == State.PLAYING:
            prefix = f"Lecture ({fmt_seconds(self.session_seconds)})"
        elif self.current_state == State.RECORDING:
            prefix = "Enregistrement..."
            
        self.status_var.set(f"{prefix} — Macro : {self._macro_label()} — Loop {loop_state}")

    def _is_protected_macro(self, name: Optional[str]) -> bool:
        if not name: return False
        # --- MODIFICATION ---
        name_lower = name.strip().lower()
        return name_lower in (
            RECHARGER_MACRO_NAME.strip().lower(),
            VALIDER_MACRO_NAME.strip().lower()
        )
        # --- FIN MODIFICATION ---

    def _load_steps_from_path(self, path: Path) -> List[dict]:
        try:
            _, steps, _, _ = read_macro_file(path)
            return steps
        except Exception:
            return []

    def play_macro_once(self, name: str, notify_tg: bool = True):
        """Joue une macro par nom, sans changer la sélection UI."""
        if self.current_state != State.IDLE:
            msg = "Veuillez arrêter l'opération en cours avant de lancer cette macro."
            self.toast(msg)
            if notify_tg: self.tg.send(f"Erreur : {msg}")
            return
            
        path = LEGACY_MACRO if name == LEGACY_MACRO.stem else macro_path_from_name(name)
        if not path.exists():
            self.toast(f"Macro « {name} » introuvable.")
            if notify_tg: self.tg.send(f"Erreur : macro « {name} » introuvable.")
            return
            
        steps = self._load_steps_from_path(path)
        if not steps:
            self.toast(f"Macro « {name} » vide.")
            if notify_tg: self.tg.send(f"Erreur : macro « {name} » vide.")
            return
            
        if not self.transition_to(State.PLAYING):
            return
            
        self.cycle_count.set(0)
        self.session_seconds = 0.0
        self.session_elapsed.set("00:00")
        
        self.player.play(steps, loop=False) # Joue une seule fois
        
        self._start_play_timer()
        self.toast(f"Lecture de « {name} » démarrée.")
        if notify_tg:
            self.tg.replace_controls(self._compose_status_for_tg(f"Lecture « {name} »"))

    # ---------- Gestion fenêtres Singleton ----------
    def open_singleton_window(self, name: str, window_class: type[BaseToplevel], *args, **kwargs):
        """Ouvre une popup, en garantissant qu'une seule instance existe."""
        if name in self.open_windows:
            win = self.open_windows[name]
            try:
                if win.winfo_exists():
                    win.lift()
                    win.attributes("-topmost", True)
                    win.focus_force()
                    self.toast(f"La fenêtre « {name} » est déjà ouverte.")
                    return
                else:
                    del self.open_windows[name]
            except Exception:
                if name in self.open_windows:
                    del self.open_windows[name]
        
        # Wrapper pour le callback de fermeture
        user_on_close = kwargs.pop("on_close_cb", None)
        
        def _master_on_close(win_name):
            if win_name in self.open_windows:
                del self.open_windows[win_name]
                log.info(f"Fenêtre singleton fermée: {win_name}")
            if callable(user_on_close):
                user_on_close() # Appelle le callback de l'utilisateur

        kwargs["name"] = name
        kwargs["on_close_cb"] = _master_on_close
        
        log.info(f"Ouverture fenêtre singleton: {name}")
        win = window_class(self, *args, **kwargs)
        self.open_windows[name] = win
        win.focus_force()

    # ---------- Paramètres & Telegram (Req 2, 4) ----------
    def _update_tg_status_vars(self):
        status, color = self.tg.get_status()
        self.tg_status_var.set(status)
        self.tg_status_color_var.set(color)

    def _reconfigure_telegram(self):
        """Applique les nouveaux params (token/chat_id) à Telegram."""
        log.info("Reconfiguration du bridge Telegram...")
        
        # --- CORRECTION 409 Conflict ---
        old_poller_thread = None
        # Récupérer le thread poller de l'ANCIENNE instance de tg
        if hasattr(self.tg, '_poll_thread') and self.tg._poll_thread is not None:
            old_poller_thread = self.tg._poll_thread
            if old_poller_thread.is_alive():
                log.info(f"Signal d'arrêt envoyé à l'ancien poller (TID: {old_poller_thread.ident}) pour reconfiguration.")

        self.tg.stop() # Envoie le signal d'arrêt à l'ancienne instance
        
        # Attendre la fin de l'ancien poller avant d'en créer un nouveau
        if old_poller_thread and old_poller_thread.is_alive():
            log.info(f"Attente de la fin de l'ancien poller (TID: {old_poller_thread.ident})...")
            old_poller_thread.join(timeout=30.0) # Attente max 30s
            if old_poller_thread.is_alive():
                log.warning("L'ancien poller n'a pas pu être arrêté à temps. Un 409 Conflict est possible.")
        # --- FIN CORRECTION ---

        self.tg = TelegramBridge(
            token=self.params.get("telegram_bot_token", ""),
            chat_id=self.params.get("telegram_chat_id", ""),
            on_command=self._on_tg_command
        )
        self._update_tg_status_vars()
        
        if self.tg.is_configured():
            self.tg.discard_backlog()
            self.tg.start()
            if self.tg.ready():
                self.tg.send("Configuration Telegram enregistrée.")
                self.tg.replace_controls(self._compose_status_for_tg("Prêt"))
            self._update_tg_status_vars()
        else:
            self.toast("Telegram non configuré.")

    def pause_telegram_poller(self):
        log.warning("Pause du poller Telegram demandée.")
        self.tg.stop()
        self._update_tg_status_vars()

    def resume_telegram_poller(self):
        log.info("Reprise du poller Telegram demandée (lancement en arrière-plan).")
        # Démarrer dans un thread pour permettre au join() de ne pas geler l'UI
        # lors de l'attente de la fin de l'ancien long-poll.
        threading.Thread(target=self._threaded_resume, daemon=True, name="TG-Resume").start()
        
    def _threaded_resume(self):
        """S'exécute dans un thread pour redémarrer le pont TG."""
        try:
            # La méthode start() va maintenant bloquer (dans ce thread)
            # le temps que l'ancien poller meure.
            self.tg.start()
            
            # Mettre à jour l'UI depuis le thread principal
            self.after(0, self._update_tg_status_vars)
        except Exception as e:
            log.error(f"Erreur lors de la reprise du poller TG: {e}")
            # Mettre à jour l'UI même en cas d'échec
            self.after(0, self._update_tg_status_vars)
        
    def purge_tg_backlog(self):
        log.info("Purge manuelle du backlog Telegram demandée...")
        # Lance la purge dans un thread pour ne pas geler l'UI
        # et pour gérer l'attente du poller.
        threading.Thread(target=self._threaded_purge, daemon=True, name="TG-Purge").start()
        self.toast("Purge du backlog en cours...")
        
    def _threaded_purge(self):
        """Effectue la purge dans un thread pour éviter les conflits 409."""
        try:
            # 1. Signale au poller de s'arrêter
            self.pause_telegram_poller() # self.tg.stop()
            
            # 2. Attend que le thread du poller meure
            # C'est crucial : il faut attendre la fin de son long-poll (timeout 30s)
            old_thread = self.tg._poll_thread
            if old_thread and old_thread.is_alive():
                log.info(f"Purge: Attente de la fin du poller (TID: {old_thread.ident})...")
                old_thread.join(timeout=30.0) # Attente max 30s
                if old_thread.is_alive():
                    log.warning("Purge: Timeout en attente de l'arrêt du poller.")
            
            # 3. Le poller est (normalement) mort. On peut purger.
            log.info("Purge: Poller arrêté, exécution de discard_backlog().")
            self.tg.discard_backlog()
            
            # 4. Notifier l'UI (via self.after)
            self.after(0, lambda: self.toast("Backlog Telegram purgé."))
            
        except Exception as e:
            log.error(f"Erreur durant la purge threadée: {e}")
            self.after(0, lambda: self.toast("Échec de la purge."))
            
        finally:
            # 5. Redémarre le poller quoi qu'il arrive
            log.info("Purge: Redémarrage du poller...")
            self.resume_telegram_poller()
            if self.tg.ready():
                 self.tg.send("Backlog purgé manuellement.")

    def open_settings(self):
        def _on_save_general(p: dict):
            self.params.update(p)
            self._save_params()
            self.update_ui_for_state() 
            self.toast("Paramètres enregistrés.")
            
        def _open_tg_dialog():
            def _on_save_tg(p: dict):
                self.params.update(p)
                self._save_params()
                self._reconfigure_telegram()
                self.toast("Configuration Telegram enregistrée.")
            
            # Req 2: Arrête le poller avant, relance après fermeture
            self.pause_telegram_poller() 
            self.open_singleton_window(
                "Automatisation Telegram",
                TelegramAutomationDialog,
                params=dict(self.params),
                on_save=_on_save_tg,
                on_close_cb=self.resume_telegram_poller # Relance à la fermeture
            )
            
        def _open_diag_dialog():
            self.open_singleton_window(
                "Diagnostic",
                DiagnosticsDialog,
                tg_status=self.tg_status_var.get()
            )

        self.open_singleton_window(
            "Paramètres",
            SettingsDialog,
            params=dict(self.params),
            on_save=_on_save_general,
            open_tg_dialog_cb=_open_tg_dialog,
            open_diag_dialog_cb=_open_diag_dialog,
            tg_status_var=self.tg_status_var,
            tg_status_color_var=self.tg_status_color_var,
            purge_tg_backlog_cb=self.purge_tg_backlog
        )
        
    def _save_params(self):
        """Wrapper pour l'écriture atomique des params."""
        write_params_csv(PARAMS_PATH, self.params)
        self.update_ui_for_state()

    def _auto_loop_enabled(self) -> bool:
        return as_bool(self.params.get("auto_loop", "0"), False)

    # ---------- Bootstrap ----------
    def _bootstrap(self):
        """Initialisation post-lancement UI."""
        
        # Lancer le démarrage de Telegram dans un thread
        # pour ne pas geler l'interface (UI)
        if self.tg.is_configured():
            threading.Thread(target=self._bootstrap_telegram, daemon=True, name="TG-Bootstrap").start()

        # --- NOUVELLE LIGNE ---
        # S'assurer que les fichiers JSON des macros protégées existent
        self._ensure_protected_macros()
        # --- FIN NOUVELLE LIGNE ---

        # Le chargement des macros (local) peut rester ici
        self._refresh_macro_sidebar()
        wanted = (self.params.get("last_macro_played") or "").strip().strip('"')
        if not wanted:
            wanted = (self.params.get("last_macro") or "").strip().strip('"')
        
        target = wanted if wanted in self.all_macro_names else (self.all_macro_names[0] if self.all_macro_names else None)
        
        if target:
            self.macro_list.select(target)  # déclenche chargement
            self.toast(f"Macro « {target} » chargée.")
        else:
            self.toast("Aucune macro. Créez-en une.")
            log.warning("Aucune macro trouvée au démarrage.")

    def _bootstrap_telegram(self):
        """S'exécute dans un thread pour ne pas bloquer l'UI."""
        try:
            self.tg.discard_backlog()
            self.tg.start()
            if self.tg.ready():
                log.info("TG Bridge prêt, envoi du message de démarrage.")
                # TÂCHE 6: Message de démarrage
                self.tg.send(f"Macro COC v{APP_VERSION} lancée.")
                # TÂCHE 4: État CoC
                self.tg.replace_controls(self._compose_status_for_tg("Prêt"), coc_launched=self._coc_is_running_tg)
            self.after(0, self._update_tg_status_vars) # Mettre à jour l'UI
        except Exception as e:
            log.error(f"Échec du bootstrap Telegram: {e}")

    # ---------- Gestion Macros (Req 3, 4, 9) ----------

    def _refresh_macro_sidebar(self, keep_selection: bool = True):
        """Met à jour la liste des macros et le cache de noms."""
        items = list_macros()
        self.all_macro_names = [n for n, _ in items]
        
        meta = {n: read_macro_meta(p) for n, p in items}
        self.macro_list.set_meta(meta)
        
        sel = self.current_macro_name if keep_selection else None
        self.macro_list.refresh(self.all_macro_names, selected=sel)
        self._filter_macro_list() # Appliquer filtre
        
    def _filter_macro_list(self, event=None):
        """Applique le filtre de recherche (Req 4)."""
        term = self.search_entry.get()
        self.macro_list.refresh(self.all_macro_names, self.current_macro_name, filter_term=term)

    def _show_macro_context_menu(self, event, name: str):
        """Affiche le menu clic-droit (Req 9)."""
        if self.current_state != State.IDLE:
            return # Pas de menu en cours d'opération
            
        # Sélectionne l'item cliqué
        self.macro_list.select(name, fire=True)
        
        # Affiche le menu
        self.macro_menu.place(x=event.x_root - self.winfo_x(), 
                              y=event.y_root - self.winfo_y())

    def _select_macro_by_name(self, name: str):
        """Chargement de la macro lors de la sélection."""
        if self.current_state != State.IDLE:
            log.warning("Ignoré: Sélection de macro pendant une opération.")
            # Resélectionne l'ancienne pour éviter un changement d'UI
            self.macro_list.select(self.current_macro_name, fire=False)
            return
            
        path = LEGACY_MACRO if name == LEGACY_MACRO.stem else macro_path_from_name(name)
        if not path.exists():
            self.toast("Fichier de macro introuvable.")
            log.error(f"Fichier macro {path} non trouvé pour '{name}'")
            return
            
        if self.model.load(path):
            self.current_macro_name = name
            self.current_macro_path = path
            self._update_macro_info_from_model()
            self.params["last_macro"] = name
            self._save_params() # Utilise le wrapper atomique
            self.toast(f"Macro « {name} » prête.")
        else:
            self.toast(f"Erreur chargement macro « {name} ».")

    def _update_macro_info_from_model(self):
        """Met à jour l'UI centrale avec les infos du modèle chargé."""
        if not self.current_macro_name:
            n, d = 0, 0.0
            name_txt = "Macro sélectionnée : —"
            meta_txt = "Non enregistrée"
        else:
            n = len(self.model.get_steps())
            d = self.model.duration()
            name_txt = f"Macro sélectionnée : {self.current_macro_name}"
            meta_txt = "Non enregistrée" if n <= 0 else f"Événements : {n} | Durée : {fmt_seconds(d)}"
        
        self.macro_events.set(str(n))
        self.macro_duration.set(fmt_dur_for_list(d))
        self.lbl_macro_name.configure(text=name_txt)
        self.lbl_macro_meta.configure(text=meta_txt)
        
        # Mettre à jour la liste
        if self.current_macro_name:
            meta = getattr(self.macro_list, "_meta", {})
            meta[self.current_macro_name] = (n, d)
            self.macro_list.set_meta(meta)
            self.macro_list.update_one(self.current_macro_name)

    def macro_new(self):           
        dlg = TextInputDialog(self, "Nouvelle macro", "Nom de la macro :", "Nouvelle Macro")
        name = dlg.show()
        if not name: return
        
        name = sanitize_macro_name(name)
        if self._is_protected_macro(name):
            self.toast("Ce nom est réservé.")
            return
            
        path = macro_path_from_name(name)
        if path.exists():
            if not messagebox.askyesno("Existe déjà", "Ce nom existe déjà. Écraser ?", icon="question"):
                return
                
        self.model.name = name
        self.model.clear()
        self.model.save(path) # Sauvegarde atomique
        
        self.current_macro_name = name
        self.current_macro_path = path
        
        self._refresh_macro_sidebar(keep_selection=False) # Ajoute la nouvelle
        self.macro_list.select(name, fire=False) # Sélectionne
        self._update_macro_info_from_model()
        
        self.params["last_macro"] = name
        self._save_params()
        self.toast(f"Macro « {name} » créée.")

    def macro_rename(self):
        if not self.current_macro_path or not self.current_macro_name:
            self.toast("Aucune macro à renommer.")
            return
        if self._is_protected_macro(self.current_macro_name):
            self.toast("Cette macro est protégée et ne peut pas être renommée.")
            return
            
        dlg = TextInputDialog(self, "Renommer la macro", "Nouveau nom :", self.current_macro_name)
        new_name = dlg.show()
        if not new_name: return
        
        new_name = sanitize_macro_name(new_name)
        if self._is_protected_macro(new_name):
            self.toast("Ce nom est réservé.")
            return
            
        new_path = macro_path_from_name(new_name)
        if new_path.exists() and new_path.resolve() != self.current_macro_path.resolve():
            if not messagebox.askyesno("Existe déjà", "Ce nom existe déjà. Écraser ?", icon="question"):
                return
                
        try:
            self.model.name = new_name
            
            # --- MODIFICATION: Forcer l'écriture ---
            # force_new_hash=True ignore le hash du contenu et force
            # la création du nouveau fichier avant de supprimer l'ancien.
            self.model.save(new_path, force_new_hash=True)
            # --- FIN MODIFICATION ---
            
            # Supprime l'ancien SEULEMENT s'il est différent
            if self.current_macro_path.exists() and self.current_macro_path.resolve() != new_path.resolve():
                try: os.remove(self.current_macro_path)
                except Exception as e: log.warning(f"Échec suppression ancien fichier macro: {e}")
                
            self.current_macro_name = new_name
            self.current_macro_path = new_path
            
            self._refresh_macro_sidebar(keep_selection=False)
            self.macro_list.select(new_name, fire=False)
            self._update_macro_info_from_model()
            
            self.params["last_macro"] = new_name
            self._save_params()
            self.toast(f"Macro renommée en « {new_name} ».")
        except Exception as e:
            log.error(f"Erreur renommage: {e}\n{traceback.format_exc()}")
            self.toast(f"Erreur renommage : {e}")

    def macro_delete(self):
        # Req 9: Interdire suppression si non IDLE
        if self.current_state != State.IDLE:
            self.toast("Arrêtez la macro avant de la supprimer.")
            return
        if not self.current_macro_path or not self.current_macro_name:
            self.toast("Aucune macro à supprimer.")
            return
        if self._is_protected_macro(self.current_macro_name):
            self.toast("Cette macro est protégée et ne peut pas être supprimée.")
            return
            
        if not messagebox.askyesno("Supprimer", f"Supprimer définitivement « {self.current_macro_name} » ?", icon="warning"):
            return
            
        try:
            if self.current_macro_path.exists():
                os.remove(self.current_macro_path)
        except Exception as e:
            self.toast(f"Suppression impossible : {e}")
            log.error(f"Échec suppression macro: {e}\n{traceback.format_exc()}")
            return
            
        deleted = self.current_macro_name
        self.current_macro_name = None
        self.current_macro_path = None
        self.model.clear()
        
        self._refresh_macro_sidebar(keep_selection=False)
        self._update_macro_info_from_model() # Met l'UI à "Non enregistrée"

        # Sélectionne la première macro restante
        if self.all_macro_names:
            self.macro_list.select(self.all_macro_names[0])
        else:
            self.params["last_macro"] = ""
            self._save_params()
            self.toast("Aucune macro restante.")
            
        self.toast(f"Macro « {deleted} » supprimée.")

    def macro_duplicate(self):
        """Duplique la macro sélectionnée (Req 9)."""
        if not self.current_macro_path or not self.current_macro_name:
            self.toast("Aucune macro à dupliquer.")
            return
            
        dlg = TextInputDialog(self, "Dupliquer la macro", "Nom de la copie :", f"{self.current_macro_name} (Copie)")
        new_name = dlg.show()
        if not new_name: return
        
        new_name = sanitize_macro_name(new_name)
        if self._is_protected_macro(new_name):
            self.toast("Ce nom est réservé.")
            return
            
        new_path = macro_path_from_name(new_name)
        if new_path.exists():
            if not messagebox.askyesno("Existe déjà", "Ce nom existe déjà. Écraser ?", icon="question"):
                return
        try:
            shutil.copy(self.current_macro_path, new_path)
            self._refresh_macro_sidebar(keep_selection=False)
            self.macro_list.select(new_name, fire=True) # Sélectionne la nouvelle
            self.toast(f"Macro « {new_name} » créée.")
        except Exception as e:
            log.error(f"Erreur duplication: {e}\n{traceback.format_exc()}")
            self.toast(f"Erreur duplication: {e}")

    def macro_export(self):
        """Exporte la macro en JSON (Req 9)."""
        if not self.current_macro_path or not self.current_macro_name:
            self.toast("Aucune macro à exporter.")
            return
            
        save_path = filedialog.asksaveasfilename(
            title="Exporter la macro",
            initialdir=BASE_DIR,
            initialfile=f"{self.current_macro_name}.json",
            defaultextension=".json",
            filetypes=[("Fichiers JSON", "*.json")]
        )
        if not save_path: return
        
        try:
            shutil.copy(self.current_macro_path, save_path)
            self.toast(f"Macro exportée vers {Path(save_path).name}")
        except Exception as e:
            log.error(f"Erreur exportation: {e}\n{traceback.format_exc()}")
            self.toast(f"Erreur exportation: {e}")

    def macro_import(self):
        """Importe une macro JSON (Req 9)."""
            
        open_path_str = filedialog.askopenfilename(
            title="Importer une macro",
            initialdir=BASE_DIR,
            defaultextension=".json",
            filetypes=[("Fichiers JSON", "*.json")]
        )
        if not open_path_str: return
        
        open_path = Path(open_path_str)
        try:
            # On lit pour valider le format
            name, steps, _, _ = read_macro_file(open_path)
            if not steps:
                raise ValueError("Le fichier JSON ne contient pas d'étapes (steps).")
            
            # On utilise le nom du *fichier* comme nom de macro, pas le nom interne
            new_name = sanitize_macro_name(open_path.stem)
            new_path = macro_path_from_name(new_name)
            
            if new_path.exists():
                if not messagebox.askyesno("Existe déjà", f"La macro « {new_name} » existe déjà. Écraser ?", icon="question"):
                    return
            
            # On ré-écrit le fichier dans le dossier macros
            # pour garantir le bon format (avec hash, etc.)
            write_macro_file(new_path, new_name, steps)
            
            self._refresh_macro_sidebar(keep_selection=False)
            self.macro_list.select(new_name, fire=True)
            self.toast(f"Macro « {new_name} » importée.")
            
        except Exception as e:
            log.error(f"Erreur importation: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Erreur d'importation", f"Impossible d'importer le fichier: {e}")

    # ---------- Enregistrement (Req 1, 7) ----------
    def toggle_record(self):
        """Bascule l'enregistrement (protégé par la machine à états)."""
        
        # Cas 1: On arrête l'enregistrement
        if self.current_state == State.RECORDING:
            self._stop_record_with_tail_trim()
            # _stop_record... s'occupe de la transition vers IDLE
            
        # Cas 2: On démarre l'enregistrement
        elif self.current_state == State.IDLE:
            if not self.current_macro_path:
                self.toast("Choisis ou crée une macro avant d’enregistrer.")
                return
            
            # Tente la transition
            if not self.transition_to(State.RECORDING):
                return
                
            self.model.clear()
            self._update_macro_info_from_model()
            
            try:
                self.rec.start()
                self.toast("Prépare-toi… démarrage dans 3 s.")
                threading.Thread(target=self._grace_countdown, daemon=True, name="RecGrace").start()

            except Exception as e:
                msg = f"Erreur pynput: {e}"
                log.error(f"{msg}\n{traceback.format_exc()}")
                self.toast(msg)
                messagebox.showerror("Erreur pynput", f"Impossible de démarrer les écouteurs (pynput).\n{e}")
                self.transition_to(State.IDLE) 

        # Cas 3: Bloqué (ex: en lecture)
        else:
            self.toast("Opération impossible dans l'état actuel.")

    def _grace_countdown(self):
        """Thread pour le compte à rebours avant enregistrement."""
        for remaining in (3, 2, 1):
            if self.current_state != State.RECORDING: # Vérifie si annulé
                return
            self.toast(f"Démarrage dans {remaining} s")
            time.sleep(1.0)
        if self.current_state == State.RECORDING:
            self.toast("Enregistrement en cours.")

    def _stop_record_with_tail_trim(self):
        """Arrête l'enregistrement, traite les données, et sauvegarde."""
        if self.current_state != State.RECORDING:
            return
            
        self.rec.stop()
        trimmed = self._trim_tail(self.rec.steps, 3.0)
        
        self.model.set_steps(trimmed)
        
        try:
            self.model.save(self.current_macro_path)
        except Exception as e:
            self.toast(f"Erreur sauvegarde : {e}")
            log.error(f"Échec sauvegarde macro: {e}\n{traceback.format_exc()}")
            
        self._update_macro_info_from_model()
        self.toast(f"Enregistrement arrêté. Événements gardés : {len(trimmed)}.")
        
        self.transition_to(State.IDLE) # Terminé

    # ---------- Commandes Telegram (Req 2, 5, 6) ----------
    
    def _compose_status_for_tg(self, prefix: str) -> str:
        """Helper pour générer le texte du clavier Telegram."""
        loop_state = "ON" if self._auto_loop_enabled() else "OFF"
        return f"{prefix} | Macro: {self._macro_label()} | Loop: {loop_state}"

    def _on_tg_command(self, cmd: str, meta: dict):
        """Handler central pour les commandes reçues de Telegram."""
        
        # Helper pour Tâches 2, 3, 4
        def cleanup_menu_and_replace_controls(status_text: str):
            """Supprime le msg menu et remplace le msg de contrôles."""
            if self.tg._last_menu_id:
                self.tg.delete_message(self.tg._last_menu_id)
                self.tg._last_menu_id = None
            self.tg.replace_controls(self._compose_status_for_tg(status_text), coc_launched=self._coc_is_running_tg) # TÂCHE 4

        if cmd == "STOP":
            self.after(0, lambda: self.force_stop_all(notify_tg=False))
            self.tg.replace_controls(self._compose_status_for_tg("Lecture arrêtée"), coc_launched=self._coc_is_running_tg) # TÂCHE 4
                
        elif cmd == "GO":
            self.after(0, lambda: self.on_play(notify_tg=False))
            self.tg.replace_controls(self._compose_status_for_tg("Lecture démarrée"), coc_launched=self._coc_is_running_tg) # TÂCHE 4
                
        elif cmd == "MENU":
            # TÂCHE 2: Passer l'état loop
            self.tg.replace_menu("Paramètres", loop_state=self._auto_loop_enabled())

        elif cmd == "BACK":
            # Supprime le message "Paramètres"
            if self.tg._last_menu_id:
                self.tg.delete_message(self.tg._last_menu_id)
                self.tg._last_menu_id = None
            self.tg.replace_controls(self._compose_status_for_tg("Prêt"), coc_launched=self._coc_is_running_tg) # TÂCHE 4

        elif cmd == "SHUTDOWN_ASK":
            self.tg.push_shutdown_confirm(origin_msg_id=meta.get("message_id"))
            # Supprime le menu d'où on vient
            if self.tg._last_menu_id:
                self.tg.delete_message(self.tg._last_menu_id)
                self.tg._last_menu_id = None
                
        elif cmd == "SHUTDOWN_CONFIRM":
            chat_id = meta.get("from")
            self.tg.send(f"🛑 Extinction du PC demandée par {chat_id}...")
            # Lancement de l'extinction
            self.after(0, self.perform_shutdown, "Telegram", chat_id)
            
        elif cmd == "SHUTDOWN_CANCEL":
            self.tg.send("Extinction annulée.")
            cleanup_menu_and_replace_controls("Prêt")

        # Commandes exécutées en thread pour ne pas bloquer l'UI
        elif cmd == "CAPTURE":
            threading.Thread(target=self._send_capture_to_telegram, daemon=True, name="TGCapture").start()

        elif cmd == "LAUNCH_COC":
            # TÂCHE 4: Lancer la vérification
            self.after(0, self.launch_coc_for_telegram)
            # (le message "Lancement..." est géré dans la fonction)

        elif cmd == "DUMMY_COC_STATUS":
            # TÂCHE 4: Clic sur le bouton "Déjà lancé", ne rien faire
            pass 

        elif cmd == "RELOAD_COC":
            self.after(0, lambda: self.play_macro_once(RECHARGER_MACRO_NAME, notify_tg=True))
            cleanup_menu_and_replace_controls("Rechargement CoC...")

        # --- NOUVEAU HANDLER ---
        elif cmd == "VALIDATE_ARRIVAL":
            self.after(0, lambda: self.play_macro_once(VALIDER_MACRO_NAME, notify_tg=True))
            cleanup_menu_and_replace_controls(f"Lancement {VALIDER_MACRO_NAME}...")
        # --- FIN NOUVEAU HANDLER ---

        # TÂCHE 3: Suppression AutoCapture
        # ... Handlers AUTO_CAP_ON/OFF supprimés ...

        # TÂCHE 2: Remplacement Loop
        # ... Handlers LOOP_ON/OFF supprimés ...
        
        elif cmd == "TOGGLE_LOOP":
            # TÂCHE 2: Nouveau handler
            current_state = self._auto_loop_enabled()
            new_state = not current_state
            self.params["auto_loop"] = "1" if new_state else "0"
            self._save_params()
            self.tg.send(f"Loop {'ON' if new_state else 'OFF'}.")
            # Rafraîchir le menu pour mettre à jour le libellé
            self.tg.replace_menu("Paramètres", loop_state=new_state)

        elif cmd == "SELECT_MACRO_LIST":
            all_macros = [name for name, path in list_macros()]
            self.tg.push_macro_selection(all_macros)
            # Supprime le menu d'où on vient
            if self.tg._last_menu_id:
                self.tg.delete_message(self.tg._last_menu_id)
                self.tg._last_menu_id = None
                
        elif cmd == "CANCEL_SELECTION":
            cleanup_menu_and_replace_controls("Sélection annulée")
                
        elif cmd.startswith("SELECT_MACRO:"):
            macro_name = cmd[13:]
            self.after(0, lambda n=macro_name: self._select_macro_by_name_from_tg(n))

    def _select_macro_by_name_from_tg(self, name: str):
        """Change la macro sélectionnée depuis Telegram."""
        if self.current_state != State.IDLE:
            self.tg.send(f"Impossible de changer de macro: opération en cours.")
            self.tg.replace_controls(self._compose_status_for_tg("Erreur"), coc_launched=self._coc_is_running_tg) # TÂCHE 4
            return
            
        if name in self.all_macro_names:
            self.macro_list.select(name, fire=True) # fire=True déclenche _select_macro_by_name
            self.tg.replace_controls(self._compose_status_for_tg(f"Prêt (Macro: {name})"), coc_launched=self._coc_is_running_tg) # TÂCHE 4
        else:
            self.tg.send(f"Erreur : Macro « {name} » introuvable.")
            self.tg.replace_controls(self._compose_status_for_tg("Erreur"), coc_launched=self._coc_is_running_tg) # TÂCHE 4

    def _send_capture_to_telegram(self):
        """Prend et envoie une capture (thread-safe)."""
        if not self.tg.ready():
            self.after(0, lambda: self.toast("Telegram non configuré."))
            return
            
        png = grab_screenshot_png_bytes()
        if png:
            caption = f"📸 Capture — {self._macro_label()}"
            
            # TÂCHE 3: Remplacement send_or_edit_photo par send_photo
            self.tg.send_photo(png, caption=caption)

            self.after(0, lambda: self.toast("Capture envoyée sur Telegram."))
            self.tg.replace_controls(self._compose_status_for_tg("Capture envoyée"), coc_launched=self._coc_is_running_tg) # TÂCHE 4
        else:
            self.tg.send("Échec de la capture d’écran.")
            self.after(0, lambda: self.toast("Impossible de capturer l’écran."))

    def _send_gif_to_telegram(self, duration_sec: int = 4, fps: int = 5):
        """Crée et envoie un GIF (Req 6)."""
        if not self.tg.ready():
            self.after(0, lambda: self.toast("Telegram non configuré."))
            return
            
        if not PIL_AVAILABLE:
            log.warning("Demande de GIF, mais Pillow n'est pas installée.")
            self.tg.send("Erreur: Librairie Pillow (PIL) manquante pour créer le GIF. Envoi d'une photo...")
            self._send_capture_to_telegram()
            return
            
        log.info(f"Démarrage capture GIF ({duration_sec}s @ {fps}fps)...")
        self.tg.send(f"⏳ Création du GIF ({duration_sec}s)...")
        
        frames = []
        start_time = time.perf_counter()
        end_time = start_time + duration_sec
        frame_delay = 1.0 / fps
        
        try:
            while time.perf_counter() < end_time:
                frame_start = time.perf_counter()
                png_bytes = grab_screenshot_png_bytes()
                if png_bytes:
                    img = Image.open(BytesIO(png_bytes))
                    frames.append(img)
                
                # Attente
                elapsed = time.perf_counter() - frame_start
                wait = max(0, frame_delay - elapsed)
                time.sleep(wait)

            if not frames:
                self.tg.send("Échec: Aucune image capturée pour le GIF.")
                return
            
            log.info(f"Assemblage de {len(frames)} images en GIF...")
            bio = BytesIO()
            frames[0].save(
                bio, 
                format="GIF", 
                save_all=True, 
                append_images=frames[1:], 
                duration=int(1000 / fps), 
                loop=0
            )
            
            log.info("Envoi du GIF...")
            self.tg.send_animation(bio.getvalue(), caption=f"GIF Capture — {self._macro_label()}")
            self.tg.replace_controls(self._compose_status_for_tg("GIF envoyé"))

        except Exception as e:
            log.error(f"Échec création/envoi GIF: {e}\n{traceback.format_exc()}")
            self.tg.send(f"Erreur lors de la création du GIF: {e}. Fallback photo...")
            self._send_capture_to_telegram()

    # ---------- Lancer / Recharger CoC (Req 5) ----------
    def launch_coc_once(self):
            if self._coc_launched_once:
                self.toast("CoC déjà lancé (depuis cette session).")
                if self.tg.ready(): self.tg.send("ℹ️ CoC déjà lancé.")
                return
            
            path = (self.params.get("coc_path", "") or "").strip()
            if not path:
                msg = "Chemin CoC manquant dans Paramètres."
                self.toast(msg)
                if self.tg.ready(): self.tg.send(f"❗ {msg}")
                return
                
            # --- MODIFICATION (Appel simple) ---
            # Laisse notify_tg=True (par défaut)
            self._actually_launch_coc(path)
            # --- FIN MODIFICATION ---
            self._coc_launched_once = True # Marqué pour cette session

    def restart_coc(self):
        """Tente de tuer le processus puis de le relancer (Req 5)."""
        path = (self.params.get("coc_path", "") or "").strip()
        if not path:
            msg = "Chemin CoC manquant dans Paramètres."
            self.toast(msg)
            if self.tg.ready(): self.tg.send(f"❗ {msg}")
            return

        if os.name == 'nt':
            exe_name = resolve_exe_from_path(path)
            if exe_name:
                # kill_process_by_name gère taskkill + wmic fallback (Req 5)
                kill_process_by_name(exe_name)
                time.sleep(2.0) # Laisse le temps au processus de mourir
            else:
                log.warning(f"Impossible de résoudre le nom de l'exécutable depuis {path}")

        self._actually_launch_coc(path)
        self._coc_launched_once = True

    def _actually_launch_coc(self, path: str, notify_tg: bool = True):
        """
        Exécute le lancement.
        notify_tg=False est utilisé par launch_coc_for_telegram
        pour éviter les messages en double.
        """
        try:
            self.toast(f"Lancement de « {path} »...")
            if os.name == 'nt':
                os.startfile(path)
            else:
                subprocess.Popen([path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # --- MODIFICATION ---
            if notify_tg and self.tg.ready():
                self.tg.send("🚀 CoC lancé.")
            # --- FIN MODIFICATION ---
            
        except FileNotFoundError:
            msg = f"Erreur : Fichier/chemin non trouvé {path}"
            log.error(msg)
            self.toast(msg)
            if self.tg.ready(): self.tg.send(f"❌ {msg}")
        except Exception as e:
            msg = f"Impossible de lancer : {e}"
            log.error(f"{msg}\n{traceback.format_exc()}")
            self.toast(msg)
            if self.tg.ready(): self.tg.send(f"❌ {msg}")

    # ---------- Lecture (Req 1) ----------
    def on_play(self, notify_tg: bool = True):
        """Démarre la lecture (protégé par la machine à états)."""
        if self.current_state != State.IDLE:
            self.toast("Arrête l'opération en cours avant de lire.")
            return
            
        steps = self.model.get_steps()
        if not steps:
            self.toast("Aucune action enregistrée.")
            return
            
        if not self.transition_to(State.PLAYING):
            return
            
        self.cycle_count.set(0)
        self.session_seconds = 0.0
        self.session_elapsed.set("00:00")
        
        loop = self._auto_loop_enabled()
        self.player.play(steps, loop=loop)
        self._start_play_timer()
        
        self.toast(f"Lecture démarrée. Boucle = {loop}.")
        if notify_tg and self.tg.ready():
            self.tg.replace_controls(self._compose_status_for_tg(f"Lecture (Loop={loop})"), coc_launched=self._coc_is_running_tg) # TÂCHE 4
            
        if self.current_macro_name:
            self.params["last_macro_played"] = self.current_macro_name
            self._save_params()
    # ---- Timer de lecture UI (Req 1) ----
    def _start_play_timer(self):
        self._play_timer_running = True
        self._last_tick = time.perf_counter()
        # Req 1: Remplacer after(1000) par after(200)
        self.after(200, self._tick_play_timer)

    def _stop_play_timer(self):
        self._play_timer_running = False
        self._last_tick = None

    def _tick_play_timer(self):
        """Timer UI précis (Req 1)."""
        if not self._play_timer_running:
            return
            
        now = time.perf_counter()
        if self._last_tick is None:
            self._last_tick = now
            
        # Req 1: Accumulation précise
        dt = now - self._last_tick
        self._last_tick = now
        
        if self.current_state != State.PLAYING:
            self._stop_play_timer()
            return
            
        self.session_seconds += dt
        self.session_elapsed.set(fmt_seconds(self.session_seconds))
        self._update_status_bar() # Met à jour le temps dans la barre de statut
        
        self.after(200, self._tick_play_timer) # Boucle rapide

    # ---------- Aides ----------
    def _trim_tail(self, steps, tail_seconds: float):
        """Coupe les X dernières secondes d'un enregistrement."""
        if not steps or tail_seconds <= 0:
            return list(steps)
            
        total = sum(max(0.0, ev.get("t", 0.0)) for ev in steps)
        cutoff = max(0.0, total - tail_seconds)
        if cutoff <= 0.0:
            return []
            
        trimmed, acc = [], 0.0
        for ev in steps:
            dt = max(0.0, ev.get("t", 0.0))
            if acc + dt < cutoff:
                trimmed.append(ev)
                acc += dt
            else:
                # Ajoute un dernier step partiel pour atteindre le cutoff
                remaining = cutoff - acc
                if remaining > 0.001:
                    ev_copy = ev.copy()
                    ev_copy["t"] = remaining
                    trimmed.append(ev_copy)
                break # Terminé
        return trimmed

    # ---------- Hotkeys / Arrêt global (Req 1) ----------
    def _register_hotkeys(self):
        self.player.on_cycle = self._on_player_cycle
        self.player.on_stopped = self._on_player_stopped
        try:
            keyboard.add_hotkey("f1", self.on_toggle_global)
            keyboard.add_hotkey("ctrl+shift+1", self._hk_play)
            keyboard.add_hotkey("ctrl+shift+0", lambda: self.force_stop_all(notify_tg=True))
        except Exception as e:
            msg = f"Échec enregistrement hotkeys: {e}"
            log.error(f"{msg}\n{traceback.format_exc()}")
            self.toast(msg)
            messagebox.showerror("Erreur Hotkeys", f"Impossible d'enregistrer les raccourcis clavier (F1, etc.).\n{e}")

    def _on_player_cycle(self):
        self.after(0, lambda: self.cycle_count.set(self.cycle_count.get() + 1))

    def _on_player_stopped(self):
        """Callback du Player quand il s'arrête (fin ou stop())."""
        log.info("Player a signalé 'stopped'.")
        # Le Player s'exécute dans un thread, on notifie l'UI
        self.after(0, self._handle_player_stopped_ui)

    def _handle_player_stopped_ui(self):
        """Met à jour l'UI après l'arrêt du player."""
        self._stop_play_timer()
        # On ne transitionne que si on était en lecture
        # (évite double transition si force_stop_all est appelé)
        if self.current_state == State.PLAYING:
            self.transition_to(State.IDLE)
        self.toast("Lecture terminée.")

    def on_toggle_global(self):
        """Hotkey F1: bascule Play/Stop."""
        if self.current_state == State.PLAYING:
            self.force_stop_all(notify_tg=True)
        elif self.current_state == State.IDLE:
            self.on_play(notify_tg=True)

    def _hk_play(self):
        """Hotkey Ctrl+Shift+1: Force Play."""
        if self.current_state == State.IDLE:
            self.on_play(notify_tg=True)
        else:
            self.toast("Opération en cours, 'Play' ignoré.")

    def force_stop_all(self, notify_tg: bool = True):
        """Arrêt d'urgence (UI, hotkey, ou TG)."""
        log.info(f"Force stop all demandé (notify_tg={notify_tg})")
        did_stop = False
        
        if self.current_state == State.RECORDING:
            self._stop_record_with_tail_trim()
            did_stop = True
            
        if self.current_state == State.PLAYING:
            self.player.stop() # Appelle on_stopped, qui transitionne vers IDLE
            did_stop = True

        # Assure que le timer UI s'arrête dans tous les cas
        self._stop_play_timer()
        
        if did_stop:
            self.toast("Opération arrêtée.")
            if notify_tg and self.tg.ready():
                self.tg.replace_controls(self._compose_status_for_tg("Opération arrêtée"), coc_launched=self._coc_is_running_tg) # TÂCHE 4
        else:
            self.toast("Rien à arrêter.")

    # ---------- Extinction système (Req 5) ----------
    def request_local_shutdown(self):
        """Demande d'extinction depuis l'UI (double confirmation Req 5)."""
            
        if not messagebox.askyesno("Confirmer l'extinction", "Éteindre l'ordinateur ?", icon="warning"):
            return
        if not messagebox.askyesno("Double confirmation", "Veuillez confirmer.\nL'ordinateur va s'éteindre immédiatement.", icon="error"):
            return
            
        self.perform_shutdown(origin="UI (Local)", chat_id=None)

    def perform_shutdown(self, origin: str, chat_id: Optional[int]):
        """Exécute l'extinction (Req 5: log)."""
        log.warning(f"EXTINCTION SYSTÈME demandée par: {origin} (ChatID: {chat_id})")
        if self.tg.ready():
            self.tg.send(f"⚠️ EXTINCTION SYSTÈME ⚠️\nDemandé par: {origin}")
            
        try:
            self.force_stop_all(notify_tg=False)
        except Exception:
            pass
            
        threading.Thread(target=self._shutdown_system, daemon=True, name="Shutdown").start()

    def _shutdown_system(self):
        time.sleep(2) # Laisse le temps au message TG de partir
        try:
            if os.name == "nt":
                os.system("shutdown /s /t 1") # 1s délai
            else:
                ret = os.system("systemctl poweroff")
                if ret != 0:
                    os.system("shutdown -h now")
        except Exception as e:
            log.error(f"Échec de l'extinction: {e}")
            self.toast(f"Échec de l’extinction: {e}")

    # ---------- Divers ----------
    def toast(self, msg: str):
        """Affiche un message dans la barre de statut (thread-safe)."""
        def _set_toast():
            self.status_var.set(msg)
        self.after(0, _set_toast)

    def safe_quit(self):
        """Nettoyage avant de quitter."""
        log.info("Demande de fermeture de l'application.")
        try:
            self.force_stop_all(notify_tg=False)
        except Exception:
            pass
        
        # TÂCHE 6: Envoyer message de fermeture
        if self.tg.ready():
            try:
                log.info("Envoi du message de fermeture Telegram...")
                self.tg.send("Application fermée.")
                # L'appel est synchrone mais non bloquant (utilise la session)
                # Laisse 100ms pour être sûr que la requête parte
                time.sleep(0.1) 
            except Exception as e:
                log.error(f"Échec envoi message fermeture: {e}")
        
        try:
            self.tg.stop()
        except Exception:
            pass
            
        # TÂCHE 1: Nettoyage logs à la fermeture
        try:
            log.info(f"Nettoyage des logs > {24}h lors de la fermeture...")
            clean_old_logs(LOG_PATH.parent, "app.log*", 24)
        except Exception as e:
            log.error(f"Erreur clean_old_logs (shutdown): {e}")

        # Fermer les hotkeys (sinon le process peut rester)
        try:
            keyboard.remove_all_hotkeys()
        except Exception as e:
            log.warning(f"Échec suppression hotkeys: {e}")
            
        log.info("Application fermée.\n" + "="*30)
        self.destroy()

    # --- TÂCHE 4: Helpers de vérification CoC ---
    
    def is_process_running(self, exe_name: str) -> bool:
        """Vérifie si un processus est en cours via tasklist (Windows)."""
        if os.name != 'nt' or not exe_name:
            return False
        try:
            cmd = ["tasklist", "/NH", "/FI", f"IMAGENAME eq {exe_name}"]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=3, 
                creationflags=subprocess.CREATE_NO_WINDOW, 
                encoding='utf-8'
            )
            if exe_name.lower() in result.stdout.lower():
                return True
        except Exception as e:
            log.error(f"Erreur tasklist: {e}")
        return False

    def launch_coc_for_telegram(self):
        """Lance CoC et vérifie son état pour Telegram."""
        path = (self.params.get("coc_path", "") or "").strip()
        if not path:
            self.tg.send("❗ Chemin CoC manquant dans Paramètres.")
            self.tg.replace_controls(self._compose_status_for_tg("Erreur"), coc_launched=self._coc_is_running_tg)
            return
        
        exe_name = resolve_exe_from_path(path)
        if not exe_name:
            log.warning(f"Impossible de résoudre l'exe pour {path}. Lancement en aveugle.")
            # --- MODIFICATION ---
            self._actually_launch_coc(path, notify_tg=False) # notify_tg=False
            # --- FIN MODIFICATION ---
            self.tg.send("🚀 CoC lancé (validation impossible).")
            # On met le bouton à jour en supposant que c'est bon
            self._coc_is_running_tg = True
            self.tg.replace_controls(self._compose_status_for_tg("Prêt"), coc_launched=True)
            return

        self.tg.replace_controls(self._compose_status_for_tg("Lancement CoC..."), coc_launched=False)
        # --- MODIFICATION ---
        self._actually_launch_coc(path, notify_tg=False) # notify_tg=False
        # --- FIN MODIFICATION ---
        
        # Start checker thread
        log.info(f"Lancement du thread de vérification pour {exe_name}")
        threading.Thread(target=self._check_coc_process, args=(exe_name,), daemon=True, name="CoC-Checker").start()

    def _check_coc_process(self, exe_name: str):
        """Thread: attend et vérifie si le processus CoC est lancé."""
        time.sleep(2.0) # Attente 2s (demandé)
        is_running = self.is_process_running(exe_name)
        log.info(f"Vérification CoC ({exe_name}): {'Lancé' if is_running else 'Échec'}")
        # Notifier le thread UI
        self.after(0, self._update_coc_status_on_tg, is_running)

    def _update_coc_status_on_tg(self, is_running: bool):
        """Mise à jour de l'UI Telegram depuis le thread principal."""
        self._coc_is_running_tg = is_running
        if is_running:
            self.tg.replace_controls(self._compose_status_for_tg("Prêt"), coc_launched=True)
        else:
            self.tg.send("❌ Échec du lancement de CoC (processus non détecté).")
            self.tg.replace_controls(self._compose_status_for_tg("Échec lancement"), coc_launched=False)

    def _ensure_protected_macros(self):
        """Vérifie que les macros système (protégées) existent sur le disque."""
        log.info("Vérification des macros protégées...")
        protected_macros = [RECHARGER_MACRO_NAME, VALIDER_MACRO_NAME]
        
        for name in protected_macros:
            path = macro_path_from_name(name)
            if not path.exists():
                try:
                    log.warning(f"Macro protégée '{name}' manquante. Création du fichier vide.")
                    # Crée un fichier de macro vide
                    write_macro_file(path, name, steps=[]) 
                except Exception as e:
                    log.error(f"Échec de la création de la macro protégée '{name}': {e}")

# =========================
#     Selftest (Req 8)
# =========================
def run_selftest():
    """
    Exécute une suite de tests non-UI (Req 8).
    Lance via: python votre_script.py --selftest
    """
    if Mock is None or patch is None:
        print("Erreur: Le module 'unittest.mock' est requis pour --selftest.")
        sys.exit(1)

    print("="*30)
    print("       RUNNING SELFTESTS")
    print("="*30)
    
    test_dir = CONFIG_DIR / "selftest"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_csv_path = test_dir / "test_data.csv"
    test_json_path = test_dir / "test_macro.json"
    
    # --- 1. Test I/O Atomique (Req 3, 8) ---
    print("\n[TEST] 1/4: I/O Atomique (CSV/JSON)...")
    try:
        # CSV
        params = {"test_key": "val1", "timer": "100"}
        write_params_csv(test_csv_path, params)
        assert test_csv_path.exists()
        assert not test_csv_path.with_suffix(".csv.tmp").exists()
        read_p = read_params_csv(test_csv_path)
        assert read_p.get("test_key") == "val1"
        
        # JSON (Hash check)
        steps1 = [{"t": 1.0, "type": "mouse_move", "data": {"x": 1, "y": 1}}]
        steps2 = [{"t": 1.0, "type": "mouse_move", "data": {"x": 2, "y": 2}}]
        
        # Écriture 1
        written1 = write_macro_file(test_json_path, "Test", steps1)
        assert written1 is True
        n, s, h1, _ = read_macro_file(test_json_path)
        assert h1 == get_macro_hash(steps1)
        
        # Écriture 2 (identique)
        written2 = write_macro_file(test_json_path, "Test", steps1, current_hash=h1)
        assert written2 is False # Ne doit pas ré-écrire
        
        # Écriture 3 (différente)
        written3 = write_macro_file(test_json_path, "Test", steps2, current_hash=h1)
        assert written3 is True
        n, s, h2, _ = read_macro_file(test_json_path)
        assert h2 != h1
        
        print("  [PASS] I/O Atomique OK.")
    except Exception as e:
        print(f"  [FAIL] I/O Atomique: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    # --- 2. Test Précision Timer (Req 1, 8) ---
    print("\n[TEST] 2/4: Précision Timer Player (10s)...")
    try:
        dummy_recorder = Mock(spec=Recorder)
        dummy_recorder.mark_self_play = Mock()
        player = Player(dummy_recorder)
        
        # 10 steps de 1s = 10s total
        steps = [{"t": 1.0, "type": "nop", "data": {}}] * 10
        player._apply = lambda ev: None # Stub
        
        t_start = time.perf_counter()
        player._run_one_cycle(steps)
        t_end = time.perf_counter()
        
        duration = t_end - t_start
        target_duration = 10.0
        drift_pct = (abs(duration - target_duration) / target_duration) * 100
        
        print(f"  Durée cible: {target_duration:.3f}s")
        print(f"  Durée mesurée: {duration:.3f}s")
        print(f"  Dérive: {drift_pct:.2f}%")
        
        assert drift_pct < 5.0 # Tolérance 5% (Req 8)
        print("  [PASS] Précision Timer OK.")
    except Exception as e:
        print(f"  [FAIL] Précision Timer: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    # --- 3. Test Stub Telegram (Req 2, 8) ---
    print("\n[TEST] 3/4: Stub Telegram (editMessageMedia)...")
    try:
        mock_session = Mock(spec=requests.Session)
        bridge = TelegramBridge(token="test", chat_id="123", on_command=lambda c, m: None)
        bridge.session = mock_session
        bridge._rate_limiter_queue.put(True) # Pré-charge le jeton
        
        dummy_png = b"dummy_png_bytes"
        
        # --- Cas 1: sendPhoto (pas d'ID)
        mock_session.request.return_value = Mock(
            ok=True, 
            status_code=200, 
            json=lambda: {"ok": True, "result": {"message_id": 12345}}
        )
        
        bridge.send_or_edit_photo(dummy_png, "Caption 1")
        
        # Vérifie que sendPhoto a été appelé
        call_args = mock_session.request.call_args
        assert call_args[0][1].endswith("/sendPhoto")
        assert bridge.last_photo_message_id == 12345
        
        print("  [PASS] sendPhoto OK.")

        # --- Cas 2: editMessageMedia (ID existant)
        bridge._rate_limiter_queue.put(True) # Jet_on
        mock_session.request.return_value = Mock(
            ok=True, 
            status_code=200, 
            json=lambda: {"ok": True, "result": {"message_id": 12345}}
        )
        
        bridge.send_or_edit_photo(dummy_png, "Caption 2")

        # Vérifie que editMessageMedia a été appelé
        call_args = mock_session.request.call_args
        assert call_args[0][1].endswith("/editMessageMedia")
        assert 'message_id": 12345' in call_args[1]['data']['media'] # Vérifie qu'on édite le bon
        print("  [PASS] editMessageMedia (stub) OK.")

        # --- Cas 3: Fallback (edit échoue)
        bridge._rate_limiter_queue.put(True) # Jeton
        # Simule un échec de l'edit
        mock_session.request.side_effect = [
            # 1er appel (editMessageMedia)
            Mock(ok=False, status_code=400, json=lambda: {"ok": False, "description": "Bad request"}),
            # 2e appel (sendPhoto)
            Mock(ok=True, status_code=200, json=lambda: {"ok": True, "result": {"message_id": 54321}})
        ]
        
        bridge.send_or_edit_photo(dummy_png, "Caption 3")
        assert bridge.last_photo_message_id == 54321 # Doit avoir le nouvel ID
        assert mock_session.request.call_count == 2
        print("  [PASS] editMessageMedia (fallback) OK.")

    except Exception as e:
        print(f"  [FAIL] Stub Telegram: {e}\n{traceback.format_exc()}")
        sys.exit(1)
        
    # --- 4. Test Validation Callback (Req 7) ---
    print("\n[TEST] 4/4: Validation Callback Data (64 octets)...")
    try:
        bridge = TelegramBridge(token="test", chat_id="123", on_command=lambda c, m: None)
        
        short = "SELECT_MACRO:MaMacro"
        hashed_short = bridge._hash_callback_data(short)
        assert hashed_short == short
        assert len(hashed_short.encode('utf-8')) <= 64
        
        long = "SELECT_MACRO:Ceci est un nom de macro extrêmement long qui va dépasser la limite de 64 octets"
        hashed_long = bridge._hash_callback_data(long)
        
        assert hashed_long != long
        assert len(hashed_long.encode('utf-8')) == 64
        assert hashed_long.startswith("SELECT_MACRO:Ceci est un nom de macro extrêmement long")
        assert bridge._callback_data_map[hashed_long] == long
        
        # Test UTF-8 (caractères multi-octets)
        long_utf8 = "SELECT_MACRO:Ma macro spéciale avec des émojis 😊😊😊😊😊😊😊😊😊😊😊😊"
        hashed_utf8 = bridge._hash_callback_data(long_utf8)
        assert hashed_utf8 != long_utf8
        assert len(hashed_utf8.encode('utf-8')) <= 64
        assert bridge._callback_data_map[hashed_utf8] == long_utf8
        
        print("  [PASS] Validation Callback Data OK.")
    except Exception as e:
        print(f"  [FAIL] Validation Callback Data: {e}\n{traceback.format_exc()}")
        sys.exit(1)
        
    # --- Cleanup ---
    try:
        shutil.rmtree(test_dir)
        print(f"\nNettoyage {test_dir} OK.")
    except Exception: pass
    
    print("\n" + "="*30)
    print("     ALL SELFTESTS PASSED")
    print("="*30)


# =========================
#         Lancement
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Macro COC Application")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Exécute les tests internes non-UI et quitte."
    )
    args = parser.parse_args()

    # Initialiser le logging (sera utilisé par selftest ou App)
    setup_logging()

    if args.selftest:
        try:
            run_selftest()
            sys.exit(0)
        except AssertionError:
            log.error("SELFTEST FAILED (AssertionError)")
            sys.exit(1)
        except Exception as e:
            log.error(f"SELFTEST FAILED (Exception): {e}\n{traceback.format_exc()}")
            # ----- CORRECTION BONUS -----
            sys.exit(1) # Ajouté pour quitter en cas d'échec du test
            # --------------------------

    # ----- CORRECTION PRINCIPALE -----
    # Si on n'est pas en selftest, on lance l'application
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        log.error(f"Erreur fatale lors du lancement de l'application: {e}\n{traceback.format_exc()}")
        sys.exit(1)
    # -------------------------------