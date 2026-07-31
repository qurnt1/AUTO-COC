# -*- coding: utf-8 -*-
"""
AUTO-COC system utilities

Gestion des processus système, extinction PC, et capture d'écran.
"""

import os
import subprocess
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

# Libs optionnelles
try:
    from PIL import Image, ImageGrab
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image, ImageGrab = None, None

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    mss = None


# =========================
#     Process Management
# =========================

def resolve_exe_from_path(path_str: str) -> Optional[str]:
    """
    À partir d'un .exe ou d'un .lnk, tente de déduire le nom d'exe à fermer.
    
    Args:
        path_str: Chemin vers le fichier (.exe ou .lnk)
        
    Returns:
        Nom de l'exécutable (ex: "game.exe") ou None
    """
    log = get_logger()
    
    if not path_str:
        return None
    
    p = Path(path_str)
    s = str(p)
    
    if s.lower().endswith(".exe"):
        return p.name
    
    if os.name == "nt" and s.lower().endswith(".lnk"):
        try:
            escaped = s.replace("'", "''")
            ps = f"(New-Object -ComObject WScript.Shell).CreateShortcut('{escaped}').TargetPath"
            cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps]
            
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


def kill_process_by_name(exe_name: str) -> bool:
    """
    Tente de tuer un processus par son nom.
    Utilise taskkill + wmic fallback.
    
    Args:
        exe_name: Nom de l'exécutable (ex: "game.exe")
        
    Returns:
        True si au moins une méthode a réussi
    """
    log = get_logger()
    
    if os.name != 'nt' or not exe_name:
        return False
    
    log.info(f"Tentative de fermeture de '{exe_name}'...")
    success = False
    
    # Méthode 1: taskkill
    try:
        cmd_taskkill = ["taskkill", "/F", "/IM", exe_name]
        result = subprocess.run(
            cmd_taskkill,
            capture_output=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        if result.returncode in (0, 128):
            log.info(f"taskkill pour '{exe_name}' exécuté (code {result.returncode}).")
            success = True
        else:
            log.warning(f"taskkill a échoué (code {result.returncode})")
    except Exception as e:
        log.error(f"Erreur taskkill: {e}")
    
    # Méthode 2: wmic (fallback)
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
            success = True
        else:
            log.warning(f"WMIC a échoué (code {result.returncode})")
    except Exception as e:
        log.error(f"Erreur WMIC: {e}")
    
    return success


def is_process_running(exe_name: str) -> bool:
    """
    Vérifie si un processus est en cours via tasklist (Windows).
    
    Args:
        exe_name: Nom de l'exécutable
        
    Returns:
        True si le processus est en cours
    """
    log = get_logger()
    
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


# =========================
#     System Shutdown
# =========================

def perform_shutdown(delay_seconds: float = 2.0) -> None:
    """
    Éteint le système après un délai.
    
    Args:
        delay_seconds: Délai avant extinction
    """
    log = get_logger()
    log.warning(f"EXTINCTION SYSTÈME dans {delay_seconds}s...")
    
    time.sleep(delay_seconds)
    
    try:
        if os.name == "nt":
            os.system("shutdown /s /t 1")
        else:
            ret = os.system("systemctl poweroff")
            if ret != 0:
                os.system("shutdown -h now")
    except Exception as e:
        log.error(f"Échec de l'extinction: {e}")


# =========================
#     Screenshot Capture
# =========================

def grab_screenshot_png_bytes() -> Optional[bytes]:
    """
    Retourne une capture d'écran en PNG (bytes) ou None si impossible.
    
    Utilise PIL ImageGrab en priorité, avec fallback MSS.
    
    Returns:
        Bytes PNG de la capture ou None
    """
    log = get_logger()
    
    # 1) Pillow ImageGrab
    if PIL_AVAILABLE and ImageGrab is not None:
        try:
            img = ImageGrab.grab()
            bio = BytesIO()
            img.save(bio, format="PNG")
            return bio.getvalue()
        except Exception as e:
            log.warning(f"Échec ImageGrab (Pillow): {e}")
    
    # 2) Fallback MSS
    if MSS_AVAILABLE and mss is not None:
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[0]
                raw = sct.grab(monitor)
                if Image is None:
                    log.warning("MSS a capturé, mais PIL manque pour convertir.")
                    return None
                img = Image.frombytes("RGB", (raw.width, raw.height), raw.rgb)
                bio = BytesIO()
                img.save(bio, format="PNG")
                return bio.getvalue()
        except Exception as e:
            log.warning(f"Échec capture MSS: {e}")
    
    log.error("Aucune méthode de capture d'écran n'a fonctionné.")
    return None


def play_beep(frequency: int = 1000, duration_ms: int = 200) -> None:
    """Joue un bip sonore (Windows uniquement)."""
    try:
        import winsound
        winsound.Beep(frequency, duration_ms)
    except Exception:
        pass


# =========================
#     Utility
# =========================

def ensure_directories(*paths: Path) -> None:
    """Crée les dossiers s'ils n'existent pas."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
