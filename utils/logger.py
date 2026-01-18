# -*- coding: utf-8 -*-
"""
Macro COC v3.0 — Utils / Logger

Configuration du logging rotatif avec nettoyage automatique.
"""

import logging
import logging.handlers
import sys
import time
from glob import glob
from pathlib import Path
from typing import Optional

# Logger global de l'application
log: Optional[logging.Logger] = None


def setup_logging(log_path: Path, max_bytes: int = 1_048_576, backup_count: int = 3) -> logging.Logger:
    """
    Configure le logger principal avec rotation.
    
    Args:
        log_path: Chemin vers le fichier de log
        max_bytes: Taille maximale avant rotation (défaut: 1 Mo)
        backup_count: Nombre de backups à conserver
        
    Returns:
        Logger configuré
    """
    global log
    
    # Créer le dossier parent si nécessaire
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Nettoyer les anciens logs au démarrage
    try:
        clean_old_logs(log_path.parent, "app.log*", max_age_hours=24)
    except Exception as e:
        print(f"Avertissement: Échec nettoyage logs: {e}")
    
    # Créer le logger
    logger = logging.getLogger("MacroApp")
    logger.setLevel(logging.INFO)
    
    # Éviter les handlers dupliqués
    if logger.handlers:
        return logger
    
    # Format
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Handler 1: Fichier rotatif
    try:
        handler_file = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        handler_file.setFormatter(formatter)
        logger.addHandler(handler_file)
    except Exception as e:
        print(f"Erreur: Impossible de créer le logger fichier: {e}")
    
    # Handler 2: Console
    handler_console = logging.StreamHandler(sys.stdout)
    handler_console.setFormatter(formatter)
    logger.addHandler(handler_console)
    
    log = logger
    return logger


def clean_old_logs(log_dir: Path, pattern: str, max_age_hours: int = 24) -> int:
    """
    Supprime les fichiers logs plus anciens que max_age_hours.
    
    Args:
        log_dir: Dossier contenant les logs
        pattern: Pattern glob (ex: "app.log*")
        max_age_hours: Âge maximum en heures
        
    Returns:
        Nombre de fichiers supprimés
    """
    deleted_count = 0
    cutoff = time.time() - (max_age_hours * 3600)
    search_path = log_dir / pattern
    
    for log_file in glob(str(search_path)):
        try:
            p = Path(log_file)
            if not p.is_file():
                continue
            
            if p.stat().st_mtime < cutoff:
                p.unlink()
                deleted_count += 1
                if log:
                    log.info(f"Nettoyage ancien log: {p.name}")
        except Exception as e:
            if log:
                log.warning(f"Échec suppression log {log_file}: {e}")
    
    return deleted_count


def log_uncaught_exception(exc_type, exc_value, exc_traceback):
    """Callback pour sys.excepthook pour logger les crashs."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    if log:
        log.error("EXCEPTION NON CAPTURÉE", exc_info=(exc_type, exc_value, exc_traceback))
    else:
        # Fallback si le logger n'est pas initialisé
        import traceback
        print("EXCEPTION NON CAPTURÉE:")
        traceback.print_exception(exc_type, exc_value, exc_traceback)


def install_exception_hook():
    """Installe le hook pour capturer les exceptions non gérées."""
    sys.excepthook = log_uncaught_exception


def get_logger() -> logging.Logger:
    """Retourne le logger de l'application."""
    global log
    if log is None:
        log = logging.getLogger("MacroApp")
    return log
