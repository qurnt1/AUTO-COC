# -*- coding: utf-8 -*-
"""
Macro COC v3.0 — Main Entry Point

Orchestrateur principal qui lie :
- Le service Telegram (asyncio dans un thread)
- L'application GUI (Tkinter mainloop)
"""

import argparse
import sys
import threading
from pathlib import Path

# Configuration des chemins
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
MACROS_DIR = CONFIG_DIR / "macros"
PARAMS_PATH = CONFIG_DIR / "data.csv"
LOG_PATH = CONFIG_DIR / "app.log"
ICON_PATH = CONFIG_DIR / "icon.ico"
ICON_PNG_PATH = CONFIG_DIR / "image.png"
GUIDE_HTML_PATH = CONFIG_DIR / "guide_telegram.html"
LEGACY_MACRO = CONFIG_DIR / "macro.json"

# Version
APP_VERSION = "3.0.0"
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

# Macros protégées
RECHARGER_MACRO_NAME = "Recharger COC"
VALIDER_MACRO_NAME = "Valider arrivée"


def setup_environment():
    """Configure l'environnement avant le démarrage."""
    # Créer les dossiers nécessaires
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    MACROS_DIR.mkdir(parents=True, exist_ok=True)


def run_selftest():
    """
    Exécute une suite de tests non-UI.
    Lance via: python main.py --selftest
    """
    from utils.logger import setup_logging
    
    # Initialiser le logger pour les tests
    setup_logging(LOG_PATH)
    
    print("=" * 30)
    print("       RUNNING SELFTESTS")
    print("=" * 30)
    
    # --- 1. Test Models ---
    print("\n[TEST] 1/4: Models (Macro/Step)...")
    try:
        from models.macro import Macro, Step, StepType
        
        step = Step(time_delta=1.0, step_type=StepType.MOUSE_MOVE, data={"x": 100, "y": 200})
        assert step.to_dict()["t"] == 1.0
        assert step.to_dict()["type"] == "mouse_move"
        
        macro = Macro(name="Test")
        macro.set_steps([step])
        assert macro.event_count() == 1
        assert macro.duration() == 1.0
        
        # Sérialisation
        json_str = macro.to_json()
        macro2 = Macro.from_json(json_str)
        assert macro2.name == "Test"
        assert macro2.event_count() == 1
        
        print("  [PASS] Models OK.")
    except Exception as e:
        print(f"  [FAIL] Models: {e}")
        sys.exit(1)
    
    # --- 2. Test Config I/O ---
    print("\n[TEST] 2/4: Config I/O (CSV/JSON)...")
    try:
        import tempfile
        from utils.config import (
            read_params_csv, write_params_csv,
            read_macro_file, write_macro_file, get_macro_hash
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # CSV
            csv_path = tmp_path / "test.csv"
            params = {"key1": "value1", "key2": "value2"}
            write_params_csv(csv_path, params)
            assert csv_path.exists()
            
            read_p = read_params_csv(csv_path)
            assert read_p.get("key1") == "value1"
            
            # JSON
            json_path = tmp_path / "test.json"
            steps = [{"t": 1.0, "type": "nop", "data": {}}]
            write_macro_file(json_path, "Test", steps)
            assert json_path.exists()
            
            name, read_steps, sha1, _ = read_macro_file(json_path)
            assert name == "Test"
            assert len(read_steps) == 1
            assert sha1 == get_macro_hash(steps)
        
        print("  [PASS] Config I/O OK.")
    except Exception as e:
        print(f"  [FAIL] Config I/O: {e}")
        sys.exit(1)
    
    # --- 3. Test Recorder/Player ---
    print("\n[TEST] 3/4: Recorder/Player (timing 5s)...")
    try:
        import time
        from services.recorder_service import Recorder, Player
        
        recorder = Recorder()
        player = Player(recorder)
        
        # 5 steps de 1s = 5s total
        steps = [{"t": 1.0, "type": "nop", "data": {}}] * 5
        player._apply = lambda ev: None  # Stub
        
        t_start = time.perf_counter()
        player._run_one_cycle(steps)
        t_end = time.perf_counter()
        
        duration = t_end - t_start
        target = 5.0
        drift_pct = (abs(duration - target) / target) * 100
        
        print(f"  Durée cible: {target:.3f}s")
        print(f"  Durée mesurée: {duration:.3f}s")
        print(f"  Dérive: {drift_pct:.2f}%")
        
        assert drift_pct < 5.0  # Tolérance 5%
        print("  [PASS] Recorder/Player OK.")
    except Exception as e:
        print(f"  [FAIL] Recorder/Player: {e}")
        sys.exit(1)
    
    # --- 4. Test Theme ---
    print("\n[TEST] 4/4: Theme (contraste)...")
    try:
        from gui.theme import Theme, get_contrast_ratio
        
        # Vérifier que les couleurs du thème ont un contraste suffisant
        ratio = get_contrast_ratio(Theme.ROW_NAME_COLOR, Theme.ROW_BG)
        print(f"  Ratio ROW_NAME/ROW_BG: {ratio:.2f} (min: 4.5)")
        assert ratio >= 4.5
        
        print("  [PASS] Theme OK.")
    except Exception as e:
        print(f"  [FAIL] Theme: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 30)
    print("     ALL SELFTESTS PASSED")
    print("=" * 30)


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description="Macro COC v3.0 Application")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Exécute les tests internes non-UI et quitte."
    )
    args = parser.parse_args()
    
    # Setup de base
    setup_environment()
    
    # Mode selftest
    if args.selftest:
        try:
            run_selftest()
            sys.exit(0)
        except AssertionError as e:
            print(f"SELFTEST FAILED (AssertionError): {e}")
            sys.exit(1)
        except Exception as e:
            print(f"SELFTEST FAILED (Exception): {e}")
            sys.exit(1)
    
    # Mode normal : lancer l'application
    try:
        # Initialiser le logging
        from utils.logger import setup_logging, install_exception_hook
        log = setup_logging(LOG_PATH)
        install_exception_hook()
        
        log.info(f"=== Démarrage Macro COC v{APP_VERSION} ===")
        log.info(f"Python: {PYTHON_VERSION}")
        log.info(f"Base dir: {BASE_DIR}")
        
        # Charger la configuration
        from utils.config import read_params_csv
        params = read_params_csv(PARAMS_PATH)
        
        # Créer le service Telegram
        from services.telegram_service import TelegramBotService
        
        token = params.get("telegram_bot_token", "")
        chat_id_str = params.get("telegram_chat_id", "")
        chat_id = None
        try:
            chat_id = int(chat_id_str) if chat_id_str else None
        except ValueError:
            pass
        
        tg_service = TelegramBotService(token=token, chat_id=chat_id)
        
        # Démarrer le service Telegram en arrière-plan
        if tg_service.is_configured:
            log.info("Démarrage du service Telegram...")
            tg_service.start()
        else:
            log.warning("Service Telegram non configuré (token manquant).")
        
        # Lancer l'application GUI
        from gui.app import App
        
        app = App(
            params=params,
            tg_service=tg_service,
            base_dir=BASE_DIR,
            config_dir=CONFIG_DIR,
            macros_dir=MACROS_DIR,
            params_path=PARAMS_PATH,
            log_path=LOG_PATH,
            icon_path=ICON_PATH,
            icon_png_path=ICON_PNG_PATH,
            guide_html_path=GUIDE_HTML_PATH,
            legacy_macro_path=LEGACY_MACRO,
            app_version=APP_VERSION,
            python_version=PYTHON_VERSION,
            protected_macro_names=[RECHARGER_MACRO_NAME, VALIDER_MACRO_NAME]
        )
        
        # Boucle principale Tkinter
        app.mainloop()
        
        # Cleanup
        log.info("Arrêt du service Telegram...")
        tg_service.stop()
        
        log.info("Application fermée.\n" + "=" * 30)
        
    except Exception as e:
        import traceback
        print(f"Erreur fatale: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
