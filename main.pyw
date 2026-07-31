# -*- coding: utf-8 -*-
"""AUTO-COC entry point and headless self-tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
MACROS_DIR = CONFIG_DIR / "macros"
PARAMS_PATH = CONFIG_DIR / "data.csv"
LOG_PATH = CONFIG_DIR / "app.log"
ICON_PATH = CONFIG_DIR / "icon.ico"
GUIDE_HTML_PATH = CONFIG_DIR / "guide_telegram.html"

RELOAD_COC_MACRO_NAME = "Reload CoC"


def setup_environment() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    MACROS_DIR.mkdir(parents=True, exist_ok=True)


def run_selftest() -> None:
    from models.macro import Macro, Step, StepType
    from services.recorder_service import Player, Recorder
    from services.coc.models import CocLaunchProfile
    from utils.config import get_macro_hash, read_macro_file, read_params_csv, write_macro_file, write_params_csv
    from gui.theme import Theme, get_contrast_ratio
    import tempfile
    import time

    print("=" * 34)
    print("       AUTO-COC SELF-TESTS")
    print("=" * 34)

    step = Step(time_delta=1.0, step_type=StepType.MOUSE_MOVE, data={"x": 100, "y": 200})
    macro = Macro(name="Test")
    macro.set_steps([step])
    assert macro.event_count() == 1 and macro.duration() == 1.0
    assert Macro.from_json(macro.to_json()).event_count() == 1
    print("[PASS] Macro/Step model")

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        params_path = root / "params.csv"
        params = {"key1": "value1", "key2": "value2"}
        assert write_params_csv(params_path, params)
        assert read_params_csv(params_path)["key1"] == "value1"
        macro_path = root / "test.json"
        steps = [{"t": 1.0, "type": "nop", "data": {}}]
        assert write_macro_file(macro_path, "Test", steps)
        name, loaded_steps, sha1, _ = read_macro_file(macro_path)
        assert name == "Test" and len(loaded_steps) == 1 and sha1 == get_macro_hash(steps)
    print("[PASS] atomic CSV/JSON persistence")

    recorder = Recorder()
    player = Player(recorder)
    player._apply = lambda event: None
    started = time.perf_counter()
    player._run_one_cycle([{"t": 0.05, "type": "nop", "data": {}}] * 4)
    elapsed = time.perf_counter() - started
    assert abs(elapsed - 0.2) < 0.08
    print("[PASS] Player absolute timing")

    assert get_contrast_ratio(Theme.TEXT, Theme.BG) >= 4.5
    assert get_contrast_ratio(Theme.ACCENT, Theme.BG) >= 3.0
    print("[PASS] theme contrast")

    profile = CocLaunchProfile.from_params({
        "coc_path": "C:/Games/CoC.lnk",
        "coc_process_names": "wsaClient.exe | ClashOfClans.exe",
        "coc_window_titles": "Clash of Clans | Google Play Games",
        "coc_missing_tolerance": "4",
    })
    assert profile.launcher_path.endswith("CoC.lnk")
    assert profile.process_names == ("wsaClient.exe", "ClashOfClans.exe")
    assert profile.window_titles == ("Clash of Clans", "Google Play Games")
    assert profile.missing_tolerance == 4
    print("[PASS] CoC launch and detection profile")
    print("ALL SELFTESTS PASSED")


def build_window():
    from gui.app import MainWindow
    from services.telegram_service import TelegramBotService
    from utils.config import read_params_csv

    params = read_params_csv(PARAMS_PATH)
    chat_id = None
    try:
        if params.get("telegram_chat_id", "").strip():
            chat_id = int(params["telegram_chat_id"].strip())
    except ValueError:
        pass
    telegram = TelegramBotService(token=params.get("telegram_bot_token", ""), chat_id=chat_id)
    if telegram.is_configured:
        telegram.start()
    window = MainWindow(
        params=params,
        telegram=telegram,
        macros_dir=MACROS_DIR,
        params_path=PARAMS_PATH,
        log_path=LOG_PATH,
        icon_path=ICON_PATH,
        guide_path=GUIDE_HTML_PATH,
        system_names=[RELOAD_COC_MACRO_NAME],
    )
    return window


def main() -> int:
    parser = argparse.ArgumentParser(description="AUTO-COC operator console")
    parser.add_argument("--selftest", action="store_true", help="Run headless checks and exit.")
    args = parser.parse_args()
    setup_environment()
    if args.selftest:
        run_selftest()
        return 0

    from PyQt6.QtWidgets import QApplication
    from gui.theme import build_stylesheet
    from utils.logger import install_exception_hook, setup_logging

    logger = setup_logging(LOG_PATH)
    install_exception_hook()
    app = QApplication(sys.argv)
    app.setApplicationName("AUTO-COC")
    app.setApplicationDisplayName("AUTO-COC")
    app.setStyle("Fusion")
    app.setStyleSheet(build_stylesheet())
    window = None
    try:
        window = build_window()
        window.show()
        logger.info("AUTO-COC started.")
        return app.exec()
    except Exception:
        if window is not None:
            window.controller.shutdown()
        logger.exception("Fatal startup error.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
