# -*- coding: utf-8 -*-
"""Checks for editable macro role assignments."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from gui.controller import RuntimeController
from utils.config import read_params_csv, write_macro_file


class _TelegramStub:
    is_ready = False
    command_queue = None

    def stop(self) -> None:
        return None


class EditableMacroRoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_renaming_and_deleting_role_macro_updates_assignment(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            macros_dir = root / "macros"
            params_path = root / "data.csv"
            params = {"coc_safeguard": "0"}
            controller = RuntimeController(
                params=params,
                telegram=_TelegramStub(),
                macros_dir=macros_dir,
                params_path=params_path,
                system_names=["Reload CoC", "Validate arrival"],
            )
            controller._ensure_system_macros()
            write_macro_file(macros_dir / "Reload CoC.json", "Reload CoC", [{"t": 0.1, "type": "key", "data": {}}])
            controller.refresh_macros()
            controller.select_macro("Reload CoC")

            self.assertTrue(controller.rename_macro("Reload routine"))
            self.assertEqual(params["system_reload_coc_macro"], "Reload routine")
            self.assertTrue((macros_dir / "Reload routine.json").exists())
            self.assertFalse((macros_dir / "Reload CoC.json").exists())
            self.assertEqual(read_params_csv(params_path)["system_reload_coc_macro"], "Reload routine")

            self.assertTrue(controller.delete_macro())
            self.assertEqual(params["system_reload_coc_macro"], "")
            self.assertFalse((macros_dir / "Reload routine.json").exists())
            controller.shutdown()


if __name__ == "__main__":
    unittest.main()
