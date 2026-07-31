# -*- coding: utf-8 -*-
"""Checks for private Telegram routine assignments."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from gui.controller import RuntimeController
from utils.config import read_params_csv, write_macro_file


class _TelegramStub:
    is_ready = False
    command_queue = None

    def stop(self) -> None:
        return None


class TelegramMacroRoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _controller(self, root: Path, params: dict[str, str]) -> RuntimeController:
        return RuntimeController(
            params=params,
            telegram=_TelegramStub(),
            macros_dir=root / "macros",
            params_path=root / "data.csv",
            system_names=["Reload CoC"],
        )

    def test_role_macro_is_private_and_assignment_follows_rename_and_delete(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            macros_dir = root / "macros"
            write_macro_file(macros_dir / "Reload CoC.json", "Reload CoC", [{"t": 0.1, "type": "key", "data": {}}])
            write_macro_file(macros_dir / "Attack.json", "Attack", [{"t": 0.1, "type": "key", "data": {}}])
            params = {"coc_safeguard": "0", "system_validate_arrival_macro": "Obsolete"}
            controller = self._controller(root, params)
            controller._ensure_system_macros()
            controller.refresh_macros()

            self.assertNotIn("system_validate_arrival_macro", params)
            self.assertEqual([item.name for item in controller.user_summaries()], ["Attack"])
            self.assertEqual(controller.system_summary("reload_coc").name, "Reload CoC")
            self.assertTrue(controller.select_system_macro("reload_coc"))

            self.assertTrue(controller.rename_macro("Reload routine"))
            self.assertEqual(params["system_reload_coc_macro"], "Reload routine")
            self.assertTrue((macros_dir / "Reload routine.json").exists())
            self.assertFalse((macros_dir / "Reload CoC.json").exists())
            self.assertEqual(read_params_csv(root / "data.csv")["system_reload_coc_macro"], "Reload routine")

            self.assertTrue(controller.delete_macro())
            self.assertEqual(params["system_reload_coc_macro"], "")
            self.assertFalse((macros_dir / "Reload routine.json").exists())
            self.assertEqual(controller.current_macro_name, "Attack")
            controller.shutdown()

    def test_missing_assigned_routine_is_cleared_not_recreated(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            params = {"system_reload_coc_macro": "Missing", "coc_safeguard": "0"}
            controller = self._controller(root, params)
            controller._ensure_system_macros()

            self.assertEqual(params["system_reload_coc_macro"], "")
            self.assertFalse((root / "macros" / "Missing.json").exists())
            controller.shutdown()

    def test_rename_rolls_back_when_role_settings_cannot_be_saved(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            macros_dir = root / "macros"
            old_path = macros_dir / "Reload CoC.json"
            write_macro_file(old_path, "Reload CoC", [{"t": 0.1, "type": "key", "data": {}}])
            params = {"system_reload_coc_macro": "Reload CoC", "coc_safeguard": "0"}
            controller = self._controller(root, params)
            controller.refresh_macros()
            controller.select_system_macro("reload_coc")

            with patch("gui.controller.write_params_csv", return_value=False):
                self.assertFalse(controller.rename_macro("New routine"))

            self.assertTrue(old_path.exists())
            self.assertFalse((macros_dir / "New routine.json").exists())
            self.assertEqual(params["system_reload_coc_macro"], "Reload CoC")
            controller.shutdown()

    def test_delete_restores_file_when_role_settings_cannot_be_saved(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            macro_path = root / "macros" / "Reload CoC.json"
            write_macro_file(macro_path, "Reload CoC", [{"t": 0.1, "type": "key", "data": {}}])
            params = {"system_reload_coc_macro": "Reload CoC", "coc_safeguard": "0"}
            controller = self._controller(root, params)
            controller.refresh_macros()
            controller.select_system_macro("reload_coc")

            with patch("gui.controller.write_params_csv", return_value=False):
                self.assertFalse(controller.delete_macro())

            self.assertTrue(macro_path.exists())
            self.assertEqual(params["system_reload_coc_macro"], "Reload CoC")
            self.assertEqual(list((root / "macros").glob("*.delete.*")), [])
            controller.shutdown()


if __name__ == "__main__":
    unittest.main()
