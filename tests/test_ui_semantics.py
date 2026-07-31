# -*- coding: utf-8 -*-
"""Headless checks for the simplified operator interface."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLabel

from gui.components import MacroListModel
from gui.models import MacroSummary
from gui.pages.console_page import ConsolePage
from gui.pages.telegram_page import TelegramPage


class UiSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_home_uses_one_contextual_run_stop_button(self) -> None:
        page = ConsolePage()
        page.set_macros([MacroSummary("Attack", 12, 3.0)], "Attack")
        page._event_count = 12
        page.set_state("Ready", "IDLE")
        self.assertEqual(page.run_button.text(), "Run macro · F1")
        self.assertTrue(page.run_button.isEnabled())
        self.assertFalse(hasattr(page, "play_button"))
        self.assertFalse(hasattr(page, "stop_button"))

        page.set_state("Playing", "PLAYING")
        self.assertEqual(page.run_button.text(), "Stop macro · F1")
        self.assertTrue(page.run_button.isEnabled())
        page.deleteLater()

    def test_telegram_action_is_visually_distinct_from_user_macros(self) -> None:
        page = TelegramPage()
        page.set_action(MacroSummary("Reload CoC", 8, 2.5, role="reload_coc"))
        tags = [label.text() for label in page.findChildren(QLabel) if label.objectName() == "Tag"]
        self.assertEqual(tags, ["TELEGRAM ACTION"])
        self.assertFalse(page.record_button.isHidden())
        self.assertTrue(page.create_button.isHidden())
        page.deleteLater()

    def test_macro_model_exposes_accessible_display_text(self) -> None:
        model = MacroListModel()
        model.set_items([MacroSummary("Attack", 12, 3.0)])
        index = model.index(0, 0)
        self.assertEqual(model.data(index, Qt.ItemDataRole.DisplayRole), "Attack")
        self.assertEqual(model.data(index, Qt.ItemDataRole.UserRole).events, 12)


if __name__ == "__main__":
    unittest.main()
