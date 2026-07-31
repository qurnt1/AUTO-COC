# -*- coding: utf-8 -*-
"""Headless checks for the user/system macro boundary."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from gui.components import MacroLibrary
from gui.models import MacroSummary


class MacroLibrarySemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_all_macros_share_one_editable_library(self) -> None:
        library = MacroLibrary()
        library.set_items(
            [
                MacroSummary("Recharger COC", 10, 2.0),
                MacroSummary("Attaque", 12, 3.0),
            ]
        )
        self.assertEqual(library.model.rowCount(), 2)
        self.assertIsNone(library.selected_name())
        self.assertFalse(library.rename_button.isEnabled())
        self.assertFalse(library.delete_button.isEnabled())
        library.search.setText("Attaque")
        self.assertEqual(library.model.rowCount(), 1)
        library.deleteLater()


if __name__ == "__main__":
    unittest.main()
