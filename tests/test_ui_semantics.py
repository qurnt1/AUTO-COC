# -*- coding: utf-8 -*-
"""Headless checks for the user/system macro boundary."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QListView

from gui.components import MacroLibrary
from gui.models import MacroSummary


class MacroLibrarySemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_system_routines_are_not_selectable_or_editable(self) -> None:
        library = MacroLibrary()
        library.set_items(
            [
                MacroSummary("Recharger COC", 10, 2.0, "system", "Routine intégrée."),
                MacroSummary("Attaque", 12, 3.0, "user", "Macro créée par toi."),
            ]
        )
        self.assertEqual(library.system_view.selectionMode(), QListView.SelectionMode.NoSelection)
        self.assertIsNone(library.selected_name())
        self.assertFalse(library.rename_button.isEnabled())
        self.assertFalse(library.delete_button.isEnabled())
        library.search.setText("Attaque")
        self.assertEqual(library.model.rowCount(), 1)
        self.assertEqual(library.system_model.rowCount(), 0)
        library.deleteLater()


if __name__ == "__main__":
    unittest.main()
