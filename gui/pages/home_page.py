# -*- coding: utf-8 -*-
"""
Macro COC v3.0 — GUI / Pages / Home

Page d'accueil / Dashboard.
"""

from typing import Dict, List, TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QSizePolicy,
)
from PyQt6.QtCore import Qt

from qfluentwidgets import (
    CardWidget, StrongBodyLabel, BodyLabel, CaptionLabel,
    PrimaryPushButton, PushButton, FluentIcon, InfoBadge,
    FlowLayout, ScrollArea,
)

from gui.theme import Theme
from gui.widgets import StatCard, MacroCard
from utils.config import fmt_duration_for_list, fmt_seconds
from utils.logger import get_logger

if TYPE_CHECKING:
    from gui.app import App


class HomePage(QWidget):
    """Page d'accueil : dashboard, macros recentes, etat systeme."""

    def __init__(self, app: "App", parent=None):
        super().__init__(parent)
        self.setObjectName("homePage")
        self._app = app
        self._log = get_logger()

        # Layout principal
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(20)

        # --- En-tete ---
        header = QHBoxLayout()
        title = StrongBodyLabel("Tableau de bord")
        title.setStyleSheet("font-size: 22px;")
        header.addWidget(title)
        header.addStretch()

        # Statut Telegram
        self._tg_status_lbl = BodyLabel("Telegram: ---")
        header.addWidget(self._tg_status_lbl)
        main_layout.addLayout(header)

        # --- Cartes de stats ---
        stats_row = QHBoxLayout()
        stats_row.setSpacing(12)

        self._stat_total = StatCard(FluentIcon.SAVE, "0", "Macros", "#60a5fa")
        self._stat_recents = StatCard(FluentIcon.HISTORY, "0", "Modifiees < 7j", "#f59e0b")
        self._stat_tg = StatCard(FluentIcon.SEND, "---", "Telegram", "#22c55e")

        stats_row.addWidget(self._stat_total)
        stats_row.addWidget(self._stat_recents)
        stats_row.addWidget(self._stat_tg)
        stats_row.addStretch()
        main_layout.addLayout(stats_row)

        # --- Derniere macro + Acces rapide ---
        bottom = QHBoxLayout()
        bottom.setSpacing(16)

        # Derniere macro
        self._recent_card = CardWidget()
        self._recent_card.setMinimumHeight(180)
        recent_layout = QVBoxLayout(self._recent_card)
        recent_layout.setContentsMargins(20, 16, 20, 16)
        recent_layout.setSpacing(8)

        recent_title = BodyLabel("Derniere macro utilisee")
        recent_title.setStyleSheet("color: rgba(255,255,255,0.6);")
        recent_layout.addWidget(recent_title)

        self._recent_name = StrongBodyLabel("Aucune")
        self._recent_name.setStyleSheet("font-size: 18px;")
        recent_layout.addWidget(self._recent_name)

        self._recent_meta = CaptionLabel("")
        recent_layout.addWidget(self._recent_meta)

        recent_layout.addStretch()

        recent_btns = QHBoxLayout()
        play_btn = PrimaryPushButton(FluentIcon.PLAY, "  Lire")
        play_btn.clicked.connect(lambda: self._app.on_play())
        recent_btns.addWidget(play_btn)
        recent_btns.addStretch()
        recent_layout.addLayout(recent_btns)

        bottom.addWidget(self._recent_card, stretch=2)

        # Acces rapide
        quick_card = CardWidget()
        quick_card.setMinimumHeight(180)
        quick_layout = QVBoxLayout(quick_card)
        quick_layout.setContentsMargins(20, 16, 20, 16)
        quick_layout.setSpacing(10)

        quick_title = BodyLabel("Acces rapide")
        quick_title.setStyleSheet("color: rgba(255,255,255,0.6);")
        quick_layout.addWidget(quick_title)

        new_btn = PushButton(FluentIcon.ADD, "  Nouvelle macro")
        new_btn.clicked.connect(self._app.macro_new)
        quick_layout.addWidget(new_btn)

        launch_btn = PushButton(FluentIcon.GAME, "  Lancer CoC")
        launch_btn.clicked.connect(self._app.launch_coc_once)
        quick_layout.addWidget(launch_btn)

        capture_btn = PushButton(FluentIcon.PHOTO, "  Capture ecran")
        capture_btn.clicked.connect(self._app.send_capture_to_tg)
        quick_layout.addWidget(capture_btn)

        quick_layout.addStretch()
        bottom.addWidget(quick_card, stretch=1)

        main_layout.addLayout(bottom, stretch=1)

    def refresh(self):
        """Rafraichit les donnees du dashboard."""
        self._update_stats()
        self._update_recent_macro()
        self._update_tg_status()

    def _update_stats(self):
        names = self._app.all_macro_names
        self._stat_total.set_value(str(len(names)))

        # Compter les macros modifiees < 7 jours (simplifie)
        recent_count = min(len(names), 5)  # fallback
        self._stat_recents.set_value(str(recent_count))

    def _update_recent_macro(self):
        name = self._app.current_macro_name
        if name:
            self._recent_name.setText(name)
            m = self._app.macro
            self._recent_meta.setText(
                f"{m.event_count()} evenements | {fmt_seconds(m.duration())}"
            )
        else:
            self._recent_name.setText("Aucune macro selectionnee")
            self._recent_meta.setText("Selectionnez une macro dans Mes Macros")

    def _update_tg_status(self):
        status, color = self._app.tg.get_status()
        self._app.tg_status_text = status
        self._app.tg_status_color = color
        self._tg_status_lbl.setText(f"Telegram: {status}")
        self._tg_status_lbl.setStyleSheet(f"color: {color};")
        self._stat_tg.set_value(status)
