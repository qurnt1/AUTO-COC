# -*- coding: utf-8 -*-
"""
Macro COC v3.0 — GUI / Pages / Macros

Page principale de gestion des macros.
"""

from typing import Dict, List, TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QSizePolicy, QLabel,
)
from PyQt6.QtCore import Qt

from qfluentwidgets import (
    CardWidget, StrongBodyLabel, BodyLabel, CaptionLabel,
    PrimaryPushButton, PushButton, FluentIcon, InfoBadge,
    SearchLineEdit, ScrollArea, ComboBox, ProgressBar,
)

from gui.theme import Theme
from gui.components import MacroList
from gui.widgets import (
    StatCard, ActionButton, DangerButton,
    RecordingIndicator, MacroTimeline, StepListWidget,
)
from gui.dialogs import TextInputDialog, show_toast
from models.macro import Macro, is_protected_macro
from utils.config import (
    read_macro_file, write_macro_file, read_macro_meta,
    macro_path_from_name, sanitize_macro_name, fmt_seconds,
)
from utils.logger import get_logger

if TYPE_CHECKING:
    from gui.app import App


class MacrosPage(QWidget):
    """Page de gestion des macros : liste, detail, actions, stats."""

    def __init__(self, app: "App", parent=None):
        super().__init__(parent)
        self.setObjectName("macrosPage")
        self._app = app
        self._log = get_logger()

        # Layout vertical : contenu principal + status bar
        wrapper = QVBoxLayout(self)
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.setSpacing(0)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(12)
        wrapper.addLayout(main_layout, stretch=1)

        # --- Panneau gauche : liste macros ---
        left = QFrame()
        left.setFixedWidth(280)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # En-tete + recherche
        header_row = QHBoxLayout()
        title = StrongBodyLabel("Mes Macros")
        header_row.addWidget(title)
        header_row.addStretch()
        left_layout.addLayout(header_row)

        self._search = SearchLineEdit()
        self._search.setPlaceholderText("Rechercher...")
        self._search.textChanged.connect(self._filter_macros)
        left_layout.addWidget(self._search)

        # Liste
        self._macro_list = MacroList()
        self._macro_list.macro_selected.connect(self._on_macro_selected)
        left_layout.addWidget(self._macro_list, stretch=1)

        # Boutons CRUD
        crud = QHBoxLayout()
        crud.setSpacing(4)
        new_btn = PushButton(FluentIcon.ADD, "  Nouveau")
        new_btn.clicked.connect(self._app.macro_new)
        crud.addWidget(new_btn)

        rename_btn = PushButton(FluentIcon.EDIT, "  Renommer")
        rename_btn.clicked.connect(self._app.macro_rename)
        crud.addWidget(rename_btn)

        del_btn = DangerButton(FluentIcon.DELETE, "  Supprimer")
        del_btn.clicked.connect(self._app.macro_delete)
        crud.addWidget(del_btn)
        left_layout.addLayout(crud)

        main_layout.addWidget(left)

        # --- Centre : detail macro ---
        center = QFrame()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(10)

        # En-tete macro
        self._detail_header = QWidget()
        dh_layout = QVBoxLayout(self._detail_header)
        dh_layout.setContentsMargins(0, 0, 0, 0)
        dh_layout.setSpacing(2)

        self._macro_title = StrongBodyLabel("Selectionnez une macro")
        self._macro_title.setStyleSheet("font-size: 20px;")
        dh_layout.addWidget(self._macro_title)

        self._macro_subtitle = CaptionLabel("")
        self._macro_subtitle.setStyleSheet("color: rgba(255,255,255,0.5);")
        dh_layout.addWidget(self._macro_subtitle)
        center_layout.addWidget(self._detail_header)

        # Timeline visuelle
        timeline_card = CardWidget()
        tl_layout = QVBoxLayout(timeline_card)
        tl_layout.setContentsMargins(16, 10, 16, 10)
        tl_label = CaptionLabel("Chronologie des evenements")
        tl_label.setStyleSheet("color: rgba(255,255,255,0.5);")
        tl_layout.addWidget(tl_label)
        self._timeline = MacroTimeline()
        tl_layout.addWidget(self._timeline)

        # Legende
        legend = QHBoxLayout()
        legend.setSpacing(16)
        for label_text, color in [
            ("Clics", Theme.TIMELINE_CLICK),
            ("Mouvements", Theme.TIMELINE_MOVE),
            ("Touches", Theme.TIMELINE_KEY),
            ("Scroll", Theme.TIMELINE_SCROLL),
        ]:
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 12px;")
            legend.addWidget(dot)
            lbl = CaptionLabel(label_text)
            legend.addWidget(lbl)
        legend.addStretch()
        tl_layout.addLayout(legend)

        center_layout.addWidget(timeline_card)

        # Liste des steps
        steps_card = CardWidget()
        steps_layout = QVBoxLayout(steps_card)
        steps_layout.setContentsMargins(16, 10, 16, 10)
        steps_label = CaptionLabel("Etapes detaillees")
        steps_label.setStyleSheet("color: rgba(255,255,255,0.5);")
        steps_layout.addWidget(steps_label)
        self._step_list = StepListWidget()
        self._step_list.setMinimumHeight(100)
        steps_layout.addWidget(self._step_list, stretch=1)
        center_layout.addWidget(steps_card, stretch=1)

        main_layout.addWidget(center, stretch=1)

        # --- Panneau droit : actions + stats ---
        right = QFrame()
        right.setFixedWidth(200)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Actions
        actions_card = CardWidget()
        actions_layout = QVBoxLayout(actions_card)
        actions_layout.setContentsMargins(12, 12, 12, 12)
        actions_layout.setSpacing(8)

        actions_label = BodyLabel("Actions")
        actions_layout.addWidget(actions_label)

        self._btn_play = ActionButton(FluentIcon.PLAY, "  Lire")
        self._btn_play.clicked.connect(lambda: self._app.on_play())
        actions_layout.addWidget(self._btn_play)

        self._btn_record = ActionButton(FluentIcon.MICROPHONE, "  Enregistrer")
        self._btn_record.clicked.connect(self._app.toggle_record)
        actions_layout.addWidget(self._btn_record)

        self._btn_stop = DangerButton(FluentIcon.CLOSE, "  Stop")
        self._btn_stop.clicked.connect(lambda: self._app.force_stop_all())
        actions_layout.addWidget(self._btn_stop)

        right_layout.addWidget(actions_card)

        # Stats
        stats_card = CardWidget()
        stats_layout = QVBoxLayout(stats_card)
        stats_layout.setContentsMargins(12, 12, 12, 12)
        stats_layout.setSpacing(6)

        stats_label = BodyLabel("Session")
        stats_layout.addWidget(stats_label)

        self._stat_duration = StatCard(FluentIcon.STOP_WATCH, "00:00", "Duree macro", "#93c5fd")
        self._stat_duration.setFixedWidth(176)
        stats_layout.addWidget(self._stat_duration)

        self._stat_events = StatCard(FluentIcon.LIBRARY, "0", "Evenements", "#86efac")
        self._stat_events.setFixedWidth(176)
        stats_layout.addWidget(self._stat_events)

        self._stat_cycles = StatCard(FluentIcon.SYNC, "0", "Cycles", "#f59e0b")
        self._stat_cycles.setFixedWidth(176)
        stats_layout.addWidget(self._stat_cycles)

        self._stat_elapsed = StatCard(FluentIcon.STOP_WATCH, "00:00", "Temps total", "#c084fc")
        self._stat_elapsed.setFixedWidth(176)
        stats_layout.addWidget(self._stat_elapsed)

        right_layout.addWidget(stats_card)

        # Launch CoC
        launch_btn = PushButton(FluentIcon.GAME, "  Lancer CoC")
        launch_btn.clicked.connect(self._app.launch_coc_once)
        right_layout.addWidget(launch_btn)

        right_layout.addStretch()
        main_layout.addWidget(right)

        # --- Status bar (en bas de la page) ---
        self._status_bar = QFrame()
        self._status_bar.setFixedHeight(36)
        sb_layout = QHBoxLayout(self._status_bar)
        sb_layout.setContentsMargins(12, 0, 12, 0)
        sb_layout.setSpacing(12)

        self._status_label = BodyLabel("Pret")
        self._status_label.setStyleSheet("color: rgba(255,255,255,0.8);")
        sb_layout.addWidget(self._status_label)

        self._status_progress = ProgressBar()
        self._status_progress.setFixedWidth(200)
        self._status_progress.setFixedHeight(6)
        self._status_progress.hide()
        sb_layout.addWidget(self._status_progress)

        sb_layout.addStretch()

        self._recording_indicator = RecordingIndicator()
        self._recording_indicator.hide()
        sb_layout.addWidget(self._recording_indicator)

        self._status_macro_label = CaptionLabel("")
        self._status_macro_label.setStyleSheet("color: rgba(255,255,255,0.5);")
        sb_layout.addWidget(self._status_macro_label)

        wrapper.addWidget(self._status_bar)

    # =========================
    #     Accesseurs publics
    # =========================

    @property
    def macro_list(self) -> MacroList:
        return self._macro_list

    @property
    def search(self) -> SearchLineEdit:
        return self._search

    def update_ui_state(self):
        """Met a jour l'etat des boutons selon le state machine."""
        from gui.app import State
        idle = self._app.current_state == State.IDLE

        self._btn_play.setEnabled(idle)
        self._btn_record.setEnabled(idle or self._app.current_state == State.RECORDING)

        if self._app.current_state == State.RECORDING:
            self._btn_record.setText("  Arreter")
        else:
            self._btn_record.setText("  Enregistrer")

    def update_macro_info(self, name: str, event_count: int, duration: float):
        """Met a jour les infos de la macro selectionnee."""
        self._macro_title.setText(name)
        n = event_count
        self._macro_subtitle.setText(
            "Non enregistree" if n == 0
            else f"{n} evenements | {fmt_seconds(duration)}"
        )
        self._stat_duration.set_value(fmt_seconds(duration))
        self._stat_events.set_value(str(n))

    def update_timeline(self, steps: List[dict]):
        """Met a jour la timeline et la liste des steps."""
        self._timeline.set_steps(steps)
        self._step_list.set_steps(steps)

    def update_session_stats(self, cycles: int, elapsed: float):
        """Met a jour les stats de session."""
        self._stat_cycles.set_value(str(cycles))
        self._stat_elapsed.set_value(fmt_seconds(elapsed))

    def clear_details(self):
        """Vide le panneau de detail."""
        self._macro_title.setText("Selectionnez une macro")
        self._macro_subtitle.setText("")
        self._timeline.clear()
        self._step_list.clear()
        self._stat_duration.set_value("00:00")
        self._stat_events.set_value("0")

    # =========================
    #     Callbacks internes
    # =========================

    def _on_macro_selected(self, name: str):
        """Une macro a ete selectionnee dans la liste."""
        self._app._select_macro_by_name(name)

    def _filter_macros(self):
        """Filtre la liste de macros."""
        self._app._refresh_macro_sidebar()
