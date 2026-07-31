# -*- coding: utf-8 -*-
"""Operator console page."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from gui.components import ActivityFeed, MacroLibrary, MetricCard, StatusPill
from gui.theme import Theme
from utils.config import fmt_seconds


def _label(text: str, object_name: str = "") -> QLabel:
    label = QLabel(text)
    if object_name:
        label.setObjectName(object_name)
    return label


class ConsolePage(QWidget):
    record_requested = pyqtSignal()
    play_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    coc_requested = pyqtSignal()
    loop_toggled = pyqtSignal(bool)
    safeguard_toggled = pyqtSignal(bool)
    macro_selected = pyqtSignal(str)
    create_requested = pyqtSignal()
    rename_requested = pyqtSignal()
    delete_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QGridLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setHorizontalSpacing(16)
        root.setVerticalSpacing(16)

        heading = QVBoxLayout()
        heading.setSpacing(4)
        heading.addWidget(_label("CENTRE D’EXÉCUTION", "PageEyebrow"))
        heading.addWidget(_label("Lancer une séquence, en confiance", "PageTitle"))
        heading.addWidget(_label("Choisis ta macro, vérifie CoC, puis garde le contrôle sur chaque exécution.", "PageSubtitle"))
        root.addLayout(heading, 0, 0, 1, 2)

        self.library = MacroLibrary()
        self.library.setMinimumWidth(320)
        self.library.setMaximumWidth(390)
        root.addWidget(self.library, 1, 0, 2, 1)

        runtime = QFrame()
        runtime.setObjectName("CardRaised")
        runtime_layout = QVBoxLayout(runtime)
        runtime_layout.setContentsMargins(20, 18, 20, 18)
        runtime_layout.setSpacing(14)

        header = QHBoxLayout()
        title_column = QVBoxLayout()
        title_column.setSpacing(4)
        self.macro_name = _label("Aucune macro sélectionnée", "CardTitle")
        self.macro_name.setStyleSheet("font-size: 20px;")
        self.macro_meta = _label("Crée ou sélectionne une macro pour commencer.", "CardCaption")
        title_column.addWidget(self.macro_name)
        title_column.addWidget(self.macro_meta)
        header.addLayout(title_column)
        header.addStretch()
        self.state_pill = StatusPill("État", "Prêt", Theme.ACCENT)
        header.addWidget(self.state_pill)
        runtime_layout.addLayout(header)

        self.state_message = _label("Prêt à exécuter", "PageSubtitle")
        self.state_message.setStyleSheet(f"font-size: 16px; color: {Theme.TEXT};")
        runtime_layout.addWidget(self.state_message)

        context = QHBoxLayout()
        self.coc_status = StatusPill("CoC", "Vérification…", Theme.TEXT_MUTED)
        self.safeguard_status = StatusPill("Safeguard", "Désactivé", Theme.TEXT_MUTED)
        context.addWidget(self.coc_status)
        context.addWidget(self.safeguard_status)
        context.addStretch()
        runtime_layout.addLayout(context)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        runtime_layout.addWidget(self.progress)

        metrics = QHBoxLayout()
        metrics.setSpacing(8)
        self.elapsed_metric = MetricCard("TEMPS ÉCOULÉ", "00:00", Theme.INFO)
        self.events_metric = MetricCard("ÉVÉNEMENTS", "0", Theme.ACCENT)
        self.duration_metric = MetricCard("DURÉE MACRO", "00:00", Theme.WARNING)
        self.cycles_metric = MetricCard("CYCLES", "0", Theme.ACCENT)
        for metric in (self.elapsed_metric, self.events_metric, self.duration_metric, self.cycles_metric):
            metrics.addWidget(metric, 1)
        runtime_layout.addLayout(metrics)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.record_button = QPushButton("Enregistrer")
        self.record_button.setObjectName("PrimaryButton")
        self.play_button = QPushButton("Lancer la macro")
        self.play_button.setObjectName("PrimaryButton")
        self.stop_button = QPushButton("Stopper")
        self.stop_button.setObjectName("DangerButton")
        for button in (self.record_button, self.play_button, self.stop_button):
            button.setMinimumHeight(44)
        controls.addWidget(self.record_button, 1)
        controls.addWidget(self.play_button, 1)
        controls.addWidget(self.stop_button, 1)
        runtime_layout.addLayout(controls)

        lower = QHBoxLayout()
        self.loop = QCheckBox("Lecture en boucle")
        lower.addWidget(self.loop)
        self.safeguard = QCheckBox("Arrêter si CoC disparaît")
        self.safeguard.setToolTip("Coupe automatiquement la macro après plusieurs contrôles sans CoC détecté.")
        lower.addWidget(self.safeguard)
        lower.addStretch()
        self.launch_button = QPushButton("Ouvrir CoC")
        lower.addWidget(self.launch_button)
        runtime_layout.addLayout(lower)

        root.addWidget(runtime, 1, 1)

        self.activity = ActivityFeed()
        root.addWidget(self.activity, 2, 1)
        root.setColumnStretch(1, 1)

        self.library.macro_selected.connect(self.macro_selected)
        self.library.create_requested.connect(self.create_requested)
        self.library.rename_requested.connect(self.rename_requested)
        self.library.delete_requested.connect(self.delete_requested)
        self.record_button.clicked.connect(self.record_requested)
        self.play_button.clicked.connect(self.play_requested)
        self.stop_button.clicked.connect(self.stop_requested)
        self.launch_button.clicked.connect(self.coc_requested)
        self.loop.toggled.connect(self.loop_toggled)
        self.safeguard.toggled.connect(self.safeguard_toggled)

    def set_macro(self, macro) -> None:
        if macro and getattr(macro, "name", ""):
            self.macro_name.setText(macro.name)
            self.macro_meta.setText(f"{macro.event_count():,} événements  ·  {fmt_seconds(macro.duration())}")
            self.duration_metric.set_value(fmt_seconds(macro.duration()))
            self.events_metric.set_value(f"{macro.event_count():,}")
        else:
            self.macro_name.setText("Aucune macro sélectionnée")
            self.macro_meta.setText("Crée ou sélectionne une macro pour commencer.")
            self.duration_metric.set_value("00:00")
            self.events_metric.set_value("0")
            self.progress.setValue(0)

    def set_state(self, label: str, color: str, message: str, state_name: str) -> None:
        self.state_pill.set_status(label, color)
        self.state_message.setText(message)
        recording = state_name == "RECORDING"
        playing = state_name == "PLAYING"
        busy = recording or playing or state_name == "STOPPING"
        self.record_button.setText("Arrêter l’enregistrement" if recording else "Enregistrer")
        self.record_button.setEnabled(not playing and state_name != "STOPPING")
        self.play_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        self.library.setEnabled(not busy)

    def set_coc_presence(self, snapshot) -> None:
        if snapshot.error:
            self.coc_status.set_status("Détection indisponible", Theme.WARNING)
        elif snapshot.present:
            detail = "Processus détecté" if snapshot.process_found else "Fenêtre détectée"
            self.coc_status.set_status(detail, Theme.ACCENT)
        else:
            self.coc_status.set_status("Non détecté", Theme.WARNING)

    def set_safeguard(self, enabled: bool) -> None:
        self.safeguard.blockSignals(True)
        self.safeguard.setChecked(enabled)
        self.safeguard.blockSignals(False)
        self.safeguard_status.set_status("Actif" if enabled else "Désactivé", Theme.ACCENT if enabled else Theme.TEXT_MUTED)

    def set_metrics(self, elapsed: float, events: int, duration: float, cycles: int, state_name: str, loop: bool) -> None:
        self.elapsed_metric.set_value(fmt_seconds(elapsed))
        self.events_metric.set_value(f"{events:,}")
        self.duration_metric.set_value(fmt_seconds(duration))
        self.cycles_metric.set_value(str(cycles))
        if state_name == "PLAYING" and duration > 0:
            current = elapsed % duration if loop else min(elapsed, duration)
            self.progress.setValue(max(0, min(100, int((current / duration) * 100))))
        elif state_name != "RECORDING":
            self.progress.setValue(0)
