# -*- coding: utf-8 -*-
"""Focused operator console page."""

from __future__ import annotations

from collections.abc import Iterable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from gui.components import MetricCard, StatusPill
from gui.models import MacroSummary
from gui.theme import Theme
from utils.config import fmt_seconds


def _label(text: str, object_name: str = "") -> QLabel:
    label = QLabel(text)
    if object_name:
        label.setObjectName(object_name)
    return label


class ConsolePage(QWidget):
    record_requested = pyqtSignal()
    run_requested = pyqtSignal()
    coc_requested = pyqtSignal()
    loop_toggled = pyqtSignal(bool)
    safeguard_toggled = pyqtSignal(bool)
    macro_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state_name = "IDLE"
        self._event_count = 0
        self._coc_launching = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)

        heading = QVBoxLayout()
        heading.setSpacing(3)
        heading.addWidget(_label("OPERATOR CONSOLE", "PageEyebrow"))
        heading.addWidget(_label("Ready when you are", "PageTitle"))
        heading.addWidget(_label("Choose a macro, check Clash of Clans, then start.", "PageSubtitle"))
        root.addLayout(heading)

        coc_card = QFrame()
        coc_card.setObjectName("Card")
        coc_layout = QHBoxLayout(coc_card)
        coc_layout.setContentsMargins(16, 12, 16, 12)
        coc_layout.setSpacing(12)
        self.coc_status = StatusPill("CoC", "Checking…", Theme.TEXT_MUTED)
        self.coc_status.setAccessibleName("Clash of Clans status")
        coc_layout.addWidget(self.coc_status)
        coc_layout.addStretch()
        self.launch_button = QPushButton("Launch Clash of Clans")
        self.launch_button.setAccessibleName("Launch Clash of Clans")
        self.launch_button.setMinimumHeight(40)
        coc_layout.addWidget(self.launch_button)
        root.addWidget(coc_card)

        runtime = QFrame()
        runtime.setObjectName("CardRaised")
        runtime_layout = QVBoxLayout(runtime)
        runtime_layout.setContentsMargins(20, 18, 20, 18)
        runtime_layout.setSpacing(14)

        top = QHBoxLayout()
        selector_column = QVBoxLayout()
        selector_column.setSpacing(6)
        selector_label = _label("MACRO", "SectionTitle")
        selector_column.addWidget(selector_label)
        self.macro_selector = QComboBox()
        self.macro_selector.setPlaceholderText("Select a macro")
        self.macro_selector.setAccessibleName("Selected macro")
        self.macro_selector.setMinimumWidth(320)
        selector_column.addWidget(self.macro_selector)
        self.macro_meta = _label("Create or record a macro from the Macros page.", "CardCaption")
        selector_column.addWidget(self.macro_meta)
        top.addLayout(selector_column, 1)
        top.addStretch()
        self.record_button = QPushButton("Record")
        self.record_button.setAccessibleName("Record selected macro")
        self.record_button.setMinimumHeight(40)
        top.addWidget(self.record_button, alignment=Qt.AlignmentFlag.AlignTop)
        runtime_layout.addLayout(top)

        status_row = QHBoxLayout()
        self.runtime_status = _label("Ready", "PageSubtitle")
        self.runtime_status.setStyleSheet(f"color: {Theme.TEXT}; font-weight: 600;")
        status_row.addWidget(self.runtime_status)
        status_row.addStretch()
        self.shortcut_hint = _label("Esc stops immediately", "CardCaption")
        status_row.addWidget(self.shortcut_hint)
        runtime_layout.addLayout(status_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setAccessibleName("Macro progress")
        runtime_layout.addWidget(self.progress)

        metrics = QHBoxLayout()
        metrics.setSpacing(8)
        self.events_metric = MetricCard("EVENTS", "0", Theme.ACCENT)
        self.duration_metric = MetricCard("DURATION", "00:00", Theme.WARNING)
        self.elapsed_metric = MetricCard("ELAPSED", "00:00", Theme.INFO)
        self.cycles_metric = MetricCard("CYCLES", "0", Theme.ACCENT)
        for metric in (self.events_metric, self.duration_metric, self.elapsed_metric, self.cycles_metric):
            metrics.addWidget(metric, 1)
        runtime_layout.addLayout(metrics)

        options = QHBoxLayout()
        options.setSpacing(18)
        self.loop = QCheckBox("Loop playback")
        self.loop.setAccessibleName("Loop playback")
        options.addWidget(self.loop)
        self.safeguard = QCheckBox("Stop if CoC disappears")
        self.safeguard.setAccessibleName("Stop macro if Clash of Clans disappears")
        self.safeguard.setToolTip("Stops playback after the configured number of failed CoC checks.")
        options.addWidget(self.safeguard)
        options.addStretch()
        runtime_layout.addLayout(options)

        self.run_button = QPushButton("Run macro · F1")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.setAccessibleName("Run selected macro, keyboard shortcut F1")
        self.run_button.setMinimumHeight(48)
        runtime_layout.addWidget(self.run_button)
        root.addWidget(runtime, 1)

        self.macro_selector.currentTextChanged.connect(self._emit_macro_selection)
        self.record_button.clicked.connect(self.record_requested)
        self.run_button.clicked.connect(self.run_requested)
        self.launch_button.clicked.connect(self.coc_requested)
        self.loop.toggled.connect(self.loop_toggled)
        self.safeguard.toggled.connect(self.safeguard_toggled)
        self._update_contextual_metrics(False)
        self._update_controls()

    def set_macros(self, items: Iterable[MacroSummary], selected_name: str = "") -> None:
        names = [item.name for item in items]
        self.macro_selector.blockSignals(True)
        self.macro_selector.clear()
        self.macro_selector.addItems(names)
        if selected_name in names:
            self.macro_selector.setCurrentText(selected_name)
        elif names:
            self.macro_selector.setCurrentIndex(0)
        self.macro_selector.blockSignals(False)
        self._update_controls()

    def set_macro(self, macro) -> None:
        if macro and getattr(macro, "name", ""):
            self._event_count = macro.event_count()
            self.macro_selector.blockSignals(True)
            if self.macro_selector.findText(macro.name) >= 0:
                self.macro_selector.setCurrentText(macro.name)
            self.macro_selector.blockSignals(False)
            self.macro_meta.setText(f"{self._event_count:,} events  ·  {fmt_seconds(macro.duration())}")
            self.duration_metric.set_value(fmt_seconds(macro.duration()))
            self.events_metric.set_value(f"{self._event_count:,}")
        else:
            self._event_count = 0
            self.macro_meta.setText("Create or record a macro from the Macros page.")
            self.duration_metric.set_value("00:00")
            self.events_metric.set_value("0")
            self.progress.setValue(0)
        self._update_controls()

    def set_state(self, label: str, state_name: str) -> None:
        self._state_name = state_name
        self.runtime_status.setText(label)
        if state_name == "RECORDING":
            self.progress.setValue(0)
        self._update_contextual_metrics(self.loop.isChecked())
        self._update_controls()

    def set_coc_presence(self, snapshot) -> None:
        if snapshot.error:
            self.coc_status.set_status("Detection unavailable", Theme.WARNING)
        elif snapshot.present:
            self.coc_status.set_status("Detected", Theme.ACCENT)
        else:
            self.coc_status.set_status("Not detected", Theme.WARNING)

    def set_coc_launching(self, launching: bool) -> None:
        self._coc_launching = launching
        self.launch_button.setText("Launching…" if launching else "Launch Clash of Clans")
        self._update_controls()

    def set_safeguard(self, enabled: bool) -> None:
        self.safeguard.blockSignals(True)
        self.safeguard.setChecked(enabled)
        self.safeguard.blockSignals(False)

    def set_loop(self, enabled: bool) -> None:
        self.loop.blockSignals(True)
        self.loop.setChecked(enabled)
        self.loop.blockSignals(False)
        self._update_contextual_metrics(enabled)

    def set_metrics(self, elapsed: float, events: int, duration: float, cycles: int, state_name: str, loop: bool) -> None:
        self.elapsed_metric.set_value(fmt_seconds(elapsed))
        self.events_metric.set_value(f"{events:,}")
        self.duration_metric.set_value(fmt_seconds(duration))
        self.cycles_metric.set_value(str(cycles))
        self._update_contextual_metrics(loop)
        if state_name == "PLAYING" and duration > 0:
            current = elapsed % duration if loop else min(elapsed, duration)
            self.progress.setValue(max(0, min(100, int((current / duration) * 100))))
        elif state_name != "RECORDING":
            self.progress.setValue(0)

    def _emit_macro_selection(self, name: str) -> None:
        if name:
            self.macro_selected.emit(name)

    def _update_contextual_metrics(self, loop: bool) -> None:
        active = self._state_name in {"RECORDING", "PLAYING", "STOPPING"}
        self.elapsed_metric.setVisible(active)
        self.cycles_metric.setVisible(loop or self._state_name in {"PLAYING", "STOPPING"})

    def _update_controls(self) -> None:
        recording = self._state_name == "RECORDING"
        playing = self._state_name == "PLAYING"
        stopping = self._state_name == "STOPPING"
        busy = recording or playing or stopping
        has_macro = bool(self.macro_selector.currentText())

        self.macro_selector.setEnabled(not busy)
        self.loop.setEnabled(not recording and not stopping)
        self.safeguard.setEnabled(not recording and not stopping)
        self.launch_button.setEnabled(not busy and not self._coc_launching)
        self.record_button.setText("Stop recording" if recording else "Record")
        self.record_button.setObjectName("DangerButton" if recording else "")
        self.record_button.setAccessibleName("Stop recording selected macro" if recording else "Record selected macro")
        self.record_button.setEnabled(has_macro and not playing and not stopping)

        if playing:
            self.run_button.setText("Stop macro · F1")
            self.run_button.setObjectName("DangerButton")
            self.run_button.setAccessibleName("Stop running macro, keyboard shortcut F1")
            self.run_button.setEnabled(True)
        elif stopping:
            self.run_button.setText("Stopping…")
            self.run_button.setObjectName("DangerButton")
            self.run_button.setEnabled(False)
        else:
            self.run_button.setText("Run macro · F1")
            self.run_button.setObjectName("PrimaryButton")
            self.run_button.setAccessibleName("Run selected macro, keyboard shortcut F1")
            self.run_button.setEnabled(has_macro and self._event_count > 0 and not recording)
        for button in (self.record_button, self.run_button):
            button.style().unpolish(button)
            button.style().polish(button)
