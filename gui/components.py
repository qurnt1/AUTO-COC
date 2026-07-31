# -*- coding: utf-8 -*-
"""Reusable PyQt6 widgets for the AUTO-COC console."""

from __future__ import annotations

from typing import Callable, Iterable

from PyQt6.QtCore import QAbstractListModel, QModelIndex, QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QPushButton,
    QStyledItemDelegate,
    QStyle,
    QStyleOptionViewItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui.theme import Theme
from gui.models import MacroSummary
from utils.config import fmt_duration_for_list


class NavButton(QToolButton):
    def __init__(self, label: str, tooltip: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("NavButton")
        self.setText(label)
        self.setToolTip(tooltip)
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(42)


class StatusPill(QFrame):
    def __init__(self, label: str, value: str = "—", color: str = Theme.TEXT_MUTED, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(7)
        self.dot = QLabel("●")
        self.dot.setStyleSheet(f"color: {color}; font-size: 12px;")
        self.label = QLabel(label.upper())
        self.label.setStyleSheet(f"color: {Theme.TEXT_SUBTLE}; font-size: 10px; font-weight: 700;")
        self.value = QLabel(value)
        self.value.setStyleSheet(f"color: {Theme.TEXT}; font-weight: 600;")
        layout.addWidget(self.dot)
        layout.addWidget(self.label)
        layout.addWidget(self.value)

    def set_status(self, value: str, color: str) -> None:
        self.value.setText(value)
        self.dot.setStyleSheet(f"color: {color}; font-size: 12px;")


class MetricCard(QFrame):
    def __init__(self, label: str, value: str = "—", accent: str = Theme.ACCENT, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 13, 14, 13)
        layout.setSpacing(4)
        self.value = QLabel(value)
        self.value.setObjectName("MetricValue")
        self.value.setStyleSheet(f"color: {accent};")
        caption = QLabel(label)
        caption.setObjectName("MetricLabel")
        layout.addWidget(self.value)
        layout.addWidget(caption)

    def set_value(self, value: str) -> None:
        self.value.setText(value)


class MacroListModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[MacroSummary] = []

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._items)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._items)):
            return None
        item = self._items[index.row()]
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.UserRole:
            return item
        return None

    def set_items(self, items: Iterable[MacroSummary]) -> None:
        self.beginResetModel()
        self._items = list(items)
        self.endResetModel()

    def item_at(self, row: int) -> MacroSummary | None:
        return self._items[row] if 0 <= row < len(self._items) else None


class MacroDelegate(QStyledItemDelegate):
    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        item: MacroSummary = index.data(Qt.ItemDataRole.UserRole)
        rect = option.rect.adjusted(6, 4, -6, -4)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        background = Theme.ACCENT_SOFT if selected else Theme.SURFACE_RAISED if hovered else Theme.SURFACE
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(background))
        painter.drawRoundedRect(rect, 9, 9)
        if selected:
            painter.setPen(QPen(QColor(Theme.ACCENT), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect.adjusted(0, 0, -1, -1), 9, 9)
        painter.setPen(QColor(Theme.TEXT))
        painter.setFont(painter.font())
        name_rect = QRect(rect.left() + 13, rect.top() + 10, rect.width() - 26, 21)
        painter.drawText(name_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, item.name)
        meta = f"{item.events:,} events  ·  {fmt_duration_for_list(item.duration)}"
        painter.setPen(QColor(Theme.TEXT_MUTED))
        painter.setFont(painter.font())
        meta_rect = QRect(rect.left() + 13, rect.bottom() - 26, rect.width() - 26, 17)
        painter.drawText(meta_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, meta)
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        return QSize(260, 72)


class MacroLibrary(QFrame):
    macro_selected = pyqtSignal(str)
    create_requested = pyqtSignal()
    rename_requested = pyqtSignal()
    delete_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("Macros")
        title.setObjectName("CardTitle")
        self.count = QLabel("0 macros")
        self.count.setObjectName("CardCaption")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.count)
        layout.addLayout(header)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search macros…")
        self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName("Search macros")
        layout.addWidget(self.search)

        self.model = MacroListModel(self)
        self._all_items: list[MacroSummary] = []
        self._filter_callback: Callable[[], None] | None = None
        self._updating = False
        self.view = QListView()
        self.view.setModel(self.model)
        self.view.setItemDelegate(MacroDelegate(self.view))
        self.view.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self.view.setVerticalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        self.view.setMouseTracking(True)
        self.view.setSpacing(3)
        self.view.setAccessibleName("Macro list")
        layout.addWidget(self.view, 1)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.create_button = QPushButton("New")
        self.create_button.setObjectName("PrimaryButton")
        self.rename_button = QPushButton("Rename")
        self.delete_button = QPushButton("Delete")
        self.delete_button.setObjectName("DangerButton")
        for button in (self.create_button, self.rename_button, self.delete_button):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        actions.addWidget(self.create_button)
        actions.addWidget(self.rename_button)
        actions.addWidget(self.delete_button)
        layout.addLayout(actions)

        self.search.textChanged.connect(self._filter)
        self.view.selectionModel().currentChanged.connect(lambda current, _previous: self._select(current) if current.isValid() else None)
        self.create_button.clicked.connect(self.create_requested)
        self.rename_button.clicked.connect(self.rename_requested)
        self.delete_button.clicked.connect(self.delete_requested)

    def set_items(self, items: Iterable[MacroSummary], selected_name: str | None = None) -> None:
        self._updating = True
        self._all_items = list(items)
        all_items = list(self._all_items)
        query = self.search.text().strip().lower()
        if query:
            all_items = [item for item in all_items if query in item.name.lower()]
        self.model.set_items(all_items)
        self.count.setText(f"{len(all_items)} macro{'' if len(all_items) == 1 else 's'}")
        if selected_name:
            for row in range(self.model.rowCount()):
                item = self.model.item_at(row)
                if item and item.name == selected_name:
                    self.view.setCurrentIndex(self.model.index(row, 0))
                    break
        self._updating = False
        self._update_actions()

    def selected_name(self) -> str | None:
        user_indexes = self.view.selectionModel().selectedIndexes()
        item = self.model.item_at(user_indexes[0].row()) if user_indexes else None
        if item:
            return item.name
        return None

    def _filter(self) -> None:
        if self._filter_callback:
            self._filter_callback()
        else:
            self.set_items(self._all_items, self.selected_name())

    def set_filter_callback(self, callback: Callable[[], None]) -> None:
        self._filter_callback = callback

    def _select(self, index: QModelIndex) -> None:
        if self._updating:
            return
        item = self.model.item_at(index.row())
        if item:
            self.macro_selected.emit(item.name)
        self._update_actions()

    def _update_actions(self) -> None:
        user_selected = bool(self.view.selectionModel().selectedIndexes())
        self.rename_button.setEnabled(user_selected)
        self.delete_button.setEnabled(user_selected)


class ActivityFeed(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 10)
        layout.setSpacing(8)
        header = QHBoxLayout()
        title = QLabel("Activity")
        title.setObjectName("CardTitle")
        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("QuietButton")
        self.clear_button.clicked.connect(self.clear)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.clear_button)
        layout.addLayout(header)
        self.list = QListView()
        self.list.setObjectName("ActivityList")
        self.list.setMaximumHeight(180)
        self.list.setEditTriggers(QListView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.list)
        self._model = _ActivityModel(self)
        self.list.setModel(self._model)

    def add(self, message: str, level: str = "info") -> None:
        self._model.add(message, level)
        self.list.scrollToTop()

    def clear(self) -> None:
        self._model.clear()


class _ActivityModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[tuple[str, str]] = []

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._items)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self._items):
            return None
        text, level = self._items[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return text
        if role == Qt.ItemDataRole.ForegroundRole:
            return QColor({"success": Theme.ACCENT, "warning": Theme.WARNING, "error": Theme.DANGER}.get(level, Theme.TEXT_MUTED))
        return None

    def add(self, message: str, level: str) -> None:
        self.beginResetModel()
        self._items.insert(0, (message, level))
        self._items = self._items[:30]
        self.endResetModel()

    def clear(self) -> None:
        self.beginResetModel()
        self._items.clear()
        self.endResetModel()
