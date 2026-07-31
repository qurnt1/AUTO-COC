# -*- coding: utf-8 -*-
"""Visual system for the PyQt6 operator console."""

from __future__ import annotations

class Theme:
    BG = "#121417"
    SURFACE = "#1A1E22"
    SURFACE_RAISED = "#22282E"
    SURFACE_HOVER = "#2A3239"
    BORDER = "#343C43"
    BORDER_STRONG = "#4C5962"
    TEXT = "#F5F1E8"
    TEXT_MUTED = "#B5B8B0"
    TEXT_SUBTLE = "#7F8987"
    ACCENT = "#C7F36B"
    ACCENT_DARK = "#7DAA3A"
    ACCENT_SOFT = "#28331F"
    WARNING = "#F5B967"
    WARNING_SOFT = "#45371F"
    DANGER = "#FF7A68"
    DANGER_DARK = "#B64F49"
    DANGER_SOFT = "#452927"
    INFO = "#B19CFF"
    INFO_SOFT = "#302B4D"
    WHITE = "#FFFFFF"


def hex_to_rgb(hex_code: str) -> tuple[int, int, int]:
    value = hex_code.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def get_luminance(r: int, g: int, b: int) -> float:
    values = []
    for value in (r, g, b):
        normalized = value / 255.0
        values.append(
            normalized / 12.92
            if normalized <= 0.03928
            else ((normalized + 0.055) / 1.055) ** 2.4
        )
    return 0.2126 * values[0] + 0.7152 * values[1] + 0.0722 * values[2]


def get_contrast_ratio(foreground: str, background: str) -> float:
    foreground_luminance = get_luminance(*hex_to_rgb(foreground))
    background_luminance = get_luminance(*hex_to_rgb(background))
    lighter = max(foreground_luminance, background_luminance)
    darker = min(foreground_luminance, background_luminance)
    return (lighter + 0.05) / (darker + 0.05)


def build_stylesheet() -> str:
    return f"""
    * {{
        font-family: "Segoe UI";
        font-size: 13px;
        color: {Theme.TEXT};
    }}
    QMainWindow, QWidget#WindowRoot {{ background: {Theme.BG}; }}
    QToolTip {{
        background: {Theme.SURFACE_RAISED};
        color: {Theme.TEXT};
        border: 1px solid {Theme.BORDER_STRONG};
        padding: 6px 8px;
    }}
    QFrame#Card {{
        background: {Theme.SURFACE};
        border: 1px solid {Theme.BORDER};
        border-radius: 12px;
    }}
    QFrame#CardRaised {{
        background: {Theme.SURFACE_RAISED};
        border: 1px solid {Theme.BORDER};
        border-radius: 12px;
    }}
    QLabel#PageEyebrow {{
        color: {Theme.ACCENT};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
    }}
    QLabel#PageTitle {{
        color: {Theme.TEXT};
        font-size: 28px;
        font-weight: 700;
    }}
    QLabel#PageSubtitle {{ color: {Theme.TEXT_MUTED}; font-size: 13px; }}
    QLabel#CardTitle {{ color: {Theme.TEXT}; font-size: 15px; font-weight: 700; }}
    QLabel#SectionTitle {{ color: {Theme.TEXT}; font-size: 12px; font-weight: 700; }}
    QLabel#CardCaption {{ color: {Theme.TEXT_MUTED}; font-size: 12px; }}
    QLabel#MetricValue {{ color: {Theme.TEXT}; font-size: 24px; font-weight: 700; }}
    QLabel#MetricLabel {{ color: {Theme.TEXT_MUTED}; font-size: 11px; }}
    QLabel#Mono {{ font-family: "Cascadia Mono"; color: {Theme.TEXT_MUTED}; }}
    QLineEdit, QSpinBox, QPlainTextEdit, QTextEdit, QComboBox {{
        background: {Theme.BG};
        border: 1px solid {Theme.BORDER};
        border-radius: 8px;
        padding: 9px 11px;
        selection-background-color: {Theme.ACCENT_DARK};
        selection-color: {Theme.WHITE};
    }}
    QLineEdit:focus, QSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus {{
        border: 1px solid {Theme.ACCENT};
    }}
    QLineEdit::placeholder {{ color: {Theme.TEXT_SUBTLE}; }}
    QComboBox QAbstractItemView {{
        background: {Theme.SURFACE_RAISED};
        border: 1px solid {Theme.BORDER};
        selection-background-color: {Theme.ACCENT_DARK};
    }}
    QPushButton {{
        background: {Theme.SURFACE_RAISED};
        border: 1px solid {Theme.BORDER};
        border-radius: 8px;
        padding: 9px 14px;
        font-weight: 600;
    }}
    QPushButton:hover {{ background: {Theme.SURFACE_HOVER}; border-color: {Theme.BORDER_STRONG}; }}
    QPushButton:pressed {{ background: {Theme.BORDER}; }}
    QPushButton:focus {{ border: 1px solid {Theme.ACCENT}; }}
    QPushButton:disabled {{ color: {Theme.TEXT_SUBTLE}; background: {Theme.SURFACE}; border-color: {Theme.BORDER}; }}
    QPushButton#PrimaryButton {{ background: {Theme.ACCENT}; color: #07131A; border: none; }}
    QPushButton#PrimaryButton:hover {{ background: #D9FF8A; }}
    QPushButton#PrimaryButton:pressed {{ background: {Theme.ACCENT_DARK}; color: {Theme.WHITE}; }}
    QPushButton#DangerButton {{ background: {Theme.DANGER_SOFT}; color: #FFD8DB; border-color: {Theme.DANGER_DARK}; }}
    QPushButton#DangerButton:hover {{ background: {Theme.DANGER_DARK}; color: {Theme.WHITE}; }}
    QPushButton#QuietButton {{ background: transparent; border-color: transparent; color: {Theme.TEXT_MUTED}; }}
    QPushButton#QuietButton:hover {{ background: {Theme.SURFACE_HOVER}; color: {Theme.TEXT}; }}
    QToolButton#NavButton {{
        text-align: left;
        background: transparent;
        border: 1px solid transparent;
        border-radius: 8px;
        padding: 10px 12px;
        color: {Theme.TEXT_MUTED};
        font-weight: 600;
    }}
    QToolButton#NavButton:hover {{ background: {Theme.SURFACE_HOVER}; color: {Theme.TEXT}; }}
    QToolButton#NavButton:focus {{ border: 1px solid {Theme.ACCENT}; }}
    QToolButton#NavButton:checked {{ background: {Theme.ACCENT_SOFT}; color: {Theme.ACCENT}; border-color: #285D4C; }}
    QListView {{ background: transparent; border: none; outline: none; }}
    QListView#SystemRoutineList {{ background: {Theme.ACCENT_SOFT}; border: 1px solid #44552B; border-radius: 8px; padding: 4px; }}
    QListView:focus {{ border: 1px solid {Theme.ACCENT}; border-radius: 8px; }}
    QListView::item {{ border: none; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px; }}
    QScrollBar::handle:vertical {{ background: {Theme.BORDER}; border-radius: 5px; min-height: 24px; }}
    QScrollBar::handle:vertical:hover {{ background: {Theme.BORDER_STRONG}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QProgressBar {{ background: {Theme.BG}; border: none; border-radius: 4px; text-align: center; color: {Theme.TEXT}; height: 8px; }}
    QProgressBar::chunk {{ background: {Theme.ACCENT}; border-radius: 4px; }}
    QTableWidget {{ background: transparent; border: none; gridline-color: {Theme.BORDER}; }}
    QHeaderView::section {{ background: {Theme.SURFACE_RAISED}; color: {Theme.TEXT_MUTED}; border: none; padding: 8px; font-weight: 600; }}
    QTableWidget::item {{ padding: 8px; border-bottom: 1px solid {Theme.BORDER}; }}
    QTableWidget::item:selected {{ background: {Theme.ACCENT_SOFT}; color: {Theme.TEXT}; }}
    QCheckBox {{ spacing: 8px; color: {Theme.TEXT}; }}
    QCheckBox::indicator {{ width: 18px; height: 18px; border: 1px solid {Theme.BORDER_STRONG}; border-radius: 5px; background: {Theme.BG}; }}
    QCheckBox::indicator:checked {{ background: {Theme.ACCENT}; border-color: {Theme.ACCENT}; }}
    QStatusBar {{ background: {Theme.SURFACE}; color: {Theme.TEXT_MUTED}; border-top: 1px solid {Theme.BORDER}; }}
    QDialog {{ background: {Theme.BG}; }}
    QDialogButtonBox QPushButton {{ min-width: 92px; }}
    """
