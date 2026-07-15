# -*- coding: utf-8 -*-
"""
Macro COC v3.0 — GUI / Theme

Constantes de couleurs et utilitaires WCAG pour les widgets custom.
Le theme global (sombre, Material) est gere par qfluentwidgets.
"""


class Theme:
    """Theme sombre pour Macro COC — constantes de reference."""

    # === Dimensions ===
    HEADER_HEIGHT = 88
    ICON_PNG_SIZE = (140, 88)

    # === Couleurs principales ===
    APP_BG = "#0b1220"
    HEADER_BG = "#0b1220"
    STATUS_BG = "#0b0f19"

    # === Panneau gauche (liste macros) ===
    LEFT_CONTAINER_BG = "#0e1624"
    LEFT_HEADER_TEXT = "#e5e7eb"
    LEFT_ACTIONS_BG = "#132033"

    # === Lignes de macro ===
    ROW_BG = "#152235"
    ROW_HOVER = "#1d2d45"
    ROW_SELECTED = "#2a4365"
    ROW_NAME_COLOR = "#e5e7eb"
    ROW_DUR_COLOR = "#93c5fd"

    # === Zone centrale ===
    CENTER_BG = "#111827"
    INFO_BG = "#0e1624"
    INFO_TEXT_MUTED = "#cbd5e1"

    # === Boutons primaires (vert COC) ===
    BTN_PRIMARY_BG = "#22c55e"
    BTN_PRIMARY_HOVER = "#16a34a"

    # === Boutons stop/danger ===
    BTN_STOP_BG = "#ef4444"
    BTN_STOP_HOVER = "#b91c1c"

    # === Boutons lancement ===
    BTN_LAUNCH_BG = "#0ea5e9"
    BTN_LAUNCH_HOVER = "#0284c7"

    # === Boutons desactives ===
    BTN_DISABLED_BG = "#374151"
    BTN_DISABLED_TEXT = "#9ca3af"

    # === Divers ===
    CARD_BG = "#0b1220"
    DIVIDER = "#334155"

    # === Couleurs de statut ===
    STATUS_OK = "#22c55e"
    STATUS_WARN = "#f59e0b"
    STATUS_ERROR = "#ef4444"

    # === Texte accessible (WCAG) ===
    TEXT_COMPLIANT = "#e5e7eb"
    TEXT_MUTED_COMPLIANT = "#cbd5e1"

    # === Timeline visuelle (couleurs par type d'evenement) ===
    TIMELINE_CLICK = "#60a5fa"     # bleu — mouse_click
    TIMELINE_MOVE = "#818cf8"      # indigo — mouse_move
    TIMELINE_KEY = "#f59e0b"       # orange — key_down/up
    TIMELINE_SCROLL = "#34d399"    # vert — scroll


# =========================
#     WCAG Contrast
# =========================

def hex_to_rgb(hex_code: str) -> tuple:
    """Convertit #RRGGBB en (R, G, B)."""
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


def get_luminance(r: int, g: int, b: int) -> float:
    """Calcule la luminance relative (WCAG)."""
    vals = []
    for v in (r, g, b):
        v = v / 255.0
        v = v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
        vals.append(v)
    return 0.2126 * vals[0] + 0.7152 * vals[1] + 0.0722 * vals[2]


def get_contrast_ratio(hex_fg: str, hex_bg: str) -> float:
    """Calcule le ratio de contraste (WCAG)."""
    try:
        lum1 = get_luminance(*hex_to_rgb(hex_fg))
        lum2 = get_luminance(*hex_to_rgb(hex_bg))
        if lum1 > lum2:
            return (lum1 + 0.05) / (lum2 + 0.05)
        return (lum2 + 0.05) / (lum1 + 0.05)
    except Exception:
        return 1.0


def ensure_contrast(fg: str, bg: str, target_ratio: float = 4.5) -> str:
    """
    Verifie le contraste. Si insuffisant, retourne un fallback (blanc/noir).

    Args:
        fg: Couleur foreground (#RRGGBB)
        bg: Couleur background (#RRGGBB)
        target_ratio: Ratio WCAG cible (4.5 pour AA)

    Returns:
        Couleur fg corrigee si necessaire
    """
    ratio = get_contrast_ratio(fg, bg)
    if ratio >= target_ratio:
        return fg

    bg_lum = get_luminance(*hex_to_rgb(bg))
    if bg_lum > 0.5:
        return "#000000"
    else:
        return "#FFFFFF"
