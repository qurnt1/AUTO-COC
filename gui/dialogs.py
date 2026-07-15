# -*- coding: utf-8 -*-
"""
Macro COC v3.0 — GUI / Dialogs

Fenetres popup basees sur qfluentwidgets (Dialog, MessageBox, InfoBar).
"""

import webbrowser
from pathlib import Path
from typing import Callable, Dict, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QTextEdit,
)
from PyQt6.QtCore import Qt

from qfluentwidgets import (
    Dialog, MessageBox, InfoBar, InfoBarPosition,
    LineEdit, PushButton, PrimaryPushButton,
    BodyLabel, StrongBodyLabel, CaptionLabel,
    FluentIcon, TextEdit, ScrollArea,
)

from gui.theme import Theme
from utils.config import as_bool
from utils.logger import get_logger


# =========================
#     Toast / InfoBar
# =========================

def show_toast(parent: QWidget, title: str, content: str,
               level: str = "success", duration: int = 3000):
    """
    Affiche une notification InfoBar Fluent.

    Args:
        parent: Widget parent (doit etre visible)
        title: Titre de la notification
        content: Message
        level: 'success', 'warning', 'error', 'info'
        duration: Duree d'affichage en ms
    """
    bar_map = {
        "success": InfoBar.success,
        "warning": InfoBar.warning,
        "error":   InfoBar.error,
        "info":    InfoBar.info,
    }
    creator = bar_map.get(level, InfoBar.info)

    creator(
        title=title,
        content=content,
        parent=parent,
        duration=duration,
        position=InfoBarPosition.TOP_RIGHT,
        isClosable=True,
    )


# =========================
#     TextInputDialog
# =========================

class TextInputDialog:
    """
    Popup modal pour demander un nom.
    Compatible API avec l'ancien TextInputDialog.show() -> Optional[str].
    """

    def __init__(self, master, title: str, prompt: str, initial: str = ""):
        self._master = master
        self._result: Optional[str] = None

        self._dialog = Dialog(title, "", master)
        self._dialog.titleLabel.setStyleSheet("font-size: 16px; font-weight: bold;")

        # Contenu
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(12)

        # Prompt
        lbl = BodyLabel(prompt)
        layout.addWidget(lbl)

        # Entry
        self._entry = LineEdit()
        self._entry.setText(initial)
        self._entry.setClearButtonEnabled(True)
        layout.addWidget(self._entry)

        self._dialog.setContentView(content)

        # Boutons
        self._dialog.yesButton.setText("Valider")
        self._dialog.cancelButton.setText("Annuler")
        self._dialog.yesButton.clicked.connect(self._on_accept)
        self._dialog.cancelButton.clicked.connect(self._on_reject)

    def _on_accept(self):
        self._result = self._entry.text().strip() or None

    def _on_reject(self):
        self._result = None

    def show(self) -> Optional[str]:
        """Affiche le dialog et retourne le resultat."""
        self._result = None
        if self._dialog.exec():
            return self._result
        return None


# =========================
#     TelegramAutomationDialog
# =========================

class TelegramAutomationDialog:
    """
    Fenetre de configuration Telegram.
    """

    def __init__(
        self,
        master,
        params: Dict[str, str],
        on_save: Callable,
        guide_html_path: Optional[Path] = None,
        icon_path=None,
        name=None,
        on_close_cb=None,
    ):
        self._params = params
        self._on_save = on_save
        self._guide_path = guide_html_path
        self._on_close_cb = on_close_cb
        self._window_name = name

        self._dialog = Dialog("Automatisation Telegram", "", master)
        self._dialog.titleLabel.setStyleSheet("font-size: 16px; font-weight: bold;")

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(12)

        # Token
        layout.addWidget(BodyLabel("Bot token"))
        self._tg_token = LineEdit()
        self._tg_token.setText(params.get("telegram_bot_token", ""))
        self._tg_token.setClearButtonEnabled(True)
        layout.addWidget(self._tg_token)

        # Chat ID
        layout.addWidget(BodyLabel("Chat ID"))
        self._tg_chat = LineEdit()
        self._tg_chat.setText(params.get("telegram_chat_id", ""))
        self._tg_chat.setClearButtonEnabled(True)
        layout.addWidget(self._tg_chat)

        # Bouton guide
        guide_btn = PushButton(FluentIcon.LINK, "  Ouvrir le guide (page HTML)")
        guide_btn.clicked.connect(self._open_guide)
        layout.addWidget(guide_btn)

        self._dialog.setContentView(content)

        # Boutons
        self._dialog.yesButton.setText("Enregistrer")
        self._dialog.cancelButton.setText("Fermer")
        self._dialog.yesButton.clicked.connect(self._save)
        self._dialog.cancelButton.clicked.connect(self._close)

    def _open_guide(self):
        if self._guide_path and self._guide_path.exists():
            try:
                webbrowser.open(self._guide_path.resolve().as_uri())
            except Exception as e:
                MessageBox("Erreur", f"Impossible d'ouvrir le guide: {e}", self._dialog).exec()
        else:
            MessageBox("Guide introuvable", "Le fichier guide_telegram.html n'a pas ete trouve.", self._dialog).exec()

    def _save(self):
        self._params["telegram_bot_token"] = self._tg_token.text().strip()
        self._params["telegram_chat_id"] = self._tg_chat.text().strip()
        if callable(self._on_save):
            self._on_save(self._params)
        self._dialog.accept()
        self._notify_close()

    def _close(self):
        self._dialog.reject()
        self._notify_close()

    def _notify_close(self):
        if callable(self._on_close_cb):
            try:
                self._on_close_cb(self._window_name)
            except Exception as e:
                get_logger().error(f"Erreur on_close_cb ({self._window_name}): {e}")

    def show(self):
        self._dialog.exec()


# =========================
#     DiagnosticsDialog
# =========================

class DiagnosticsDialog:
    """
    Fenetre de diagnostic systeme.
    """

    def __init__(
        self,
        master,
        tg_status: str,
        app_version: str,
        python_version: str,
        pil_available: bool,
        mss_available: bool,
        log_path: Path,
        base_dir: Path,
        icon_path=None,
        name=None,
        on_close_cb=None,
    ):
        self._on_close_cb = on_close_cb
        self._window_name = name

        self._dialog = Dialog("Etat du systeme", "", master)
        self._dialog.titleLabel.setStyleSheet("font-size: 16px; font-weight: bold;")

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(4)

        def _row(k: str, v: str, color: str = "rgba(255,255,255,0.93)"):
            r = QHBoxLayout()
            r.setSpacing(8)
            kl = CaptionLabel(k)
            kl.setFixedWidth(180)
            kl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            r.addWidget(kl)
            vl = CaptionLabel(v)
            vl.setStyleSheet(f"color: {color};")
            vl.setWordWrap(True)
            r.addWidget(vl, stretch=1)
            layout.addLayout(r)

        _row("Version App:", app_version)
        _row("Version Python:", python_version)

        pil_txt = "Oui" if pil_available else "Non (CAPTURE INDISPONIBLE)"
        pil_col = Theme.STATUS_OK if pil_available else Theme.STATUS_ERROR
        _row("Pillow:", pil_txt, pil_col)

        mss_txt = "Oui" if mss_available else "Non (fallback)"
        mss_col = Theme.STATUS_OK if mss_available else Theme.STATUS_WARN
        _row("MSS:", mss_txt, mss_col)

        _row("Telegram:", tg_status)

        try:
            import shutil
            disk = shutil.disk_usage(base_dir)
            free_gb = disk.free / (1024**3)
            _row("Espace disque:", f"{free_gb:.2f} Go libres")
        except Exception as e:
            _row("Espace disque:", f"Erreur: {e}", Theme.STATUS_ERROR)

        # Logs
        layout.addWidget(StrongBodyLabel("Derniers logs (5)"))
        log_text = TextEdit()
        log_text.setReadOnly(True)
        log_text.setMinimumHeight(200)
        try:
            if log_path.exists():
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    log_text.setPlainText("".join(lines[-5:]))
            else:
                log_text.setPlainText("Fichier log non encore cree.")
        except Exception as e:
            log_text.setPlainText(f"Erreur: {e}")
        layout.addWidget(log_text)

        self._dialog.setContentView(content)

        self._dialog.yesButton.setText("Fermer")
        self._dialog.cancelButton.hide()
        self._dialog.yesButton.clicked.connect(self._close)

    def _close(self):
        self._dialog.accept()
        if callable(self._on_close_cb):
            try:
                self._on_close_cb(self._window_name)
            except Exception as e:
                get_logger().error(f"Erreur on_close_cb ({self._window_name}): {e}")

    def show(self):
        self._dialog.exec()
