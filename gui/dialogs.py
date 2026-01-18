# -*- coding: utf-8 -*-
"""
Macro COC v3.0 — GUI / Dialogs

Fenêtres popup (Paramètres, Configuration Telegram, Diagnostic, etc.)
"""

import webbrowser
from pathlib import Path
from tkinter import messagebox
from typing import Callable, Dict, Optional

import customtkinter as ctk

from gui.theme import Theme
from utils.config import as_bool
from utils.logger import get_logger


class BaseToplevel(ctk.CTkToplevel):
    """
    Fenêtre popup de base avec gestion d'icône et de fermeture.
    """
    
    def __init__(self, master, *args, **kwargs):
        self._window_name = kwargs.pop("name", None)
        self._on_close_cb = kwargs.pop("on_close_cb", None)
        
        super().__init__(master, *args, **kwargs)
        
        # Appliquer l'icône
        self._apply_icon()
        
        self.protocol("WM_DELETE_WINDOW", self._on_wm_close)
        
        # Centrer sur le parent
        self.transient(master)
        self.grab_set()
    
    def _apply_icon(self):
        """Applique l'icône à la fenêtre."""
        # Sera configuré par App avec le chemin correct
        pass
    
    def _on_wm_close(self):
        """Appelé par le 'X' de la fenêtre."""
        self.grab_release()
        if callable(self._on_close_cb):
            try:
                self._on_close_cb(self._window_name)
            except Exception as e:
                get_logger().error(f"Erreur on_close_cb ({self._window_name}): {e}")
            self._on_close_cb = None
        super().destroy()
    
    def destroy(self):
        """Appelé par les boutons (Sauver, Annuler, Fermer)."""
        self.grab_release()
        if callable(self._on_close_cb):
            try:
                self._on_close_cb(self._window_name)
            except Exception as e:
                get_logger().error(f"Erreur on_close_cb ({self._window_name}): {e}")
            self._on_close_cb = None
        super().destroy()


class TextInputDialog(BaseToplevel):
    """
    Popup pour demander un nom (macro, etc.).
    """
    
    def __init__(
        self,
        master,
        title: str,
        prompt: str,
        initial: str = ""
    ):
        super().__init__(master, name="TextInput")
        self.title(title)
        self.geometry("380x160")
        self.resizable(False, False)
        
        wrap = ctk.CTkFrame(self, corner_radius=12)
        wrap.pack(fill="both", expand=True, padx=12, pady=12)
        
        ctk.CTkLabel(
            wrap,
            text=prompt,
            text_color=Theme.TEXT_COMPLIANT
        ).pack(anchor="w", pady=(4, 6))
        
        self._var = ctk.StringVar(value=initial)
        self._entry = ctk.CTkEntry(wrap, textvariable=self._var)
        self._entry.pack(fill="x")
        
        btns = ctk.CTkFrame(wrap, fg_color="transparent")
        btns.pack(fill="x", pady=(12, 0))
        
        ctk.CTkButton(
            btns,
            text="Annuler",
            width=100,
            fg_color="#374151",
            hover_color="#4b5563",
            command=self._cancel
        ).pack(side="right")
        
        ctk.CTkButton(
            btns,
            text="Valider",
            width=110,
            command=self._ok
        ).pack(side="right", padx=(0, 8))
        
        self._entry.focus_set()
        self._result: Optional[str] = None
        
        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self._cancel())
    
    def _ok(self):
        self._result = self._var.get().strip() or None
        self.destroy()
    
    def _cancel(self):
        self._result = None
        self.destroy()
    
    def show(self) -> Optional[str]:
        """Affiche le dialog et retourne le résultat."""
        self.wait_window(self)
        return self._result


class TelegramAutomationDialog(BaseToplevel):
    """
    Fenêtre de configuration Telegram.
    """
    
    def __init__(
        self,
        master,
        params: Dict[str, str],
        on_save: Callable,
        guide_html_path: Optional[Path] = None,
        **kwargs
    ):
        super().__init__(master, **kwargs)
        self.title("Automatisation Telegram")
        self.geometry("640x260")
        self.resizable(False, False)
        
        self._params = params
        self._on_save = on_save
        self._guide_path = guide_html_path
        
        root = ctk.CTkFrame(self, corner_radius=12)
        root.pack(fill="both", expand=True, padx=12, pady=12)
        root.grid_columnconfigure(0, weight=1)
        
        # Champs
        grid = ctk.CTkFrame(root, fg_color="transparent")
        grid.grid(row=0, column=0, sticky="ew")
        grid.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(grid, text="Bot token").grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=(2, 6)
        )
        self._tg_token = ctk.StringVar(value=self._params.get("telegram_bot_token", ""))
        ctk.CTkEntry(grid, textvariable=self._tg_token).grid(
            row=0, column=1, sticky="ew", pady=(2, 6)
        )
        
        ctk.CTkLabel(grid, text="Chat ID").grid(
            row=1, column=0, sticky="w", padx=(0, 10), pady=(2, 6)
        )
        self._tg_chat = ctk.StringVar(value=self._params.get("telegram_chat_id", ""))
        ctk.CTkEntry(grid, textvariable=self._tg_chat).grid(
            row=1, column=1, sticky="ew", pady=(2, 6)
        )
        
        # Bouton guide
        btns2 = ctk.CTkFrame(root, fg_color="transparent")
        btns2.grid(row=1, column=0, sticky="ew", pady=(10, 8))
        
        ctk.CTkButton(
            btns2,
            text="Ouvrir le guide (page HTML locale)",
            width=260,
            command=self._open_guide
        ).pack(side="left", padx=(0, 8))
        
        # Boutons bas
        bottom = ctk.CTkFrame(root, fg_color="transparent")
        bottom.grid(row=2, column=0, sticky="ew")
        
        ctk.CTkButton(
            bottom,
            text="Fermer",
            width=120,
            fg_color="#374151",
            hover_color="#4b5563",
            command=self.destroy
        ).pack(side="right")
        
        ctk.CTkButton(
            bottom,
            text="Enregistrer",
            width=130,
            command=self._save
        ).pack(side="right", padx=(0, 8))
    
    def _open_guide(self):
        """Ouvre le guide HTML local."""
        if self._guide_path and self._guide_path.exists():
            try:
                webbrowser.open(self._guide_path.resolve().as_uri())
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible d'ouvrir le guide: {e}")
        else:
            messagebox.showwarning(
                "Guide introuvable",
                "Le fichier guide_telegram.html n'a pas été trouvé."
            )
    
    def _save(self):
        self._params["telegram_bot_token"] = self._tg_token.get().strip()
        self._params["telegram_chat_id"] = self._tg_chat.get().strip()
        if callable(self._on_save):
            self._on_save(self._params)
        self.destroy()


class SettingsDialog(BaseToplevel):
    """
    Fenêtre de paramètres généraux.
    """
    
    def __init__(
        self,
        master,
        params: Dict[str, str],
        on_save: Callable,
        open_tg_dialog_cb: Callable,
        open_diag_dialog_cb: Callable,
        tg_status_var: ctk.StringVar,
        tg_status_color_var: ctk.StringVar,
        purge_tg_backlog_cb: Callable,
        request_shutdown_cb: Callable,
        **kwargs
    ):
        super().__init__(master, **kwargs)
        self.title("Paramètres")
        self.geometry("540x520")
        self.resizable(False, False)
        
        self._params = params
        self._on_save = on_save
        self._open_tg = open_tg_dialog_cb
        self._open_diag = open_diag_dialog_cb
        self._purge_tg = purge_tg_backlog_cb
        self._request_shutdown = request_shutdown_cb
        
        root = ctk.CTkFrame(self, corner_radius=12)
        root.pack(fill="both", expand=True, padx=12, pady=12)
        
        # === Lecture en boucle ===
        row1 = ctk.CTkFrame(root, fg_color="transparent")
        row1.pack(fill="x", pady=(4, 8))
        
        ctk.CTkLabel(
            row1,
            text="Lecture en boucle (auto_loop)",
            text_color=Theme.LEFT_HEADER_TEXT
        ).pack(side="left")
        
        self.auto_loop_var = ctk.BooleanVar(
            value=as_bool(self._params.get("auto_loop", "0"), False)
        )
        ctk.CTkSwitch(row1, text="", variable=self.auto_loop_var).pack(side="right")
        
        # === Séparateur ===
        ctk.CTkFrame(root, height=1, fg_color=Theme.DIVIDER).pack(fill="x", pady=8)
        
        # === Section Telegram ===
        row2 = ctk.CTkFrame(root, fg_color="transparent")
        row2.pack(fill="x", pady=(4, 8))
        
        ctk.CTkLabel(
            row2,
            text="Automatisation Telegram",
            text_color=Theme.LEFT_HEADER_TEXT,
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w")
        
        # Statut TG
        status_frame = ctk.CTkFrame(row2, fg_color="transparent")
        status_frame.pack(fill="x", pady=(4, 0))
        
        ctk.CTkLabel(status_frame, text="Statut:").pack(side="left", padx=(0, 6))
        
        self.lbl_tg_status = ctk.CTkLabel(
            status_frame,
            textvariable=tg_status_var,
            text_color=Theme.STATUS_OK
        )
        self.lbl_tg_status.pack(side="left")
        
        # Garder référence pour le trace
        self.tg_status_color_var = tg_status_color_var
        self._tg_trace_id = self.tg_status_color_var.trace_add("write", self._update_tg_color)
        self._update_tg_color(None, None, None)
        
        # Boutons TG
        btn_frame_tg = ctk.CTkFrame(row2, fg_color="transparent")
        btn_frame_tg.pack(fill="x", pady=(8, 0))
        
        ctk.CTkButton(
            btn_frame_tg,
            text="Configurer Telegram…",
            width=200,
            command=self._open_tg
        ).pack(side="left")
        
        ctk.CTkButton(
            btn_frame_tg,
            text="Purger backlog",
            width=140,
            fg_color="#374151",
            hover_color="#4b5563",
            command=self._purge_tg
        ).pack(side="left", padx=(8, 0))
        
        # === Chemin CoC ===
        ctk.CTkFrame(root, height=1, fg_color=Theme.DIVIDER).pack(fill="x", pady=12)
        
        row3 = ctk.CTkFrame(root, fg_color="transparent")
        row3.pack(fill="x", pady=(4, 0))
        
        ctk.CTkLabel(
            row3,
            text="Chemin de lancement CoC",
            text_color=Theme.LEFT_HEADER_TEXT,
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w")
        
        ctk.CTkLabel(
            row3,
            text="Raccourci (.lnk) ou .exe",
            text_color="#94a3b8"
        ).pack(anchor="w")
        
        self.coc_path_var = ctk.StringVar(value=self._params.get("coc_path", ""))
        ctk.CTkEntry(row3, textvariable=self.coc_path_var).pack(fill="x", pady=(4, 0))
        
        # === Diag / Shutdown ===
        ctk.CTkFrame(root, height=1, fg_color=Theme.DIVIDER).pack(fill="x", pady=12)
        
        row4 = ctk.CTkFrame(root, fg_color="transparent")
        row4.pack(fill="x", pady=(4, 0))
        
        ctk.CTkButton(
            row4,
            text="État du système…",
            width=180,
            fg_color="#1f2937",
            hover_color="#374151",
            command=self._open_diag
        ).pack(side="left")
        
        ctk.CTkButton(
            row4,
            text="📴 Éteindre le PC",
            width=160,
            fg_color=Theme.BTN_STOP_BG,
            hover_color=Theme.BTN_STOP_HOVER,
            command=self._request_shutdown
        ).pack(side="right", padx=(8, 0))
        
        # === Boutons bas ===
        btns = ctk.CTkFrame(root, fg_color="transparent")
        btns.pack(fill="x", side="bottom", pady=(10, 0))
        
        ctk.CTkButton(
            btns,
            text="Fermer",
            width=110,
            fg_color="#374151",
            hover_color="#4b5563",
            command=self.destroy
        ).pack(side="right")
        
        ctk.CTkButton(
            btns,
            text="Enregistrer",
            width=120,
            command=self._save
        ).pack(side="right", padx=(0, 8))
    
    def _save(self):
        self._params["auto_loop"] = "1" if self.auto_loop_var.get() else "0"
        self._params["coc_path"] = self.coc_path_var.get().strip()
        
        if callable(self._on_save):
            self._on_save(self._params)
        self.destroy()
    
    def _update_tg_color(self, var, idx, mode):
        """Met à jour la couleur du label de statut."""
        try:
            if self.lbl_tg_status.winfo_exists():
                color = self.tg_status_color_var.get()
                self.lbl_tg_status.configure(text_color=color)
            else:
                self._remove_tg_trace()
        except Exception:
            self._remove_tg_trace()
    
    def _remove_tg_trace(self):
        """Désenregistre le callback de trace."""
        if hasattr(self, '_tg_trace_id') and self._tg_trace_id:
            try:
                self.tg_status_color_var.trace_remove("write", self._tg_trace_id)
            except Exception:
                pass
            self._tg_trace_id = None
    
    def destroy(self):
        self._remove_tg_trace()
        super().destroy()


class DiagnosticsDialog(BaseToplevel):
    """
    Fenêtre de diagnostic système.
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
        **kwargs
    ):
        super().__init__(master, **kwargs)
        self.title("État du système")
        self.geometry("700x500")
        
        root = ctk.CTkFrame(self, corner_radius=12)
        root.pack(fill="both", expand=True, padx=12, pady=12)
        
        scroll = ctk.CTkScrollableFrame(root, fg_color=Theme.CENTER_BG)
        scroll.pack(fill="both", expand=True)
        
        def add_row(key: str, value: str, color: str = Theme.TEXT_COMPLIANT):
            f = ctk.CTkFrame(scroll, fg_color="transparent")
            f.pack(fill="x")
            ctk.CTkLabel(
                f,
                text=key,
                text_color=Theme.TEXT_MUTED_COMPLIANT,
                width=180,
                anchor="e"
            ).pack(side="left", padx=(0, 10))
            ctk.CTkLabel(
                f,
                text=value,
                text_color=color,
                anchor="w"
            ).pack(side="left", expand=True, fill="x")
        
        # Infos système
        add_row("Version App:", app_version)
        add_row("Version Python:", python_version)
        
        pil_txt = "Oui" if pil_available else "Non (CAPTURE D'ÉCRAN INDISPONIBLE)"
        pil_col = Theme.STATUS_OK if pil_available else Theme.STATUS_ERROR
        add_row("Librairie Pillow:", pil_txt, pil_col)
        
        mss_txt = "Oui" if mss_available else "Non (fallback capture)"
        mss_col = Theme.STATUS_OK if mss_available else Theme.STATUS_WARN
        add_row("Librairie MSS:", mss_txt, mss_col)
        
        add_row("Statut Telegram:", tg_status)
        
        # Espace disque
        try:
            import shutil
            disk = shutil.disk_usage(base_dir)
            free_gb = disk.free / (1024**3)
            add_row("Espace disque (app):", f"{free_gb:.2f} Go libres")
        except Exception as e:
            add_row("Espace disque (app):", f"Erreur: {e}", Theme.STATUS_ERROR)
        
        # Logs récents
        ctk.CTkLabel(
            scroll,
            text="Derniers logs (5)",
            font=ctk.CTkFont(weight="bold")
        ).pack(pady=(15, 5), anchor="w")
        
        log_text = ctk.CTkTextbox(
            scroll,
            height=200,
            fg_color=Theme.APP_BG,
            text_color=Theme.TEXT_MUTED_COMPLIANT,
            font=ctk.CTkFont(family="Courier New", size=12)
        )
        log_text.pack(fill="x", expand=True)
        
        try:
            if log_path.exists():
                with log_path.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
                    log_text.insert("1.0", "".join(lines[-5:]))
            else:
                log_text.insert("1.0", "Fichier log non encore créé.")
        except Exception as e:
            log_text.insert("1.0", f"Erreur lecture log: {e}")
        
        log_text.configure(state="disabled")
        
        # Bouton fermer
        ctk.CTkButton(
            root,
            text="Fermer",
            command=self.destroy
        ).pack(side="bottom", anchor="e", pady=(10, 0))
