# -*- coding: utf-8 -*-
"""
Macro COC v3.0 — GUI / Components

Widgets réutilisables pour l'interface (MacroRow, MacroList).
"""

from typing import Callable, Dict, List, Optional, Tuple

import customtkinter as ctk

from gui.theme import Theme
from utils.config import fmt_duration_for_list


class MacroRow:
    """
    Widget représentant une ligne de macro dans la liste.
    Affiche le nom à gauche et la durée à droite.
    """
    
    def __init__(
        self,
        parent: ctk.CTkFrame,
        name: str,
        duration_txt: str,
        on_click: Callable[[str], None],
        on_rclick: Callable,
    ):
        self.name = name
        self._selected = False
        
        # Frame principale
        self.frame = ctk.CTkFrame(parent, corner_radius=8, fg_color=Theme.ROW_BG)
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=0)
        
        # Label nom
        self.lbl_name = ctk.CTkLabel(
            self.frame,
            text=name,
            anchor="w",
            text_color=Theme.TEXT_COMPLIANT
        )
        self.lbl_name.grid(row=0, column=0, sticky="ew", padx=(10, 6), pady=8)
        
        # Label durée
        self.lbl_dur = ctk.CTkLabel(
            self.frame,
            text=duration_txt,
            anchor="e",
            text_color=Theme.ROW_DUR_COLOR
        )
        self.lbl_dur.grid(row=0, column=1, sticky="e", padx=(6, 10))
        
        # Événements
        def bind_all(widget):
            widget.bind("<Button-1>", lambda e: on_click(self.name))
            widget.bind("<Enter>", lambda e: self._hover(True))
            widget.bind("<Leave>", lambda e: self._hover(False))
            widget.bind("<Button-3>", lambda e: on_rclick(e, self.name))
        
        bind_all(self.frame)
        bind_all(self.lbl_name)
        bind_all(self.lbl_dur)
    
    def pack(self, **kwargs):
        """Pack le widget."""
        self.frame.pack(**kwargs)
    
    def destroy(self):
        """Détruit le widget."""
        self.frame.destroy()
    
    def set_selected(self, selected: bool):
        """Change l'état de sélection."""
        self._selected = selected
        self.frame.configure(
            fg_color=Theme.ROW_SELECTED if selected else Theme.ROW_BG
        )
    
    def _hover(self, enter: bool):
        """Gère le survol."""
        if self._selected:
            return
        self.frame.configure(
            fg_color=Theme.ROW_HOVER if enter else Theme.ROW_BG
        )
    
    def set_duration(self, dur_txt: str):
        """Met à jour l'affichage de la durée."""
        self.lbl_dur.configure(text=dur_txt)


class MacroList(ctk.CTkScrollableFrame):
    """
    Liste scrollable des macros avec recherche.
    """
    
    def __init__(
        self,
        master: ctk.CTkFrame,
        on_select: Callable[[str], None],
        on_rclick: Callable,
    ):
        super().__init__(master, corner_radius=12, fg_color=Theme.LEFT_CONTAINER_BG)
        
        self._on_select = on_select
        self._on_rclick = on_rclick
        self._rows: Dict[str, MacroRow] = {}
        self._selected: Optional[str] = None
        self._meta: Dict[str, Tuple[int, float]] = {}
    
    def set_meta(self, meta: Dict[str, Tuple[int, float]]):
        """Définit les métadonnées (nb_events, duration) pour chaque macro."""
        self._meta = dict(meta)
    
    def refresh(
        self,
        names: List[str],
        selected: Optional[str],
        filter_term: Optional[str] = None
    ):
        """
        Rafraîchit la liste, en appliquant un filtre optionnel.
        
        Args:
            names: Liste des noms de macros
            selected: Nom de la macro à sélectionner
            filter_term: Terme de recherche (optionnel)
        """
        # Purge
        for row in self._rows.values():
            row.destroy()
        self._rows.clear()
        
        filter_term = filter_term.lower() if filter_term else None
        
        # Rebuild
        for name in names:
            # Filtre de recherche
            if filter_term and filter_term not in name.lower():
                continue
            
            _, d = self._meta.get(name, (0, 0.0))
            row = MacroRow(
                self,
                name,
                fmt_duration_for_list(d),
                on_click=self.select,
                on_rclick=self._on_rclick
            )
            row.pack(fill="x", padx=6, pady=4)
            self._rows[name] = row
        
        if selected and selected in self._rows:
            self.select(selected, fire=False)
    
    def update_one(self, name: str):
        """Met à jour la durée d'une seule ligne."""
        if name in self._rows:
            _, d = self._meta.get(name, (0, 0.0))
            self._rows[name].set_duration(fmt_duration_for_list(d))
    
    def select(self, name: str, fire: bool = True):
        """
        Sélectionne une macro dans la liste.
        
        Args:
            name: Nom de la macro
            fire: Si True, appelle le callback on_select
        """
        # Désélectionner l'ancien
        if self._selected and self._selected in self._rows:
            self._rows[self._selected].set_selected(False)
        
        self._selected = name
        
        # Sélectionner le nouveau
        if name in self._rows:
            self._rows[name].set_selected(True)
            if fire and callable(self._on_select):
                try:
                    self._on_select(name)
                except Exception as e:
                    from utils.logger import get_logger
                    get_logger().error(f"Erreur on_select({name}): {e}")
    
    def get_selected(self) -> Optional[str]:
        """Retourne le nom de la macro sélectionnée."""
        return self._selected
