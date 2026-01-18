# -*- coding: utf-8 -*-
"""
Macro COC v3.0 — Services / Vision Service

Placeholder pour la Computer Vision (v3.1+).
Intégration future avec OpenCV/pyautogui.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from utils.logger import get_logger


@dataclass
class VisualAnchor:
    """
    Point de référence visuel pour calibration des macros.
    
    Attributes:
        name: Nom de l'ancre (ex: "bouton_ok")
        template_path: Chemin vers l'image template
        relative_offset: Offset (x, y) par rapport au centre trouvé
        confidence: Seuil de confiance pour la détection (0.0 - 1.0)
    """
    name: str
    template_path: str
    relative_offset: Tuple[int, int] = (0, 0)
    confidence: float = 0.8


class VisionService:
    """
    Service de Computer Vision.
    
    Placeholder pour la v3.1.
    Sera utilisé pour rendre les macros résilientes aux changements de résolution.
    """
    
    def __init__(self):
        self._log = get_logger()
        self._opencv_available = False
        self._pyautogui_available = False
        
        # Tenter d'importer les dépendances optionnelles
        try:
            import cv2
            self._opencv_available = True
        except ImportError:
            pass
        
        try:
            import pyautogui
            self._pyautogui_available = True
        except ImportError:
            pass
    
    @property
    def is_available(self) -> bool:
        """Vérifie si au moins une méthode de vision est disponible."""
        return self._opencv_available or self._pyautogui_available
    
    def find_element(
        self,
        template_path: str,
        confidence: float = 0.8
    ) -> Optional[Tuple[int, int]]:
        """
        Cherche un élément visuel et retourne ses coordonnées.
        
        Args:
            template_path: Chemin vers l'image template
            confidence: Seuil de confiance
            
        Returns:
            Tuple (x, y) du centre de l'élément ou None
        """
        # TODO: Implémenter avec OpenCV ou pyautogui
        self._log.info(f"VisionService.find_element() appelé (non implémenté)")
        return None
    
    def find_all_elements(
        self,
        template_path: str,
        confidence: float = 0.8
    ) -> List[Tuple[int, int]]:
        """
        Cherche toutes les occurrences d'un élément visuel.
        
        Args:
            template_path: Chemin vers l'image template
            confidence: Seuil de confiance
            
        Returns:
            Liste de tuples (x, y)
        """
        # TODO: Implémenter
        self._log.info(f"VisionService.find_all_elements() appelé (non implémenté)")
        return []
    
    def wait_for_element(
        self,
        template_path: str,
        timeout_seconds: float = 10.0,
        confidence: float = 0.8
    ) -> Optional[Tuple[int, int]]:
        """
        Attend qu'un élément apparaisse à l'écran.
        
        Args:
            template_path: Chemin vers l'image template
            timeout_seconds: Timeout en secondes
            confidence: Seuil de confiance
            
        Returns:
            Tuple (x, y) du centre de l'élément ou None si timeout
        """
        # TODO: Implémenter
        self._log.info(f"VisionService.wait_for_element() appelé (non implémenté)")
        return None
    
    def calibrate_coordinates(
        self,
        original_coords: Tuple[int, int],
        anchor: VisualAnchor,
        original_anchor_pos: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Recalcule les coordonnées en fonction d'une ancre visuelle.
        
        Args:
            original_coords: Coordonnées originales (x, y)
            anchor: Ancre visuelle de référence
            original_anchor_pos: Position originale de l'ancre
            
        Returns:
            Nouvelles coordonnées calibrées
        """
        # Trouver la position actuelle de l'ancre
        current_anchor_pos = self.find_element(anchor.template_path, anchor.confidence)
        
        if not current_anchor_pos:
            # Ancre non trouvée, retourner les coordonnées originales
            return original_coords
        
        # Calculer le décalage
        dx = current_anchor_pos[0] - original_anchor_pos[0]
        dy = current_anchor_pos[1] - original_anchor_pos[1]
        
        # Appliquer le décalage
        new_x = original_coords[0] + dx
        new_y = original_coords[1] + dy
        
        return (new_x, new_y)
