# -*- coding: utf-8 -*-
"""
AUTO-COC recorder and player services

Classes Recorder et Player pour l'enregistrement/lecture de macros avec Pynput.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, List, Optional, Set

from pynput import mouse, keyboard as pyn_keyboard
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyController, Key

from models.macro import Step, StepType
from utils.logger import get_logger
from utils.system import play_beep


# =========================
#     Helpers Pynput
# =========================

def button_to_str(btn: Button) -> str:
    """Convertit un Button pynput en string."""
    if btn == Button.left:
        return "left"
    if btn == Button.right:
        return "right"
    if btn == Button.middle:
        return "middle"
    return "left"


def str_to_button(s: str) -> Button:
    """Convertit un string en Button pynput."""
    return {
        "left": Button.left,
        "right": Button.right,
        "middle": Button.middle
    }.get(s, Button.left)


def key_to_str(k) -> str:
    """Convertit une Key pynput en string."""
    try:
        if hasattr(k, 'char') and k.char is not None:
            return k.char
        else:
            return str(k).split('.')[-1]
    except Exception:
        return str(k)


def str_to_key(s: str):
    """Convertit un string en Key pynput."""
    if not s:
        return s
    if len(s) == 1:
        return s
    
    aliases = {"control": "ctrl", "win": "cmd", "altgr": "alt_gr"}
    s = aliases.get(s, s)
    
    try:
        return getattr(Key, s)
    except AttributeError:
        pass
    
    specials = {
        "enter": Key.enter, "shift": Key.shift, "ctrl": Key.ctrl, "alt": Key.alt,
        "esc": Key.esc, "tab": Key.tab, "space": Key.space, "backspace": Key.backspace,
        "delete": Key.delete, "up": Key.up, "down": Key.down, "left": Key.left,
        "right": Key.right, "cmd": getattr(Key, "cmd", None),
        "alt_gr": getattr(Key, "alt_gr", Key.alt),
        "caps_lock": getattr(Key, "caps_lock", None),
        "num_lock": getattr(Key, "num_lock", None),
        "scroll_lock": getattr(Key, "scroll_lock", None),
        "f1": Key.f1, "f2": Key.f2, "f3": Key.f3, "f4": Key.f4, "f5": Key.f5,
        "f6": Key.f6, "f7": Key.f7, "f8": Key.f8, "f9": Key.f9, "f10": Key.f10,
        "f11": Key.f11, "f12": Key.f12,
    }
    k = specials.get(s)
    if k is not None:
        return k
    
    # Gérer les KeyCode (ex: <65>)
    if s.startswith('<') and s.endswith('>'):
        try:
            vk = int(s[1:-1])
            return pyn_keyboard.KeyCode(vk=vk)
        except ValueError:
            pass
    
    get_logger().warning(f"Touche spéciale non reconnue: '{s}'")
    return s


# =========================
#     Recorder
# =========================

class Recorder:
    """
    Enregistre les événements clavier/souris.
    
    Attributes:
        recording: État d'enregistrement
        steps: Liste des événements enregistrés
        grace_seconds: Délai avant de commencer l'enregistrement réel
    """
    
    def __init__(self):
        self.recording = False
        self.steps: List[Step] = []
        self.grace_seconds = 3.0
        
        self._mouse_listener = None
        self._keyboard_listener = None
        self._t0: Optional[float] = None
        self._last_t: Optional[float] = None
        self._self_playing_flag = False
        self._grace_until: Optional[float] = None
        self._grace_started = False
        self._log = get_logger()
    
    def _now(self) -> float:
        return time.perf_counter()
    
    def start(self):
        """Démarre l'enregistrement."""
        if self.recording:
            return
        
        self.steps = []
        self.recording = True
        self._t0 = self._now()
        self._last_t = self._t0
        self._grace_until = self._t0 + self.grace_seconds
        self._grace_started = True
        
        try:
            self._mouse_listener = mouse.Listener(
                on_move=self._on_move,
                on_click=self._on_click,
                on_scroll=self._on_scroll
            )
            self._mouse_listener.start()
            
            self._keyboard_listener = pyn_keyboard.Listener(
                on_press=self._on_key_down,
                on_release=self._on_key_up
            )
            self._keyboard_listener.start()
            
        except Exception as e:
            self._log.error(f"Échec démarrage listeners pynput: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Arrête l'enregistrement."""
        self.recording = False
        self._grace_started = False
        
        try:
            if self._mouse_listener:
                self._mouse_listener.stop()
        finally:
            self._mouse_listener = None
        
        try:
            if self._keyboard_listener:
                self._keyboard_listener.stop()
        finally:
            self._keyboard_listener = None
    
    def mark_self_play(self, flag: bool):
        """Marque si le player joue (pour ignorer ses propres événements)."""
        self._self_playing_flag = flag
    
    def get_steps_as_dicts(self) -> List[dict]:
        """Retourne les steps en format dictionnaire (compat v2.1)."""
        return [s.to_dict() for s in self.steps]
    
    def _push(self, step_type: StepType, data: dict):
        """Ajoute un événement."""
        if not self.recording or self._self_playing_flag:
            return
        
        t = self._now()
        
        # Période de grâce
        if self._grace_until is not None and t < self._grace_until:
            return
        
        if self._grace_started and t >= self._grace_until:
            self._last_t = t  # Le premier step a dt=0
            self._grace_started = False
            play_beep()  # Signal de début
        
        dt = t - self._last_t
        self._last_t = t
        
        self.steps.append(Step(
            time_delta=dt,
            step_type=step_type,
            data=data
        ))
    
    # === Callbacks ===
    
    def _on_move(self, x: int, y: int):
        self._push(StepType.MOUSE_MOVE, {"x": int(x), "y": int(y)})
    
    def _on_click(self, x: int, y: int, button: Button, pressed: bool):
        self._push(StepType.MOUSE_CLICK, {
            "x": int(x),
            "y": int(y),
            "button": button_to_str(button),
            "action": "down" if pressed else "up"
        })
    
    def _on_scroll(self, x: int, y: int, dx: int, dy: int):
        self._push(StepType.SCROLL, {
            "x": int(x),
            "y": int(y),
            "dx": int(dx),
            "dy": int(dy)
        })
    
    def _on_key_down(self, key):
        self._push(StepType.KEY_DOWN, {"key": key_to_str(key)})
    
    def _on_key_up(self, key):
        self._push(StepType.KEY_UP, {"key": key_to_str(key)})


# =========================
#     Player
# =========================

class Player:
    """
    Lit les macros avec timing haute-précision.
    
    Utilise une boucle de timing absolue pour éviter la dérive.
    """
    
    def __init__(self, recorder: Recorder):
        self._mouse = MouseController()
        self._keys = KeyController()
        self._recorder = recorder
        self._log = get_logger()
        
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._lifecycle_lock = threading.RLock()
        self._stopped_callback_sent = True
        self._loop = False
        
        # Callbacks
        self.on_cycle: Optional[Callable] = None
        self.on_stopped: Optional[Callable] = None
        
        # État des inputs (pour release en cas d'arrêt)
        self._pressed_keys: Set[Any] = set()
        self._pressed_buttons: Set[Button] = set()
    
    def is_playing(self) -> bool:
        """Vérifie si une lecture est en cours."""
        with self._lifecycle_lock:
            thread = self._thread
        return thread is not None and thread.is_alive()
    
    def stop(self):
        """Arrête la lecture et relâche toutes les touches/boutons."""
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None:
                return
            self._stop_flag.set()

        if thread is threading.current_thread():
            return

        try:
            thread.join(timeout=2.0)
        except Exception:
            pass

        if thread.is_alive():
            self._log.warning("Arrêt Player différé : le worker est encore actif après le timeout.")
            return

        # Le worker nettoie les inputs et notifie on_stopped dans son finally.
        # Le flag reste positionné jusqu'au prochain play(), après confirmation
        # que ce worker n'est plus vivant.
    
    def play(self, steps: List[dict], loop: bool = False) -> bool:
        """
        Démarre la lecture d'une macro.
        
        Args:
            steps: Liste des événements (format dict)
            loop: Si True, boucle indéfiniment
        """
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return False

            self._stop_flag.clear()
            self._loop = loop
            self._stopped_callback_sent = False
            thread = threading.Thread(
                target=self._run,
                args=(steps,),
                daemon=True,
                name="MacroPlayer"
            )
            self._thread = thread
            try:
                thread.start()
            except Exception:
                self._thread = None
                self._stopped_callback_sent = True
                self._stop_flag.set()
                raise
            return True

    def play_steps(self, steps: List[Step], loop: bool = False) -> bool:
        """
        Démarre la lecture d'une macro (format Step).
        
        Args:
            steps: Liste des Step
            loop: Si True, boucle indéfiniment
        """
        return self.play([s.to_dict() for s in steps], loop)
    
    def _wait(self, seconds: float):
        """Attente interruptible."""
        self._stop_flag.wait(timeout=max(0.0, seconds))
    
    def _run_one_cycle(self, steps: List[dict]) -> bool:
        """
        Exécute un cycle de macro avec timing absolu.
        
        Returns:
            True si le cycle s'est terminé normalement
        """
        if not steps:
            return False
        
        # Calculer les timestamps absolus
        abs_times = []
        acc = 0.0
        for ev in steps:
            dt = float(ev.get("t", 0.0))
            acc += dt
            abs_times.append(acc)
        
        start = time.perf_counter()
        
        # Boucle d'exécution
        for i, ev in enumerate(steps):
            if self._stop_flag.is_set():
                return False
            
            # Calculer le temps d'attente
            target_time_abs = abs_times[i]
            now = time.perf_counter()
            time_elapsed = now - start
            remain = target_time_abs - time_elapsed
            
            if remain > 0.001:  # Seuil minimal
                self._wait(remain)
                if self._stop_flag.is_set():
                    return False
            
            # Appliquer l'événement
            self._apply(ev)
        
        return True
    
    def _run(self, steps: List[dict]):
        """Thread principal de lecture."""
        try:
            self._recorder.mark_self_play(True)
            while not self._stop_flag.is_set():
                cycle_completed = self._run_one_cycle(steps)
                
                if self._stop_flag.is_set():
                    break
                
                if cycle_completed and self.on_cycle:
                    try:
                        self.on_cycle()
                    except Exception as e:
                        self._log.error(f"Erreur callback on_cycle: {e}")
                
                if not self._loop:
                    break
                    
        finally:
            try:
                self._recorder.mark_self_play(False)
            except Exception as e:
                self._log.error(f"Erreur arrêt état self-play: {e}")
            self._release_inputs()
            self._notify_stopped_once()

    def _release_inputs(self) -> None:
        """Relâche les entrées uniquement depuis un worker terminé."""
        try:
            for k in list(self._pressed_keys):
                try:
                    self._keys.release(k)
                except Exception:
                    pass
            self._pressed_keys.clear()

            for b in list(self._pressed_buttons):
                try:
                    self._mouse.release(b)
                except Exception:
                    pass
            self._pressed_buttons.clear()
        except Exception as e:
            self._log.error(f"Erreur release inputs: {e}")

    def _notify_stopped_once(self) -> None:
        with self._lifecycle_lock:
            if self._stopped_callback_sent:
                return
            self._stopped_callback_sent = True
            callback = self.on_stopped
        if callback:
            try:
                callback()
            except Exception as e:
                self._log.error(f"Erreur callback on_stopped: {e}")
    
    def _apply(self, ev: dict):
        """Applique un événement."""
        typ = ev.get("type")
        data = ev.get("data")
        
        if not typ or not data:
            return
        
        try:
            if typ == "mouse_move":
                self._mouse.position = (data["x"], data["y"])
                
            elif typ == "mouse_click":
                btn = str_to_button(data["button"])
                if data["action"] == "down":
                    self._mouse.press(btn)
                    self._pressed_buttons.add(btn)
                else:
                    self._mouse.release(btn)
                    self._pressed_buttons.discard(btn)
                    
            elif typ == "scroll":
                self._mouse.scroll(data["dx"], data["dy"])
                
            elif typ == "key_down":
                k = str_to_key(data["key"])
                self._keys.press(k)
                self._pressed_keys.add(k)
                
            elif typ == "key_up":
                k = str_to_key(data["key"])
                self._keys.release(k)
                self._pressed_keys.discard(k)
                
        except Exception as e:
            self._log.warning(f"Erreur application step {typ}: {e}")


def trim_tail(steps: List[dict], tail_seconds: float) -> List[dict]:
    """
    Coupe les X dernières secondes d'un enregistrement.
    
    Args:
        steps: Liste des événements
        tail_seconds: Durée à couper à la fin
        
    Returns:
        Liste des événements sans la fin
    """
    if not steps or tail_seconds <= 0:
        return list(steps)
    
    total = sum(max(0.0, ev.get("t", 0.0)) for ev in steps)
    cutoff = max(0.0, total - tail_seconds)
    
    if cutoff <= 0.0:
        return []
    
    trimmed = []
    acc = 0.0
    
    for ev in steps:
        dt = max(0.0, ev.get("t", 0.0))
        if acc + dt < cutoff:
            trimmed.append(ev)
            acc += dt
        else:
            remaining = cutoff - acc
            if remaining > 0.001:
                ev_copy = ev.copy()
                ev_copy["t"] = remaining
                trimmed.append(ev_copy)
            break
    
    return trimmed
