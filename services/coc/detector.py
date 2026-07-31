# -*- coding: utf-8 -*-
"""Windows process and window detection for Clash of Clans."""

from __future__ import annotations

import ctypes
import os

import psutil

from services.coc.models import CocLaunchProfile, CocPresence
from utils.logger import get_logger


class CocDetector:
    """Find CoC by process identity and/or a matching top-level window."""

    def __init__(self, profile: CocLaunchProfile):
        self.profile = profile
        self.log = get_logger()

    def snapshot(self) -> CocPresence:
        processes = self._matching_processes()
        windows = self._matching_windows()
        process_names = tuple(sorted({name for _, name in processes}))
        pids = tuple(sorted({pid for pid, _ in processes}))
        window_titles = tuple(title for title, _ in windows)
        present = bool(processes or windows)
        reason = "Processus CoC détecté" if processes else "Fenêtre CoC détectée" if windows else "CoC introuvable"
        return CocPresence(
            present=present,
            process_found=bool(processes),
            window_found=bool(windows),
            pids=pids,
            process_names=process_names,
            window_titles=window_titles,
            reason=reason,
        )

    def _matching_processes(self) -> list[tuple[int, str]]:
        expected_names = {name.casefold() for name in self.profile.process_names}
        path_hint = self.profile.process_path_hint.casefold()
        matches: list[tuple[int, str]] = []
        try:
            for process in psutil.process_iter(["pid", "name", "exe"]):
                try:
                    name = (process.info.get("name") or "").strip()
                    exe = (process.info.get("exe") or "").strip()
                    name_match = bool(expected_names and name.casefold() in expected_names)
                    path_match = bool(path_hint and path_hint in exe.casefold())
                    if name_match or path_match:
                        matches.append((int(process.info["pid"]), name or exe))
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as exc:
            self.log.warning("Détection processus CoC impossible : %s", exc)
        return matches

    def _matching_windows(self) -> list[tuple[str, int]]:
        if os.name != "nt" or not self.profile.window_titles:
            return []
        patterns = tuple(pattern.casefold() for pattern in self.profile.window_titles)
        matches: list[tuple[str, int]] = []
        user32 = ctypes.windll.user32
        enum_proc_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        enum_windows = user32.EnumWindows
        enum_windows.argtypes = [enum_proc_type, ctypes.c_void_p]
        enum_windows.restype = ctypes.c_bool
        get_window_text_length = user32.GetWindowTextLengthW
        get_window_text = user32.GetWindowTextW
        get_pid = user32.GetWindowThreadProcessId

        @enum_proc_type
        def callback(hwnd, _lparam):
            length = get_window_text_length(hwnd)
            if length <= 0:
                return True
            buffer = ctypes.create_unicode_buffer(length + 1)
            get_window_text(hwnd, buffer, length + 1)
            title = buffer.value.strip()
            if title and any(pattern in title.casefold() for pattern in patterns):
                pid = ctypes.c_ulong()
                get_pid(hwnd, ctypes.byref(pid))
                matches.append((title, int(pid.value)))
            return True

        try:
            enum_windows(callback, 0)
        except Exception as exc:
            self.log.warning("Détection fenêtre CoC impossible : %s", exc)
        return matches
