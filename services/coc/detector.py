# -*- coding: utf-8 -*-
"""Windows process and window detection for Clash of Clans."""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

import psutil

from services.coc.models import CocLaunchProfile, CocPresence
from utils.logger import get_logger


class CocDetector:
    """Find CoC by process identity and/or a matching top-level window."""

    def __init__(self, profile: CocLaunchProfile):
        self.profile = profile
        self.log = get_logger()

    def snapshot(self) -> CocPresence:
        processes, process_error = self._matching_processes()
        process_pids = {pid for pid, _ in processes}
        pid_filter = process_pids if (self.profile.process_names or self.profile.process_path_hint) else None
        windows, window_error = self._matching_windows(pid_filter)
        process_names = tuple(sorted({name for _, name in processes}))
        pids = tuple(sorted({pid for pid, _ in processes}))
        window_titles = tuple(title for title, _ in windows)
        present = bool(processes or windows)
        errors = tuple(error for error in (process_error, window_error) if error)
        error = " · ".join(errors)
        if error and not present:
            reason = "CoC detection unavailable"
        else:
            reason = "CoC process detected" if processes else "CoC window detected" if windows else "CoC not found"
        return CocPresence(
            present=present,
            process_found=bool(processes),
            window_found=bool(windows),
            pids=pids,
            process_names=process_names,
            window_titles=window_titles,
            reason=reason,
            error=error,
        )

    def _matching_processes(self) -> tuple[list[tuple[int, str]], str]:
        expected_names = {name.casefold() for name in self.profile.process_names}
        path_hint = self.profile.process_path_hint.casefold()
        matches: list[tuple[int, str]] = []
        error = ""
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
            error = f"processus : {exc}"
            self.log.warning("Détection processus CoC impossible : %s", exc)
        return matches, error

    def _matching_windows(self, allowed_pids: set[int] | None = None) -> tuple[list[tuple[str, int]], str]:
        if os.name != "nt" or not self.profile.window_titles:
            return [], ""
        patterns = tuple(pattern.casefold() for pattern in self.profile.window_titles)
        matches: list[tuple[str, int]] = []
        errors: list[str] = []
        try:
            user32 = ctypes.windll.user32
            enum_proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
            enum_windows = user32.EnumWindows
            enum_windows.argtypes = [enum_proc_type, wintypes.LPARAM]
            enum_windows.restype = wintypes.BOOL
            get_window_text_length = user32.GetWindowTextLengthW
            get_window_text_length.argtypes = [wintypes.HWND]
            get_window_text_length.restype = ctypes.c_int
            get_window_text = user32.GetWindowTextW
            get_window_text.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
            get_window_text.restype = ctypes.c_int
            get_pid = user32.GetWindowThreadProcessId
            get_pid.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
            get_pid.restype = wintypes.DWORD
            is_window_visible = user32.IsWindowVisible
            is_window_visible.argtypes = [wintypes.HWND]
            is_window_visible.restype = wintypes.BOOL
        except Exception as exc:
            error = f"fenêtres : {exc}"
            self.log.warning("Détection fenêtre CoC impossible : %s", exc)
            return [], error

        @enum_proc_type
        def callback(hwnd, _lparam):
            try:
                if not is_window_visible(hwnd):
                    return True
                length = get_window_text_length(hwnd)
                if length <= 0:
                    return True
                buffer = ctypes.create_unicode_buffer(length + 1)
                get_window_text(hwnd, buffer, length + 1)
                title = buffer.value.strip()
                if not title or not any(pattern in title.casefold() for pattern in patterns):
                    return True
                pid = wintypes.DWORD()
                get_pid(hwnd, ctypes.byref(pid))
                if allowed_pids is not None and int(pid.value) not in allowed_pids:
                    return True
                matches.append((title, int(pid.value)))
            except Exception as exc:
                errors.append(str(exc))
            return True

        try:
            enum_windows(callback, 0)
        except Exception as exc:
            errors.append(str(exc))
        error = f"fenêtres : {'; '.join(errors)}" if errors else ""
        if error:
            self.log.warning("Détection fenêtre CoC impossible : %s", error)
        return matches, error
