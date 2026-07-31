# AUTO-COC

AUTO-COC is a Windows desktop console for recording and playing keyboard/mouse macros, with optional Telegram control for Clash of Clans workflows.

## Features

- PyQt6 operator console with a focused dark theme.
- Keyboard and mouse recording with absolute-timing playback.
- One-shot or looped playback, with immediate local and global stop controls.
- One editable macro library. System actions are regular macros assigned to roles, so they can be renamed, edited or deleted.
- Telegram control and screenshots.
- Configurable CoC launcher (`.exe` or `.lnk`) with process, executable-path and window-title detection.
- Optional safeguard that stops playback when CoC disappears for the configured number of checks.
- Atomic persistence for macro JSON files and settings CSV.

## Install

Requirements: Python 3.10+ on Windows 10/11.

```bash
pip install -r requirements.txt
python main.pyw
```

## Use

Create or select a macro in `Macros`, then use `Record`, `Run macro`, `Stop`, or `Open CoC` from `Home`.

In `Settings`, configure the CoC launcher and detection profile. Enable `Stop if CoC disappears` to arm the safeguard during playback. The launcher is considered ready only after CoC is detected, not merely after a process is started.

Telegram shortcuts use the assigned system roles. Renaming a role macro keeps its Telegram action linked. Deleting it clears the assignment, so the action becomes unavailable until a macro is assigned again.

## Self-test

```bash
python main.pyw --selftest
```

## Shortcuts

| Shortcut | Action |
|---|---|
| `F1` | Run or stop |
| `Ctrl+Shift+1` | Run selected macro |
| `Ctrl+Shift+0` | Stop immediately |
| `Ctrl+N` | Create a macro |
| `Ctrl+F` | Search macros |
| `Ctrl+Enter` | Run selected macro |
| `Esc` | Stop or close the current action |

## Layout

```text
main.pyw                  # QApplication bootstrap and self-tests
gui/                      # Main window, controller, dialogs, theme and pages
models/                   # Macro data models
services/                 # Recording, playback, Telegram and CoC services
  coc/                    # Launching, detection and safeguard
utils/                    # Persistence, logging and system helpers
config/macros/            # Local macro JSON files
```

Macro files and settings stay local in `config/`. The Telegram token is never committed and remains masked in the settings dialog.

## License

MIT.
