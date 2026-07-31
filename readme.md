# AUTO-COC

AUTO-COC is a Windows desktop console for recording and playing keyboard/mouse macros, with optional Telegram control for Clash of Clans workflows.

## Features

- PyQt6 operator console with a focused dark theme.
- Keyboard and mouse recording with absolute-timing playback.
- One-shot or looped playback, with immediate local and global stop controls.
- A focused user macro library, separate from Telegram-only routines.
- Telegram control with one compact control panel and one replaceable latest screenshot.
- Automatic CoC shortcut discovery, with an optional explicit `.exe` or `.lnk` override.
- Optional safeguard that stops playback when CoC disappears for the configured number of checks.
- Atomic persistence for macro JSON files and settings CSV.

## Install

Requirements: Python 3.10+ on Windows 10/11.

```bash
pip install -r requirements.txt
python main.pyw
```

## Use

Create or select a macro in `Macros`. `Home` provides a compact selector, `Record`, a single contextual `Run macro · F1` / `Stop macro · F1` button, and `Launch Clash of Clans`.

In `Settings`, tune the CoC detection profile. Leave the launcher path empty to discover a `Clash of Clans` Windows shortcut automatically, or choose an explicit launcher. Enable `Stop if CoC disappears` to arm the safeguard during playback. A launch is confirmed only after CoC is detected.

`Telegram` owns bot credentials, screenshots and the private `Reload CoC` routine. This routine is tagged `TELEGRAM ACTION` and never appears in the user macro selectors. Renaming or removing it keeps the role assignment consistent.

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
gui/                      # PyQt6 shell, controller, dialogs, theme and focused pages
models/                   # Macro data models
services/                 # Recording, playback, Telegram and CoC services
  coc/                    # Auto-discovery, launching, detection and safeguard
utils/                    # Persistence, logging and system helpers
config/macros/            # Local macro JSON files
```

Macro files and settings stay local in `config/`. The Telegram token is never committed and remains masked in the settings dialog.

## License

MIT.
