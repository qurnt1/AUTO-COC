# Product

## Platform

Windows desktop application.

## Users

Windows users who run keyboard and mouse automation while another application is in the foreground.

## Purpose

AUTO-COC records, stores and replays input sequences with local controls and optional Telegram commands. Success means reliable execution, immediate state visibility and an unambiguous stop action.

## Operating context

The application runs on Windows while the user interacts with Clash of Clans or another foreground application. Macros are stored locally as JSON and settings as CSV.

## Capabilities and constraints

- Keyboard and mouse recording through `pynput`.
- Absolute-timing playback, optional looping and emergency stop.
- A user macro library plus editable Telegram-only routines, separated by intent.
- Global shortcuts: `F1`, `Ctrl+Shift+1` and `Ctrl+Shift+0`.
- Asynchronous Telegram control in a dedicated thread.
- Screenshots, CoC launch and Windows shutdown.
- CoC shortcut auto-discovery with an optional explicit path, plus process and window detection.
- Presence detection and safeguard that stops playback after repeated missing checks.
- PyQt6-only interface.

## Design principles

1. Execution state and CoC presence must be understandable in under one second.
2. Run and stop share one contextual control whose label always includes `F1`.
3. Remote and local actions must use the same runtime state.
4. Existing macro data must remain readable.
5. Telegram routines remain editable from Telegram settings but never look like user-launchable macros.
6. Telegram keeps one active panel and one latest screenshot whenever deletion is permitted by Telegram.

## Accessibility

Keyboard navigation, visible focus, strong contrast, Windows scaling support and explicit error messages.
