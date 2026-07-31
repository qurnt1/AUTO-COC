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
- A single editable macro library. System actions are role assignments, not a separate locked macro type.
- Global shortcuts: `F1`, `Ctrl+Shift+1` and `Ctrl+Shift+0`.
- Asynchronous Telegram control in a dedicated thread.
- Screenshots, CoC launch and Windows shutdown.
- Configurable CoC launch profile using path, process names and window titles.
- Presence detection and safeguard that stops playback after repeated missing checks.
- PyQt6-only interface.

## Design principles

1. Execution state and CoC presence must be understandable in under one second.
2. Stop must remain visible, immediate and unambiguous.
3. Remote and local actions must use the same runtime state.
4. Existing macro data must remain readable.
5. Every macro is user-editable. System actions point to macros by role and follow rename/delete operations safely.

## Accessibility

Keyboard navigation, visible focus, strong contrast, Windows scaling support and explicit error messages.
