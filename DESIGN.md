# AUTO-COC Design System

## Direction

AUTO-COC is a calm, readable operator console. It shows what will happen, what is detected and how to stop automation.

The visual language uses warm graphite surfaces, a soft lime action accent, amber for attention and coral for stop or recording. User macros stay in the library. Telegram-only routines live in Telegram settings under an explicit `TELEGRAM ACTION` tag.

The first view keeps one CoC status, a launch button, a compact macro selector, recording and a single Run/Stop control visible. Secondary information belongs in `Macros`, `Telegram` and `Diagnostics`.

## Tokens

- Background: `#121417`
- Surface: `#1A1E22`
- Raised surface: `#22282E`
- Border: `#343C43`
- Text: `#F5F1E8`
- Muted text: `#B5B8B0`
- Accent: `#C7F36B`
- Warning: `#F5B967`
- Danger: `#FF7A68`
- Info: `#B19CFF`
- Radius: 12px
- Base spacing: 8px

## Motion and accessibility

Transitions stay short and purposeful. No decorative animation is used in dense macro data. Every action has a text label or accessible name, focus rings use the accent color, and status color is always paired with text.
