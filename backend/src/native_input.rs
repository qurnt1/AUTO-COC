use std::{
    collections::HashSet,
    io,
    time::{Duration, Instant},
};

use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::types::ShortcutSettings;

const PREPARATION: Duration = Duration::from_secs(3);
const TAIL_TRIM_SECONDS: f64 = 3.0;
const MAX_CAPTURE_EVENTS: usize = 250_000;
const MAX_PLAYBACK_SECONDS: f64 = 86_400.0;

pub(crate) fn try_accumulate_playback_duration(total: &mut f64, delta: f64) -> bool {
    if !delta.is_finite() || delta < 0.0 {
        return false;
    }
    let accumulated = *total + delta;
    if accumulated > MAX_PLAYBACK_SECONDS {
        return false;
    }
    *total = accumulated;
    true
}

const MOD_ALT: u32 = 0x0001;
const MOD_CONTROL: u32 = 0x0002;
const MOD_SHIFT: u32 = 0x0004;
const MOD_WIN: u32 = 0x0008;
const MOD_NOREPEAT: u32 = 0x4000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeEvent {
    Toggle,
    Play,
    Stop,
    Cycle,
    PlaybackEnded,
}

pub struct RecordingCapture {
    pub steps: Vec<Value>,
    pub overflowed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ParsedHotkey {
    modifiers: u32,
    virtual_key: u16,
}

pub struct NativeInput {
    #[cfg(windows)]
    inner: win::NativeInputInner,
}

impl NativeInput {
    pub fn start(
        shortcuts: ShortcutSettings,
        events: mpsc::UnboundedSender<NativeEvent>,
    ) -> io::Result<Self> {
        let parsed = parse_shortcuts(&shortcuts)?;

        #[cfg(windows)]
        {
            Ok(Self {
                inner: win::NativeInputInner::start(shortcuts, parsed, events)?,
            })
        }

        #[cfg(not(windows))]
        {
            let _ = (parsed, events);
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "native input is only available on Windows",
            ))
        }
    }

    pub async fn start_recording(&self) -> io::Result<()> {
        #[cfg(windows)]
        {
            self.inner.start_recording()
        }
        #[cfg(not(windows))]
        {
            Err(unsupported())
        }
    }

    pub async fn stop_recording(&self) -> io::Result<RecordingCapture> {
        #[cfg(windows)]
        {
            self.inner.stop_recording()
        }
        #[cfg(not(windows))]
        {
            Err(unsupported())
        }
    }

    pub async fn start_playback(&self, steps: Vec<Value>, looped: bool) -> io::Result<()> {
        #[cfg(windows)]
        {
            self.inner.start_playback(steps, looped).await
        }
        #[cfg(not(windows))]
        {
            let _ = (steps, looped);
            Err(unsupported())
        }
    }

    pub async fn stop_playback(&self) -> io::Result<()> {
        #[cfg(windows)]
        {
            self.inner.stop_playback().await
        }
        #[cfg(not(windows))]
        {
            Err(unsupported())
        }
    }

    pub async fn set_shortcuts(&self, shortcuts: ShortcutSettings) -> io::Result<()> {
        let parsed = parse_shortcuts(&shortcuts)?;
        #[cfg(windows)]
        {
            self.inner.set_shortcuts(shortcuts, parsed).await
        }
        #[cfg(not(windows))]
        {
            let _ = (shortcuts, parsed);
            Err(unsupported())
        }
    }

    pub async fn shutdown(&mut self) -> io::Result<()> {
        #[cfg(windows)]
        {
            self.inner.shutdown().await
        }
        #[cfg(not(windows))]
        {
            Ok(())
        }
    }
}

#[cfg(not(windows))]
fn unsupported() -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        "native input is only available on Windows",
    )
}

fn invalid_input(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn parse_shortcuts(shortcuts: &ShortcutSettings) -> io::Result<[ParsedHotkey; 3]> {
    let parsed = [
        parse_hotkey(&shortcuts.toggle)?,
        parse_hotkey(&shortcuts.play)?,
        parse_hotkey(&shortcuts.stop)?,
    ];
    if parsed[0] == parsed[1] || parsed[0] == parsed[2] || parsed[1] == parsed[2] {
        return Err(invalid_input("shortcut bindings must be distinct"));
    }
    Ok(parsed)
}

pub(crate) fn validate_shortcuts(shortcuts: &ShortcutSettings) -> io::Result<()> {
    parse_shortcuts(shortcuts).map(|_| ())
}

fn parse_hotkey(value: &str) -> io::Result<ParsedHotkey> {
    let mut modifiers = 0;
    let mut key = None;
    for part in value.split('+').map(str::trim) {
        if part.is_empty() {
            return Err(invalid_input("shortcut contains an empty part"));
        }
        let bit = match part.to_ascii_lowercase().as_str() {
            "ctrl" | "control" => Some(MOD_CONTROL),
            "shift" => Some(MOD_SHIFT),
            "alt" => Some(MOD_ALT),
            "win" | "windows" | "cmd" | "super" => Some(MOD_WIN),
            _ => None,
        };
        if let Some(bit) = bit {
            if modifiers & bit != 0 {
                return Err(invalid_input("shortcut repeats a modifier"));
            }
            modifiers |= bit;
            continue;
        }
        if key.replace(parse_key_code(part)?).is_some() {
            return Err(invalid_input("shortcut must contain exactly one key"));
        }
    }
    let virtual_key = key.ok_or_else(|| invalid_input("shortcut must contain a key"))?;
    if virtual_key == 0
        || virtual_key == 0x5b
        || virtual_key == 0x5c
        || virtual_key == 0x10
        || virtual_key == 0x11
        || virtual_key == 0x12
    {
        return Err(invalid_input("shortcut key is not supported"));
    }
    Ok(ParsedHotkey {
        modifiers: modifiers | MOD_NOREPEAT,
        virtual_key,
    })
}

pub(crate) fn parse_key_code(value: &str) -> io::Result<u16> {
    let normalized = value.trim().to_ascii_lowercase();
    if let Some(digit) = normalized.strip_prefix("numpad")
        && digit.len() == 1
        && let Some(digit) = digit.as_bytes().first()
        && digit.is_ascii_digit()
    {
        return Ok(0x60 + u16::from(*digit - b'0'));
    }
    if normalized.len() == 1 {
        let byte = normalized.as_bytes()[0];
        if byte.is_ascii_alphanumeric() {
            return Ok(byte.to_ascii_uppercase() as u16);
        }
        return match byte {
            b';' | b':' => Ok(0xba),
            b'=' | b'+' => Ok(0xbb),
            b',' | b'<' => Ok(0xbc),
            b'-' | b'_' => Ok(0xbd),
            b'.' | b'>' => Ok(0xbe),
            b'/' | b'?' => Ok(0xbf),
            b'`' | b'~' => Ok(0xc0),
            b'[' | b'{' => Ok(0xdb),
            b'\\' | b'|' => Ok(0xdc),
            b']' | b'}' => Ok(0xdd),
            b'\'' | b'"' => Ok(0xde),
            _ => Err(invalid_input("unsupported shortcut key")),
        };
    }

    if let Some(number) = normalized.strip_prefix('f')
        && let Ok(number) = number.parse::<u16>()
        && (1..=24).contains(&number)
    {
        return Ok(0x70 + number - 1);
    }

    if let Some(decimal) = normalized
        .strip_prefix('<')
        .and_then(|key| key.strip_suffix('>'))
    {
        let vk = decimal
            .parse::<u16>()
            .map_err(|_| invalid_input("invalid virtual-key code"))?;
        if vk > 0 && vk < 0x100 {
            return Ok(vk);
        }
        return Err(invalid_input("invalid virtual-key code"));
    }
    if let Some(hex) = normalized.strip_prefix("vk:0x") {
        let vk =
            u16::from_str_radix(hex, 16).map_err(|_| invalid_input("invalid virtual-key code"))?;
        if vk > 0 && vk < 0x100 {
            return Ok(vk);
        }
        return Err(invalid_input("invalid virtual-key code"));
    }

    match normalized.as_str() {
        "esc" | "escape" => Ok(0x1b),
        "enter" | "return" => Ok(0x0d),
        "space" => Ok(0x20),
        "tab" => Ok(0x09),
        "backspace" => Ok(0x08),
        "delete" | "del" => Ok(0x2e),
        "insert" | "ins" => Ok(0x2d),
        "home" => Ok(0x24),
        "end" => Ok(0x23),
        "page_up" | "pageup" | "prior" => Ok(0x21),
        "page_down" | "pagedown" | "next" => Ok(0x22),
        "up" => Ok(0x26),
        "down" => Ok(0x28),
        "left" => Ok(0x25),
        "right" => Ok(0x27),
        "ctrl" | "control" => Ok(0x11),
        "ctrl_l" => Ok(0xa2),
        "ctrl_r" => Ok(0xa3),
        "shift" | "shift_l" => Ok(0x10),
        "shift_r" => Ok(0xa1),
        "alt" => Ok(0x12),
        "alt_l" => Ok(0xa4),
        "alt_r" | "alt_gr" => Ok(0xa5),
        "cmd" | "cmd_l" | "win" | "win_l" => Ok(0x5b),
        "cmd_r" | "win_r" => Ok(0x5c),
        "caps_lock" => Ok(0x14),
        "num_lock" => Ok(0x90),
        "scroll_lock" => Ok(0x91),
        "print_screen" | "snapshot" => Ok(0x2c),
        "pause" => Ok(0x13),
        "menu" => Ok(0x5d),
        "media_volume_mute" => Ok(0xad),
        "media_volume_down" => Ok(0xae),
        "media_volume_up" => Ok(0xaf),
        "media_next" => Ok(0xb0),
        "media_previous" => Ok(0xb1),
        "media_stop" => Ok(0xb2),
        "media_play_pause" => Ok(0xb3),
        _ => Err(invalid_input("unsupported key name")),
    }
}

fn key_name(virtual_key: u16) -> String {
    match virtual_key {
        0x08 => "backspace".into(),
        0x09 => "tab".into(),
        0x0d => "enter".into(),
        0x10 | 0xa0 | 0xa1 => "shift".into(),
        0x11 | 0xa2 | 0xa3 => "ctrl".into(),
        0x12 | 0xa4 | 0xa5 => "alt".into(),
        0x13 => "pause".into(),
        0x14 => "caps_lock".into(),
        0x1b => "esc".into(),
        0x20 => "space".into(),
        0x21 => "page_up".into(),
        0x22 => "page_down".into(),
        0x23 => "end".into(),
        0x24 => "home".into(),
        0x25 => "left".into(),
        0x26 => "up".into(),
        0x27 => "right".into(),
        0x28 => "down".into(),
        0x2c => "print_screen".into(),
        0x2d => "insert".into(),
        0x2e => "delete".into(),
        0x5b | 0x5c => "cmd".into(),
        0x5d => "menu".into(),
        0x90 => "num_lock".into(),
        0x91 => "scroll_lock".into(),
        0x30..=0x39 => char::from_u32(virtual_key as u32)
            .unwrap_or('?')
            .to_string(),
        0x41..=0x5a => char::from_u32((virtual_key + 32) as u32)
            .unwrap_or('?')
            .to_string(),
        0x70..=0x87 => format!("f{}", virtual_key - 0x70 + 1),
        _ => format!("<{virtual_key}>"),
    }
}

fn trim_tail(steps: Vec<Value>, tail_seconds: f64) -> Vec<Value> {
    if steps.is_empty() || !tail_seconds.is_finite() || tail_seconds <= 0.0 {
        return steps;
    }
    let total = steps
        .iter()
        .map(|step| {
            step.get("t")
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite() && *value >= 0.0)
                .unwrap_or(0.0)
        })
        .sum::<f64>();
    let cutoff = (total - tail_seconds).max(0.0);
    if cutoff <= 0.0 {
        return Vec::new();
    }

    let mut trimmed = Vec::new();
    let mut elapsed = 0.0;
    for step in steps {
        let delta = step
            .get("t")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value >= 0.0)
            .unwrap_or(0.0);
        if elapsed + delta < cutoff {
            elapsed += delta;
            trimmed.push(step);
        } else {
            let remaining = cutoff - elapsed;
            if remaining > 0.001 {
                let mut retained = step;
                retained["t"] = json!(remaining);
                trimmed.push(retained);
            }
            break;
        }
    }
    trimmed
}

fn event_value(delta: f64, kind: &str, data: Value) -> Value {
    json!({"t": delta.max(0.0), "type": kind, "data": data})
}

fn append_recording_event(
    steps: &mut Vec<Value>,
    last_event_at: &mut Option<Instant>,
    overflowed: &mut bool,
    captures_at: Instant,
    now: Instant,
    max_events: usize,
    kind: &str,
    payload: Value,
) {
    if now < captures_at || *overflowed {
        return;
    }
    if steps.len() >= max_events {
        *overflowed = true;
        return;
    }
    let delta = last_event_at
        .map(|last| now.duration_since(last).as_secs_f64())
        .unwrap_or(0.0);
    *last_event_at = Some(now);
    steps.push(event_value(delta, kind, payload));
}

fn finish_recording(steps: Vec<Value>, overflowed: bool) -> RecordingCapture {
    RecordingCapture {
        steps: trim_tail(steps, TAIL_TRIM_SECONDS),
        overflowed,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MouseButton {
    Left,
    Right,
    Middle,
}

#[derive(Debug, Clone)]
enum ReplayEvent {
    Move { x: i32, y: i32 },
    Click { button: MouseButton, down: bool },
    Scroll { dx: i32, dy: i32 },
    Key { virtual_key: u16, down: bool },
    Nop,
}

#[derive(Debug, Clone)]
struct ReplayStep {
    delay: Duration,
    event: ReplayEvent,
}

fn parse_steps(steps: &[Value]) -> io::Result<Vec<ReplayStep>> {
    if steps.is_empty() {
        return Err(invalid_input("macro has no events"));
    }
    if steps.len() > MAX_CAPTURE_EVENTS {
        return Err(invalid_input("macro contains too many events"));
    }
    let mut total = 0.0;
    let mut parsed = Vec::with_capacity(steps.len());
    for step in steps {
        let delta = step
            .get("t")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value >= 0.0)
            .ok_or_else(|| invalid_input("macro event time is invalid"))?;
        if !try_accumulate_playback_duration(&mut total, delta) {
            return Err(invalid_input("macro duration is too long"));
        }
        let kind = step
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_input("macro event type is missing"))?;
        let data = step
            .get("data")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid_input("macro event data is invalid"))?;
        let event = match kind {
            "mouse_move" => ReplayEvent::Move {
                x: read_i32(data.get("x"))?,
                y: read_i32(data.get("y"))?,
            },
            "mouse_click" => {
                let button = match data.get("button").and_then(Value::as_str) {
                    Some("left") => MouseButton::Left,
                    Some("right") => MouseButton::Right,
                    Some("middle") => MouseButton::Middle,
                    _ => return Err(invalid_input("macro mouse button is invalid")),
                };
                read_i32(data.get("x"))?;
                read_i32(data.get("y"))?;
                let down = match data.get("action").and_then(Value::as_str) {
                    Some("down") => true,
                    Some("up") => false,
                    _ => return Err(invalid_input("macro mouse action is invalid")),
                };
                ReplayEvent::Click { button, down }
            }
            "scroll" => {
                read_i32(data.get("x"))?;
                read_i32(data.get("y"))?;
                ReplayEvent::Scroll {
                    dx: read_i32(data.get("dx"))?,
                    dy: read_i32(data.get("dy"))?,
                }
            }
            "key_down" | "key_up" => {
                let key = data
                    .get("key")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid_input("macro key is missing"))?;
                let virtual_key = parse_key_code(key)?;
                ReplayEvent::Key {
                    virtual_key,
                    down: kind == "key_down",
                }
            }
            "nop" => ReplayEvent::Nop,
            _ => return Err(invalid_input("macro event type is unsupported")),
        };
        parsed.push(ReplayStep {
            delay: Duration::from_secs_f64(delta),
            event,
        });
    }
    Ok(parsed)
}

fn read_i32(value: Option<&Value>) -> io::Result<i32> {
    value
        .and_then(Value::as_i64)
        .and_then(|number| i32::try_from(number).ok())
        .ok_or_else(|| invalid_input("macro coordinate or delta is invalid"))
}

#[cfg(windows)]
mod win {
    use super::*;
    use std::{
        cell::RefCell,
        collections::HashMap,
        sync::{
            Arc, Mutex, PoisonError,
            atomic::{AtomicU64, Ordering},
            mpsc as std_mpsc,
        },
        thread::{self, JoinHandle},
        time::Instant,
    };

    use tokio::{
        sync::{Mutex as AsyncMutex, watch},
        task::JoinHandle as AsyncJoinHandle,
        time::sleep_until,
    };
    use windows_sys::Win32::{
        Foundation::{LPARAM, WPARAM},
        System::{LibraryLoader::GetModuleHandleW, Threading::GetCurrentThreadId},
        UI::{
            Input::KeyboardAndMouse::{
                INPUT, INPUT_0, INPUT_KEYBOARD, INPUT_MOUSE, KEYBDINPUT, KEYEVENTF_EXTENDEDKEY,
                KEYEVENTF_KEYUP, MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_HWHEEL, MOUSEEVENTF_LEFTDOWN,
                MOUSEEVENTF_LEFTUP, MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP, MOUSEEVENTF_MOVE,
                MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, MOUSEEVENTF_VIRTUALDESK,
                MOUSEEVENTF_WHEEL, MOUSEINPUT, RegisterHotKey, SendInput, UnregisterHotKey,
            },
            WindowsAndMessaging::{
                CallNextHookEx, DispatchMessageW, KBDLLHOOKSTRUCT, LLKHF_INJECTED,
                LLKHF_LOWER_IL_INJECTED, LLMHF_INJECTED, LLMHF_LOWER_IL_INJECTED, MSG,
                MSLLHOOKSTRUCT, PM_NOREMOVE, PM_REMOVE, PeekMessageW, PostThreadMessageW,
                SetWindowsHookExW, TranslateMessage, UnhookWindowsHookEx, WH_KEYBOARD_LL,
                WH_MOUSE_LL, WM_HOTKEY, WM_KEYDOWN, WM_KEYUP, WM_LBUTTONDOWN, WM_LBUTTONUP,
                WM_MBUTTONDOWN, WM_MBUTTONUP, WM_MOUSEHWHEEL, WM_MOUSEMOVE, WM_MOUSEWHEEL, WM_QUIT,
                WM_RBUTTONDOWN, WM_RBUTTONUP, WM_SYSKEYDOWN, WM_SYSKEYUP,
            },
        },
    };

    const HOTKEY_TOGGLE: i32 = 1;
    const HOTKEY_PLAY: i32 = 2;
    const HOTKEY_STOP: i32 = 3;
    const HOTKEY_IDS: [i32; 3] = [HOTKEY_TOGGLE, HOTKEY_PLAY, HOTKEY_STOP];
    const INJECTION_TAG: usize = 0x4155_4343;
    const WHEEL_DELTA: i32 = 120;
    const SM_XVIRTUALSCREEN: i32 = 76;
    const SM_YVIRTUALSCREEN: i32 = 77;
    const SM_CXVIRTUALSCREEN: i32 = 78;
    const SM_CYVIRTUALSCREEN: i32 = 79;

    #[derive(Clone)]
    struct Recording {
        captures_at: Instant,
        last_event_at: Option<Instant>,
        steps: Vec<Value>,
        overflowed: bool,
    }

    struct Shared {
        recording: Mutex<Option<Recording>>,
        shortcuts: Mutex<[ParsedHotkey; 3]>,
        injected_keys: Mutex<HashSet<u16>>,
        suppressed_hotkeys: Mutex<HashMap<i32, Instant>>,
        events: mpsc::UnboundedSender<NativeEvent>,
        playback: AsyncMutex<Option<PlaybackControl>>,
        next_playback_id: AtomicU64,
    }

    struct PlaybackControl {
        id: u64,
        cancel: watch::Sender<bool>,
        task: AsyncJoinHandle<()>,
    }

    enum WorkerCommand {
        SetShortcuts {
            bindings: [ParsedHotkey; 3],
            reply: std_mpsc::SyncSender<io::Result<()>>,
        },
        Shutdown,
    }

    pub struct NativeInputInner {
        shared: Arc<Shared>,
        worker_tx: std_mpsc::Sender<WorkerCommand>,
        worker_thread_id: u32,
        worker: Option<JoinHandle<()>>,
    }

    thread_local! {
        static HOOK_SHARED: RefCell<Option<Arc<Shared>>> = const { RefCell::new(None) };
    }

    impl NativeInputInner {
        pub fn start(
            _shortcuts: ShortcutSettings,
            bindings: [ParsedHotkey; 3],
            events: mpsc::UnboundedSender<NativeEvent>,
        ) -> io::Result<Self> {
            let shared = Arc::new(Shared {
                recording: Mutex::new(None),
                shortcuts: Mutex::new(bindings),
                injected_keys: Mutex::new(HashSet::new()),
                suppressed_hotkeys: Mutex::new(HashMap::new()),
                events,
                playback: AsyncMutex::new(None),
                next_playback_id: AtomicU64::new(1),
            });
            let (worker_tx, worker_rx) = std_mpsc::channel();
            let (ready_tx, ready_rx) = std_mpsc::sync_channel(1);
            let worker_shared = Arc::clone(&shared);
            let worker = thread::Builder::new()
                .name("auto-coc-input".into())
                .spawn(move || {
                    run_worker(worker_shared, worker_rx, ready_tx, bindings);
                })?;
            let worker_thread_id = match ready_rx.recv() {
                Ok(Ok(thread_id)) => thread_id,
                Ok(Err((kind, message))) => {
                    let _ = worker.join();
                    return Err(io::Error::new(kind, message));
                }
                Err(_) => {
                    let _ = worker.join();
                    return Err(io::Error::other(
                        "native input worker failed during startup",
                    ));
                }
            };
            Ok(Self {
                shared,
                worker_tx,
                worker_thread_id,
                worker: Some(worker),
            })
        }

        pub fn start_recording(&self) -> io::Result<()> {
            let mut recording = lock(&self.shared.recording);
            if recording.is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "recording is already active",
                ));
            }
            *recording = Some(Recording {
                captures_at: Instant::now() + PREPARATION,
                last_event_at: None,
                steps: Vec::new(),
                overflowed: false,
            });
            Ok(())
        }

        pub fn stop_recording(&self) -> io::Result<RecordingCapture> {
            let mut recording = lock(&self.shared.recording);
            let Some(recording) = recording.take() else {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    "recording is not active",
                ));
            };
            Ok(finish_recording(recording.steps, recording.overflowed))
        }

        pub async fn start_playback(&self, steps: Vec<Value>, looped: bool) -> io::Result<()> {
            let parsed = parse_steps(&steps)?;
            let mut playback = self.shared.playback.lock().await;
            if playback.is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "playback is already active",
                ));
            }
            let id = self.shared.next_playback_id.fetch_add(1, Ordering::Relaxed);
            let (cancel, cancel_rx) = watch::channel(false);
            let shared = Arc::clone(&self.shared);
            let task = tokio::spawn(async move {
                run_playback(shared, id, parsed, looped, cancel_rx).await;
            });
            *playback = Some(PlaybackControl { id, cancel, task });
            Ok(())
        }

        pub async fn stop_playback(&self) -> io::Result<()> {
            let control = self.shared.playback.lock().await.take();
            let Some(control) = control else {
                return Ok(());
            };
            control.cancel.send_replace(true);
            control
                .task
                .await
                .map_err(|_| io::Error::other("playback task failed"))
        }

        pub async fn set_shortcuts(
            &self,
            _shortcuts: ShortcutSettings,
            bindings: [ParsedHotkey; 3],
        ) -> io::Result<()> {
            let (reply, response) = std_mpsc::sync_channel(1);
            self.worker_tx
                .send(WorkerCommand::SetShortcuts { bindings, reply })
                .map_err(|_| {
                    io::Error::new(io::ErrorKind::BrokenPipe, "native input worker has stopped")
                })?;
            tokio::task::spawn_blocking(move || {
                response.recv().unwrap_or_else(|_| {
                    Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "native input worker has stopped",
                    ))
                })
            })
            .await
            .map_err(|_| io::Error::other("shortcut update task failed"))?
        }

        pub async fn shutdown(&mut self) -> io::Result<()> {
            self.stop_playback().await?;
            let Some(worker) = self.worker.take() else {
                return Ok(());
            };
            if self.worker_tx.send(WorkerCommand::Shutdown).is_err() {
                tracing::warn!("AUTO-COC native input worker was already stopped");
            }
            if unsafe { PostThreadMessageW(self.worker_thread_id, WM_QUIT, 0, 0) } == 0 {
                tracing::warn!(
                    error = %io::Error::last_os_error(),
                    "Could not wake the AUTO-COC native input worker"
                );
            }
            tokio::task::spawn_blocking(move || worker.join())
                .await
                .map_err(|_| io::Error::other("native input worker join task failed"))?
                .map_err(|_| io::Error::other("native input worker panicked"))
        }
    }

    impl Drop for NativeInputInner {
        fn drop(&mut self) {
            if let Ok(mut playback) = self.shared.playback.try_lock()
                && let Some(control) = playback.take()
            {
                control.cancel.send_replace(true);
            }
            if self.worker.is_some() {
                if self.worker_tx.send(WorkerCommand::Shutdown).is_err() {
                    tracing::warn!("AUTO-COC native input worker was already stopped");
                }
                if unsafe { PostThreadMessageW(self.worker_thread_id, WM_QUIT, 0, 0) } == 0 {
                    tracing::warn!(
                        error = %io::Error::last_os_error(),
                        "Could not wake the AUTO-COC native input worker"
                    );
                }
            }
            drop(self.worker.take());
        }
    }

    fn run_worker(
        shared: Arc<Shared>,
        commands: std_mpsc::Receiver<WorkerCommand>,
        ready: std_mpsc::SyncSender<Result<u32, (io::ErrorKind, String)>>,
        initial: [ParsedHotkey; 3],
    ) {
        let instance = unsafe { GetModuleHandleW(std::ptr::null()) };
        if instance.is_null() {
            let error = io::Error::last_os_error();
            let _ = ready.send(Err((error.kind(), error.to_string())));
            return;
        }
        let keyboard_hook =
            unsafe { SetWindowsHookExW(WH_KEYBOARD_LL, Some(keyboard_hook), instance, 0) };
        if keyboard_hook.is_null() {
            let error = io::Error::last_os_error();
            let _ = ready.send(Err((error.kind(), error.to_string())));
            return;
        }
        let mouse_hook = unsafe { SetWindowsHookExW(WH_MOUSE_LL, Some(mouse_hook), instance, 0) };
        if mouse_hook.is_null() {
            let error = io::Error::last_os_error();
            unsafe {
                UnhookWindowsHookEx(keyboard_hook);
            }
            let _ = ready.send(Err((error.kind(), error.to_string())));
            return;
        }
        HOOK_SHARED.with(|current| *current.borrow_mut() = Some(Arc::clone(&shared)));
        let mut message: MSG = unsafe { std::mem::zeroed() };
        unsafe {
            PeekMessageW(&mut message, std::ptr::null_mut(), 0, 0, PM_NOREMOVE);
        }
        if let Err(error) = register_all(&initial) {
            HOOK_SHARED.with(|current| *current.borrow_mut() = None);
            unsafe {
                UnhookWindowsHookEx(mouse_hook);
                UnhookWindowsHookEx(keyboard_hook);
            }
            let _ = ready.send(Err((error.kind(), error.to_string())));
            return;
        }
        let thread_id = unsafe { GetCurrentThreadId() };
        if ready.send(Ok(thread_id)).is_err() {
            let _ = unregister_all();
            HOOK_SHARED.with(|current| *current.borrow_mut() = None);
            unsafe {
                UnhookWindowsHookEx(mouse_hook);
                UnhookWindowsHookEx(keyboard_hook);
            }
            return;
        }

        let mut should_stop = false;
        let mut active_bindings = initial;
        while !should_stop {
            loop {
                match commands.try_recv() {
                    Ok(WorkerCommand::SetShortcuts { bindings, reply }) => {
                        let result = transactional_rebind(&active_bindings, &bindings);
                        if result.is_ok() {
                            active_bindings = bindings;
                            *lock(&shared.shortcuts) = bindings;
                        }
                        let _ = reply.send(result);
                    }
                    Ok(WorkerCommand::Shutdown) => {
                        should_stop = true;
                        break;
                    }
                    Err(std_mpsc::TryRecvError::Empty) => break,
                    Err(std_mpsc::TryRecvError::Disconnected) => {
                        should_stop = true;
                        break;
                    }
                }
            }
            if should_stop {
                break;
            }
            let received =
                unsafe { PeekMessageW(&mut message, std::ptr::null_mut(), 0, 0, PM_REMOVE) };
            if received == 0 {
                thread::sleep(Duration::from_millis(10));
                continue;
            }
            if message.message == WM_QUIT {
                break;
            }
            if message.message == WM_HOTKEY {
                let id = message.wParam as i32;
                if consume_suppressed_hotkey(&shared, id) {
                    continue;
                }
                let event = match id {
                    HOTKEY_TOGGLE => NativeEvent::Toggle,
                    HOTKEY_PLAY => NativeEvent::Play,
                    HOTKEY_STOP => NativeEvent::Stop,
                    _ => continue,
                };
                let _ = shared.events.send(event);
            } else {
                unsafe {
                    TranslateMessage(&message);
                    DispatchMessageW(&message);
                }
            }
        }
        let _ = unregister_all();
        HOOK_SHARED.with(|current| *current.borrow_mut() = None);
        unsafe {
            UnhookWindowsHookEx(mouse_hook);
            UnhookWindowsHookEx(keyboard_hook);
        }
    }

    fn register_all(bindings: &[ParsedHotkey; 3]) -> io::Result<()> {
        register_selected(bindings, [true; 3])
    }

    fn register_selected(bindings: &[ParsedHotkey; 3], selected: [bool; 3]) -> io::Result<()> {
        let mut registered = Vec::new();
        for (index, (binding, should_register)) in bindings.iter().zip(selected).enumerate() {
            if !should_register {
                continue;
            }
            if unsafe {
                RegisterHotKey(
                    std::ptr::null_mut(),
                    HOTKEY_IDS[index],
                    binding.modifiers,
                    binding.virtual_key as u32,
                )
            } == 0
            {
                let error = io::Error::last_os_error();
                for id in registered {
                    if unsafe { UnregisterHotKey(std::ptr::null_mut(), id) } == 0 {
                        return Err(io::Error::other(format!(
                            "shortcut registration failed and a partial binding could not be removed: {error}"
                        )));
                    }
                }
                return Err(error);
            }
            registered.push(HOTKEY_IDS[index]);
        }
        Ok(())
    }

    fn unregister_all() -> ([bool; 3], Option<io::Error>) {
        let mut removed = [false; 3];
        let mut first_error = None;
        for (index, id) in HOTKEY_IDS.into_iter().enumerate() {
            if unsafe { UnregisterHotKey(std::ptr::null_mut(), id) } != 0 {
                removed[index] = true;
            } else if first_error.is_none() {
                first_error = Some(io::Error::last_os_error());
            }
        }
        (removed, first_error)
    }

    fn transactional_rebind(old: &[ParsedHotkey; 3], new: &[ParsedHotkey; 3]) -> io::Result<()> {
        let (removed, unregister_error) = unregister_all();
        if let Some(error) = unregister_error {
            if let Err(restore_error) = register_selected(old, removed) {
                return Err(io::Error::other(format!(
                    "shortcut unregister failed and prior bindings could not be restored: {restore_error}"
                )));
            }
            return Err(error);
        }
        if let Err(error) = register_all(new) {
            if let Err(restore_error) = register_all(old) {
                return Err(io::Error::other(format!(
                    "shortcut registration failed and prior bindings could not be restored: {restore_error}"
                )));
            }
            return Err(error);
        }
        Ok(())
    }

    unsafe extern "system" fn keyboard_hook(code: i32, message: WPARAM, data: LPARAM) -> isize {
        if code >= 0 && data != 0 {
            let event = unsafe { &*(data as *const KBDLLHOOKSTRUCT) };
            let msg = message as u32;
            let down = matches!(msg, WM_KEYDOWN | WM_SYSKEYDOWN);
            let up = matches!(msg, WM_KEYUP | WM_SYSKEYUP);
            if (down || up) && event.vkCode <= u16::MAX as u32 {
                HOOK_SHARED.with(|slot| {
                    if let Some(shared) = slot.borrow().as_ref() {
                        if event.flags & (LLKHF_INJECTED | LLKHF_LOWER_IL_INJECTED) != 0 {
                            if event.dwExtraInfo == INJECTION_TAG {
                                note_injected_key(shared, event.vkCode as u16, down);
                            }
                        } else {
                            let name = key_name(event.vkCode as u16);
                            record(
                                shared,
                                if down { "key_down" } else { "key_up" },
                                json!({"key":name}),
                            );
                        }
                    }
                });
            }
        }
        unsafe { CallNextHookEx(std::ptr::null_mut(), code, message, data) }
    }

    unsafe extern "system" fn mouse_hook(code: i32, message: WPARAM, data: LPARAM) -> isize {
        if code >= 0 && data != 0 {
            let event = unsafe { &*(data as *const MSLLHOOKSTRUCT) };
            let msg = message as u32;
            if event.flags & (LLMHF_INJECTED | LLMHF_LOWER_IL_INJECTED) == 0 {
                let x = event.pt.x;
                let y = event.pt.y;
                let captured = match msg {
                    WM_MOUSEMOVE => Some(("mouse_move", json!({"x":x,"y":y}))),
                    WM_LBUTTONDOWN => Some((
                        "mouse_click",
                        json!({"x":x,"y":y,"button":"left","action":"down"}),
                    )),
                    WM_LBUTTONUP => Some((
                        "mouse_click",
                        json!({"x":x,"y":y,"button":"left","action":"up"}),
                    )),
                    WM_RBUTTONDOWN => Some((
                        "mouse_click",
                        json!({"x":x,"y":y,"button":"right","action":"down"}),
                    )),
                    WM_RBUTTONUP => Some((
                        "mouse_click",
                        json!({"x":x,"y":y,"button":"right","action":"up"}),
                    )),
                    WM_MBUTTONDOWN => Some((
                        "mouse_click",
                        json!({"x":x,"y":y,"button":"middle","action":"down"}),
                    )),
                    WM_MBUTTONUP => Some((
                        "mouse_click",
                        json!({"x":x,"y":y,"button":"middle","action":"up"}),
                    )),
                    WM_MOUSEWHEEL => Some((
                        "scroll",
                        json!({"x":x,"y":y,"dx":0,"dy":wheel_steps(event.mouseData)}),
                    )),
                    WM_MOUSEHWHEEL => Some((
                        "scroll",
                        json!({"x":x,"y":y,"dx":wheel_steps(event.mouseData),"dy":0}),
                    )),
                    _ => None,
                };
                if let Some((kind, payload)) = captured {
                    record_event(kind, payload);
                }
            }
        }
        unsafe { CallNextHookEx(std::ptr::null_mut(), code, message, data) }
    }

    fn wheel_steps(data: u32) -> i32 {
        ((data >> 16) as u16 as i16 as i32) / WHEEL_DELTA
    }

    fn record_event(kind: &str, payload: Value) {
        HOOK_SHARED.with(|current| {
            if let Some(shared) = current.borrow().as_ref() {
                record(shared, kind, payload);
            }
        });
    }

    fn record(shared: &Shared, kind: &str, payload: Value) {
        let mut current = lock(&shared.recording);
        let Some(recording) = current.as_mut() else {
            return;
        };
        append_recording_event(
            &mut recording.steps,
            &mut recording.last_event_at,
            &mut recording.overflowed,
            recording.captures_at,
            Instant::now(),
            MAX_CAPTURE_EVENTS,
            kind,
            payload,
        );
    }

    fn note_injected_key(shared: &Shared, virtual_key: u16, down: bool) {
        let mut pressed = lock(&shared.injected_keys);
        if down {
            pressed.insert(virtual_key);
        } else {
            pressed.remove(&virtual_key);
        }
        if !down {
            return;
        }
        let modifiers = pressed_modifiers(&pressed);
        for (index, binding) in lock(&shared.shortcuts).iter().enumerate() {
            let wanted = binding.modifiers & !MOD_NOREPEAT;
            if binding.virtual_key == virtual_key && wanted == modifiers {
                lock(&shared.suppressed_hotkeys).insert(HOTKEY_IDS[index], Instant::now());
            }
        }
    }

    fn pressed_modifiers(pressed: &HashSet<u16>) -> u32 {
        let mut modifiers = 0;
        if [0x10, 0xa0, 0xa1].iter().any(|key| pressed.contains(key)) {
            modifiers |= MOD_SHIFT;
        }
        if [0x11, 0xa2, 0xa3].iter().any(|key| pressed.contains(key)) {
            modifiers |= MOD_CONTROL;
        }
        if [0x12, 0xa4, 0xa5].iter().any(|key| pressed.contains(key)) {
            modifiers |= MOD_ALT;
        }
        if [0x5b, 0x5c].iter().any(|key| pressed.contains(key)) {
            modifiers |= MOD_WIN;
        }
        modifiers
    }

    fn consume_suppressed_hotkey(shared: &Shared, id: i32) -> bool {
        lock(&shared.suppressed_hotkeys)
            .remove(&id)
            .is_some_and(|at| Instant::now().duration_since(at) <= Duration::from_secs(1))
    }

    async fn run_playback(
        shared: Arc<Shared>,
        id: u64,
        steps: Vec<ReplayStep>,
        looped: bool,
        mut cancel: watch::Receiver<bool>,
    ) {
        let mut guard = PlaybackGuard::default();
        'playback: loop {
            let start = tokio::time::Instant::now();
            let mut elapsed = Duration::ZERO;
            for step in &steps {
                elapsed = elapsed.saturating_add(step.delay);
                tokio::select! {
                    changed = cancel.changed() => {
                        if changed.is_err() || *cancel.borrow() { break 'playback; }
                    }
                    _ = sleep_until(start + elapsed) => {}
                }
                if *cancel.borrow() {
                    break 'playback;
                }
                if apply_event(&step.event, &mut guard.held).is_err() {
                    break 'playback;
                }
            }
            let _ = shared.events.send(NativeEvent::Cycle);
            if !looped {
                break;
            }
        }
        guard.release();
        let _ = shared.events.send(NativeEvent::PlaybackEnded);
        let mut playback = shared.playback.lock().await;
        if playback.as_ref().is_some_and(|control| control.id == id) {
            playback.take();
        }
    }

    #[derive(Default)]
    struct HeldInputs {
        keys: HashSet<u16>,
        buttons: HashSet<MouseButton>,
    }

    #[derive(Default)]
    struct PlaybackGuard {
        held: HeldInputs,
    }

    impl PlaybackGuard {
        fn release(&mut self) {
            let _ = release_inputs(&mut self.held);
        }
    }

    impl Drop for PlaybackGuard {
        fn drop(&mut self) {
            self.release();
        }
    }

    fn apply_event(event: &ReplayEvent, held: &mut HeldInputs) -> io::Result<()> {
        match event {
            ReplayEvent::Move { x, y } => send_mouse_move(*x, *y),
            ReplayEvent::Click { button, down } => {
                send_mouse_button(*button, *down)?;
                if *down {
                    held.buttons.insert(*button);
                } else {
                    held.buttons.remove(button);
                }
                Ok(())
            }
            ReplayEvent::Scroll { dx, dy } => send_scroll(*dx, *dy),
            ReplayEvent::Key { virtual_key, down } => {
                send_key(*virtual_key, *down)?;
                if *down {
                    held.keys.insert(*virtual_key);
                } else {
                    held.keys.remove(virtual_key);
                }
                Ok(())
            }
            ReplayEvent::Nop => Ok(()),
        }
    }

    fn release_inputs(held: &mut HeldInputs) -> io::Result<()> {
        let mut first_error = None;
        for key in held.keys.iter().copied().collect::<Vec<_>>() {
            match send_inputs(&[key_input(key, false)]) {
                Ok(()) => {
                    held.keys.remove(&key);
                }
                Err(error) if first_error.is_none() => first_error = Some(error),
                Err(_) => {}
            }
        }
        for button in held.buttons.iter().copied().collect::<Vec<_>>() {
            match send_inputs(&[mouse_button_input(button, false)]) {
                Ok(()) => {
                    held.buttons.remove(&button);
                }
                Err(error) if first_error.is_none() => first_error = Some(error),
                Err(_) => {}
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn send_key(virtual_key: u16, down: bool) -> io::Result<()> {
        send_inputs(&[key_input(virtual_key, down)])
    }

    fn key_input(virtual_key: u16, down: bool) -> INPUT {
        let mut flags = if down { 0 } else { KEYEVENTF_KEYUP };
        if is_extended_key(virtual_key) {
            flags |= KEYEVENTF_EXTENDEDKEY;
        }
        INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: virtual_key,
                    wScan: 0,
                    dwFlags: flags,
                    time: 0,
                    dwExtraInfo: INJECTION_TAG,
                },
            },
        }
    }

    fn is_extended_key(key: u16) -> bool {
        matches!(
            key,
            0x21..=0x2e | 0x5b..=0x5d | 0x6f | 0x90..=0x92 | 0xa3 | 0xa5 | 0xad..=0xb3
        )
    }

    fn send_mouse_move(x: i32, y: i32) -> io::Result<()> {
        let left = unsafe {
            windows_sys::Win32::UI::WindowsAndMessaging::GetSystemMetrics(SM_XVIRTUALSCREEN)
        };
        let top = unsafe {
            windows_sys::Win32::UI::WindowsAndMessaging::GetSystemMetrics(SM_YVIRTUALSCREEN)
        };
        let width = unsafe {
            windows_sys::Win32::UI::WindowsAndMessaging::GetSystemMetrics(SM_CXVIRTUALSCREEN)
        };
        let height = unsafe {
            windows_sys::Win32::UI::WindowsAndMessaging::GetSystemMetrics(SM_CYVIRTUALSCREEN)
        };
        if width <= 0 || height <= 0 {
            return Err(io::Error::other(
                "virtual desktop dimensions are unavailable",
            ));
        }
        let normalized_x = normalize_coordinate(x, left, width);
        let normalized_y = normalize_coordinate(y, top, height);
        send_inputs(&[INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dx: normalized_x,
                    dy: normalized_y,
                    mouseData: 0,
                    dwFlags: MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
                    time: 0,
                    dwExtraInfo: INJECTION_TAG,
                },
            },
        }])
    }

    fn normalize_coordinate(value: i32, origin: i32, extent: i32) -> i32 {
        if extent <= 1 {
            return 0;
        }
        (((i64::from(value) - i64::from(origin)).clamp(0, i64::from(extent - 1)) * 65_535)
            / i64::from(extent - 1)) as i32
    }

    fn send_mouse_button(button: MouseButton, down: bool) -> io::Result<()> {
        send_inputs(&[mouse_button_input(button, down)])
    }

    fn mouse_button_input(button: MouseButton, down: bool) -> INPUT {
        let flags = match (button, down) {
            (MouseButton::Left, true) => MOUSEEVENTF_LEFTDOWN,
            (MouseButton::Left, false) => MOUSEEVENTF_LEFTUP,
            (MouseButton::Right, true) => MOUSEEVENTF_RIGHTDOWN,
            (MouseButton::Right, false) => MOUSEEVENTF_RIGHTUP,
            (MouseButton::Middle, true) => MOUSEEVENTF_MIDDLEDOWN,
            (MouseButton::Middle, false) => MOUSEEVENTF_MIDDLEUP,
        };
        INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dx: 0,
                    dy: 0,
                    mouseData: 0,
                    dwFlags: flags,
                    time: 0,
                    dwExtraInfo: INJECTION_TAG,
                },
            },
        }
    }

    fn send_scroll(dx: i32, dy: i32) -> io::Result<()> {
        let mut inputs = Vec::with_capacity(2);
        if dy != 0 {
            inputs.push(scroll_input(dy, false)?);
        }
        if dx != 0 {
            inputs.push(scroll_input(dx, true)?);
        }
        send_inputs(&inputs)
    }

    fn scroll_input(amount: i32, horizontal: bool) -> io::Result<INPUT> {
        let wheel = amount
            .checked_mul(WHEEL_DELTA)
            .ok_or_else(|| invalid_input("scroll amount is too large"))?;
        Ok(INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dx: 0,
                    dy: 0,
                    mouseData: wheel as u32,
                    dwFlags: if horizontal {
                        MOUSEEVENTF_HWHEEL
                    } else {
                        MOUSEEVENTF_WHEEL
                    },
                    time: 0,
                    dwExtraInfo: INJECTION_TAG,
                },
            },
        })
    }

    fn send_inputs(inputs: &[INPUT]) -> io::Result<()> {
        if inputs.is_empty() {
            return Ok(());
        }
        let sent = unsafe {
            SendInput(
                inputs.len() as u32,
                inputs.as_ptr(),
                std::mem::size_of::<INPUT>() as i32,
            )
        };
        if sent != inputs.len() as u32 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        mutex.lock().unwrap_or_else(PoisonError::into_inner)
    }

    #[cfg(test)]
    mod tests {
        use super::is_extended_key;

        #[test]
        fn pynput_media_keys_use_extended_key_flags() {
            for key in 0xad..=0xb3 {
                assert!(is_extended_key(key));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_default_and_custom_shortcuts_and_rejects_collisions() {
        let defaults = parse_shortcuts(&ShortcutSettings::default()).unwrap();
        assert_eq!(
            defaults[0],
            ParsedHotkey {
                modifiers: MOD_NOREPEAT,
                virtual_key: 0x70
            }
        );
        assert_eq!(
            defaults[1],
            ParsedHotkey {
                modifiers: MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT,
                virtual_key: b'1' as u16
            }
        );
        let custom = ShortcutSettings {
            toggle: "Alt+F2".into(),
            play: "Ctrl+P".into(),
            stop: "Ctrl+Shift+Escape".into(),
        };
        assert_eq!(
            parse_hotkey(&custom.toggle).unwrap(),
            ParsedHotkey {
                modifiers: MOD_ALT | MOD_NOREPEAT,
                virtual_key: 0x71
            }
        );
        assert!(
            parse_shortcuts(&ShortcutSettings {
                toggle: "Ctrl+K".into(),
                play: "control+k".into(),
                stop: "F2".into()
            })
            .is_err()
        );
    }

    #[test]
    fn rejects_malformed_shortcuts_and_preserves_legacy_key_forms() {
        for invalid in [
            "",
            "Ctrl+",
            "Ctrl+Ctrl+A",
            "A+B",
            "Ctrl+Shift",
            "Ctrl+<999>",
            "Ctrl+F25",
        ] {
            assert!(parse_hotkey(invalid).is_err(), "accepted {invalid}");
        }
        assert_eq!(parse_key_code("<65>").unwrap(), 65);
        assert_eq!(parse_key_code("vk:0x41").unwrap(), 65);
        assert_eq!(parse_key_code("ctrl_l").unwrap(), 0xa2);
        assert_eq!(parse_key_code("F12").unwrap(), 0x7b);
    }

    #[test]
    fn parses_pynput_media_key_names() {
        for (name, virtual_key) in [
            ("media_volume_mute", 0xad),
            ("media_volume_down", 0xae),
            ("media_volume_up", 0xaf),
            ("media_next", 0xb0),
            ("media_previous", 0xb1),
            ("media_stop", 0xb2),
            ("media_play_pause", 0xb3),
        ] {
            assert_eq!(parse_key_code(name).unwrap(), virtual_key);
        }
        assert!(parse_key_code("é").is_err());
    }

    #[test]
    fn preserves_legacy_modifier_side_in_virtual_key_mapping() {
        for (name, virtual_key) in [
            ("ctrl", 0x11),
            ("ctrl_l", 0xa2),
            ("ctrl_r", 0xa3),
            ("shift", 0x10),
            ("shift_r", 0xa1),
            ("alt", 0x12),
            ("alt_l", 0xa4),
            ("alt_r", 0xa5),
            ("alt_gr", 0xa5),
            ("cmd", 0x5b),
            ("cmd_r", 0x5c),
        ] {
            assert_eq!(parse_key_code(name).unwrap(), virtual_key);
        }
    }

    #[test]
    fn distinguishes_numpad_digits_from_the_digit_row() {
        for digit in 0u16..=9 {
            assert_eq!(
                parse_key_code(&format!("Numpad{digit}")).unwrap(),
                0x60 + digit
            );
            assert_eq!(parse_key_code(&digit.to_string()).unwrap(), 0x30 + digit);
        }
    }

    #[test]
    fn playback_parser_enforces_the_shared_24_hour_limit() {
        let at_limit = vec![event_value(86_400.0, "nop", json!({}))];
        assert!(parse_steps(&at_limit).is_ok());

        let over_limit = vec![
            event_value(86_400.0, "nop", json!({})),
            event_value(0.1, "nop", json!({})),
        ];
        assert!(parse_steps(&over_limit).is_err());
    }

    #[test]
    fn trims_the_last_three_seconds_with_legacy_delta_semantics() {
        let steps = vec![
            event_value(0.0, "key_down", json!({"key":"a"})),
            event_value(2.0, "key_up", json!({"key":"a"})),
            event_value(2.0, "mouse_move", json!({"x":-20,"y":12})),
        ];
        let trimmed = trim_tail(steps, 3.0);
        assert_eq!(trimmed.len(), 2);
        assert_eq!(trimmed[1]["t"], 1.0);
        let short = vec![
            event_value(0.0, "nop", json!({})),
            event_value(2.0, "nop", json!({})),
        ];
        assert!(trim_tail(short, 3.0).is_empty());
    }

    #[test]
    fn capture_limit_preserves_collected_steps_and_marks_the_partial_capture() {
        let now = Instant::now();
        let mut steps = vec![event_value(5.0, "nop", json!({}))];
        let mut last_event_at = Some(now - Duration::from_secs(5));
        let mut overflowed = false;

        append_recording_event(
            &mut steps,
            &mut last_event_at,
            &mut overflowed,
            now - Duration::from_secs(1),
            now,
            1,
            "key_down",
            json!({"key":"a"}),
        );

        let capture = finish_recording(steps, overflowed);
        assert!(capture.overflowed);
        assert_eq!(capture.steps.len(), 1);
        assert_eq!(capture.steps[0]["type"], "nop");
        assert_eq!(capture.steps[0]["t"], 2.0);
    }

    #[test]
    fn serialized_capture_events_match_legacy_macro_shape() {
        let value = event_value(
            0.125,
            "mouse_click",
            json!({"x":-1920,"y":40,"button":"right","action":"down"}),
        );
        assert_eq!(
            value,
            json!({"t":0.125,"type":"mouse_click","data":{"x":-1920,"y":40,"button":"right","action":"down"}})
        );
        assert!(parse_steps(std::slice::from_ref(&value)).is_ok());
    }

    #[test]
    fn macro_parser_rejects_bad_events_without_turning_them_into_nops() {
        for invalid in [
            json!({"t":-1.0,"type":"nop","data":{}}),
            json!({"t":0.0,"type":"mystery","data":{}}),
            json!({"t":0.0,"type":"key_down","data":{"key":"not-a-key"}}),
            json!({"t":0.0,"type":"mouse_click","data":{"x":0,"y":0,"button":"x","action":"down"}}),
        ] {
            assert!(parse_steps(&[invalid]).is_err());
        }
    }
}
