use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    io,
    time::{Duration, Instant},
};

use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::types::ShortcutSettings;

pub(crate) const RECORDING_PREPARATION: Duration = Duration::from_secs(3);
const TAIL_TRIM_SECONDS: f64 = 3.0;
const MAX_CAPTURE_EVENTS: usize = 250_000;
const MAX_PLAYBACK_SECONDS: f64 = 86_400.0;
const OUTSIDE_COC_CLIENT_AREA: &str =
    "recorded mouse position is outside the Clash of Clans client area";

fn is_outside_coc_client_area(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::InvalidInput && error.to_string() == OUTSIDE_COC_CLIENT_AREA
}

fn ignore_outside_mouse_move(error: io::Error) -> io::Result<()> {
    if is_outside_coc_client_area(&error) {
        Ok(())
    } else {
        Err(error)
    }
}

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
    Cycle {
        playback_id: u64,
    },
    PlaybackEnded {
        playback_id: u64,
    },
    PlaybackFailed {
        playback_id: u64,
        reason: PlaybackFailure,
        detail: Option<String>,
    },
    PlaybackStopped {
        playback_id: u64,
        reason: PlaybackStopReason,
    },
    PlaybackPaused {
        playback_id: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaybackFailure {
    Input,
    Release,
    InputAndRelease,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaybackStopReason {
    TargetLost,
}

fn playback_completion_event(
    playback_id: u64,
    input_failed: bool,
    release_failed: bool,
    stop_reason: Option<PlaybackStopReason>,
    input_detail: Option<String>,
) -> NativeEvent {
    let input_detail = if input_failed {
        Some(input_detail.unwrap_or_else(|| {
            "le moteur de lecture n’a pas transmis le détail de l’étape en échec".into()
        }))
    } else {
        input_detail
    };
    if release_failed {
        NativeEvent::PlaybackFailed {
            playback_id,
            reason: if input_failed {
                PlaybackFailure::InputAndRelease
            } else {
                PlaybackFailure::Release
            },
            detail: input_detail.or_else(|| {
                (stop_reason == Some(PlaybackStopReason::TargetLost)).then(|| {
                    "La fenêtre Clash of Clans a disparu avant la libération des commandes maintenues."
                        .into()
                })
            }),
        }
    } else if input_failed {
        NativeEvent::PlaybackFailed {
            playback_id,
            reason: PlaybackFailure::Input,
            detail: input_detail,
        }
    } else if let Some(reason) = stop_reason {
        NativeEvent::PlaybackStopped {
            playback_id,
            reason,
        }
    } else {
        NativeEvent::PlaybackEnded { playback_id }
    }
}

fn finalized_playback_event(
    playback_id: u64,
    input_failed: bool,
    release_failed: bool,
    stop_reason: Option<PlaybackStopReason>,
    input_detail: Option<String>,
    release: impl FnOnce() -> io::Result<()>,
    publish: impl FnOnce(&NativeEvent),
) -> NativeEvent {
    let final_release_failed = release().is_err();
    let event = playback_completion_event(
        playback_id,
        input_failed,
        release_failed || final_release_failed,
        stop_reason,
        input_detail,
    );
    publish(&event);
    event
}

fn playback_cycle_event(
    playback_id: u64,
    input_failed: bool,
    cancelled: bool,
) -> Option<NativeEvent> {
    (!input_failed && !cancelled).then_some(NativeEvent::Cycle { playback_id })
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

    pub fn coc_is_foreground(&self) -> bool {
        #[cfg(windows)]
        {
            self.inner.resolve_coc_foreground()
        }
        #[cfg(not(windows))]
        {
            false
        }
    }

    pub fn cached_coc_is_foreground(&self) -> bool {
        #[cfg(windows)]
        {
            self.inner.cached_coc_is_foreground()
        }
        #[cfg(not(windows))]
        {
            false
        }
    }

    pub async fn start_recording(&self) -> io::Result<Instant> {
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

    pub async fn start_playback(
        &self,
        steps: &[Value],
        looped: bool,
        allow_background_playback: bool,
    ) -> io::Result<u64> {
        #[cfg(windows)]
        {
            self.inner
                .start_playback(steps, looped, allow_background_playback)
                .await
        }
        #[cfg(not(windows))]
        {
            let _ = (steps, looped, allow_background_playback);
            Err(unsupported())
        }
    }

    pub async fn stop_playback(&self) -> io::Result<Option<NativeEvent>> {
        #[cfg(windows)]
        {
            self.inner.stop_playback().await
        }
        #[cfg(not(windows))]
        {
            Err(unsupported())
        }
    }

    pub async fn acknowledge_playback_terminal(&self, playback_id: u64) -> io::Result<()> {
        #[cfg(windows)]
        {
            self.inner.acknowledge_playback_terminal(playback_id).await
        }
        #[cfg(not(windows))]
        {
            let _ = playback_id;
            Err(unsupported())
        }
    }

    pub async fn resume_playback(&self, playback_id: u64) -> io::Result<()> {
        #[cfg(windows)]
        {
            self.inner.resume_playback(playback_id).await
        }
        #[cfg(not(windows))]
        {
            let _ = playback_id;
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

fn ensure_coc_foreground(required: bool, foreground: bool) -> io::Result<()> {
    if required && !foreground {
        Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "Clash of Clans is not the foreground window",
        ))
    } else {
        Ok(())
    }
}

pub(crate) fn is_coc_window_identity(title: &str, image_path: &str) -> bool {
    let process_name = image_path.rsplit(['\\', '/']).next().unwrap_or(image_path);
    title.to_lowercase().contains("clash of clans")
        && process_name.eq_ignore_ascii_case("crosvm.exe")
}

fn decode_window_title(length: i32, last_error: u32, buffer: &[u16]) -> io::Result<String> {
    if length == 0 {
        return if last_error == 0 {
            Ok(String::new())
        } else {
            Err(io::Error::from_raw_os_error(last_error as i32))
        };
    }
    let length = usize::try_from(length)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid window title length"))?;
    let title = buffer
        .get(..length)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid window title length"))?;
    Ok(String::from_utf16_lossy(title))
}

pub(crate) fn coc_windows() -> io::Result<Vec<isize>> {
    #[cfg(windows)]
    {
        win::coc_windows()
    }
    #[cfg(not(windows))]
    {
        Err(unsupported())
    }
}

pub(crate) fn focus_coc_window(window: isize) -> io::Result<()> {
    #[cfg(windows)]
    {
        win::focus_coc_window(window)
    }
    #[cfg(not(windows))]
    {
        let _ = window;
        Err(unsupported())
    }
}

fn wait_for_coc_foreground_with(
    mut is_foreground: impl FnMut() -> io::Result<bool>,
    mut wait: impl FnMut(),
    deadline: Instant,
) -> io::Result<()> {
    loop {
        if is_foreground()? {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Windows did not allow Clash of Clans to become the foreground window",
            ));
        }
        wait();
    }
}

pub(crate) fn close_coc_window(window: isize) -> io::Result<()> {
    #[cfg(windows)]
    {
        win::close_coc_window(window)
    }
    #[cfg(not(windows))]
    {
        let _ = window;
        Err(unsupported())
    }
}

pub(crate) fn wait_for_coc_windows() -> io::Result<()> {
    #[cfg(windows)]
    {
        win::wait_for_coc_windows()
    }
    #[cfg(not(windows))]
    {
        Err(unsupported())
    }
}

fn same_process_instance(
    expected_pid: u32,
    expected_creation_time: u64,
    current_pid: u32,
    current_creation_time: u64,
) -> bool {
    expected_pid != 0
        && expected_pid == current_pid
        && expected_creation_time != 0
        && expected_creation_time == current_creation_time
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
) -> bool {
    if now < captures_at || *overflowed {
        return false;
    }
    if steps.len() >= max_events {
        *overflowed = true;
        return false;
    }
    let delta = last_event_at
        .map(|last| now.duration_since(last).as_secs_f64())
        .unwrap_or(0.0);
    *last_event_at = Some(now);
    steps.push(event_value(delta, kind, payload));
    true
}

fn should_play_recording_beep(beeped: &mut bool, event_accepted: bool) -> bool {
    if !event_accepted || *beeped {
        return false;
    }
    *beeped = true;
    true
}

fn finish_recording(steps: Vec<Value>, overflowed: bool) -> RecordingCapture {
    let mut steps = trim_tail(steps, TAIL_TRIM_SECONDS);
    let overflowed = overflowed || balance_recorded_inputs(&mut steps, MAX_CAPTURE_EVENTS);
    RecordingCapture { steps, overflowed }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MouseButton {
    Left,
    Right,
    Middle,
}

fn balance_recorded_inputs(steps: &mut Vec<Value>, max_steps: usize) -> bool {
    let mut discarded_events = false;

    loop {
        let mut held_keys = BTreeSet::new();
        let mut held_buttons = BTreeMap::new();
        for step in steps.iter() {
            let Some(kind) = step.get("type").and_then(Value::as_str) else {
                continue;
            };
            let Some(data) = step.get("data") else {
                continue;
            };
            match kind {
                "key_down" => {
                    if let Some(key) = data.get("key").and_then(Value::as_str) {
                        held_keys.insert(key.to_owned());
                    }
                }
                "key_up" => {
                    if let Some(key) = data.get("key").and_then(Value::as_str) {
                        held_keys.remove(key);
                    }
                }
                "mouse_click" => {
                    let (Some(button), Some(action)) = (
                        data.get("button").and_then(Value::as_str),
                        data.get("action").and_then(Value::as_str),
                    ) else {
                        continue;
                    };
                    if action == "down" {
                        held_buttons.insert(button.to_owned(), data.clone());
                    } else if action == "up" {
                        held_buttons.remove(button);
                    }
                }
                _ => {}
            }
        }

        let release_count = held_keys.len() + held_buttons.len();
        if steps.len() + release_count > max_steps {
            if steps.pop().is_some() {
                discarded_events = true;
                continue;
            }
            return discarded_events;
        }

        for key in held_keys {
            steps.push(event_value(0.0, "key_up", json!({"key": key})));
        }
        for mut data in held_buttons.into_values() {
            data["action"] = json!("up");
            steps.push(event_value(0.0, "mouse_click", data));
        }
        return discarded_events;
    }
}

#[derive(Debug, Clone)]
enum ReplayEvent {
    Move {
        point: RecordedMousePoint,
    },
    Click {
        point: RecordedMousePoint,
        button: MouseButton,
        down: bool,
    },
    Scroll {
        point: RecordedMousePoint,
        dx: i32,
        dy: i32,
    },
    Key {
        virtual_key: u16,
        down: bool,
    },
    Nop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RecordedMousePoint {
    x: i32,
    y: i32,
    window_origin: Option<(i32, i32)>,
}

impl RecordedMousePoint {
    fn client_position(self) -> io::Result<Option<(i32, i32)>> {
        let Some((origin_x, origin_y)) = self.window_origin else {
            return Ok(None);
        };
        let x = self
            .x
            .checked_sub(origin_x)
            .ok_or_else(|| invalid_input("recorded mouse x coordinate is out of range"))?;
        let y = self
            .y
            .checked_sub(origin_y)
            .ok_or_else(|| invalid_input("recorded mouse y coordinate is out of range"))?;
        Ok(Some((x, y)))
    }
}

impl ReplayEvent {
    fn label(&self) -> &'static str {
        match self {
            Self::Move { .. } => "déplacement de souris",
            Self::Click { .. } => "clic de souris",
            Self::Scroll { .. } => "défilement de souris",
            Self::Key { .. } => "touche clavier",
            Self::Nop => "événement vide",
        }
    }
}

fn playback_input_detail(step: usize, event: &str, error: io::Error) -> String {
    let cause = match error.to_string().as_str() {
        "recorded mouse position is outside the Clash of Clans client area" => {
            "la position souris enregistrée est hors de la zone de jeu CoC; réenregistrez la macro avec Clash of Clans au premier plan"
                .to_owned()
        }
        "the captured Clash of Clans window is no longer available" => {
            "la fenêtre Clash of Clans utilisée lors du démarrage du replay n’est plus disponible"
                .to_owned()
        }
        _ => error.to_string(),
    };
    format!("Étape {step} ({event}) : {cause}")
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
                point: read_mouse_point(data)?,
            },
            "mouse_click" => {
                let button = match data.get("button").and_then(Value::as_str) {
                    Some("left") => MouseButton::Left,
                    Some("right") => MouseButton::Right,
                    Some("middle") => MouseButton::Middle,
                    _ => return Err(invalid_input("macro mouse button is invalid")),
                };
                let point = read_mouse_point(data)?;
                let down = match data.get("action").and_then(Value::as_str) {
                    Some("down") => true,
                    Some("up") => false,
                    _ => return Err(invalid_input("macro mouse action is invalid")),
                };
                ReplayEvent::Click {
                    point,
                    button,
                    down,
                }
            }
            "scroll" => ReplayEvent::Scroll {
                point: read_mouse_point(data)?,
                dx: read_i32(data.get("dx"))?,
                dy: read_i32(data.get("dy"))?,
            },
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

fn read_mouse_point(data: &serde_json::Map<String, Value>) -> io::Result<RecordedMousePoint> {
    let window_origin = match data.get("windowOrigin") {
        None => None,
        Some(Value::Object(origin)) => {
            Some((read_i32(origin.get("x"))?, read_i32(origin.get("y"))?))
        }
        Some(_) => return Err(invalid_input("macro window origin is invalid")),
    };
    Ok(RecordedMousePoint {
        x: read_i32(data.get("x"))?,
        y: read_i32(data.get("y"))?,
        window_origin,
    })
}

#[cfg(windows)]
mod win {
    use super::*;
    use std::{
        cell::RefCell,
        collections::HashMap,
        sync::{
            Arc, Mutex, PoisonError,
            atomic::{AtomicBool, AtomicIsize, AtomicU32, AtomicU64, Ordering},
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
        Foundation::{
            CloseHandle, FILETIME, GetLastError, HWND, LPARAM, POINT, RECT, SetLastError, WPARAM,
        },
        Graphics::Gdi::{ClientToScreen, ScreenToClient},
        System::{
            Diagnostics::Debug::Beep,
            LibraryLoader::GetModuleHandleW,
            Threading::{
                GetCurrentThreadId, GetProcessTimes, OpenProcess,
                PROCESS_QUERY_LIMITED_INFORMATION, QueryFullProcessImageNameW,
            },
        },
        UI::{
            Input::KeyboardAndMouse::{
                MAPVK_VK_TO_VSC_EX, MapVirtualKeyW, RegisterHotKey, UnregisterHotKey,
            },
            WindowsAndMessaging::{
                CallNextHookEx, DispatchMessageW, EnumWindows, GetClientRect, GetForegroundWindow,
                GetWindowTextW, GetWindowThreadProcessId, IsIconic, IsWindow, IsWindowVisible,
                KBDLLHOOKSTRUCT, LLKHF_INJECTED, LLKHF_LOWER_IL_INJECTED, LLMHF_INJECTED,
                LLMHF_LOWER_IL_INJECTED, MSG, MSLLHOOKSTRUCT, PM_NOREMOVE, PM_REMOVE, PeekMessageW,
                PostMessageW, PostThreadMessageW, SW_RESTORE, SetForegroundWindow,
                SetWindowsHookExW, ShowWindow, TranslateMessage, UnhookWindowsHookEx,
                WH_KEYBOARD_LL, WH_MOUSE_LL, WM_CLOSE, WM_HOTKEY, WM_KEYDOWN, WM_KEYUP,
                WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MBUTTONDOWN, WM_MBUTTONUP, WM_MOUSEHWHEEL,
                WM_MOUSEMOVE, WM_MOUSEWHEEL, WM_QUIT, WM_RBUTTONDOWN, WM_RBUTTONUP, WM_SYSKEYDOWN,
                WM_SYSKEYUP,
            },
        },
    };

    const HOTKEY_TOGGLE: i32 = 1;
    const HOTKEY_PLAY: i32 = 2;
    const HOTKEY_STOP: i32 = 3;
    const HOTKEY_IDS: [i32; 3] = [HOTKEY_TOGGLE, HOTKEY_PLAY, HOTKEY_STOP];
    const INJECTION_TAG: usize = 0x4155_4343;
    const WHEEL_DELTA: i32 = 120;
    const MK_LBUTTON: WPARAM = 0x0001;
    const MK_RBUTTON: WPARAM = 0x0002;
    const MK_SHIFT: WPARAM = 0x0004;
    const MK_CONTROL: WPARAM = 0x0008;
    const MK_MBUTTON: WPARAM = 0x0010;
    const COC_CLOSE_TIMEOUT: Duration = Duration::from_secs(15);
    const COC_CLOSE_POLL_INTERVAL: Duration = Duration::from_millis(200);
    const COC_FOREGROUND_TIMEOUT: Duration = Duration::from_secs(3);
    const COC_FOREGROUND_POLL_INTERVAL: Duration = Duration::from_millis(50);
    const PAUSED_TARGET_POLL_INTERVAL: Duration = Duration::from_millis(250);

    #[derive(Clone)]
    struct Recording {
        captures_at: Instant,
        last_event_at: Option<Instant>,
        steps: Vec<Value>,
        held_keys: HashSet<String>,
        held_buttons: HashMap<MouseButton, RecordedMousePoint>,
        overflowed: bool,
        beeped: bool,
    }

    struct Shared {
        recording: Mutex<Option<Recording>>,
        coc_identity_valid: AtomicBool,
        coc_window: AtomicIsize,
        coc_process_id: AtomicU32,
        coc_process_created_at: AtomicU64,
        shortcuts: Mutex<[ParsedHotkey; 3]>,
        injected_keys: Mutex<HashSet<u16>>,
        suppressed_hotkeys: Mutex<HashMap<i32, Instant>>,
        events: mpsc::UnboundedSender<NativeEvent>,
        playback: AsyncMutex<Option<PlaybackControl>>,
        next_playback_id: AtomicU64,
    }

    impl Shared {
        fn cached_coc_is_foreground(&self) -> bool {
            if !self.coc_identity_valid.load(Ordering::Acquire) {
                return false;
            }
            let target = self.coc_window.load(Ordering::Acquire) as HWND;
            let process_id = self.coc_process_id.load(Ordering::Acquire);
            if target.is_null() || process_id == 0 {
                return false;
            }
            let mut target_process_id = 0;
            if unsafe { GetWindowThreadProcessId(target, &mut target_process_id) } == 0
                || target_process_id != process_id
            {
                self.coc_identity_valid.store(false, Ordering::Release);
                return false;
            }
            if unsafe { GetForegroundWindow() } != target || unsafe { IsWindowVisible(target) } == 0
            {
                return false;
            }
            if !window_title(target)
                .ok()
                .is_some_and(|title| title.to_ascii_lowercase().contains("clash of clans"))
            {
                self.coc_identity_valid.store(false, Ordering::Release);
                return false;
            }
            true
        }

        fn cached_coc_identity_is_foreground(&self) -> bool {
            if !self.cached_coc_is_foreground() {
                return false;
            }
            let target = self.coc_window.load(Ordering::Acquire) as HWND;
            let process_id = self.coc_process_id.load(Ordering::Acquire);
            let expected_created_at = self.coc_process_created_at.load(Ordering::Acquire);
            let is_valid = window_title(target).ok().is_some_and(|title| {
                process_image_and_creation_time(process_id).is_some_and(
                    |(path, current_created_at)| {
                        is_coc_window_identity(&title, &path)
                            && same_process_instance(
                                process_id,
                                expected_created_at,
                                process_id,
                                current_created_at,
                            )
                    },
                )
            });
            self.coc_identity_valid.store(is_valid, Ordering::Release);
            is_valid
        }
    }

    fn current_foreground_coc_window() -> Option<(HWND, u32, u64)> {
        let window = unsafe { GetForegroundWindow() };
        if window.is_null() || unsafe { IsWindowVisible(window) } == 0 {
            return None;
        }
        let mut process_id = 0;
        unsafe { GetWindowThreadProcessId(window, &mut process_id) };
        let title = window_title(window).ok()?;
        if process_id == 0 || !title.to_ascii_lowercase().contains("clash of clans") {
            return None;
        }

        let (image_path, created_at) = process_image_and_creation_time(process_id)?;
        is_coc_window_identity(&title, &image_path).then_some((window, process_id, created_at))
    }

    fn playback_target(allow_background_playback: bool) -> io::Result<CocTarget> {
        if let Some((window, process_id, process_created_at)) = current_foreground_coc_window() {
            return Ok(CocTarget {
                window: window as isize,
                process_id,
                process_created_at,
            });
        }
        if !allow_background_playback {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Clash of Clans must be the foreground window to start playback",
            ));
        }

        let mut candidates = enumerate_coc_windows()?
            .into_iter()
            .filter(|window| unsafe { IsWindowVisible(*window) } != 0)
            .filter_map(|window| {
                let mut process_id = 0;
                if unsafe { GetWindowThreadProcessId(window, &mut process_id) } == 0
                    || process_id == 0
                {
                    return None;
                }
                let (_, process_created_at) = process_image_and_creation_time(process_id)?;
                Some(CocTarget {
                    window: window as isize,
                    process_id,
                    process_created_at,
                })
            })
            .collect::<Vec<_>>();
        single_background_target(&mut candidates)
    }

    fn single_background_target(candidates: &mut Vec<CocTarget>) -> io::Result<CocTarget> {
        if candidates.len() != 1 {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "no unique visible Clash of Clans window is available",
            ));
        }
        Ok(candidates.remove(0))
    }

    pub(super) fn coc_windows() -> io::Result<Vec<isize>> {
        enumerate_coc_windows()
            .map(|windows| windows.into_iter().map(|window| window as isize).collect())
    }

    pub(super) fn focus_coc_window(window: isize) -> io::Result<()> {
        let window = window as HWND;
        if !window_matches_coc(window)? {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Clash of Clans window is no longer available",
            ));
        }

        if unsafe { IsIconic(window) } != 0 || unsafe { IsWindowVisible(window) } == 0 {
            unsafe { ShowWindow(window, SW_RESTORE) };
        }
        if unsafe { GetForegroundWindow() } != window {
            unsafe { SetForegroundWindow(window) };
        }

        wait_for_coc_foreground_with(
            || {
                if !window_matches_coc(window)? {
                    return Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        "Clash of Clans window is no longer available",
                    ));
                }
                Ok(unsafe { GetForegroundWindow() } == window
                    && unsafe { IsWindowVisible(window) } != 0)
            },
            || thread::sleep(COC_FOREGROUND_POLL_INTERVAL),
            Instant::now() + COC_FOREGROUND_TIMEOUT,
        )
    }

    pub(super) fn close_coc_window(window: isize) -> io::Result<()> {
        let window = window as HWND;
        if !window_matches_coc(window)? {
            return Ok(());
        }
        // The observed and closed object is the validated CoC game window, not the shared
        // Google Play Games crosvm.exe host; real restart behavior still needs validation there.
        if unsafe { PostMessageW(window, WM_CLOSE, 0, 0) } == 0 {
            if !window_matches_coc(window)? {
                return Ok(());
            }
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    pub(super) fn wait_for_coc_windows() -> io::Result<()> {
        let deadline = Instant::now() + COC_CLOSE_TIMEOUT;
        loop {
            if enumerate_coc_windows()?.is_empty() {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "Clash of Clans did not close",
                ));
            }
            thread::sleep(COC_CLOSE_POLL_INTERVAL);
        }
    }

    fn enumerate_coc_windows() -> io::Result<Vec<HWND>> {
        struct Enumeration {
            windows: Vec<HWND>,
            error: Option<io::Error>,
        }

        unsafe extern "system" fn collect_coc_window(window: HWND, parameter: LPARAM) -> i32 {
            let enumeration = unsafe { &mut *(parameter as *mut Enumeration) };
            match window_matches_coc(window) {
                Ok(true) => enumeration.windows.push(window),
                Ok(false) => {}
                Err(error) => {
                    enumeration.error = Some(error);
                    return 0;
                }
            }
            1
        }

        let mut enumeration = Enumeration {
            windows: Vec::new(),
            error: None,
        };
        let enumerated = unsafe {
            EnumWindows(
                Some(collect_coc_window),
                (&mut enumeration as *mut Enumeration) as LPARAM,
            )
        } != 0;
        if let Some(error) = enumeration.error {
            return Err(error);
        }
        if !enumerated {
            return Err(io::Error::last_os_error());
        }
        Ok(enumeration.windows)
    }

    fn window_matches_coc(window: HWND) -> io::Result<bool> {
        if window.is_null() || unsafe { IsWindow(window) } == 0 {
            return Ok(false);
        }
        let title = window_title(window)?;
        if title.is_empty() || !title.to_ascii_lowercase().contains("clash of clans") {
            return Ok(false);
        }
        let mut process_id = 0;
        let thread_id = unsafe { GetWindowThreadProcessId(window, &mut process_id) };
        if thread_id == 0 || process_id == 0 {
            return identity_read_failed_for_live_window(
                window,
                io::Error::other("could not read the process ID for the Clash of Clans window"),
            );
        }
        let image_path = match process_image_path(process_id) {
            Ok(image_path) => image_path,
            Err(error) => return identity_read_failed_for_live_window(window, error),
        };
        Ok(is_coc_window_identity(&title, &image_path))
    }

    fn identity_read_failed_for_live_window(window: HWND, error: io::Error) -> io::Result<bool> {
        if unsafe { IsWindow(window) } == 0 {
            Ok(false)
        } else {
            Err(error)
        }
    }

    fn process_image_path(process_id: u32) -> io::Result<String> {
        let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, process_id) };
        if process.is_null() {
            return Err(io::Error::last_os_error());
        }
        let image_path = process_image_path_from_handle(process);
        unsafe { CloseHandle(process) };
        image_path
    }

    fn process_image_and_creation_time(process_id: u32) -> Option<(String, u64)> {
        let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, process_id) };
        if process.is_null() {
            return None;
        }
        let image_path = process_image_path_from_handle(process);
        let created_at = process_creation_time_from_handle(process);
        unsafe { CloseHandle(process) };
        Some((image_path.ok()?, created_at?))
    }

    fn process_image_path_from_handle(
        process: windows_sys::Win32::Foundation::HANDLE,
    ) -> io::Result<String> {
        let mut image_path = [0u16; 1024];
        let mut length = image_path.len() as u32;
        if unsafe { QueryFullProcessImageNameW(process, 0, image_path.as_mut_ptr(), &mut length) }
            == 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(String::from_utf16_lossy(&image_path[..length as usize]))
    }

    fn process_creation_time(process_id: u32) -> Option<u64> {
        let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, process_id) };
        if process.is_null() {
            return None;
        }
        let created_at = process_creation_time_from_handle(process);
        unsafe { CloseHandle(process) };
        created_at
    }

    fn process_creation_time_from_handle(
        process: windows_sys::Win32::Foundation::HANDLE,
    ) -> Option<u64> {
        let mut created = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        let mut exited = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        let mut kernel = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        let mut user = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        (unsafe { GetProcessTimes(process, &mut created, &mut exited, &mut kernel, &mut user) }
            != 0)
            .then_some((u64::from(created.dwHighDateTime) << 32) | u64::from(created.dwLowDateTime))
            .filter(|created_at| *created_at != 0)
    }

    fn window_title(window: HWND) -> io::Result<String> {
        let mut title = [0u16; 512];
        unsafe { SetLastError(0) };
        let length = unsafe { GetWindowTextW(window, title.as_mut_ptr(), title.len() as i32) };
        let last_error = unsafe { GetLastError() };
        decode_window_title(length, last_error, &title)
    }

    struct PlaybackControl {
        id: u64,
        cancel: watch::Sender<bool>,
        resume: watch::Sender<()>,
        target: CocTarget,
        task: AsyncJoinHandle<NativeEvent>,
    }

    async fn join_playback_task(
        task: &mut AsyncJoinHandle<NativeEvent>,
    ) -> io::Result<NativeEvent> {
        task.await
            .map_err(|_| io::Error::other("playback task failed"))
    }

    #[derive(Clone, Copy)]
    struct CocTarget {
        window: isize,
        process_id: u32,
        process_created_at: u64,
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
                coc_identity_valid: AtomicBool::new(false),
                coc_window: AtomicIsize::new(0),
                coc_process_id: AtomicU32::new(0),
                coc_process_created_at: AtomicU64::new(0),
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

        pub fn resolve_coc_foreground(&self) -> bool {
            if let Some((window, process_id, created_at)) = current_foreground_coc_window() {
                self.shared
                    .coc_window
                    .store(window as isize, Ordering::Release);
                self.shared
                    .coc_process_id
                    .store(process_id, Ordering::Release);
                self.shared
                    .coc_process_created_at
                    .store(created_at, Ordering::Release);
                self.shared
                    .coc_identity_valid
                    .store(true, Ordering::Release);
                true
            } else {
                false
            }
        }

        pub fn cached_coc_is_foreground(&self) -> bool {
            self.shared.cached_coc_identity_is_foreground()
        }

        pub fn start_recording(&self) -> io::Result<Instant> {
            let mut recording = lock(&self.shared.recording);
            if recording.is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "recording is already active",
                ));
            }
            ensure_coc_foreground(true, self.shared.cached_coc_is_foreground())?;
            let started = Instant::now();
            *recording = Some(Recording {
                captures_at: started + RECORDING_PREPARATION,
                last_event_at: None,
                steps: Vec::new(),
                held_keys: HashSet::new(),
                held_buttons: HashMap::new(),
                overflowed: false,
                beeped: false,
            });
            Ok(started)
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

        pub async fn start_playback(
            &self,
            steps: &[Value],
            looped: bool,
            allow_background_playback: bool,
        ) -> io::Result<u64> {
            let parsed = parse_steps(steps)?;
            let mut playback = self.shared.playback.lock().await;
            if let Some(control) = playback.as_mut() {
                if !control.task.is_finished() {
                    return Err(io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        "playback is already active",
                    ));
                }
                let joined = join_playback_task(&mut control.task).await;
                playback.take();
                joined?;
            }
            let target = playback_target(allow_background_playback)?;
            if !allow_background_playback {
                ensure_coc_foreground(true, target_is_foreground(target))?;
            }
            let id = self.shared.next_playback_id.fetch_add(1, Ordering::Relaxed);
            let (cancel, cancel_rx) = watch::channel(false);
            let (resume, resume_rx) = watch::channel(());
            let shared = Arc::clone(&self.shared);
            let task = tokio::spawn(async move {
                run_playback(
                    shared,
                    id,
                    parsed,
                    looped,
                    target,
                    allow_background_playback,
                    cancel_rx,
                    resume_rx,
                )
                .await
            });
            *playback = Some(PlaybackControl {
                id,
                cancel,
                resume,
                target,
                task,
            });
            Ok(id)
        }

        pub async fn stop_playback(&self) -> io::Result<Option<NativeEvent>> {
            let mut playback = self.shared.playback.lock().await;
            let Some(control) = playback.as_mut() else {
                return Ok(None);
            };
            control.cancel.send_replace(true);
            let joined = join_playback_task(&mut control.task).await;
            playback.take();
            joined.map(Some)
        }

        pub async fn acknowledge_playback_terminal(&self, playback_id: u64) -> io::Result<()> {
            let mut playback = self.shared.playback.lock().await;
            let Some(control) = playback
                .as_mut()
                .filter(|control| control.id == playback_id)
            else {
                return Ok(());
            };
            let joined = join_playback_task(&mut control.task).await;
            playback.take();
            joined.map(|_| ())
        }

        pub async fn resume_playback(&self, playback_id: u64) -> io::Result<()> {
            let playback = self.shared.playback.lock().await;
            let control = playback
                .as_ref()
                .filter(|control| control.id == playback_id)
                .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "playback is not active"))?;
            if control.task.is_finished() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    "playback is no longer active",
                ));
            }
            let foreground = target_is_foreground(control.target);
            signal_playback_resume(&control.resume, foreground)
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
                            record_key_event(shared, &key_name(event.vkCode as u16), down);
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
                HOOK_SHARED.with(|slot| {
                    if let Some(shared) = slot.borrow().as_ref() {
                        record_mouse_hook_event(shared, msg, event);
                    }
                });
            }
        }
        unsafe { CallNextHookEx(std::ptr::null_mut(), code, message, data) }
    }

    fn record_mouse_hook_event(shared: &Shared, msg: u32, event: &MSLLHOOKSTRUCT) {
        let x = event.pt.x;
        let y = event.pt.y;
        let button_event = match msg {
            WM_LBUTTONDOWN => Some((MouseButton::Left, true)),
            WM_LBUTTONUP => Some((MouseButton::Left, false)),
            WM_RBUTTONDOWN => Some((MouseButton::Right, true)),
            WM_RBUTTONUP => Some((MouseButton::Right, false)),
            WM_MBUTTONDOWN => Some((MouseButton::Middle, true)),
            WM_MBUTTONUP => Some((MouseButton::Middle, false)),
            _ => None,
        };
        if let Some((button, down)) = button_event {
            let foreground = shared.cached_coc_is_foreground();
            let point = foreground
                .then(|| recording_mouse_point(shared, x, y))
                .flatten();
            record_mouse_button_event(shared, button, down, point);
            return;
        }

        let Some((kind, mut payload)) = (match msg {
            WM_MOUSEMOVE => Some(("mouse_move", json!({"x":x,"y":y}))),
            WM_MOUSEWHEEL => Some((
                "scroll",
                json!({"x":x,"y":y,"dx":0,"dy":wheel_steps(event.mouseData)}),
            )),
            WM_MOUSEHWHEEL => Some((
                "scroll",
                json!({"x":x,"y":y,"dx":wheel_steps(event.mouseData),"dy":0}),
            )),
            _ => None,
        }) else {
            return;
        };
        let Some(point) = recording_mouse_point(shared, x, y) else {
            return;
        };
        let Some((origin_x, origin_y)) = point.window_origin else {
            return;
        };
        payload["windowOrigin"] = json!({"x":origin_x,"y":origin_y});
        record(shared, kind, payload);
    }

    fn recording_mouse_point(
        shared: &Shared,
        screen_x: i32,
        screen_y: i32,
    ) -> Option<RecordedMousePoint> {
        let window = shared.coc_window.load(Ordering::Acquire) as HWND;
        if window.is_null() {
            return None;
        }
        let mut client_point = POINT {
            x: screen_x,
            y: screen_y,
        };
        if unsafe { ScreenToClient(window, &mut client_point) } == 0 {
            return None;
        }
        let mut client = RECT {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        };
        if unsafe { GetClientRect(window, &mut client) } == 0
            || client_point.x < client.left
            || client_point.y < client.top
            || client_point.x >= client.right
            || client_point.y >= client.bottom
        {
            return None;
        }
        let mut origin = POINT { x: 0, y: 0 };
        if unsafe { ClientToScreen(window, &mut origin) } == 0 {
            return None;
        }
        Some(RecordedMousePoint {
            x: screen_x,
            y: screen_y,
            window_origin: Some((origin.x, origin.y)),
        })
    }

    fn wheel_steps(data: u32) -> i32 {
        ((data >> 16) as u16 as i16 as i32) / WHEEL_DELTA
    }

    fn record(shared: &Shared, kind: &str, payload: Value) {
        if !shared.cached_coc_is_foreground() {
            return;
        }
        let beep = {
            let mut current = lock(&shared.recording);
            let Some(recording) = current.as_mut() else {
                return;
            };
            let accepted = append_recording_event(
                &mut recording.steps,
                &mut recording.last_event_at,
                &mut recording.overflowed,
                recording.captures_at,
                Instant::now(),
                MAX_CAPTURE_EVENTS,
                kind,
                payload,
            );
            should_play_recording_beep(&mut recording.beeped, accepted)
        };
        if beep {
            play_recording_beep();
        }
    }

    fn record_key_event(shared: &Shared, key: &str, down: bool) {
        let foreground = shared.cached_coc_is_foreground();
        let beep = {
            let mut current = lock(&shared.recording);
            let Some(recording) = current.as_mut() else {
                return;
            };
            let accepted =
                append_recording_key_event(recording, key, down, foreground, Instant::now());
            should_play_recording_beep(&mut recording.beeped, accepted)
        };
        if beep {
            play_recording_beep();
        }
    }

    fn append_recording_key_event(
        recording: &mut Recording,
        key: &str,
        down: bool,
        foreground: bool,
        now: Instant,
    ) -> bool {
        if (down && !foreground) || (!down && !recording.held_keys.contains(key)) {
            return false;
        }
        let kind = if down { "key_down" } else { "key_up" };
        let accepted = append_recording_event(
            &mut recording.steps,
            &mut recording.last_event_at,
            &mut recording.overflowed,
            recording.captures_at,
            now,
            MAX_CAPTURE_EVENTS,
            kind,
            json!({"key": key}),
        );
        if accepted {
            if down {
                recording.held_keys.insert(key.to_owned());
            } else {
                recording.held_keys.remove(key);
            }
        }
        accepted
    }

    fn record_mouse_button_event(
        shared: &Shared,
        button: MouseButton,
        down: bool,
        point: Option<RecordedMousePoint>,
    ) {
        let beep = {
            let mut current = lock(&shared.recording);
            let Some(recording) = current.as_mut() else {
                return;
            };
            let accepted =
                append_recording_mouse_button_event(recording, button, down, point, Instant::now());
            should_play_recording_beep(&mut recording.beeped, accepted)
        };
        if beep {
            play_recording_beep();
        }
    }

    fn append_recording_mouse_button_event(
        recording: &mut Recording,
        button: MouseButton,
        down: bool,
        point: Option<RecordedMousePoint>,
        now: Instant,
    ) -> bool {
        let event_point = if down {
            point
        } else if recording.held_buttons.contains_key(&button) {
            point.or_else(|| recording.held_buttons.get(&button).copied())
        } else {
            None
        };
        let Some(point) = event_point else {
            return false;
        };
        let Some((origin_x, origin_y)) = point.window_origin else {
            return false;
        };
        let button_name = match button {
            MouseButton::Left => "left",
            MouseButton::Right => "right",
            MouseButton::Middle => "middle",
        };
        let accepted = append_recording_event(
            &mut recording.steps,
            &mut recording.last_event_at,
            &mut recording.overflowed,
            recording.captures_at,
            now,
            MAX_CAPTURE_EVENTS,
            "mouse_click",
            json!({
                "x": point.x,
                "y": point.y,
                "windowOrigin": {"x": origin_x, "y": origin_y},
                "button": button_name,
                "action": if down {"down"} else {"up"},
            }),
        );
        if accepted {
            if down {
                recording.held_buttons.insert(button, point);
            } else {
                recording.held_buttons.remove(&button);
            }
        }
        accepted
    }

    fn play_recording_beep() {
        if let Err(error) = thread::Builder::new()
            .name("auto-coc-recording-beep".into())
            .spawn(|| {
                if unsafe { Beep(1_000, 200) } == 0 {
                    tracing::warn!(
                        error = %io::Error::last_os_error(),
                        "Could not play the AUTO-COC recording beep"
                    );
                }
            })
        {
            tracing::warn!(%error, "Could not start the AUTO-COC recording beep");
        }
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
        target: CocTarget,
        allow_background_playback: bool,
        mut cancel: watch::Receiver<bool>,
        mut resume: watch::Receiver<()>,
    ) -> NativeEvent {
        let mut guard = PlaybackGuard {
            held: HeldInputs::default(),
            target,
            armed: true,
        };
        let mut input_failed = false;
        let mut input_detail = None;
        let mut release_failed = false;
        let mut stop_reason = None;
        'playback: loop {
            let mut start = tokio::time::Instant::now();
            let mut elapsed = Duration::ZERO;
            let mut step_index = 0;
            while let Some(step) = steps.get(step_index) {
                elapsed = elapsed.saturating_add(step.delay);
                let mut deadline = start + elapsed;
                'step: loop {
                    match wait_for_replay_step(
                        deadline,
                        &mut cancel,
                        !allow_background_playback,
                        || target_window(target, false).is_ok(),
                        || target_is_foreground(target),
                    )
                    .await
                    {
                        ReplayWait::Ready => {}
                        ReplayWait::Cancelled => {
                            break 'playback;
                        }
                        ReplayWait::TargetLost => {
                            stop_reason = Some(PlaybackStopReason::TargetLost);
                            break 'playback;
                        }
                        ReplayWait::ForegroundLost => {
                            match pause_for_foreground(
                                id,
                                &shared,
                                target,
                                &mut guard,
                                &mut cancel,
                                &mut resume,
                            )
                            .await
                            {
                                PlaybackPauseResult::Resumed(paused_for) => {
                                    shift_replay_timing(&mut start, &mut deadline, paused_for);
                                    continue 'step;
                                }
                                PlaybackPauseResult::Cancelled => {
                                    break 'playback;
                                }
                                PlaybackPauseResult::TargetLost => {
                                    stop_reason = Some(PlaybackStopReason::TargetLost);
                                    break 'playback;
                                }
                                PlaybackPauseResult::ReleaseFailed => {
                                    release_failed = true;
                                    break 'playback;
                                }
                            }
                        }
                    }
                    if let Err(error) = apply_event(
                        &step.event,
                        &mut guard.held,
                        target,
                        !allow_background_playback,
                    ) {
                        if error.kind() == io::ErrorKind::NotFound {
                            stop_reason = Some(PlaybackStopReason::TargetLost);
                            break 'playback;
                        }
                        if error.kind() == io::ErrorKind::PermissionDenied
                            && !allow_background_playback
                        {
                            match pause_for_foreground(
                                id,
                                &shared,
                                target,
                                &mut guard,
                                &mut cancel,
                                &mut resume,
                            )
                            .await
                            {
                                PlaybackPauseResult::Resumed(paused_for) => {
                                    shift_replay_timing(&mut start, &mut deadline, paused_for);
                                    continue 'step;
                                }
                                PlaybackPauseResult::Cancelled => {
                                    break 'playback;
                                }
                                PlaybackPauseResult::TargetLost => {
                                    stop_reason = Some(PlaybackStopReason::TargetLost);
                                    break 'playback;
                                }
                                PlaybackPauseResult::ReleaseFailed => {
                                    release_failed = true;
                                    break 'playback;
                                }
                            }
                        } else {
                            input_failed = true;
                            input_detail = Some(playback_input_detail(
                                step_index + 1,
                                step.event.label(),
                                error,
                            ));
                            break 'playback;
                        }
                    }
                    step_index += 1;
                    break 'step;
                }
            }
            let Some(cycle_event) = playback_cycle_event(id, input_failed, false) else {
                break 'playback;
            };
            let _ = shared.events.send(cycle_event);
            if !looped {
                break;
            }
        }
        finalized_playback_event(
            id,
            input_failed,
            release_failed,
            stop_reason,
            input_detail,
            || guard.finish(),
            |event| {
                let _ = shared.events.send(event.clone());
            },
        )
    }

    enum PlaybackPauseResult {
        Resumed(Duration),
        Cancelled,
        TargetLost,
        ReleaseFailed,
    }

    async fn pause_for_foreground(
        id: u64,
        shared: &Shared,
        target: CocTarget,
        guard: &mut PlaybackGuard,
        cancel: &mut watch::Receiver<bool>,
        resume: &mut watch::Receiver<()>,
    ) -> PlaybackPauseResult {
        pause_after_release(
            cancel,
            resume,
            || target_window(target, false).is_ok(),
            || guard.release(),
            || {
                let _ = shared
                    .events
                    .send(NativeEvent::PlaybackPaused { playback_id: id });
            },
        )
        .await
    }

    async fn pause_after_release(
        cancel: &mut watch::Receiver<bool>,
        resume: &mut watch::Receiver<()>,
        target_is_available: impl FnMut() -> bool,
        release: impl FnOnce() -> io::Result<()>,
        publish_paused: impl FnOnce(),
    ) -> PlaybackPauseResult {
        let paused_at = tokio::time::Instant::now();
        if release().is_err() {
            return PlaybackPauseResult::ReleaseFailed;
        }
        publish_paused();
        match wait_for_playback_resume(cancel, resume, target_is_available).await {
            PlaybackPauseResult::Resumed(_) => PlaybackPauseResult::Resumed(paused_at.elapsed()),
            result => result,
        }
    }

    fn signal_playback_resume(resume: &watch::Sender<()>, foreground: bool) -> io::Result<()> {
        ensure_coc_foreground(true, foreground)?;
        resume.send_replace(());
        Ok(())
    }

    fn shift_replay_timing(
        start: &mut tokio::time::Instant,
        deadline: &mut tokio::time::Instant,
        paused_for: Duration,
    ) {
        *start += paused_for;
        *deadline += paused_for;
    }

    async fn wait_for_playback_resume(
        cancel: &mut watch::Receiver<bool>,
        resume: &mut watch::Receiver<()>,
        mut target_is_available: impl FnMut() -> bool,
    ) -> PlaybackPauseResult {
        loop {
            if cancel.has_changed().is_err() || *cancel.borrow() {
                return PlaybackPauseResult::Cancelled;
            }
            if !target_is_available() {
                return PlaybackPauseResult::TargetLost;
            }
            tokio::select! {
                changed = cancel.changed() => {
                    if changed.is_err() || *cancel.borrow() {
                        return PlaybackPauseResult::Cancelled;
                    }
                }
                changed = resume.changed() => {
                    if changed.is_err() {
                        return PlaybackPauseResult::Cancelled;
                    }
                    if !target_is_available() {
                        return PlaybackPauseResult::TargetLost;
                    }
                    return PlaybackPauseResult::Resumed(Duration::ZERO);
                }
                _ = sleep_until(tokio::time::Instant::now() + PAUSED_TARGET_POLL_INTERVAL) => {}
            }
        }
    }

    #[derive(Default)]
    struct HeldInputs {
        keys: HashSet<u16>,
        buttons: HashMap<MouseButton, (i32, i32)>,
    }

    struct PlaybackGuard {
        held: HeldInputs,
        target: CocTarget,
        armed: bool,
    }

    impl PlaybackGuard {
        fn release(&mut self) -> io::Result<()> {
            release_targeted_inputs(self.target, &mut self.held)
        }

        fn finish(&mut self) -> io::Result<()> {
            let (armed, held, target) = (&mut self.armed, &mut self.held, self.target);
            finish_guard_release(armed, || release_targeted_inputs(target, held))
        }
    }

    impl Drop for PlaybackGuard {
        fn drop(&mut self) {
            release_guard_on_drop(self.armed, || self.release());
        }
    }

    fn finish_guard_release(
        armed: &mut bool,
        mut release: impl FnMut() -> io::Result<()>,
    ) -> io::Result<()> {
        let result = release().or_else(|_| release());
        *armed = false;
        result
    }

    fn release_guard_on_drop(armed: bool, release: impl FnOnce() -> io::Result<()>) {
        if armed {
            let _ = release();
        }
    }

    enum ReplayWait {
        Ready,
        Cancelled,
        TargetLost,
        ForegroundLost,
    }

    async fn wait_for_replay_step(
        deadline: tokio::time::Instant,
        cancel: &mut watch::Receiver<bool>,
        require_foreground: bool,
        mut target_is_available: impl FnMut() -> bool,
        mut coc_is_foreground: impl FnMut() -> bool,
    ) -> ReplayWait {
        loop {
            if cancel.has_changed().is_err() || *cancel.borrow() {
                return ReplayWait::Cancelled;
            }
            if !target_is_available() {
                return ReplayWait::TargetLost;
            }
            if require_foreground && !coc_is_foreground() {
                return ReplayWait::ForegroundLost;
            }
            let now = tokio::time::Instant::now();
            if now >= deadline {
                return ReplayWait::Ready;
            }
            let next_check = (now + Duration::from_millis(100)).min(deadline);
            tokio::select! {
                changed = cancel.changed() => {
                    if changed.is_err() || *cancel.borrow() {
                        return ReplayWait::Cancelled;
                    }
                }
                _ = sleep_until(next_check) => {}
            }
        }
    }

    fn apply_event(
        event: &ReplayEvent,
        held: &mut HeldInputs,
        target: CocTarget,
        require_foreground: bool,
    ) -> io::Result<()> {
        match event {
            ReplayEvent::Move { point } => {
                send_targeted_mouse_move(target, *point, held, require_foreground)
            }
            ReplayEvent::Click {
                point,
                button,
                down,
            } => {
                let held_position = send_targeted_mouse_button(
                    target,
                    *point,
                    *button,
                    *down,
                    held,
                    require_foreground,
                )?;
                if *down {
                    held.buttons.insert(*button, held_position);
                } else {
                    held.buttons.remove(button);
                }
                Ok(())
            }
            ReplayEvent::Scroll { point, dx, dy } => {
                send_targeted_scroll(target, *point, *dx, *dy, held, require_foreground)
            }
            ReplayEvent::Key { virtual_key, down } => {
                send_targeted_key(target, *virtual_key, *down, &held.keys, require_foreground)?;
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

    fn is_extended_key(key: u16) -> bool {
        matches!(
            key,
            0x21..=0x2e | 0x5b..=0x5d | 0x6f | 0x90..=0x92 | 0xa3 | 0xa5 | 0xad..=0xb3
        )
    }

    fn target_window(target: CocTarget, require_foreground: bool) -> io::Result<HWND> {
        let window = target.window as HWND;
        if window.is_null() || unsafe { IsWindow(window) } == 0 {
            return Err(invalid_target_window());
        }
        let mut process_id = 0;
        if unsafe { GetWindowThreadProcessId(window, &mut process_id) } == 0
            || process_id != target.process_id
            || process_creation_time(process_id) != Some(target.process_created_at)
        {
            return Err(invalid_target_window());
        }
        if !window_title(window)?
            .to_ascii_lowercase()
            .contains("clash of clans")
        {
            return Err(invalid_target_window());
        }
        let visible = unsafe { IsWindowVisible(window) } != 0;
        let foreground = unsafe { GetForegroundWindow() } == window;
        if !target_visibility_allowed(require_foreground, visible, foreground) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Clash of Clans is no longer the foreground window",
            ));
        }
        Ok(window)
    }

    fn invalid_target_window() -> io::Error {
        io::Error::new(
            io::ErrorKind::NotFound,
            "the captured Clash of Clans window is no longer available",
        )
    }

    fn target_visibility_allowed(
        require_foreground: bool,
        visible: bool,
        foreground: bool,
    ) -> bool {
        !require_foreground || (visible && foreground)
    }

    fn target_is_foreground(target: CocTarget) -> bool {
        target_window(target, true).is_ok()
    }

    fn target_client_point(
        target: CocTarget,
        recorded: RecordedMousePoint,
        require_foreground: bool,
    ) -> io::Result<(HWND, POINT)> {
        let window = target_window(target, require_foreground)?;
        let mut point = POINT {
            x: recorded.x,
            y: recorded.y,
        };
        if let Some((x, y)) = recorded.client_position()? {
            point = POINT { x, y };
        } else if unsafe { ScreenToClient(window, &mut point) } == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut client = RECT {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        };
        if unsafe { GetClientRect(window, &mut client) } == 0 {
            return Err(io::Error::last_os_error());
        }
        if point.x < client.left
            || point.y < client.top
            || point.x >= client.right
            || point.y >= client.bottom
        {
            return Err(invalid_input(OUTSIDE_COC_CLIENT_AREA));
        }
        Ok((window, point))
    }

    fn pack_window_coordinates(x: i32, y: i32) -> io::Result<LPARAM> {
        let x =
            i16::try_from(x).map_err(|_| invalid_input("mouse x coordinate is out of range"))?;
        let y =
            i16::try_from(y).map_err(|_| invalid_input("mouse y coordinate is out of range"))?;
        Ok((u32::from(x as u16) | (u32::from(y as u16) << 16)) as i32 as LPARAM)
    }

    fn post_target_message(
        target: CocTarget,
        window: HWND,
        message: u32,
        wparam: WPARAM,
        lparam: LPARAM,
        require_foreground: bool,
    ) -> io::Result<()> {
        if target_window(target, require_foreground)? != window {
            return Err(invalid_target_window());
        }
        if unsafe { PostMessageW(window, message, wparam, lparam) } == 0 {
            return Err(io::Error::other(format!(
                "could not post input to the Clash of Clans window: {}",
                io::Error::last_os_error()
            )));
        }
        Ok(())
    }

    fn send_targeted_mouse_move(
        target: CocTarget,
        recorded: RecordedMousePoint,
        held: &HeldInputs,
        require_foreground: bool,
    ) -> io::Result<()> {
        let (window, point) = match target_client_point(target, recorded, require_foreground) {
            Ok(target_point) => target_point,
            Err(error) => return ignore_outside_mouse_move(error),
        };
        post_target_message(
            target,
            window,
            WM_MOUSEMOVE,
            mouse_key_state(held, None),
            pack_window_coordinates(point.x, point.y)?,
            require_foreground,
        )
    }

    fn send_targeted_mouse_button(
        target: CocTarget,
        recorded: RecordedMousePoint,
        button: MouseButton,
        down: bool,
        held: &HeldInputs,
        require_foreground: bool,
    ) -> io::Result<(i32, i32)> {
        let (window, point) = target_client_point(target, recorded, require_foreground)?;
        let lparam = pack_window_coordinates(point.x, point.y)?;
        post_target_message(
            target,
            window,
            WM_MOUSEMOVE,
            mouse_key_state(held, None),
            lparam,
            require_foreground,
        )?;
        let message = match (button, down) {
            (MouseButton::Left, true) => WM_LBUTTONDOWN,
            (MouseButton::Left, false) => WM_LBUTTONUP,
            (MouseButton::Right, true) => WM_RBUTTONDOWN,
            (MouseButton::Right, false) => WM_RBUTTONUP,
            (MouseButton::Middle, true) => WM_MBUTTONDOWN,
            (MouseButton::Middle, false) => WM_MBUTTONUP,
        };
        post_target_message(
            target,
            window,
            message,
            mouse_key_state(held, Some((button, down))),
            lparam,
            require_foreground,
        )?;
        Ok((point.x, point.y))
    }

    fn send_targeted_scroll(
        target: CocTarget,
        recorded: RecordedMousePoint,
        dx: i32,
        dy: i32,
        held: &HeldInputs,
        require_foreground: bool,
    ) -> io::Result<()> {
        let (window, client_point) = target_client_point(target, recorded, require_foreground)?;
        let mut screen_point = if recorded.window_origin.is_some() {
            client_point
        } else {
            POINT {
                x: recorded.x,
                y: recorded.y,
            }
        };
        if recorded.window_origin.is_some()
            && unsafe { ClientToScreen(window, &mut screen_point) } == 0
        {
            return Err(io::Error::last_os_error());
        }
        let lparam = pack_window_coordinates(screen_point.x, screen_point.y)?;
        for (amount, message) in [(dy, WM_MOUSEWHEEL), (dx, WM_MOUSEHWHEEL)] {
            if amount == 0 {
                continue;
            }
            let delta = amount
                .checked_mul(WHEEL_DELTA)
                .ok_or_else(|| invalid_input("scroll amount is too large"))?;
            let wparam = mouse_key_state(held, None) | (usize::from(delta as u16) << 16);
            post_target_message(target, window, message, wparam, lparam, require_foreground)?;
        }
        Ok(())
    }

    fn mouse_key_state(held: &HeldInputs, transition: Option<(MouseButton, bool)>) -> WPARAM {
        let button_is_down = |button| {
            transition
                .filter(|(changed, _)| *changed == button)
                .map(|(_, down)| down)
                .unwrap_or_else(|| held.buttons.contains_key(&button))
        };
        let mut state = 0;
        if button_is_down(MouseButton::Left) {
            state |= MK_LBUTTON;
        }
        if button_is_down(MouseButton::Right) {
            state |= MK_RBUTTON;
        }
        if button_is_down(MouseButton::Middle) {
            state |= MK_MBUTTON;
        }
        if [0x10, 0xa0, 0xa1].iter().any(|key| held.keys.contains(key)) {
            state |= MK_SHIFT;
        }
        if [0x11, 0xa2, 0xa3].iter().any(|key| held.keys.contains(key)) {
            state |= MK_CONTROL;
        }
        state
    }

    fn send_targeted_key(
        target: CocTarget,
        virtual_key: u16,
        down: bool,
        held: &HashSet<u16>,
        require_foreground: bool,
    ) -> io::Result<()> {
        let window = target_window(target, require_foreground)?;
        let alt_down = [0x12, 0xa4, 0xa5].iter().any(|key| held.contains(key));
        let system_key = alt_down || matches!(virtual_key, 0x12 | 0xa4 | 0xa5);
        let message = match (system_key, down) {
            (false, true) => WM_KEYDOWN,
            (false, false) => WM_KEYUP,
            (true, true) => WM_SYSKEYDOWN,
            (true, false) => WM_SYSKEYUP,
        };
        let scan = unsafe { MapVirtualKeyW(u32::from(virtual_key), MAPVK_VK_TO_VSC_EX) };
        let previous_state = !down || held.contains(&virtual_key);
        let mut flags = (scan & 0xff) << 16;
        if is_extended_key(virtual_key) || matches!(scan >> 8, 0xe0 | 0xe1) {
            flags |= 1 << 24;
        }
        if alt_down {
            flags |= 1 << 29;
        }
        if previous_state {
            flags |= 1 << 30;
        }
        if !down {
            flags |= 1 << 31;
        }
        post_target_message(
            target,
            window,
            message,
            usize::from(virtual_key),
            (flags | 1) as LPARAM,
            require_foreground,
        )
    }

    fn release_targeted_inputs(target: CocTarget, held: &mut HeldInputs) -> io::Result<()> {
        if held.keys.is_empty() && held.buttons.is_empty() {
            return Ok(());
        }
        release_targeted_inputs_with(
            || target_window(target, false),
            held,
            |window, key, keys| post_target_key_release(target, window, key, keys),
            |window, button, point, held| {
                let message = match button {
                    MouseButton::Left => WM_LBUTTONUP,
                    MouseButton::Right => WM_RBUTTONUP,
                    MouseButton::Middle => WM_MBUTTONUP,
                };
                post_target_message(
                    target,
                    window,
                    message,
                    mouse_key_state(held, Some((button, false))),
                    pack_window_coordinates(point.0, point.1)?,
                    false,
                )
            },
        )
    }

    fn release_targeted_inputs_with(
        target: impl FnOnce() -> io::Result<HWND>,
        held: &mut HeldInputs,
        mut release_key: impl FnMut(HWND, u16, &HashSet<u16>) -> io::Result<()>,
        mut release_button: impl FnMut(HWND, MouseButton, (i32, i32), &HeldInputs) -> io::Result<()>,
    ) -> io::Result<()> {
        if held.keys.is_empty() && held.buttons.is_empty() {
            return Ok(());
        }
        let window = target()?;
        let mut first_error = None;
        for key in key_release_order(&held.keys) {
            let result = release_key(window, key, &held.keys);
            match result {
                Ok(()) => {
                    held.keys.remove(&key);
                }
                Err(error) if first_error.is_none() => first_error = Some(error),
                Err(_) => {}
            }
        }
        for (button, point) in held
            .buttons
            .iter()
            .map(|(button, point)| (*button, *point))
            .collect::<Vec<_>>()
        {
            let result = release_button(window, button, point, held);
            match result {
                Ok(()) => {
                    held.buttons.remove(&button);
                }
                Err(error) if first_error.is_none() => first_error = Some(error),
                Err(_) => {}
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn post_target_key_release(
        target: CocTarget,
        window: HWND,
        virtual_key: u16,
        held: &HashSet<u16>,
    ) -> io::Result<()> {
        let alt_down = [0x12, 0xa4, 0xa5].iter().any(|key| held.contains(key));
        let system_key = alt_down || matches!(virtual_key, 0x12 | 0xa4 | 0xa5);
        let message = if system_key { WM_SYSKEYUP } else { WM_KEYUP };
        let scan = unsafe { MapVirtualKeyW(u32::from(virtual_key), MAPVK_VK_TO_VSC_EX) };
        let mut flags = ((scan & 0xff) << 16) | 1 | (1 << 30) | (1 << 31);
        if is_extended_key(virtual_key) || matches!(scan >> 8, 0xe0 | 0xe1) {
            flags |= 1 << 24;
        }
        if alt_down {
            flags |= 1 << 29;
        }
        post_target_message(
            target,
            window,
            message,
            usize::from(virtual_key),
            flags as LPARAM,
            false,
        )
    }

    fn key_release_order(keys: &HashSet<u16>) -> Vec<u16> {
        let mut keys = keys.iter().copied().collect::<Vec<_>>();
        keys.sort_by_key(|key| matches!(key, 0x12 | 0xa4 | 0xa5));
        keys
    }

    fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        mutex.lock().unwrap_or_else(PoisonError::into_inner)
    }

    #[cfg(test)]
    mod tests {
        use super::{
            CocTarget, HeldInputs, MouseButton, NativeEvent, PlaybackFailure, PlaybackPauseResult,
            RecordedMousePoint, Recording, ReplayWait, Shared, append_recording_key_event,
            append_recording_mouse_button_event, current_foreground_coc_window,
            finalized_playback_event, finish_guard_release, invalid_target_window, is_extended_key,
            join_playback_task, key_release_order, pack_window_coordinates, parse_steps,
            pause_after_release, playback_completion_event, playback_target, release_guard_on_drop,
            release_targeted_inputs_with, run_playback, send_targeted_mouse_move,
            shift_replay_timing, signal_playback_resume, single_background_target,
            target_is_foreground, target_visibility_allowed, wait_for_replay_step,
        };
        use std::cell::{Cell, RefCell};
        use std::collections::{HashMap, HashSet};
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        use std::time::{Duration, Instant};
        use tokio::sync::{oneshot, watch};

        #[test]
        fn pynput_media_keys_use_extended_key_flags() {
            for key in 0xad..=0xb3 {
                assert!(is_extended_key(key));
            }
        }

        #[test]
        fn focus_loss_releases_every_held_key_and_mouse_button() {
            let mut held = HeldInputs {
                keys: HashSet::from([0x41]),
                buttons: HashMap::from([(MouseButton::Left, (10, 20))]),
            };
            let mut released_keys = Vec::new();
            let mut released_buttons = Vec::new();

            release_targeted_inputs_with(
                || Ok(std::ptr::null_mut()),
                &mut held,
                |_, key, _| {
                    released_keys.push(key);
                    Ok(())
                },
                |_, button, _, _| {
                    released_buttons.push(button);
                    Ok(())
                },
            )
            .unwrap();

            assert_eq!(released_keys, vec![0x41]);
            assert_eq!(released_buttons, vec![MouseButton::Left]);
            assert!(held.keys.is_empty());
            assert!(held.buttons.is_empty());
        }

        #[test]
        fn hidden_window_is_allowed_for_release_but_not_dispatch() {
            assert!(target_visibility_allowed(false, false, false));
            assert!(target_visibility_allowed(false, true, false));
            assert!(!target_visibility_allowed(true, false, true));
            assert!(!target_visibility_allowed(true, true, false));
        }

        #[test]
        fn background_target_selection_fails_closed_on_missing_or_ambiguous_windows() {
            let mut candidates = Vec::new();
            assert_eq!(
                single_background_target(&mut candidates)
                    .err()
                    .unwrap()
                    .kind(),
                std::io::ErrorKind::NotFound
            );

            candidates.push(CocTarget {
                window: 1,
                process_id: 2,
                process_created_at: 3,
            });
            assert_eq!(single_background_target(&mut candidates).unwrap().window, 1);

            candidates.extend([
                CocTarget {
                    window: 4,
                    process_id: 5,
                    process_created_at: 6,
                },
                CocTarget {
                    window: 7,
                    process_id: 8,
                    process_created_at: 9,
                },
            ]);
            assert_eq!(
                single_background_target(&mut candidates)
                    .err()
                    .unwrap()
                    .kind(),
                std::io::ErrorKind::NotFound
            );
        }

        #[tokio::test]
        #[ignore = "requires Clash of Clans open and another application in the foreground; posts one mouse move only"]
        async fn live_coc_background_target_accepts_mouse_move_and_noop_replay() {
            assert!(current_foreground_coc_window().is_none());
            let target = playback_target(true).unwrap();
            assert!(!target_is_foreground(target));
            let mut client = super::RECT {
                left: 0,
                top: 0,
                right: 0,
                bottom: 0,
            };
            assert_ne!(
                unsafe { super::GetClientRect(target.window as super::HWND, &mut client) },
                0
            );
            send_targeted_mouse_move(
                target,
                RecordedMousePoint {
                    x: client.right / 2,
                    y: client.bottom / 2,
                    window_origin: Some((0, 0)),
                },
                &HeldInputs::default(),
                false,
            )
            .unwrap();
            let steps = parse_steps(&[serde_json::json!({
                "t": 0.25,
                "type": "nop",
                "data": {}
            })])
            .unwrap();
            let (_cancel, cancel_rx) = watch::channel(false);
            let (_resume, resume_rx) = watch::channel(());
            let (events, _received) = tokio::sync::mpsc::unbounded_channel();
            let shared = Arc::new(Shared {
                recording: std::sync::Mutex::new(None),
                coc_identity_valid: std::sync::atomic::AtomicBool::new(false),
                coc_window: std::sync::atomic::AtomicIsize::new(0),
                coc_process_id: std::sync::atomic::AtomicU32::new(0),
                coc_process_created_at: std::sync::atomic::AtomicU64::new(0),
                shortcuts: std::sync::Mutex::new(
                    super::super::parse_shortcuts(&super::super::ShortcutSettings::default())
                        .unwrap(),
                ),
                injected_keys: std::sync::Mutex::new(HashSet::new()),
                suppressed_hotkeys: std::sync::Mutex::new(HashMap::new()),
                events,
                playback: super::AsyncMutex::new(None),
                next_playback_id: std::sync::atomic::AtomicU64::new(1),
            });

            let terminal =
                run_playback(shared, 1, steps, false, target, true, cancel_rx, resume_rx).await;

            assert_eq!(terminal, NativeEvent::PlaybackEnded { playback_id: 1 });
        }

        #[test]
        fn invalid_target_keeps_held_inputs_and_never_posts_releases() {
            let mut held = HeldInputs {
                keys: HashSet::from([0x41]),
                buttons: HashMap::from([(MouseButton::Left, (10, 20))]),
            };
            let posted = Cell::new(false);

            let result = release_targeted_inputs_with(
                || Err(invalid_target_window()),
                &mut held,
                |_, _, _| {
                    posted.set(true);
                    Ok(())
                },
                |_, _, _, _| {
                    posted.set(true);
                    Ok(())
                },
            );

            assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::NotFound);
            assert!(!posted.get());
            assert_eq!(held.keys, HashSet::from([0x41]));
            assert_eq!(held.buttons, HashMap::from([(MouseButton::Left, (10, 20))]));
        }

        #[test]
        fn final_release_succeeds_without_resolving_a_destroyed_target_when_nothing_is_held() {
            let mut held = HeldInputs::default();
            let target_checked = Cell::new(false);

            release_targeted_inputs_with(
                || {
                    target_checked.set(true);
                    Err(invalid_target_window())
                },
                &mut held,
                |_, _, _| panic!("no key is held"),
                |_, _, _, _| panic!("no mouse button is held"),
            )
            .unwrap();

            assert!(!target_checked.get());
        }

        #[test]
        fn release_failure_preserves_only_inputs_that_still_need_release() {
            let mut held = HeldInputs {
                keys: HashSet::from([0x41]),
                buttons: HashMap::from([(MouseButton::Left, (10, 20))]),
            };

            let result = release_targeted_inputs_with(
                || Ok(std::ptr::null_mut()),
                &mut held,
                |_, _, _| Ok(()),
                |_, _, _, _| Err(std::io::Error::other("release failed")),
            );

            assert!(result.is_err());
            assert!(held.keys.is_empty());
            assert_eq!(held.buttons, HashMap::from([(MouseButton::Left, (10, 20))]));
        }

        #[test]
        fn final_release_retries_remaining_inputs_once_and_disarms_drop_retry() {
            let mut armed = true;
            let mut attempts = 0;
            let order = RefCell::new(Vec::new());
            let mut held = HeldInputs {
                keys: HashSet::from([0x41, 0x42]),
                buttons: HashMap::new(),
            };
            let mut released_keys = Vec::new();

            let event = finalized_playback_event(
                8,
                false,
                false,
                None,
                None,
                || {
                    finish_guard_release(&mut armed, || {
                        attempts += 1;
                        order.borrow_mut().push("release");
                        release_targeted_inputs_with(
                            || Ok(std::ptr::null_mut()),
                            &mut held,
                            |_, key, _| {
                                released_keys.push(key);
                                if key == 0x42 && attempts == 1 {
                                    Err(std::io::Error::other("transient release failure"))
                                } else {
                                    Ok(())
                                }
                            },
                            |_, _, _, _| Ok(()),
                        )
                    })
                },
                |event| {
                    assert_eq!(event, &NativeEvent::PlaybackEnded { playback_id: 8 });
                    assert_eq!(*order.borrow(), vec!["release", "release"]);
                    order.borrow_mut().push("publish");
                },
            );
            release_guard_on_drop(armed, || {
                attempts += 1;
                Ok(())
            });

            assert_eq!(event, NativeEvent::PlaybackEnded { playback_id: 8 });
            assert!(!armed);
            assert_eq!(attempts, 2);
            assert_eq!(released_keys.iter().filter(|key| **key == 0x41).count(), 1);
            assert_eq!(released_keys.iter().filter(|key| **key == 0x42).count(), 2);
            assert!(held.keys.is_empty());
            assert_eq!(*order.borrow(), vec!["release", "release", "publish"]);
        }

        #[test]
        fn persistent_final_release_failure_is_reported_after_two_attempts() {
            let order = RefCell::new(Vec::new());
            let mut armed = true;
            let mut attempts = 0;
            let mut held = HeldInputs {
                keys: HashSet::from([0x41]),
                buttons: HashMap::new(),
            };
            let event = finalized_playback_event(
                9,
                false,
                false,
                None,
                None,
                || {
                    finish_guard_release(&mut armed, || {
                        attempts += 1;
                        order.borrow_mut().push("release");
                        release_targeted_inputs_with(
                            || Err(invalid_target_window()),
                            &mut held,
                            |_, _, _| panic!("destroyed target must not receive a release"),
                            |_, _, _, _| panic!("destroyed target must not receive a release"),
                        )
                    })
                },
                |event| {
                    assert_eq!(
                        event,
                        &NativeEvent::PlaybackFailed {
                            playback_id: 9,
                            reason: PlaybackFailure::Release,
                            detail: None,
                        }
                    );
                    order.borrow_mut().push("publish");
                },
            );
            release_guard_on_drop(armed, || {
                attempts += 1;
                Ok(())
            });

            assert_eq!(*order.borrow(), vec!["release", "release", "publish"]);
            assert_eq!(attempts, 2);
            assert!(!armed);
            assert_eq!(held.keys, HashSet::from([0x41]));
            assert_eq!(
                event,
                NativeEvent::PlaybackFailed {
                    playback_id: 9,
                    reason: PlaybackFailure::Release,
                    detail: None,
                }
            );
        }

        #[tokio::test]
        async fn joined_playback_task_returns_its_terminal_event() {
            let expected = NativeEvent::PlaybackEnded { playback_id: 23 };
            let returned = expected.clone();
            let mut task = tokio::spawn(async move { returned });

            assert_eq!(join_playback_task(&mut task).await.unwrap(), expected);
        }

        #[test]
        fn targeted_release_keeps_alt_until_system_keys_are_released() {
            let held = HashSet::from([0x12, 0x73]);
            assert_eq!(key_release_order(&held), vec![0x73, 0x12]);
        }

        #[test]
        fn targeted_mouse_coordinates_use_signed_windows_packing() {
            let packed = pack_window_coordinates(320, 240).unwrap();
            assert_eq!(packed, ((240_u32 << 16) | 320) as isize);
            assert!(pack_window_coordinates(32_768, 0).is_err());
        }

        #[tokio::test]
        async fn foreground_only_playback_pauses_when_focus_is_lost() {
            let (_cancel_tx, mut cancel) = watch::channel(false);
            let result = wait_for_replay_step(
                tokio::time::Instant::now() + Duration::from_secs(10),
                &mut cancel,
                true,
                || true,
                || false,
            )
            .await;

            assert!(matches!(result, ReplayWait::ForegroundLost));
        }

        #[tokio::test]
        async fn background_playback_ignores_focus_loss_but_stops_when_target_disappears() {
            let (_cancel_tx, mut cancel) = watch::channel(false);
            let result = wait_for_replay_step(
                tokio::time::Instant::now(),
                &mut cancel,
                false,
                || true,
                || false,
            )
            .await;
            assert!(matches!(result, ReplayWait::Ready));

            let (_cancel_tx, mut cancel) = watch::channel(false);
            let result = wait_for_replay_step(
                tokio::time::Instant::now() + Duration::from_secs(10),
                &mut cancel,
                false,
                || false,
                || false,
            )
            .await;
            assert!(matches!(result, ReplayWait::TargetLost));
        }

        #[tokio::test]
        async fn playback_stays_paused_until_foreground_is_verified() {
            let (cancel_tx, mut cancel) = watch::channel(false);
            let (resume_tx, mut resume) = watch::channel(());
            let (published_tx, published_rx) = oneshot::channel();
            let released = Arc::new(AtomicUsize::new(0));
            let worker_released = Arc::clone(&released);
            let publish_released = Arc::clone(&released);
            let injected = Arc::new(AtomicUsize::new(0));
            let worker_injected = Arc::clone(&injected);

            let worker = tokio::spawn(async move {
                let result = pause_after_release(
                    &mut cancel,
                    &mut resume,
                    || true,
                    move || {
                        worker_released.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    },
                    move || {
                        assert_eq!(publish_released.load(Ordering::SeqCst), 1);
                        published_tx.send(()).unwrap();
                    },
                )
                .await;
                if matches!(result, PlaybackPauseResult::Resumed(_)) {
                    worker_injected.fetch_add(1, Ordering::SeqCst);
                }
                result
            });

            published_rx.await.unwrap();
            assert_eq!(released.load(Ordering::SeqCst), 1);
            assert_eq!(injected.load(Ordering::SeqCst), 0);
            assert_eq!(
                signal_playback_resume(&resume_tx, false)
                    .unwrap_err()
                    .kind(),
                std::io::ErrorKind::PermissionDenied
            );
            assert_eq!(injected.load(Ordering::SeqCst), 0);
            assert!(!worker.is_finished());

            signal_playback_resume(&resume_tx, true).unwrap();
            assert!(matches!(
                worker.await.unwrap(),
                PlaybackPauseResult::Resumed(_)
            ));
            assert_eq!(injected.load(Ordering::SeqCst), 1);
            drop(cancel_tx);
        }

        #[tokio::test]
        async fn stop_cancels_a_paused_playback_without_resuming_it() {
            let (cancel_tx, mut cancel) = watch::channel(false);
            let (_resume_tx, mut resume) = watch::channel(());
            let (published_tx, published_rx) = oneshot::channel();
            let injected = Arc::new(AtomicUsize::new(0));
            let worker_injected = Arc::clone(&injected);
            let worker = tokio::spawn(async move {
                let result = pause_after_release(
                    &mut cancel,
                    &mut resume,
                    || true,
                    || Ok(()),
                    move || published_tx.send(()).unwrap(),
                )
                .await;
                if matches!(result, PlaybackPauseResult::Resumed(_)) {
                    worker_injected.fetch_add(1, Ordering::SeqCst);
                }
                result
            });

            published_rx.await.unwrap();
            cancel_tx.send_replace(true);

            assert!(matches!(
                worker.await.unwrap(),
                PlaybackPauseResult::Cancelled
            ));
            assert_eq!(injected.load(Ordering::SeqCst), 0);
        }

        #[tokio::test]
        async fn paused_playback_ends_if_coc_target_disappears_without_injecting() {
            let (_cancel_tx, mut cancel) = watch::channel(false);
            let (_resume_tx, mut resume) = watch::channel(());
            let (published_tx, published_rx) = oneshot::channel();
            let target_available = Arc::new(std::sync::atomic::AtomicBool::new(true));
            let worker_target_available = Arc::clone(&target_available);
            let injected = Arc::new(AtomicUsize::new(0));
            let worker_injected = Arc::clone(&injected);

            let worker = tokio::spawn(async move {
                let result = pause_after_release(
                    &mut cancel,
                    &mut resume,
                    move || worker_target_available.load(Ordering::SeqCst),
                    || Ok(()),
                    move || published_tx.send(()).unwrap(),
                )
                .await;
                if matches!(result, PlaybackPauseResult::Resumed(_)) {
                    worker_injected.fetch_add(1, Ordering::SeqCst);
                }
                result
            });

            published_rx.await.unwrap();
            target_available.store(false, Ordering::SeqCst);

            assert!(matches!(
                worker.await.unwrap(),
                PlaybackPauseResult::TargetLost
            ));
            assert_eq!(injected.load(Ordering::SeqCst), 0);
        }

        #[tokio::test]
        async fn release_failure_never_publishes_the_paused_state() {
            let (_cancel_tx, mut cancel) = watch::channel(false);
            let (_resume_tx, mut resume) = watch::channel(());
            let mut held = HeldInputs {
                keys: HashSet::from([0x41]),
                buttons: HashMap::new(),
            };
            let release_attempts = Cell::new(0);
            let mut published = false;
            let result = pause_after_release(
                &mut cancel,
                &mut resume,
                || true,
                || {
                    release_targeted_inputs_with(
                        || Ok(std::ptr::null_mut()),
                        &mut held,
                        |_, _, _| {
                            release_attempts.set(release_attempts.get() + 1);
                            Err(std::io::Error::other("release failed"))
                        },
                        |_, _, _, _| {
                            release_attempts.set(release_attempts.get() + 1);
                            Err(std::io::Error::other("release failed"))
                        },
                    )
                },
                || published = true,
            )
            .await;

            assert!(matches!(&result, PlaybackPauseResult::ReleaseFailed));
            assert!(!published);
            assert_eq!(release_attempts.get(), 1);
            assert_eq!(held.keys, HashSet::from([0x41]));
            let release_failed = matches!(&result, PlaybackPauseResult::ReleaseFailed);
            assert_eq!(
                playback_completion_event(5, false, release_failed, None, None),
                NativeEvent::PlaybackFailed {
                    playback_id: 5,
                    reason: PlaybackFailure::Release,
                    detail: None,
                }
            );
        }

        #[test]
        fn replay_deadline_keeps_its_remaining_delay_after_pause() {
            let started = tokio::time::Instant::now();
            let pause_at = started + Duration::from_secs(2);
            let paused_for = Duration::from_secs(20);
            let mut deadline = started + Duration::from_secs(7);
            let mut shifted_start = started;

            shift_replay_timing(&mut shifted_start, &mut deadline, paused_for);

            assert_eq!(deadline - (pause_at + paused_for), Duration::from_secs(5));
            assert_eq!(shifted_start, started + paused_for);
            assert_eq!(deadline, started + Duration::from_secs(7) + paused_for);
        }

        fn recording_for_test(started: Instant) -> Recording {
            Recording {
                captures_at: started,
                last_event_at: None,
                steps: Vec::new(),
                held_keys: HashSet::new(),
                held_buttons: HashMap::new(),
                overflowed: false,
                beeped: false,
            }
        }

        #[test]
        fn key_release_is_recorded_after_focus_leaves_coc() {
            let started = Instant::now();
            let mut recording = recording_for_test(started);

            assert!(append_recording_key_event(
                &mut recording,
                "space",
                true,
                true,
                started + Duration::from_millis(1),
            ));
            assert!(append_recording_key_event(
                &mut recording,
                "space",
                false,
                false,
                started + Duration::from_millis(10),
            ));
            assert!(!append_recording_key_event(
                &mut recording,
                "space",
                false,
                false,
                started + Duration::from_millis(11),
            ));
            assert_eq!(recording.steps[0]["type"], "key_down");
            assert_eq!(recording.steps[1]["type"], "key_up");
            assert!(recording.held_keys.is_empty());
        }

        #[test]
        fn mouse_release_outside_coc_uses_the_captured_press_position() {
            let started = Instant::now();
            let mut recording = recording_for_test(started);
            let point = RecordedMousePoint {
                x: 1240,
                y: 790,
                window_origin: Some((1000, 500)),
            };

            assert!(append_recording_mouse_button_event(
                &mut recording,
                MouseButton::Left,
                true,
                Some(point),
                started + Duration::from_millis(1),
            ));
            assert!(append_recording_mouse_button_event(
                &mut recording,
                MouseButton::Left,
                false,
                None,
                started + Duration::from_millis(10),
            ));
            assert_eq!(recording.steps[0]["data"]["action"], "down");
            assert_eq!(recording.steps[1]["data"]["action"], "up");
            assert_eq!(recording.steps[1]["data"]["x"], 1240);
            assert_eq!(recording.steps[1]["data"]["y"], 790);
            assert!(recording.held_buttons.is_empty());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_title_read_distinguishes_empty_title_from_api_error() {
        assert_eq!(decode_window_title(0, 0, &[0]).unwrap(), "");

        let error = decode_window_title(0, 5, &[0]).unwrap_err();
        assert_eq!(error.raw_os_error(), Some(5));

        assert_eq!(decode_window_title(3, 0, &[67, 111, 67]).unwrap(), "CoC");
    }

    #[test]
    fn recorded_mouse_origin_makes_coordinates_stable_when_coc_moves() {
        let point = RecordedMousePoint {
            x: 1_240,
            y: 790,
            window_origin: Some((1_000, 500)),
        };
        assert_eq!(point.client_position().unwrap(), Some((240, 290)));

        let old_macro_point = RecordedMousePoint {
            window_origin: None,
            ..point
        };
        assert_eq!(old_macro_point.client_position().unwrap(), None);
    }

    #[test]
    fn mouse_event_parser_keeps_legacy_screen_coordinates_and_reads_window_origin() {
        let legacy = serde_json::json!({"x": 500, "y": 300});
        assert_eq!(
            read_mouse_point(legacy.as_object().unwrap()).unwrap(),
            RecordedMousePoint {
                x: 500,
                y: 300,
                window_origin: None,
            }
        );

        let anchored = serde_json::json!({
            "x": 1240,
            "y": 790,
            "windowOrigin": {"x": 1000, "y": 500}
        });
        assert_eq!(
            read_mouse_point(anchored.as_object().unwrap())
                .unwrap()
                .client_position()
                .unwrap(),
            Some((240, 290))
        );
        assert!(
            read_mouse_point(
                serde_json::json!({"x": 1, "y": 2, "windowOrigin": "invalid"})
                    .as_object()
                    .unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn playback_failure_detail_names_the_step_and_explains_outside_coordinates() {
        assert_eq!(
            playback_input_detail(
                3,
                "clic de souris",
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "recorded mouse position is outside the Clash of Clans client area"
                )
            ),
            "Étape 3 (clic de souris) : la position souris enregistrée est hors de la zone de jeu CoC; réenregistrez la macro avec Clash of Clans au premier plan"
        );
    }

    #[test]
    fn only_outside_client_positions_are_ignored_for_mouse_moves() {
        assert!(
            ignore_outside_mouse_move(io::Error::new(
                io::ErrorKind::InvalidInput,
                OUTSIDE_COC_CLIENT_AREA
            ))
            .is_ok()
        );
        let out_of_range = ignore_outside_mouse_move(io::Error::new(
            io::ErrorKind::InvalidInput,
            "mouse x coordinate is out of range",
        ))
        .unwrap_err();
        assert_eq!(out_of_range.kind(), io::ErrorKind::InvalidInput);
        let permission_error = ignore_outside_mouse_move(io::Error::new(
            io::ErrorKind::PermissionDenied,
            OUTSIDE_COC_CLIENT_AREA,
        ))
        .unwrap_err();
        assert_eq!(permission_error.kind(), io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn foreground_guard_rejects_input_without_the_game_focus() {
        assert_eq!(
            ensure_coc_foreground(true, false).unwrap_err().kind(),
            io::ErrorKind::PermissionDenied
        );
        assert!(ensure_coc_foreground(true, true).is_ok());
        assert!(ensure_coc_foreground(false, false).is_ok());
    }

    #[test]
    fn foreground_activation_waits_for_confirmation_and_fails_when_denied() {
        use std::cell::Cell;

        let checks = Cell::new(0);
        let waits = Cell::new(0);
        wait_for_coc_foreground_with(
            || {
                checks.set(checks.get() + 1);
                Ok(checks.get() == 3)
            },
            || waits.set(waits.get() + 1),
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(checks.get(), 3);
        assert_eq!(waits.get(), 2);

        let denied = wait_for_coc_foreground_with(
            || Ok(false),
            || {},
            Instant::now() - Duration::from_millis(1),
        )
        .unwrap_err();
        assert_eq!(denied.kind(), io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn coc_window_identity_requires_game_title_and_google_play_process() {
        assert!(is_coc_window_identity(
            "Clash of Clans - qurnt12",
            r"C:\Program Files\Google\Play Games\crosvm.exe"
        ));
        assert!(!is_coc_window_identity(
            "Google Play Games",
            r"C:\Program Files\Google\Play Games\crosvm.exe"
        ));
        assert!(!is_coc_window_identity(
            "Clash of Clans - qurnt12",
            r"C:\Games\other.exe"
        ));
    }

    #[test]
    fn process_instance_requires_the_captured_pid_and_creation_time() {
        assert!(same_process_instance(41, 100, 41, 100));
        assert!(!same_process_instance(41, 100, 41, 101));
        assert!(!same_process_instance(41, 100, 42, 100));
        assert!(!same_process_instance(0, 100, 0, 100));
        assert!(!same_process_instance(41, 0, 41, 0));
    }

    #[test]
    fn playback_completion_distinguishes_cancellation_from_injection_failures() {
        assert_eq!(
            playback_completion_event(4, false, false, None, None),
            NativeEvent::PlaybackEnded { playback_id: 4 }
        );
        assert_eq!(
            playback_completion_event(
                4,
                true,
                false,
                None,
                Some("Étape 2 (clic de souris) : test".into()),
            ),
            NativeEvent::PlaybackFailed {
                playback_id: 4,
                reason: PlaybackFailure::Input,
                detail: Some("Étape 2 (clic de souris) : test".into()),
            }
        );
        assert_eq!(
            playback_completion_event(4, false, true, None, None),
            NativeEvent::PlaybackFailed {
                playback_id: 4,
                reason: PlaybackFailure::Release,
                detail: None,
            }
        );
        assert_eq!(
            playback_completion_event(
                4,
                true,
                true,
                None,
                Some("Étape 2 (clic de souris) : test".into()),
            ),
            NativeEvent::PlaybackFailed {
                playback_id: 4,
                reason: PlaybackFailure::InputAndRelease,
                detail: Some("Étape 2 (clic de souris) : test".into()),
            }
        );
        assert_eq!(
            playback_completion_event(4, false, false, Some(PlaybackStopReason::TargetLost), None,),
            NativeEvent::PlaybackStopped {
                playback_id: 4,
                reason: PlaybackStopReason::TargetLost,
            }
        );
        assert!(matches!(
            playback_completion_event(4, false, true, Some(PlaybackStopReason::TargetLost), None,),
            NativeEvent::PlaybackFailed {
                reason: PlaybackFailure::Release,
                detail: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn input_failure_does_not_emit_a_completed_cycle_event() {
        assert_eq!(playback_cycle_event(4, true, false), None);
        assert_eq!(playback_cycle_event(4, false, true), None);
        assert_eq!(
            playback_cycle_event(4, false, false),
            Some(NativeEvent::Cycle { playback_id: 4 })
        );
    }

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
    fn trimmed_capture_still_releases_keys_and_mouse_buttons() {
        let capture = finish_recording(
            vec![
                event_value(0.0, "key_down", json!({"key":"a"})),
                event_value(
                    1.0,
                    "mouse_click",
                    json!({"x":100,"y":200,"windowOrigin":{"x":50,"y":60},"button":"left","action":"down"}),
                ),
                event_value(4.0, "nop", json!({})),
            ],
            false,
        );

        assert_eq!(capture.steps[0]["type"], "key_down");
        assert_eq!(capture.steps[1]["data"]["action"], "down");
        assert_eq!(capture.steps[2]["type"], "nop");
        assert_eq!(capture.steps[3]["type"], "key_up");
        assert_eq!(capture.steps[3]["data"]["key"], "a");
        assert_eq!(capture.steps[4]["data"]["action"], "up");
        assert_eq!(capture.steps[4]["data"]["x"], 100);
        assert_eq!(capture.steps[4]["data"]["y"], 200);
    }

    #[test]
    fn synthetic_releases_stay_within_the_capture_limit() {
        let mut steps = vec![
            event_value(0.0, "key_down", json!({"key":"a"})),
            event_value(0.1, "nop", json!({})),
            event_value(0.1, "nop", json!({})),
        ];

        let discarded_events = balance_recorded_inputs(&mut steps, 3);

        assert!(discarded_events);
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0]["type"], "key_down");
        assert_eq!(steps[1]["type"], "nop");
        assert_eq!(steps[2]["type"], "key_up");
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
    fn replay_parser_preserves_click_and_scroll_screen_coordinates() {
        let values = [
            event_value(
                0.0,
                "mouse_click",
                json!({"x":-1920,"y":40,"button":"left","action":"down"}),
            ),
            event_value(0.5, "scroll", json!({"x":2500,"y":-10,"dx":0,"dy":1})),
        ];
        let parsed = parse_steps(&values).unwrap();

        assert!(matches!(
            &parsed[0].event,
            ReplayEvent::Click {
                point: RecordedMousePoint {
                    x: -1920,
                    y: 40,
                    window_origin: None,
                },
                button: MouseButton::Left,
                down: true,
            }
        ));
        assert!(matches!(
            &parsed[1].event,
            ReplayEvent::Scroll {
                point: RecordedMousePoint {
                    x: 2500,
                    y: -10,
                    window_origin: None,
                },
                dx: 0,
                dy: 1,
            }
        ));
    }

    #[test]
    fn recording_beep_follows_the_first_accepted_event_only() {
        let start = Instant::now();
        let captures_at = start + RECORDING_PREPARATION;
        let mut steps = Vec::new();
        let mut last_event_at = None;
        let mut overflowed = false;
        let mut beeped = false;

        let before_preparation = append_recording_event(
            &mut steps,
            &mut last_event_at,
            &mut overflowed,
            captures_at,
            captures_at - Duration::from_millis(1),
            MAX_CAPTURE_EVENTS,
            "key_down",
            json!({"key":"a"}),
        );
        assert!(!before_preparation);
        assert!(!should_play_recording_beep(&mut beeped, before_preparation));

        let first_accepted = append_recording_event(
            &mut steps,
            &mut last_event_at,
            &mut overflowed,
            captures_at,
            captures_at,
            MAX_CAPTURE_EVENTS,
            "mouse_click",
            json!({"button":"left"}),
        );
        assert!(first_accepted);
        assert!(should_play_recording_beep(&mut beeped, first_accepted));

        let next_accepted = append_recording_event(
            &mut steps,
            &mut last_event_at,
            &mut overflowed,
            captures_at,
            captures_at + Duration::from_millis(1),
            MAX_CAPTURE_EVENTS,
            "key_up",
            json!({"key":"a"}),
        );
        assert!(next_accepted);
        assert!(!should_play_recording_beep(&mut beeped, next_accepted));
        assert_eq!(steps.len(), 2);
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
