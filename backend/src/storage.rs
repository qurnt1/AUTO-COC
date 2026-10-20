use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha1::{Digest, Sha1};
use sha2::Sha256;
use thiserror::Error;

use crate::types::{MacroFile, MacroSummary, Settings};

const MAX_MACRO_BYTES: u64 = 32 * 1024 * 1024;
const MAX_PLACEHOLDER_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_PLACEHOLDER_MACRO_BYTES: u64 = 1024 * 1024;
const MAX_STEPS: usize = 250_000;
const LEGACY_TYPES: [&str; 6] = [
    "mouse_move",
    "mouse_click",
    "scroll",
    "key_down",
    "key_up",
    "nop",
];
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("macro not found")]
    NotFound,
    #[error("macro name is invalid")]
    InvalidName,
    #[error("macro name already exists")]
    Conflict,
    #[error("cannot read macro")]
    Unreadable,
    #[error("operation could not be persisted")]
    Io(#[from] io::Error),
    #[error("stored data is invalid")]
    InvalidData,
}

#[derive(Clone)]
pub struct Store {
    root: PathBuf,
    settings_path: PathBuf,
    macros_path: PathBuf,
    state: PersistedSettings,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PersistedSettings {
    schema_version: u32,
    settings: Settings,
    selected_macro: Option<String>,
    onboarding_complete: bool,
    #[serde(default)]
    present: BTreeSet<String>,
    #[serde(default)]
    telegram_owner: Option<TelegramOwner>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TelegramOwner {
    chat_id: i64,
    user_id: i64,
}

#[derive(Default, Serialize, Deserialize)]
struct PlaceholderManifest {
    files: BTreeMap<String, String>,
}

impl Default for PersistedSettings {
    fn default() -> Self {
        Self {
            schema_version: 1,
            settings: Settings::default(),
            selected_macro: None,
            onboarding_complete: false,
            present: BTreeSet::new(),
            telegram_owner: None,
        }
    }
}

impl Store {
    pub fn open() -> Result<Self, StoreError> {
        let root = std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .ok_or(StoreError::InvalidData)?
            .join("AUTO-COC");
        Self::open_inner(root)
    }

    #[cfg(test)]
    pub fn open_at(root: impl Into<PathBuf>) -> Result<Self, StoreError> {
        Self::open_inner(root.into())
    }

    fn open_inner(root: PathBuf) -> Result<Self, StoreError> {
        let macros_path = root.join("macros");
        fs::create_dir_all(&macros_path)?;
        remove_untouched_placeholders(&root, &macros_path)?;
        let settings_path = root.join("settings.json");
        let mut state = if settings_path.exists() {
            read_settings(&settings_path)?
        } else {
            PersistedSettings::default()
        };

        let was_configured = state.settings.telegram.token_configured;
        let (token_configured, status) = match load_token(&root) {
            Ok(Some(_)) => (true, crate::types::TelegramStatus::Disconnected),
            Ok(None) if was_configured => (false, crate::types::TelegramStatus::Error),
            Ok(None) => (false, crate::types::TelegramStatus::NotConfigured),
            Err(StoreError::InvalidData | StoreError::Io(_)) => {
                (false, crate::types::TelegramStatus::Error)
            }
            Err(error) => return Err(error),
        };
        if state.settings.telegram.token_configured != token_configured
            || state.settings.telegram.status != status
        {
            state.settings.telegram.token_configured = token_configured;
            state.settings.telegram.status = status;
            atomic_write_json(&settings_path, &state)?;
        }

        if !settings_path.exists() {
            atomic_write_json(&settings_path, &state)?;
        }
        // A stale selection should not leave the app in a broken state.
        if let Some(selected) = state.selected_macro.as_deref()
            && !macro_path(&macros_path, selected).is_ok_and(|path| path.exists())
        {
            state.selected_macro = None;
            atomic_write_json(&settings_path, &state)?;
        }

        Ok(Self {
            root,
            settings_path,
            macros_path,
            state,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn settings(&self) -> &Settings {
        &self.state.settings
    }
    pub(crate) fn settings_mut(&mut self) -> &mut Settings {
        &mut self.state.settings
    }
    pub fn selected_macro(&self) -> Option<&str> {
        self.state.selected_macro.as_deref()
    }
    pub fn onboarding_complete(&self) -> bool {
        self.state.onboarding_complete
    }
    pub fn telegram_owner(&self) -> Option<(i64, i64)> {
        self.state
            .telegram_owner
            .as_ref()
            .map(|owner| (owner.chat_id, owner.user_id))
    }
    pub fn set_telegram_owner(&mut self, chat_id: i64, user_id: i64) {
        self.state.telegram_owner = Some(TelegramOwner { chat_id, user_id });
    }
    pub fn clear_telegram_owner(&mut self) {
        self.state.telegram_owner = None;
    }
    pub(crate) fn persist(&self) -> Result<(), StoreError> {
        self.persist_settings()
    }

    pub fn set_loop(&mut self, value: bool) -> Result<(), StoreError> {
        let old_value = self.state.settings.loop_playback;
        let was_present = self.state.present.contains("loop");
        self.state.settings.loop_playback = value;
        self.state.present.insert("loop".into());
        if let Err(error) = self.persist_settings() {
            self.state.settings.loop_playback = old_value;
            if !was_present {
                self.state.present.remove("loop");
            }
            return Err(error);
        }
        Ok(())
    }

    pub fn update_settings(
        &mut self,
        loop_playback: Option<bool>,
        coc_path: Option<String>,
        require_coc_foreground: Option<bool>,
    ) -> Result<(), StoreError> {
        if coc_path
            .as_ref()
            .is_some_and(|value| value.len() > 2048 || value.contains('\0'))
        {
            return Err(StoreError::InvalidData);
        }
        let old_loop_playback = self.state.settings.loop_playback;
        let old_coc_path = self.state.settings.coc_path.clone();
        let old_require_coc_foreground = self.state.settings.require_coc_foreground;
        let had_loop = self.state.present.contains("loop");
        let had_coc_path = self.state.present.contains("cocPath");
        let had_require_coc_foreground = self.state.present.contains("requireCocForeground");
        if let Some(value) = loop_playback {
            self.state.settings.loop_playback = value;
            self.state.present.insert("loop".into());
        }
        if let Some(value) = coc_path {
            self.state.settings.coc_path = value;
            self.state.present.insert("cocPath".into());
        }
        if let Some(value) = require_coc_foreground {
            self.state.settings.require_coc_foreground = value;
            self.state.present.insert("requireCocForeground".into());
        }
        if let Err(error) = self.persist_settings() {
            self.state.settings.loop_playback = old_loop_playback;
            self.state.settings.coc_path = old_coc_path;
            self.state.settings.require_coc_foreground = old_require_coc_foreground;
            if !had_loop {
                self.state.present.remove("loop");
            }
            if !had_coc_path {
                self.state.present.remove("cocPath");
            }
            if !had_require_coc_foreground {
                self.state.present.remove("requireCocForeground");
            }
            return Err(error);
        }
        Ok(())
    }

    pub fn set_shortcuts(
        &mut self,
        shortcuts: crate::types::ShortcutSettings,
    ) -> Result<(), StoreError> {
        let old_shortcuts = self.state.settings.shortcuts.clone();
        let was_present = self.state.present.contains("shortcuts");
        self.state.settings.shortcuts = shortcuts;
        self.state.present.insert("shortcuts".into());
        if let Err(error) = self.persist_settings() {
            self.state.settings.shortcuts = old_shortcuts;
            if !was_present {
                self.state.present.remove("shortcuts");
            }
            return Err(error);
        }
        Ok(())
    }

    pub fn set_selected(&mut self, name: Option<String>) -> Result<(), StoreError> {
        if let Some(ref name) = name {
            self.get_macro(name)?;
        }
        let old_selected = self.state.selected_macro.clone();
        self.state.selected_macro = name;
        if let Err(error) = self.persist_settings() {
            self.state.selected_macro = old_selected;
            return Err(error);
        }
        Ok(())
    }

    pub fn complete_onboarding(&mut self) -> Result<(), StoreError> {
        let was_complete = self.state.onboarding_complete;
        self.state.onboarding_complete = true;
        if let Err(error) = self.persist_settings() {
            self.state.onboarding_complete = was_complete;
            return Err(error);
        }
        Ok(())
    }

    pub fn summaries(&self) -> Result<Vec<MacroSummary>, StoreError> {
        let mut items = Vec::new();
        for entry in fs::read_dir(&self.macros_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != "json") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let parsed = read_macro(&path);
            match parsed {
                Ok(macro_file) => {
                    let readable = macro_issue(&macro_file).is_none();
                    items.push(MacroSummary {
                        name: stem.into(),
                        event_count: macro_file.steps.len(),
                        duration_seconds: duration(&macro_file.steps),
                        readable,
                        issue: macro_issue(&macro_file),
                    });
                }
                Err(_) => items.push(MacroSummary {
                    name: stem.into(),
                    event_count: 0,
                    duration_seconds: 0.0,
                    readable: false,
                    issue: Some("Le fichier JSON ne peut pas être lu.".into()),
                }),
            }
        }
        items.sort_by(|a, b| natural_cmp(&a.name, &b.name));
        Ok(items)
    }

    pub fn get_macro(&self, name: &str) -> Result<MacroFile, StoreError> {
        let path = macro_path(&self.macros_path, name)?;
        if !path.exists() && name.eq_ignore_ascii_case("macro") {
            return read_macro(&self.macros_path.join("macro.json"));
        }
        read_macro(&path)
    }

    pub fn create_macro(&mut self, name: &str) -> Result<(), StoreError> {
        validate_name(name)?;
        let path = macro_path(&self.macros_path, name)?;
        if path.exists() || self.has_case_collision(name)? {
            return Err(StoreError::Conflict);
        }
        let bytes =
            serde_json::to_vec_pretty(&empty_macro(name)).map_err(|_| StoreError::InvalidData)?;
        create_no_clobber(&path, &bytes).map_err(no_clobber_error)?;
        self.state.selected_macro = Some(name.to_owned());
        self.persist_settings()
    }

    pub fn rename_macro(&mut self, name: &str, new_name: &str) -> Result<(), StoreError> {
        validate_name(new_name)?;
        let src = macro_path(&self.macros_path, name)?;
        if !src.exists() {
            return Err(StoreError::NotFound);
        }
        let dst = macro_path(&self.macros_path, new_name)?;
        let same_key = name.to_lowercase() == new_name.to_lowercase();
        if !same_key && (dst.exists() || self.has_case_collision(new_name)?) {
            return Err(StoreError::Conflict);
        }
        let mut data = read_macro(&src)?;
        data.name = new_name.to_owned();
        data.updated_at = now_iso();
        data.sha1 = macro_hash(&data.steps)?;
        if same_key {
            atomic_write_json(&src, &data)?;
        } else {
            rename_no_clobber(&src, &dst).map_err(no_clobber_error)?;
            atomic_write_json(&dst, &data)?;
        }
        if self.state.selected_macro.as_deref() == Some(name) {
            self.state.selected_macro = Some(new_name.to_owned());
            self.persist_settings()?;
        }
        Ok(())
    }

    pub fn delete_macro(&mut self, name: &str) -> Result<(), StoreError> {
        let path = macro_path(&self.macros_path, name)?;
        if !path.exists() {
            return Err(StoreError::NotFound);
        }
        fs::remove_file(path)?;
        if self.state.selected_macro.as_deref() == Some(name) {
            self.state.selected_macro = None;
            self.persist_settings()?;
        }
        Ok(())
    }

    pub fn save_recorded_steps(&mut self, name: &str, steps: &[Value]) -> Result<(), StoreError> {
        validate_steps(steps)?;
        let path = macro_path(&self.macros_path, name)?;
        let mut data = read_macro(&path)?;
        data.steps = steps.to_vec();
        data.updated_at = now_iso();
        data.sha1 = macro_hash(&data.steps)?;
        atomic_write_json(&path, &data)?;
        Ok(())
    }

    fn has_case_collision(&self, name: &str) -> Result<bool, StoreError> {
        let folded = name.to_lowercase();
        for entry in fs::read_dir(&self.macros_path)? {
            let entry = entry?;
            if entry.path().extension().is_some_and(|ext| ext == "json")
                && entry
                    .path()
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .is_some_and(|existing| existing.to_lowercase() == folded)
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn persist_settings(&self) -> Result<(), StoreError> {
        atomic_write_json(&self.settings_path, &self.state)?;
        Ok(())
    }
}

fn read_settings(path: &Path) -> Result<PersistedSettings, StoreError> {
    let bytes = fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(|_| StoreError::InvalidData)
}

fn validate_name(name: &str) -> Result<(), StoreError> {
    let name = name.trim();
    if name.is_empty()
        || name.len() > 160
        || name.ends_with([' ', '.'])
        || name
            .chars()
            .any(|ch| !(ch.is_alphanumeric() || matches!(ch, ' ' | '_' | '-' | '.' | '(' | ')')))
    {
        return Err(StoreError::InvalidName);
    }
    let device = name.split('.').next().unwrap_or(name).to_ascii_uppercase();
    if matches!(
        device.as_str(),
        "CON"
            | "PRN"
            | "AUX"
            | "NUL"
            | "COM1"
            | "COM2"
            | "COM3"
            | "COM4"
            | "COM5"
            | "COM6"
            | "COM7"
            | "COM8"
            | "COM9"
            | "LPT1"
            | "LPT2"
            | "LPT3"
            | "LPT4"
            | "LPT5"
            | "LPT6"
            | "LPT7"
            | "LPT8"
            | "LPT9"
    ) {
        return Err(StoreError::InvalidName);
    }
    Ok(())
}

fn macro_path(dir: &Path, name: &str) -> Result<PathBuf, StoreError> {
    validate_name(name)?;
    Ok(dir.join(format!("{name}.json")))
}

fn empty_macro(name: &str) -> MacroFile {
    MacroFile {
        name: name.into(),
        updated_at: now_iso(),
        sha1: sha1_hex(b"[]"),
        steps: Vec::new(),
        extra: BTreeMap::new(),
    }
}

fn read_macro(path: &Path) -> Result<MacroFile, StoreError> {
    let bytes = read_bounded_file(path, MAX_MACRO_BYTES).map_err(|error| match error {
        StoreError::Io(ref io_error) if io_error.kind() == io::ErrorKind::NotFound => {
            StoreError::NotFound
        }
        other => other,
    })?;
    parse_macro_bytes(&bytes)
}

#[cfg(test)]
fn read_macro_bytes(path: &Path) -> Result<Vec<u8>, StoreError> {
    let bytes = read_bounded_file(path, MAX_MACRO_BYTES)?;
    parse_macro_bytes(&bytes)?;
    Ok(bytes)
}

fn parse_macro_bytes(bytes: &[u8]) -> Result<MacroFile, StoreError> {
    let macro_file: MacroFile =
        serde_json::from_slice(bytes).map_err(|_| StoreError::Unreadable)?;
    if macro_file.steps.len() > MAX_STEPS {
        return Err(StoreError::InvalidData);
    }
    if macro_issue(&macro_file).is_some() {
        return Err(StoreError::Unreadable);
    }
    Ok(macro_file)
}

fn read_bounded_file(path: &Path, max_bytes: u64) -> Result<Vec<u8>, StoreError> {
    let metadata = fs::metadata(path)?;
    if metadata.len() > max_bytes {
        return Err(StoreError::InvalidData);
    }

    let capacity = usize::try_from(metadata.len()).map_err(|_| StoreError::InvalidData)?;
    let mut bytes = Vec::with_capacity(capacity);
    let file = fs::File::open(path)?;
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max_bytes {
        return Err(StoreError::InvalidData);
    }
    Ok(bytes)
}

fn macro_issue(data: &MacroFile) -> Option<String> {
    if data.steps.len() > MAX_STEPS {
        return Some("Cette macro contient trop d’événements.".into());
    }
    let mut playback_seconds = 0.0;
    for step in &data.steps {
        let Some(object) = step.as_object() else {
            return Some("Un événement est mal formé.".into());
        };
        let Some(time) = object.get("t").and_then(Value::as_f64) else {
            return Some("Un événement contient un délai invalide.".into());
        };
        if !time.is_finite() || time < 0.0 {
            return Some("Un événement contient un délai invalide.".into());
        }
        if !crate::native_input::try_accumulate_playback_duration(&mut playback_seconds, time) {
            return Some("Cette macro dépasse la limite de lecture de 24 heures.".into());
        }
        let Some(kind) = object.get("type").and_then(Value::as_str) else {
            return Some("Un événement ne contient pas de type.".into());
        };
        if !LEGACY_TYPES.contains(&kind) {
            return Some("Cette macro contient un type d’événement inconnu.".into());
        }
        let Some(payload) = object.get("data").and_then(Value::as_object) else {
            return Some("Les données d’un événement sont invalides.".into());
        };
        let integer = |key: &str| {
            payload
                .get(key)
                .and_then(Value::as_i64)
                .and_then(|number| i32::try_from(number).ok())
                .is_some()
        };
        match kind {
            "mouse_move" if !integer("x") || !integer("y") => {
                return Some("La position de souris est invalide.".into());
            }
            "mouse_click"
                if !integer("x")
                    || !integer("y")
                    || !payload
                        .get("button")
                        .and_then(Value::as_str)
                        .is_some_and(|button| matches!(button, "left" | "right" | "middle"))
                    || !payload
                        .get("action")
                        .and_then(Value::as_str)
                        .is_some_and(|action| matches!(action, "down" | "up")) =>
            {
                return Some("Le clic de souris est invalide.".into());
            }
            "scroll" if !integer("x") || !integer("y") || !integer("dx") || !integer("dy") => {
                return Some("Le défilement est invalide.".into());
            }
            "key_down" | "key_up"
                if !payload
                    .get("key")
                    .and_then(Value::as_str)
                    .is_some_and(|key| {
                        !key.is_empty()
                            && key.len() <= 64
                            && !key.contains('\0')
                            && crate::native_input::parse_key_code(key).is_ok()
                    }) =>
            {
                return Some("La touche enregistrée est invalide ou non prise en charge.".into());
            }
            "nop" if !payload.is_empty() => {
                return Some("L’événement vide contient des données inattendues.".into());
            }
            _ => {}
        }
    }
    None
}

fn validate_steps(steps: &[Value]) -> Result<(), StoreError> {
    if steps.len() > MAX_STEPS {
        return Err(StoreError::InvalidData);
    }
    let data = MacroFile {
        name: "_".into(),
        updated_at: String::new(),
        sha1: String::new(),
        steps: steps.to_vec(),
        extra: BTreeMap::new(),
    };
    if macro_issue(&data).is_some() {
        return Err(StoreError::InvalidData);
    }
    Ok(())
}

fn duration(steps: &[Value]) -> f64 {
    steps
        .iter()
        .filter_map(|event| event.get("t").and_then(Value::as_f64))
        .map(|seconds| seconds.max(0.0))
        .sum()
}

fn macro_hash(steps: &[Value]) -> Result<String, StoreError> {
    // The legacy Python writer used sorted JSON keys with spaces. Keep the SHA-1 as metadata;
    // compatibility is preserved by leaving imported files untouched until their first save.
    let canonical = serde_json::to_vec(steps).map_err(|_| StoreError::InvalidData)?;
    Ok(sha1_hex(&canonical))
}

fn sha1_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha1::digest(bytes))
}

fn now_iso() -> String {
    chrono::DateTime::<chrono::Utc>::from(SystemTime::now())
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn natural_cmp(left: &str, right: &str) -> std::cmp::Ordering {
    let mut l = left.chars().peekable();
    let mut r = right.chars().peekable();
    loop {
        match (l.peek(), r.peek()) {
            (None, None) => return std::cmp::Ordering::Equal,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (Some(a), Some(b)) if a.is_ascii_digit() && b.is_ascii_digit() => {
                let ln = take_digits(&mut l);
                let rn = take_digits(&mut r);
                match ln
                    .trim_start_matches('0')
                    .len()
                    .cmp(&rn.trim_start_matches('0').len())
                {
                    std::cmp::Ordering::Equal => {
                        match ln.trim_start_matches('0').cmp(rn.trim_start_matches('0')) {
                            std::cmp::Ordering::Equal => continue,
                            order => return order,
                        }
                    }
                    order => return order,
                }
            }
            (Some(a), Some(b)) => match a.to_lowercase().cmp(b.to_lowercase()) {
                std::cmp::Ordering::Equal => {
                    l.next();
                    r.next();
                }
                order => return order,
            },
        }
    }
}

fn take_digits(iter: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut out = String::new();
    while iter.peek().is_some_and(char::is_ascii_digit) {
        out.push(iter.next().unwrap());
    }
    out
}

fn atomic_write_json(path: &Path, value: &impl Serialize) -> io::Result<()> {
    let bytes = serde_json::to_vec_pretty(value).map_err(io::Error::other)?;
    atomic_write(path, &bytes)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::other("missing parent directory"))?;
    fs::create_dir_all(parent)?;
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy(),
        suffix
    ));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    let result = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        replace_file(&temp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn create_no_clobber(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::other("missing parent directory"))?;
    fs::create_dir_all(parent)?;
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp = parent.join(format!(
        ".{}.{}.{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy(),
        suffix,
        sequence
    ));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    match fs::hard_link(&temp, path) {
        Ok(()) => {
            let _ = fs::remove_file(temp);
            Ok(())
        }
        Err(error) => {
            let _ = fs::remove_file(temp);
            Err(error)
        }
    }
}

fn rename_no_clobber(source: &Path, target: &Path) -> io::Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Storage::FileSystem::MoveFileW;
        let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
        let target: Vec<u16> = target.as_os_str().encode_wide().chain(Some(0)).collect();
        let ok = unsafe { MoveFileW(source.as_ptr(), target.as_ptr()) };
        if ok == 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
    #[cfg(not(windows))]
    {
        fs::hard_link(source, target)?;
        fs::remove_file(source)
    }
}

fn no_clobber_error(error: io::Error) -> StoreError {
    if error.kind() == io::ErrorKind::AlreadyExists {
        StoreError::Conflict
    } else {
        StoreError::Io(error)
    }
}

#[cfg(windows)]
fn replace_file(source: &Path, target: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
    };
    let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let target: Vec<u16> = target.as_os_str().encode_wide().chain(Some(0)).collect();
    let ok = unsafe {
        MoveFileExW(
            source.as_ptr(),
            target.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if ok == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(windows))]
fn replace_file(source: &Path, target: &Path) -> io::Result<()> {
    fs::rename(source, target)
}

fn file_fingerprint(content: &[u8]) -> String {
    format!("{:x}", Sha256::digest(content))
}

fn read_placeholder_manifest(root: &Path) -> Result<PlaceholderManifest, StoreError> {
    let path = root.join("system-placeholders.json");
    if !path.exists() {
        return Ok(PlaceholderManifest::default());
    }
    serde_json::from_slice(&read_bounded_file(&path, MAX_PLACEHOLDER_MANIFEST_BYTES)?)
        .map_err(|_| StoreError::InvalidData)
}

fn remove_untouched_placeholders(root: &Path, macros_path: &Path) -> Result<(), StoreError> {
    let mut provenance = match read_placeholder_manifest(root) {
        Ok(provenance) => provenance,
        // A malformed manifest cannot prove that a macro is safe to delete.
        Err(StoreError::InvalidData) => return Ok(()),
        Err(error) => return Err(error),
    };
    if provenance.files.is_empty() {
        return Ok(());
    }

    let manifest_path = root.join("system-placeholders.json");
    for name in ["Recharger COC", "Valider arrivée"] {
        let Some(expected) = provenance.files.remove(name) else {
            continue;
        };
        let path = macro_path(macros_path, name)?;
        let bytes = match read_bounded_file(&path, MAX_PLACEHOLDER_MACRO_BYTES) {
            Ok(bytes) => bytes,
            Err(StoreError::Io(error)) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(StoreError::InvalidData) => continue,
            Err(error) => return Err(error),
        };
        let is_untouched_placeholder = file_fingerprint(&bytes) == expected
            && serde_json::from_slice::<MacroFile>(&bytes).is_ok_and(|data| {
                data.name == name && data.steps.is_empty() && macro_issue(&data).is_none()
            });
        if is_untouched_placeholder {
            fs::remove_file(path)?;
        }
    }

    if provenance.files.is_empty() {
        match fs::remove_file(manifest_path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    } else {
        atomic_write_json(&manifest_path, &provenance)?;
    }
    Ok(())
}

#[cfg(windows)]
fn protect_token(token: &[u8]) -> Result<Vec<u8>, ()> {
    use std::ptr;
    use windows_sys::Win32::{
        Foundation::LocalFree,
        Security::Cryptography::{CRYPT_INTEGER_BLOB, CRYPTPROTECT_UI_FORBIDDEN, CryptProtectData},
    };
    let input = CRYPT_INTEGER_BLOB {
        cbData: token.len() as u32,
        pbData: token.as_ptr() as *mut u8,
    };
    let mut output = CRYPT_INTEGER_BLOB {
        cbData: 0,
        pbData: ptr::null_mut(),
    };
    let ok = unsafe {
        CryptProtectData(
            &input,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut output,
        )
    };
    if ok == 0 || output.pbData.is_null() {
        return Err(());
    }
    let encrypted =
        unsafe { std::slice::from_raw_parts(output.pbData, output.cbData as usize).to_vec() };
    unsafe {
        LocalFree(output.pbData.cast());
    }
    Ok(encrypted)
}

pub(crate) fn save_token(root: &Path, token: &[u8]) -> Result<(), StoreError> {
    let encrypted = protect_token(token).map_err(|_| StoreError::InvalidData)?;
    atomic_write(&root.join("telegram-token.bin"), &encrypted)?;
    Ok(())
}

pub(crate) fn token_matches(root: &Path, candidate: &[u8]) -> bool {
    let Ok(encrypted) = fs::read(root.join("telegram-token.bin")) else {
        return false;
    };
    unprotect_token(&encrypted).is_ok_and(|stored| stored == candidate)
}

pub(crate) fn load_token(root: &Path) -> Result<Option<String>, StoreError> {
    let encrypted = match fs::read(root.join("telegram-token.bin")) {
        Ok(encrypted) => encrypted,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let plaintext = unprotect_token(&encrypted).map_err(|_| StoreError::InvalidData)?;
    String::from_utf8(plaintext)
        .map(Some)
        .map_err(|_| StoreError::InvalidData)
}

pub(crate) fn clear_token(root: &Path) -> Result<(), StoreError> {
    match fs::remove_file(root.join("telegram-token.bin")) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

#[cfg(not(windows))]
fn protect_token(_token: &[u8]) -> Result<Vec<u8>, ()> {
    Err(())
}

#[cfg(windows)]
fn unprotect_token(encrypted: &[u8]) -> Result<Vec<u8>, ()> {
    use std::ptr;
    use windows_sys::Win32::{
        Foundation::LocalFree,
        Security::Cryptography::{
            CRYPT_INTEGER_BLOB, CRYPTPROTECT_UI_FORBIDDEN, CryptUnprotectData,
        },
    };
    let input = CRYPT_INTEGER_BLOB {
        cbData: encrypted.len() as u32,
        pbData: encrypted.as_ptr() as *mut u8,
    };
    let mut output = CRYPT_INTEGER_BLOB {
        cbData: 0,
        pbData: ptr::null_mut(),
    };
    let ok = unsafe {
        CryptUnprotectData(
            &input,
            ptr::null_mut(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut output,
        )
    };
    if ok == 0 || output.pbData.is_null() {
        return Err(());
    }
    let plaintext =
        unsafe { std::slice::from_raw_parts(output.pbData, output.cbData as usize).to_vec() };
    unsafe {
        LocalFree(output.pbData.cast());
    }
    Ok(plaintext)
}

#[cfg(not(windows))]
fn unprotect_token(_encrypted: &[u8]) -> Result<Vec<u8>, ()> {
    Err(())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn failed_loop_and_settings_persistence_restore_memory_state() {
        let dir = tempdir().unwrap();
        let mut store = Store::open_at(dir.path()).unwrap();
        let settings_path = store.settings_path.clone();
        let blocked_parent = dir.path().join("settings-parent-is-file");
        fs::write(&blocked_parent, b"not a directory").unwrap();
        store.settings_path = blocked_parent.join("settings.json");

        assert!(matches!(store.set_loop(true), Err(StoreError::Io(_))));
        assert!(!store.settings().loop_playback);
        assert!(!store.state.present.contains("loop"));

        store.settings_path = settings_path.clone();
        store
            .update_settings(None, Some("C:\\Games\\CoC.exe".into()), None)
            .unwrap();
        store.settings_path = blocked_parent.join("settings.json");

        assert!(matches!(
            store.update_settings(Some(true), Some("C:\\Games\\Other.exe".into()), None),
            Err(StoreError::Io(_))
        ));
        assert!(!store.settings().loop_playback);
        assert_eq!(store.settings().coc_path, "C:\\Games\\CoC.exe");
        assert!(!store.state.present.contains("loop"));
        assert!(store.state.present.contains("cocPath"));

        let persisted = read_settings(&settings_path).unwrap();
        assert!(!persisted.settings.loop_playback);
        assert_eq!(persisted.settings.coc_path, "C:\\Games\\CoC.exe");
    }

    #[test]
    fn new_install_defaults_guard_on_but_legacy_missing_field_stays_off() {
        let dir = tempdir().unwrap();
        let store = Store::open_at(dir.path()).unwrap();
        assert!(store.settings().require_coc_foreground);
        let settings_path = store.settings_path.clone();
        let mut old_json = serde_json::to_value(&store.state).unwrap();
        old_json["settings"]
            .as_object_mut()
            .unwrap()
            .remove("requireCocForeground");
        fs::write(&settings_path, serde_json::to_vec(&old_json).unwrap()).unwrap();

        let mut store = Store::open_at(dir.path()).unwrap();
        assert!(!store.settings().require_coc_foreground);
        store.update_settings(None, None, Some(false)).unwrap();
        drop(store);
        let store = Store::open_at(dir.path()).unwrap();
        assert!(!store.settings().require_coc_foreground);
        assert!(
            !read_settings(&settings_path)
                .unwrap()
                .settings
                .require_coc_foreground
        );
    }

    #[test]
    fn failed_selected_macro_persistence_restores_memory_state() {
        let dir = tempdir().unwrap();
        let mut store = Store::open_at(dir.path()).unwrap();
        store.create_macro("Existing").unwrap();
        let settings_path = store.settings_path.clone();
        let blocked_parent = dir.path().join("settings-parent-is-file");
        fs::write(&blocked_parent, b"not a directory").unwrap();
        store.settings_path = blocked_parent.join("settings.json");

        assert!(matches!(store.set_selected(None), Err(StoreError::Io(_))));
        assert_eq!(store.selected_macro(), Some("Existing"));
        assert_eq!(
            read_settings(&settings_path)
                .unwrap()
                .selected_macro
                .as_deref(),
            Some("Existing")
        );
    }

    #[test]
    fn failed_onboarding_persistence_restores_memory_state() {
        let dir = tempdir().unwrap();
        let mut store = Store::open_at(dir.path()).unwrap();
        let settings_path = store.settings_path.clone();
        let blocked_parent = dir.path().join("settings-parent-is-file");
        fs::write(&blocked_parent, b"not a directory").unwrap();
        store.settings_path = blocked_parent.join("settings.json");

        assert!(matches!(
            store.complete_onboarding(),
            Err(StoreError::Io(_))
        ));
        assert!(!store.onboarding_complete());
        assert!(!read_settings(&settings_path).unwrap().onboarding_complete);
    }

    #[test]
    fn clearing_token_removes_blob_and_tolerates_a_missing_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("telegram-token.bin");
        fs::write(&path, b"synthetic protected token blob").unwrap();

        clear_token(dir.path()).unwrap();
        assert!(!path.exists());
        clear_token(dir.path()).unwrap();
    }

    #[test]
    fn clearing_token_propagates_a_directory_removal_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("telegram-token.bin");
        fs::create_dir(&path).unwrap();

        assert!(matches!(clear_token(dir.path()), Err(StoreError::Io(_))));
        assert!(path.is_dir());
    }

    #[test]
    fn creates_only_user_macros_and_natural_sort_order() {
        let dir = tempdir().unwrap();
        let store = Store::open_at(dir.path()).unwrap();
        assert!(store.summaries().unwrap().is_empty());
        let mut store = store;
        store.create_macro("Macro 10").unwrap();
        store.create_macro("Macro 2").unwrap();
        store.create_macro("Valider arrivée").unwrap();
        let names: Vec<_> = store
            .summaries()
            .unwrap()
            .into_iter()
            .map(|m| m.name)
            .collect();
        assert_eq!(names, ["Macro 2", "Macro 10", "Valider arrivée"]);
    }

    #[test]
    fn refuses_invalid_and_duplicate_macro_mutations() {
        let dir = tempdir().unwrap();
        let mut store = Store::open_at(dir.path()).unwrap();
        assert!(matches!(
            store.create_macro("../bad"),
            Err(StoreError::InvalidName)
        ));
        store.create_macro("Recharger COC").unwrap();
        store.rename_macro("Recharger COC", "Recharger").unwrap();
        store.delete_macro("Recharger").unwrap();
        store.create_macro("Test").unwrap();
        assert!(matches!(
            store.create_macro("test"),
            Err(StoreError::Conflict)
        ));
    }

    #[test]
    fn preserves_corrupt_macro_and_reports_it_unreadable() {
        let dir = tempdir().unwrap();
        let store = Store::open_at(dir.path()).unwrap();
        let path = store.macros_path.join("Corrompue.json");
        fs::write(&path, b"not-json").unwrap();
        let before = fs::read(&path).unwrap();
        let summary = store
            .summaries()
            .unwrap()
            .into_iter()
            .find(|item| item.name == "Corrompue")
            .unwrap();
        assert!(!summary.readable);
        assert_eq!(fs::read(path).unwrap(), before);
    }

    #[test]
    fn macro_readability_matches_legacy_key_playback_support() {
        let dir = tempdir().unwrap();
        let store = Store::open_at(dir.path()).unwrap();
        let fixture = |name: &str, key: &str| {
            serde_json::to_vec(&json!({
                "name": name,
                "updated_at": "2025-01-01T00:00:00Z",
                "sha1": "legacy",
                "steps": [
                    {"t": 0.0, "type": "key_down", "data": {"key": key}},
                    {"t": 0.1, "type": "key_up", "data": {"key": key}}
                ]
            }))
            .unwrap()
        };
        let media_key = fixture("Media", "media_play_pause");
        let unicode_key = fixture("Unicode", "é");
        let media_path = store.macros_path.join("Media.json");
        let unicode_path = store.macros_path.join("Unicode.json");
        fs::write(&media_path, &media_key).unwrap();
        fs::write(&unicode_path, &unicode_key).unwrap();

        let summaries = store.summaries().unwrap();
        assert!(
            summaries
                .iter()
                .find(|item| item.name == "Media")
                .unwrap()
                .readable
        );
        let unicode = summaries
            .iter()
            .find(|item| item.name == "Unicode")
            .unwrap();
        assert!(!unicode.readable);
        assert!(unicode.issue.is_some());
        assert_eq!(fs::read(media_path).unwrap(), media_key);
        assert_eq!(fs::read(unicode_path).unwrap(), unicode_key);
    }

    #[test]
    fn macro_readability_enforces_playback_duration_limit() {
        let app = tempdir().unwrap();
        let store = Store::open_at(app.path()).unwrap();
        let fixture = |name: &str, steps: Vec<Value>| {
            serde_json::to_vec(&json!({
                "name": name,
                "updated_at": "2025-01-01T00:00:00Z",
                "sha1": "legacy",
                "steps": steps
            }))
            .unwrap()
        };
        let within_limit = fixture(
            "WithinLimit",
            vec![json!({
                "t": 86_400.0,
                "type": "nop",
                "data": {}
            })],
        );
        let over_limit = fixture(
            "OverLimit",
            vec![
                json!({"t": 43_200.0, "type": "nop", "data": {}}),
                json!({"t": 43_200.1, "type": "nop", "data": {}}),
            ],
        );
        let within_path = store.macros_path.join("WithinLimit.json");
        let over_path = store.macros_path.join("OverLimit.json");
        fs::write(&within_path, &within_limit).unwrap();
        fs::write(&over_path, &over_limit).unwrap();

        let summaries = store.summaries().unwrap();
        assert!(
            summaries
                .iter()
                .find(|item| item.name == "WithinLimit")
                .unwrap()
                .readable
        );
        let over_summary = summaries
            .iter()
            .find(|item| item.name == "OverLimit")
            .unwrap();
        assert!(!over_summary.readable);
        let over_file: MacroFile = serde_json::from_slice(&over_limit).unwrap();
        assert!(macro_issue(&over_file).unwrap().contains("24 heures"));
        assert_eq!(fs::read(&within_path).unwrap(), within_limit);
        assert_eq!(fs::read(&over_path).unwrap(), over_limit);
    }

    #[test]
    fn sparse_oversized_macro_is_rejected_before_reading_its_contents() {
        let dir = tempdir().unwrap();
        let store = Store::open_at(dir.path()).unwrap();
        let path = store.macros_path.join("Oversized.json");
        let file = fs::File::create(&path).unwrap();
        file.set_len(MAX_MACRO_BYTES + 1).unwrap();

        let summary = store.summaries().unwrap().remove(0);
        assert!(!summary.readable);
        assert_eq!(
            summary.issue.as_deref(),
            Some("Le fichier JSON ne peut pas être lu.")
        );
        assert!(matches!(
            read_macro_bytes(&path),
            Err(StoreError::InvalidData)
        ));
        assert!(matches!(
            read_bounded_file(&path, MAX_MACRO_BYTES),
            Err(StoreError::InvalidData)
        ));
    }

    #[test]
    fn oversized_placeholder_manifest_does_not_block_startup_or_get_deleted() {
        let dir = tempdir().unwrap();
        let manifest = dir.path().join("system-placeholders.json");
        fs::File::create(&manifest)
            .unwrap()
            .set_len(MAX_PLACEHOLDER_MANIFEST_BYTES + 1)
            .unwrap();

        let store = Store::open_at(dir.path()).unwrap();

        assert!(store.settings().require_coc_foreground);
        assert_eq!(
            fs::metadata(manifest).unwrap().len(),
            MAX_PLACEHOLDER_MANIFEST_BYTES + 1
        );
    }

    #[test]
    fn oversized_placeholder_macro_is_preserved_on_startup() {
        let dir = tempdir().unwrap();
        let macros = dir.path().join("macros");
        fs::create_dir_all(&macros).unwrap();
        let placeholder = macros.join("Recharger COC.json");
        fs::File::create(&placeholder)
            .unwrap()
            .set_len(MAX_PLACEHOLDER_MACRO_BYTES + 1)
            .unwrap();
        atomic_write_json(
            &dir.path().join("system-placeholders.json"),
            &PlaceholderManifest {
                files: BTreeMap::from([("Recharger COC".into(), "not-a-match".into())]),
            },
        )
        .unwrap();

        Store::open_at(dir.path()).unwrap();

        assert!(placeholder.exists());
        assert_eq!(
            fs::metadata(placeholder).unwrap().len(),
            MAX_PLACEHOLDER_MACRO_BYTES + 1
        );
    }

    #[test]
    fn macro_readability_rejects_coordinates_outside_i32_without_rewriting_source() {
        let app = tempdir().unwrap();
        let store = Store::open_at(app.path()).unwrap();
        let path = store.macros_path.join("OutOfRange.json");
        let original = serde_json::to_vec(&json!({
            "name": "OutOfRange",
            "updated_at": "2025-01-01T00:00:00Z",
            "sha1": "legacy",
            "steps": [{
                "t": 0.0,
                "type": "mouse_move",
                "data": {"x": 2_147_483_648_i64, "y": 0}
            }]
        }))
        .unwrap();
        fs::write(&path, &original).unwrap();

        assert!(
            !store
                .summaries()
                .unwrap()
                .into_iter()
                .find(|item| item.name == "OutOfRange")
                .unwrap()
                .readable
        );
        assert!(matches!(
            store.get_macro("OutOfRange"),
            Err(StoreError::Unreadable)
        ));
        assert_eq!(fs::read(path).unwrap(), original);
    }

    #[test]
    fn strict_step_validation_keeps_unknown_events_out_of_replay() {
        let unknown = json!({"t": 0.1, "type": "future_event", "data": {}});
        assert!(validate_steps(&[unknown]).is_err());
    }

    #[test]
    fn concurrent_writers_never_replace_a_created_destination() {
        let dir = tempdir().unwrap();
        let destination = dir.path().join("same.json");
        let barrier = Arc::new(Barrier::new(2));
        let writers = [b"first".to_vec(), b"second".to_vec()].map(|bytes| {
            let path = destination.clone();
            let ready = barrier.clone();
            std::thread::spawn(move || {
                ready.wait();
                create_no_clobber(&path, &bytes)
            })
        });
        let results = writers.map(|writer| writer.join().unwrap());
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|result| result
                    .as_ref()
                    .is_err_and(|error| error.kind() == io::ErrorKind::AlreadyExists))
                .count(),
            1
        );
        assert!(matches!(
            fs::read(destination).unwrap().as_slice(),
            b"first" | b"second"
        ));
    }

    #[test]
    fn startup_removes_only_proven_empty_system_placeholders() {
        let app = tempdir().unwrap();
        let macros_path = app.path().join("macros");
        fs::create_dir_all(&macros_path).unwrap();

        let mut placeholders = PlaceholderManifest::default();
        for name in ["Recharger COC", "Valider arrivée"] {
            let bytes = serde_json::to_vec_pretty(&empty_macro(name)).unwrap();
            fs::write(macros_path.join(format!("{name}.json")), &bytes).unwrap();
            placeholders
                .files
                .insert(name.into(), file_fingerprint(&bytes));
        }

        let edited = json!({
            "name":"Recharger COC",
            "updated_at":"2025-01-01T00:00:00Z",
            "sha1":"legacy",
            "steps":[{"t":0.0,"type":"key_down","data":{"key":"x"}}]
        });
        let edited_bytes = serde_json::to_vec(&edited).unwrap();
        fs::write(macros_path.join("Recharger COC.json"), &edited_bytes).unwrap();
        let unproven = json!({
            "name":"Custom",
            "updated_at":"2025-01-01T00:00:00Z",
            "sha1":"legacy",
            "steps":[{"t":0.0,"type":"key_down","data":{"key":"y"}}]
        });
        let unproven_bytes = serde_json::to_vec(&unproven).unwrap();
        fs::write(macros_path.join("Custom.json"), &unproven_bytes).unwrap();
        atomic_write_json(&app.path().join("system-placeholders.json"), &placeholders).unwrap();

        let store = Store::open_at(app.path()).unwrap();

        assert!(!macros_path.join("Valider arrivée.json").exists());
        assert_eq!(
            fs::read(macros_path.join("Recharger COC.json")).unwrap(),
            edited_bytes
        );
        assert_eq!(
            fs::read(macros_path.join("Custom.json")).unwrap(),
            unproven_bytes
        );
        assert!(!app.path().join("system-placeholders.json").exists());
        let summaries = store.summaries().unwrap();
        assert_eq!(summaries.len(), 2);
    }

    #[cfg(windows)]
    #[test]
    fn startup_recovers_a_protected_token_written_before_settings_persisted() {
        let app = tempdir().unwrap();
        let store = Store::open_at(app.path()).unwrap();
        assert!(!store.settings().telegram.token_configured);
        let token = "synthetic-interrupted-telegram-token";

        save_token(app.path(), token.as_bytes()).unwrap();
        let reopened = Store::open_at(app.path()).unwrap();

        assert!(reopened.settings().telegram.token_configured);
        assert_eq!(
            reopened.settings().telegram.status,
            crate::types::TelegramStatus::Disconnected
        );
        assert_eq!(load_token(app.path()).unwrap().as_deref(), Some(token));
        let settings = fs::read(app.path().join("settings.json")).unwrap();
        assert!(
            !settings
                .windows(token.len())
                .any(|window| window == token.as_bytes())
        );
    }

    #[cfg(windows)]
    #[test]
    fn startup_clears_configured_flag_for_corrupt_protected_token() {
        let app = tempdir().unwrap();
        Store::open_at(app.path()).unwrap();
        let settings_path = app.path().join("settings.json");
        let mut persisted = read_settings(&settings_path).unwrap();
        persisted.settings.telegram.token_configured = true;
        persisted.settings.telegram.status = crate::types::TelegramStatus::Connected;
        atomic_write_json(&settings_path, &persisted).unwrap();

        let token_path = app.path().join("telegram-token.bin");
        let invalid_ciphertext = b"synthetic-corrupt-dpapi-blob";
        fs::write(&token_path, invalid_ciphertext).unwrap();
        let reopened = Store::open_at(app.path()).unwrap();

        assert!(!reopened.settings().telegram.token_configured);
        assert_eq!(
            reopened.settings().telegram.status,
            crate::types::TelegramStatus::Error
        );
        assert_eq!(fs::read(token_path).unwrap(), invalid_ciphertext);
    }

    #[cfg(windows)]
    #[test]
    fn startup_clears_configured_flag_when_protected_token_is_missing() {
        let app = tempdir().unwrap();
        Store::open_at(app.path()).unwrap();
        let settings_path = app.path().join("settings.json");
        let mut persisted = read_settings(&settings_path).unwrap();
        persisted.settings.telegram.token_configured = true;
        persisted.settings.telegram.status = crate::types::TelegramStatus::Connected;
        atomic_write_json(&settings_path, &persisted).unwrap();
        let token_path = app.path().join("telegram-token.bin");

        let reopened = Store::open_at(app.path()).unwrap();

        assert!(!reopened.settings().telegram.token_configured);
        assert_eq!(
            reopened.settings().telegram.status,
            crate::types::TelegramStatus::Error
        );
        assert!(!token_path.exists());
    }
}
