use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha1::{Digest, Sha1};
use sha2::Sha256;
use thiserror::Error;

use crate::types::{
    MacroFile, MacroSummary, MigrationCounts, MigrationMacroPreview, MigrationPreview,
    MigrationResult, Settings,
};

const MAX_MACRO_BYTES: u64 = 32 * 1024 * 1024;
const MAX_LEGACY_CSV_BYTES: u64 = 2 * 1024 * 1024;
const MAX_STEPS: usize = 250_000;
const OVERSIZED_FINGERPRINT_SAMPLE_BYTES: usize = 64 * 1024;
const OVERSIZED_FINGERPRINT_MARKER: &[u8] = b"\0AUTO-COC oversized file fingerprint v1\0";
const LEGACY_TYPES: [&str; 6] = [
    "mouse_move",
    "mouse_click",
    "scroll",
    "key_down",
    "key_up",
    "nop",
];
const PROTECTED: [&str; 2] = ["Recharger COC", "Valider arrivée"];
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("macro not found")]
    NotFound,
    #[error("macro name is invalid")]
    InvalidName,
    #[error("macro name already exists")]
    Conflict,
    #[error("protected macro cannot be changed")]
    Protected,
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
        for name in PROTECTED {
            let path = macro_path(&macros_path, name)?;
            if !path.exists() {
                let macro_file = empty_macro(name);
                let bytes = serde_json::to_vec_pretty(&macro_file).map_err(io::Error::other)?;
                match create_no_clobber(&path, &bytes) {
                    Ok(()) => {
                        let mut provenance = read_placeholder_manifest(&root)?;
                        provenance
                            .files
                            .insert(name.into(), file_fingerprint(&bytes));
                        atomic_write_json(&root.join("system-placeholders.json"), &provenance)?;
                    }
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                    Err(error) => return Err(error.into()),
                }
            }
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

    pub fn migration_was_imported(&self) -> Result<bool, StoreError> {
        let dir = self.root.join("migration-manifests");
        Ok(dir.is_dir() && fs::read_dir(dir)?.next().is_some())
    }

    pub fn preview_legacy(
        &self,
        selected: &Path,
    ) -> Result<(PathBuf, MigrationPreview, bool), StoreError> {
        let config = resolve_legacy_config_dir(selected).ok_or(StoreError::InvalidData)?;
        let settings_found = config.join("data.csv").is_file();
        let mut macros = Vec::new();
        let source_macros = config.join("macros");
        if source_macros.is_dir() {
            for entry in fs::read_dir(&source_macros)? {
                let entry = entry?;
                if entry.path().extension().is_none_or(|ext| ext != "json") {
                    continue;
                }
                let name = entry
                    .path()
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .unwrap_or("Macro")
                    .to_owned();
                macros.push(preview_macro(&entry.path(), name)?);
            }
        }
        let legacy_file = config.join("macro.json");
        if legacy_file.is_file() {
            macros.push(preview_macro(&legacy_file, "macro".into())?);
        }
        macros.sort_by(|left, right| natural_cmp(&left.name, &right.name));
        let available = settings_found || !macros.is_empty();
        let fingerprint = legacy_fingerprint(&config)?;
        let already_imported = self
            .root
            .join("migration-manifests")
            .join(format!("{fingerprint}.json"))
            .exists();
        Ok((
            config,
            MigrationPreview {
                available,
                settings_found,
                macros,
            },
            already_imported,
        ))
    }

    #[cfg(test)]
    pub fn import_legacy(&mut self, selected: &Path) -> Result<MigrationResult, StoreError> {
        self.import_legacy_with_token_installation(selected, || {})
    }

    pub(crate) fn import_legacy_with_token_installation(
        &mut self,
        selected: &Path,
        before_token_install: impl FnOnce(),
    ) -> Result<MigrationResult, StoreError> {
        let (config, preview, already_imported) = self.preview_legacy(selected)?;
        let fingerprint = legacy_fingerprint(&config)?;
        let manifest_dir = self.root.join("migration-manifests");
        fs::create_dir_all(&manifest_dir)?;
        let manifest_path = manifest_dir.join(format!("{fingerprint}.json"));
        if already_imported && manifest_path.exists() {
            let result = serde_json::from_slice(&fs::read(manifest_path)?)
                .map_err(|_| StoreError::InvalidData)?;
            return Ok(result);
        }

        let csv_path = config.join("data.csv");
        if csv_path.is_file() {
            let legacy_csv = read_bounded_file(&csv_path, MAX_LEGACY_CSV_BYTES)?;
            let backup_dir = self.root.join("migration-backups").join(format!(
                "{}-{}",
                chrono::Utc::now().format("%Y%m%dT%H%M%SZ"),
                &fingerprint[..8]
            ));
            fs::create_dir_all(&backup_dir)?;
            let backup_csv = redact_legacy_token(&legacy_csv)?;
            atomic_write(&backup_dir.join("data.csv"), &backup_csv)?;
        }

        let legacy_settings = read_legacy_settings(&csv_path)?;
        let mut imported = MigrationCounts::default();
        if let Some(value) = legacy_settings.loop_value
            && !self.state.present.contains("loop")
        {
            self.state.settings.loop_playback = value;
            self.state.present.insert("loop".into());
            imported.settings += 1;
        }
        if let Some(value) = legacy_settings.coc_path
            && !self.state.present.contains("cocPath")
        {
            self.state.settings.coc_path = value;
            self.state.present.insert("cocPath".into());
            imported.settings += 1;
        }
        if let Some(token) = legacy_settings.telegram_token {
            let needs_token_write = match load_token(&self.root) {
                Ok(Some(_)) => false,
                Ok(None) | Err(StoreError::InvalidData) => true,
                Err(error) => return Err(error),
            };
            if needs_token_write {
                self.state.settings.telegram.token_configured = false;
                self.state.settings.telegram.paired = false;
                self.state.settings.telegram.status = crate::types::TelegramStatus::Error;
                self.state.settings.telegram.pairing_code = None;
                self.state.settings.telegram.pairing_expires_at = None;
                self.clear_telegram_owner();
                self.persist_settings()?;

                before_token_install();
                save_token(&self.root, token.as_bytes())?;
                self.state.settings.telegram.token_configured = true;
                self.state.settings.telegram.status = crate::types::TelegramStatus::Disconnected;
                imported.settings += 1;
            }
        }

        let mut collisions = Vec::new();
        let mut preserved = preview
            .macros
            .iter()
            .filter(|item| !item.readable)
            .map(|item| item.name.clone())
            .collect::<Vec<_>>();
        let mut errors = Vec::new();
        let mut copy_candidates = Vec::new();
        let source_macros = config.join("macros");
        if source_macros.is_dir() {
            for entry in fs::read_dir(source_macros)? {
                let entry = entry?;
                if entry.path().extension().is_some_and(|ext| ext == "json") {
                    copy_candidates.push(entry.path());
                }
            }
        }
        let legacy_file = config.join("macro.json");
        if legacy_file.is_file() {
            copy_candidates.push(legacy_file);
        }
        let mut placeholders = read_placeholder_manifest(&self.root)?;
        let mut placeholder_manifest_changed = false;
        for source in copy_candidates {
            let Some(stem) = source
                .file_stem()
                .and_then(|value| value.to_str())
                .map(str::to_owned)
            else {
                preserved.push("Fichier de macro au nom non pris en charge".into());
                continue;
            };
            if validate_name(&stem).is_err() {
                preserved.push(stem.chars().take(80).collect());
                continue;
            }
            let source_bytes = match read_macro_bytes(&source) {
                Ok(bytes) => bytes,
                Err(_) => {
                    if !preserved.iter().any(|name| name == &stem) {
                        preserved.push(stem);
                    }
                    continue;
                }
            };
            let destination = macro_path(&self.macros_path, &stem)?;
            if destination.exists() || self.has_case_collision(&stem)? {
                if !is_protected(&stem)
                    || !can_replace_placeholder(&stem, &destination, &placeholders)?
                {
                    collisions.push(stem);
                    continue;
                }
                placeholders.files.remove(&stem);
                placeholder_manifest_changed = true;
            }
            if destination.exists() {
                atomic_write(&destination, &source_bytes)?;
                imported.macros += 1;
            } else {
                match create_no_clobber(&destination, &source_bytes) {
                    Ok(()) => imported.macros += 1,
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                        collisions.push(stem)
                    }
                    Err(error) => return Err(error.into()),
                }
            }
        }
        if placeholder_manifest_changed {
            atomic_write_json(&self.root.join("system-placeholders.json"), &placeholders)?;
        }

        if self.state.selected_macro.is_none()
            && legacy_settings.last_macro.as_deref().is_some_and(|name| {
                macro_path(&self.macros_path, name).is_ok_and(|path| path.exists())
            })
        {
            self.state.selected_macro = legacy_settings.last_macro;
            imported.settings += 1;
        }
        self.persist_settings()?;

        if !preview.available {
            errors.push("La source ne contient aucun fichier de données ou macro reconnu.".into());
        }
        let result = MigrationResult {
            imported,
            collisions,
            preserved,
            errors,
        };
        atomic_write_json(&manifest_path, &result)?;
        Ok(result)
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
    ) -> Result<(), StoreError> {
        if coc_path
            .as_ref()
            .is_some_and(|value| value.len() > 2048 || value.contains('\0'))
        {
            return Err(StoreError::InvalidData);
        }
        let old_loop_playback = self.state.settings.loop_playback;
        let old_coc_path = self.state.settings.coc_path.clone();
        let had_loop = self.state.present.contains("loop");
        let had_coc_path = self.state.present.contains("cocPath");
        if let Some(value) = loop_playback {
            self.state.settings.loop_playback = value;
            self.state.present.insert("loop".into());
        }
        if let Some(value) = coc_path {
            self.state.settings.coc_path = value;
            self.state.present.insert("cocPath".into());
        }
        if let Err(error) = self.persist_settings() {
            self.state.settings.loop_playback = old_loop_playback;
            self.state.settings.coc_path = old_coc_path;
            if !had_loop {
                self.state.present.remove("loop");
            }
            if !had_coc_path {
                self.state.present.remove("cocPath");
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
            let protected = is_protected(stem);
            match parsed {
                Ok(macro_file) => {
                    let readable = macro_issue(&macro_file).is_none();
                    items.push(MacroSummary {
                        name: stem.into(),
                        event_count: macro_file.steps.len(),
                        duration_seconds: duration(&macro_file.steps),
                        protected,
                        readable,
                        issue: macro_issue(&macro_file),
                    });
                }
                Err(_) => items.push(MacroSummary {
                    name: stem.into(),
                    event_count: 0,
                    duration_seconds: 0.0,
                    protected,
                    readable: false,
                    issue: Some("Le fichier JSON ne peut pas être lu.".into()),
                }),
            }
        }
        items.sort_by(
            |a, b| match (is_protected(&a.name), is_protected(&b.name)) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => natural_cmp(&a.name, &b.name),
            },
        );
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
        if is_protected(name) {
            return Err(StoreError::Protected);
        }
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
        if is_protected(name) || is_protected(new_name) {
            return Err(StoreError::Protected);
        }
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
        if is_protected(name) {
            return Err(StoreError::Protected);
        }
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

fn is_protected(name: &str) -> bool {
    PROTECTED
        .iter()
        .any(|protected| protected.eq_ignore_ascii_case(name.trim()))
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

#[derive(Default)]
struct LegacySettings {
    loop_value: Option<bool>,
    coc_path: Option<String>,
    last_macro: Option<String>,
    telegram_token: Option<String>,
}

fn resolve_legacy_config_dir(selected: &Path) -> Option<PathBuf> {
    if !selected.is_dir() {
        return None;
    }
    if has_legacy_content(selected) {
        return Some(selected.to_path_buf());
    }
    let config = selected.join("config");
    has_legacy_content(&config).then_some(config)
}

fn has_legacy_content(config: &Path) -> bool {
    config.join("data.csv").is_file()
        || config.join("macros").is_dir()
        || config.join("macro.json").is_file()
}

fn preview_macro(path: &Path, name: String) -> Result<MigrationMacroPreview, StoreError> {
    let bytes = match read_bounded_file(path, MAX_MACRO_BYTES) {
        Ok(bytes) => bytes,
        Err(StoreError::InvalidData) => {
            return Ok(MigrationMacroPreview {
                name,
                event_count: 0,
                readable: false,
                issue: Some("Le fichier dépasse la taille autorisée.".into()),
            });
        }
        Err(error) => return Err(error),
    };
    match serde_json::from_slice::<MacroFile>(&bytes) {
        Ok(data) => Ok(MigrationMacroPreview {
            name,
            event_count: data.steps.len(),
            readable: macro_issue(&data).is_none(),
            issue: macro_issue(&data),
        }),
        Err(_) => Ok(MigrationMacroPreview {
            name,
            event_count: 0,
            readable: false,
            issue: Some("Le fichier JSON ne peut pas être lu.".into()),
        }),
    }
}

fn read_legacy_settings(path: &Path) -> Result<LegacySettings, StoreError> {
    if !path.is_file() {
        return Ok(LegacySettings::default());
    }
    let bytes = read_bounded_file(path, MAX_LEGACY_CSV_BYTES)?;
    let text = String::from_utf8(bytes).map_err(|_| StoreError::InvalidData)?;
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b';')
        .has_headers(false)
        .from_reader(text.trim_start_matches('\u{feff}').as_bytes());
    let mut rows = reader.records();
    let Some(first) = rows.next() else {
        return Ok(LegacySettings::default());
    };
    let first = first.map_err(|_| StoreError::InvalidData)?;
    let headers: Vec<_> = first
        .iter()
        .map(|value| value.trim().to_ascii_lowercase())
        .collect();
    let name_index = headers.iter().position(|header| header == "parameter_name");
    let value_index = headers
        .iter()
        .position(|header| header == "parameter_value");
    let (name_index, value_index) = match (name_index, value_index) {
        (Some(name), Some(value)) => (name, value),
        _ => (0, 1),
    };
    let mut values = BTreeMap::new();
    if name_index == 0
        && value_index == 1
        && first
            .get(0)
            .is_some_and(|value| !value.trim().eq_ignore_ascii_case("parameter_name"))
    {
        insert_csv_row(&mut values, &first, name_index, value_index);
    }
    for row in rows {
        insert_csv_row(
            &mut values,
            &row.map_err(|_| StoreError::InvalidData)?,
            name_index,
            value_index,
        );
    }

    Ok(LegacySettings {
        loop_value: values.get("auto_loop").map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "y" | "on" | "t" | "vrai"
            )
        }),
        coc_path: values
            .get("coc_path")
            .cloned()
            .filter(|value| !value.is_empty()),
        last_macro: values
            .get("last_macro")
            .cloned()
            .filter(|value| !value.is_empty()),
        telegram_token: values
            .get("telegram_bot_token")
            .cloned()
            .filter(|value| !value.is_empty()),
    })
}

fn redact_legacy_token(csv_bytes: &[u8]) -> Result<Vec<u8>, StoreError> {
    if csv_bytes.len() > 2 * 1024 * 1024 {
        return Err(StoreError::InvalidData);
    }
    let text = std::str::from_utf8(csv_bytes).map_err(|_| StoreError::InvalidData)?;
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b';')
        .has_headers(false)
        .from_reader(text.trim_start_matches('\u{feff}').as_bytes());
    let mut rows = reader.records();
    let Some(first) = rows.next() else {
        return Ok(csv_bytes.to_vec());
    };
    let first = first.map_err(|_| StoreError::InvalidData)?;
    let headers = first
        .iter()
        .map(|value| value.trim().to_ascii_lowercase())
        .collect::<Vec<_>>();
    let named_columns = headers.iter().any(|header| header == "parameter_name")
        && headers.iter().any(|header| header == "parameter_value");
    let name_index = headers
        .iter()
        .position(|header| header == "parameter_name")
        .unwrap_or(0);
    let value_index = headers
        .iter()
        .position(|header| header == "parameter_value")
        .unwrap_or(1);

    let mut output = Vec::new();
    {
        let mut writer = csv::WriterBuilder::new()
            .delimiter(b';')
            .from_writer(&mut output);
        if named_columns {
            writer
                .write_record(&first)
                .map_err(|_| StoreError::InvalidData)?;
        } else {
            write_redacted_row(&mut writer, &first, name_index, value_index)?;
        }
        for row in rows {
            let row = row.map_err(|_| StoreError::InvalidData)?;
            write_redacted_row(&mut writer, &row, name_index, value_index)?;
        }
        writer.flush()?;
    }
    Ok(output)
}

fn write_redacted_row<W: Write>(
    writer: &mut csv::Writer<W>,
    row: &csv::StringRecord,
    name_index: usize,
    value_index: usize,
) -> Result<(), StoreError> {
    let mut fields = row.iter().map(str::to_owned).collect::<Vec<_>>();
    if fields
        .get(name_index)
        .is_some_and(|name| name.trim().eq_ignore_ascii_case("telegram_bot_token"))
        && let Some(value) = fields.get_mut(value_index)
    {
        *value = "[REDACTED]".into();
    }
    writer
        .write_record(fields)
        .map_err(|_| StoreError::InvalidData)
}

fn insert_csv_row(
    values: &mut BTreeMap<String, String>,
    row: &csv::StringRecord,
    name_index: usize,
    value_index: usize,
) {
    let Some(key) = row.get(name_index) else {
        return;
    };
    let Some(value) = row.get(value_index) else {
        return;
    };
    let key = key.trim().to_ascii_lowercase();
    if !key.is_empty() && !key.eq_ignore_ascii_case("parameter_name") {
        values.insert(key, value.trim().to_owned());
    }
}

fn legacy_fingerprint(config: &Path) -> Result<String, StoreError> {
    let mut files = Vec::new();
    let csv = config.join("data.csv");
    if csv.is_file() {
        files.push(("data.csv".to_owned(), csv, MAX_LEGACY_CSV_BYTES));
    }
    let macros = config.join("macros");
    if macros.is_dir() {
        for entry in fs::read_dir(macros)? {
            let path = entry?.path();
            if path
                .extension()
                .is_some_and(|extension| extension == "json")
            {
                let name = path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or("macro.json");
                files.push((format!("macros/{name}"), path, MAX_MACRO_BYTES));
            }
        }
    }
    let old_macro = config.join("macro.json");
    if old_macro.is_file() {
        files.push(("macro.json".into(), old_macro, MAX_MACRO_BYTES));
    }
    files.sort_by(|left, right| left.0.cmp(&right.0));
    let mut hasher = Sha256::new();
    for (name, path, max_bytes) in files {
        hasher.update(name.as_bytes());
        let mut file = fs::File::open(path)?;
        let length = file.metadata()?.len();
        hasher.update(length.to_le_bytes());
        update_fingerprint_contents(&mut file, length, max_bytes, &mut hasher)?;
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn update_fingerprint_contents(
    file: &mut (impl Read + Seek),
    length: u64,
    max_bytes: u64,
    hasher: &mut Sha256,
) -> Result<(), StoreError> {
    let mut buffer = [0; OVERSIZED_FINGERPRINT_SAMPLE_BYTES];
    if length > max_bytes {
        hasher.update(OVERSIZED_FINGERPRINT_MARKER);
        let sample_len = (length / 2).min(buffer.len() as u64) as usize;
        hasher.update((sample_len as u64).to_le_bytes());
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut buffer[..sample_len])?;
        hasher.update(&buffer[..sample_len]);
        hasher.update(b"\0omitted middle\0");
        file.seek(SeekFrom::End(-(sample_len as i64)))?;
        file.read_exact(&mut buffer[..sample_len])?;
        hasher.update(&buffer[..sample_len]);
    } else {
        let mut remaining = length;
        while remaining > 0 {
            let chunk_len = usize::try_from(remaining.min(buffer.len() as u64))
                .map_err(|_| StoreError::InvalidData)?;
            let read = file.read(&mut buffer[..chunk_len])?;
            if read == 0 {
                return Err(StoreError::InvalidData);
            }
            hasher.update(&buffer[..read]);
            remaining -= read as u64;
        }
        let mut extra = [0; 1];
        if file.read(&mut extra)? != 0 {
            return Err(StoreError::InvalidData);
        }
    }
    if file.seek(SeekFrom::End(0))? != length {
        return Err(StoreError::InvalidData);
    }
    Ok(())
}

fn file_fingerprint(content: &[u8]) -> String {
    format!("{:x}", Sha256::digest(content))
}

fn read_placeholder_manifest(root: &Path) -> Result<PlaceholderManifest, StoreError> {
    let path = root.join("system-placeholders.json");
    if !path.exists() {
        return Ok(PlaceholderManifest::default());
    }
    serde_json::from_slice(&fs::read(path)?).map_err(|_| StoreError::InvalidData)
}

fn can_replace_placeholder(
    name: &str,
    path: &Path,
    provenance: &PlaceholderManifest,
) -> Result<bool, StoreError> {
    let Some(expected) = provenance.files.get(name) else {
        return Ok(false);
    };
    let bytes = fs::read(path)?;
    if file_fingerprint(&bytes) != *expected {
        return Ok(false);
    }
    let data: MacroFile = serde_json::from_slice(&bytes).map_err(|_| StoreError::InvalidData)?;
    Ok(data.steps.is_empty() && macro_issue(&data).is_none())
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
            .update_settings(None, Some("C:\\Games\\CoC.exe".into()))
            .unwrap();
        store.settings_path = blocked_parent.join("settings.json");

        assert!(matches!(
            store.update_settings(Some(true), Some("C:\\Games\\Other.exe".into())),
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
    fn creates_protected_macros_and_natural_sort_order() {
        let dir = tempdir().unwrap();
        let store = Store::open_at(dir.path()).unwrap();
        let mut store = store;
        store.create_macro("Macro 10").unwrap();
        store.create_macro("Macro 2").unwrap();
        let names: Vec<_> = store
            .summaries()
            .unwrap()
            .into_iter()
            .map(|m| m.name)
            .collect();
        assert_eq!(&names[..2], ["Recharger COC", "Valider arrivée"]);
        assert!(
            names.iter().position(|n| n == "Macro 2") < names.iter().position(|n| n == "Macro 10")
        );
    }

    #[test]
    fn refuses_invalid_or_protected_macro_mutations() {
        let dir = tempdir().unwrap();
        let mut store = Store::open_at(dir.path()).unwrap();
        assert!(matches!(
            store.create_macro("../bad"),
            Err(StoreError::InvalidName)
        ));
        assert!(matches!(
            store.delete_macro("Recharger COC"),
            Err(StoreError::Protected)
        ));
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
    fn macro_preview_and_readability_enforce_playback_duration_limit() {
        let app = tempdir().unwrap();
        let old = tempdir().unwrap();
        let source_macros = old.path().join("config/macros");
        fs::create_dir_all(&source_macros).unwrap();
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
        let within_path = source_macros.join("WithinLimit.json");
        let over_path = source_macros.join("OverLimit.json");
        fs::write(&within_path, &within_limit).unwrap();
        fs::write(&over_path, &over_limit).unwrap();

        let (_, preview, _) = store.preview_legacy(old.path()).unwrap();
        assert!(
            preview
                .macros
                .iter()
                .find(|item| item.name == "WithinLimit")
                .unwrap()
                .readable
        );
        let over_preview = preview
            .macros
            .iter()
            .find(|item| item.name == "OverLimit")
            .unwrap();
        assert!(!over_preview.readable);
        assert!(over_preview.issue.as_deref().unwrap().contains("24 heures"));
        assert_eq!(fs::read(&within_path).unwrap(), within_limit);
        assert_eq!(fs::read(&over_path).unwrap(), over_limit);

        let local_path = store.macros_path.join("OverLimit.json");
        fs::write(&local_path, &over_limit).unwrap();
        let summary = store
            .summaries()
            .unwrap()
            .into_iter()
            .find(|item| item.name == "OverLimit")
            .unwrap();
        assert!(!summary.readable);
        assert_eq!(fs::read(local_path).unwrap(), over_limit);
    }

    #[test]
    fn sparse_oversized_macro_is_rejected_before_reading_its_contents() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("Oversized.json");
        let file = fs::File::create(&path).unwrap();
        file.set_len(MAX_MACRO_BYTES + 1).unwrap();

        let preview = preview_macro(&path, "Oversized".into()).unwrap();
        assert!(!preview.readable);
        assert_eq!(
            preview.issue.as_deref(),
            Some("Le fichier dépasse la taille autorisée.")
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
    fn sparse_oversized_legacy_csv_is_rejected_before_backup_creation() {
        let app = tempdir().unwrap();
        let old = tempdir().unwrap();
        let config = old.path().join("config");
        fs::create_dir_all(&config).unwrap();
        let csv_path = config.join("data.csv");
        let file = fs::File::create(&csv_path).unwrap();
        file.set_len(MAX_LEGACY_CSV_BYTES + 1).unwrap();
        let mut store = Store::open_at(app.path()).unwrap();

        assert!(matches!(
            store.import_legacy(old.path()),
            Err(StoreError::InvalidData)
        ));
        assert!(!app.path().join("migration-backups").exists());
        assert!(matches!(
            read_legacy_settings(&csv_path),
            Err(StoreError::InvalidData)
        ));
    }

    #[test]
    fn streaming_legacy_fingerprint_matches_the_previous_content_hash() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("config");
        let macros = config.join("macros");
        fs::create_dir_all(&macros).unwrap();
        fs::write(config.join("data.csv"), b"settings\0content").unwrap();
        fs::write(macros.join("Macro 2.json"), b"macro two").unwrap();
        fs::write(config.join("macro.json"), b"legacy macro").unwrap();

        let mut files = vec![
            ("data.csv".to_owned(), config.join("data.csv")),
            (
                "macros/Macro 2.json".to_owned(),
                macros.join("Macro 2.json"),
            ),
            ("macro.json".to_owned(), config.join("macro.json")),
        ];
        files.sort_by(|left, right| left.0.cmp(&right.0));
        let mut expected = Sha256::new();
        for (name, path) in files {
            let content = fs::read(path).unwrap();
            expected.update(name.as_bytes());
            expected.update((content.len() as u64).to_le_bytes());
            expected.update(content);
        }

        assert_eq!(
            legacy_fingerprint(&config).unwrap(),
            format!("{:x}", expected.finalize())
        );
    }

    #[test]
    fn oversized_fingerprint_reads_only_bounded_samples() {
        struct CountingReader {
            position: u64,
            length: u64,
            bytes_read: u64,
            grow_after_samples: bool,
        }

        impl Read for CountingReader {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                let remaining = self.length.saturating_sub(self.position);
                let read = buffer
                    .len()
                    .min(remaining.min(buffer.len() as u64) as usize);
                buffer[..read].fill(0x5a);
                self.position += read as u64;
                self.bytes_read += read as u64;
                if self.grow_after_samples
                    && self.bytes_read >= (OVERSIZED_FINGERPRINT_SAMPLE_BYTES * 2) as u64
                {
                    self.length += 1;
                    self.grow_after_samples = false;
                }
                Ok(read)
            }
        }

        impl Seek for CountingReader {
            fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
                let next = match position {
                    SeekFrom::Start(position) => position as i128,
                    SeekFrom::Current(offset) => self.position as i128 + offset as i128,
                    SeekFrom::End(offset) => self.length as i128 + offset as i128,
                };
                if next < 0 || next > u64::MAX as i128 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid seek"));
                }
                self.position = next as u64;
                Ok(self.position)
            }
        }

        let length = 1_u64 << 40;
        let mut reader = CountingReader {
            position: 0,
            length,
            bytes_read: 0,
            grow_after_samples: false,
        };
        let mut first_hash = Sha256::new();
        update_fingerprint_contents(&mut reader, length, MAX_MACRO_BYTES, &mut first_hash).unwrap();
        let first_hash = first_hash.finalize();
        assert_eq!(
            reader.bytes_read,
            (OVERSIZED_FINGERPRINT_SAMPLE_BYTES * 2) as u64
        );

        reader.position = 0;
        reader.bytes_read = 0;
        let mut second_hash = Sha256::new();
        update_fingerprint_contents(&mut reader, length, MAX_MACRO_BYTES, &mut second_hash)
            .unwrap();
        assert_eq!(first_hash, second_hash.finalize());
        assert_eq!(
            reader.bytes_read,
            (OVERSIZED_FINGERPRINT_SAMPLE_BYTES * 2) as u64
        );

        reader.position = 0;
        reader.bytes_read = 0;
        reader.grow_after_samples = true;
        let mut changed_hash = Sha256::new();
        assert!(matches!(
            update_fingerprint_contents(&mut reader, length, MAX_MACRO_BYTES, &mut changed_hash),
            Err(StoreError::InvalidData)
        ));
    }

    #[test]
    fn preview_legacy_marks_sparse_oversized_macro_unreadable() {
        let app = tempdir().unwrap();
        let old = tempdir().unwrap();
        let source_macros = old.path().join("config/macros");
        fs::create_dir_all(&source_macros).unwrap();
        let path = source_macros.join("Oversized.json");
        fs::File::create(&path)
            .unwrap()
            .set_len(MAX_MACRO_BYTES + 1)
            .unwrap();
        let store = Store::open_at(app.path()).unwrap();

        let (_, preview, _) = store.preview_legacy(old.path()).unwrap();
        let macro_preview = preview
            .macros
            .iter()
            .find(|item| item.name == "Oversized")
            .unwrap();
        assert!(!macro_preview.readable);
        assert_eq!(
            macro_preview.issue.as_deref(),
            Some("Le fichier dépasse la taille autorisée.")
        );
    }

    #[test]
    fn macro_preview_rejects_coordinates_outside_i32_without_rewriting_source() {
        let app = tempdir().unwrap();
        let old = tempdir().unwrap();
        let source_macros = old.path().join("config/macros");
        fs::create_dir_all(&source_macros).unwrap();
        let store = Store::open_at(app.path()).unwrap();
        let source = source_macros.join("OutOfRange.json");
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
        fs::write(&source, &original).unwrap();

        let (_, preview, _) = store.preview_legacy(old.path()).unwrap();
        let item = preview
            .macros
            .iter()
            .find(|item| item.name == "OutOfRange")
            .unwrap();
        assert!(!item.readable);
        assert!(
            item.issue
                .as_deref()
                .unwrap()
                .contains("position de souris")
        );
        assert_eq!(fs::read(&source).unwrap(), original);

        let local = store.macros_path.join("OutOfRange.json");
        fs::write(&local, &original).unwrap();
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
        assert_eq!(fs::read(local).unwrap(), original);
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
    fn import_replaces_only_untouched_protected_placeholder_and_is_idempotent() {
        let app = tempdir().unwrap();
        let old = tempdir().unwrap();
        let old_config = old.path().join("config");
        let old_macros = old_config.join("macros");
        fs::create_dir_all(&old_macros).unwrap();
        fs::write(
            old_config.join("data.csv"),
            "parameter_name;parameter_value\nauto_loop;true\n",
        )
        .unwrap();
        let historical = json!({
            "name":"Recharger COC", "updated_at":"2025-01-01T00:00:00Z", "sha1":"legacy",
            "steps":[{"t":0.0,"type":"nop","data":{}}]
        });
        let historical_bytes = serde_json::to_vec_pretty(&historical).unwrap();
        fs::write(old_macros.join("Recharger COC.json"), &historical_bytes).unwrap();

        let mut store = Store::open_at(app.path()).unwrap();
        let result = store.import_legacy(old.path()).unwrap();
        assert_eq!(result.imported.macros, 1);
        assert!(result.collisions.is_empty());
        assert_eq!(
            fs::read(app.path().join("macros/Recharger COC.json")).unwrap(),
            historical_bytes
        );
        assert!(store.settings().loop_playback);

        let before = fs::read(app.path().join("macros/Recharger COC.json")).unwrap();
        let repeated = store.import_legacy(old.path()).unwrap();
        let after = fs::read(app.path().join("macros/Recharger COC.json")).unwrap();
        assert_eq!(repeated.imported.macros, 1);
        assert_eq!(before, after);
        assert_eq!(
            fs::read(old_macros.join("Recharger COC.json")).unwrap(),
            historical_bytes
        );
        assert!(
            app.path()
                .join("migration-backups")
                .read_dir()
                .unwrap()
                .next()
                .is_some()
        );
    }

    #[test]
    fn migration_refuses_nonempty_protected_user_macro() {
        let app = tempdir().unwrap();
        let old = tempdir().unwrap();
        let old_macros = old.path().join("config/macros");
        fs::create_dir_all(&old_macros).unwrap();
        let custom = json!({"name":"Valider arrivée","steps":[{"t":0.0,"type":"key_down","data":{"key":"enter"}}]});
        fs::write(
            old_macros.join("Valider arrivée.json"),
            serde_json::to_vec(&custom).unwrap(),
        )
        .unwrap();
        let mut store = Store::open_at(app.path()).unwrap();
        let user_file = json!({"name":"Valider arrivée","steps":[{"t":0.0,"type":"key_down","data":{"key":"x"}}]});
        let user_bytes = serde_json::to_vec(&user_file).unwrap();
        fs::write(app.path().join("macros/Valider arrivée.json"), &user_bytes).unwrap();

        let result = store.import_legacy(old.path()).unwrap();
        assert!(result.collisions.contains(&"Valider arrivée".to_owned()));
        assert_eq!(
            fs::read(app.path().join("macros/Valider arrivée.json")).unwrap(),
            user_bytes
        );
    }

    #[cfg(windows)]
    #[test]
    fn migration_backup_redacts_the_token_and_keeps_source_for_recovery() {
        let app = tempdir().unwrap();
        let old = tempdir().unwrap();
        let config = old.path().join("config");
        fs::create_dir_all(&config).unwrap();
        let token = "synthetic-migration-token";
        let source_csv =
            format!("parameter_value;parameter_name\n{token};telegram_bot_token\n1;auto_loop\n");
        let source_path = config.join("data.csv");
        fs::write(&source_path, &source_csv).unwrap();

        let mut store = Store::open_at(app.path()).unwrap();
        store.import_legacy(old.path()).unwrap();

        let backup_dir = fs::read_dir(app.path().join("migration-backups"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let backup_csv = fs::read(backup_dir.join("data.csv")).unwrap();
        assert!(
            !backup_csv
                .windows(token.len())
                .any(|window| window == token.as_bytes())
        );
        let mut backup_reader = csv::ReaderBuilder::new()
            .delimiter(b';')
            .from_reader(backup_csv.as_slice());
        assert_eq!(
            backup_reader.headers().unwrap().iter().collect::<Vec<_>>(),
            ["parameter_value", "parameter_name"]
        );
        let rows = backup_reader
            .records()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(&rows[0][0], "[REDACTED]");
        assert_eq!(&rows[0][1], "telegram_bot_token");
        assert_eq!(&rows[1][0], "1");
        assert_eq!(&rows[1][1], "auto_loop");
        assert!(fs::read_to_string(&source_path).unwrap().contains(token));
        assert!(
            load_token(app.path())
                .unwrap()
                .is_some_and(|stored| stored == token)
        );
    }

    #[cfg(windows)]
    #[test]
    fn startup_recovers_a_protected_token_written_before_settings_persisted() {
        let app = tempdir().unwrap();
        let store = Store::open_at(app.path()).unwrap();
        assert!(!store.settings().telegram.token_configured);
        let token = "synthetic-interrupted-migration-token";

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
    fn startup_preserves_an_invalid_protected_token_until_explicit_migration_replaces_it() {
        let app = tempdir().unwrap();
        Store::open_at(app.path()).unwrap();
        let token_path = app.path().join("telegram-token.bin");
        let invalid_ciphertext = b"synthetic-invalid-protected-token";
        fs::write(&token_path, invalid_ciphertext).unwrap();

        let mut store = Store::open_at(app.path()).unwrap();
        assert!(!store.settings().telegram.token_configured);
        assert_eq!(
            store.settings().telegram.status,
            crate::types::TelegramStatus::Error
        );
        assert_eq!(fs::read(&token_path).unwrap(), invalid_ciphertext);

        let old = tempdir().unwrap();
        let config = old.path().join("config");
        fs::create_dir_all(&config).unwrap();
        let replacement = "synthetic-imported-replacement-token";
        fs::write(
            config.join("data.csv"),
            format!("parameter_name;parameter_value\ntelegram_bot_token;{replacement}\n"),
        )
        .unwrap();
        store.import_legacy(old.path()).unwrap();

        assert!(store.settings().telegram.token_configured);
        assert_eq!(
            load_token(app.path()).unwrap().as_deref(),
            Some(replacement)
        );
        assert_ne!(fs::read(token_path).unwrap(), invalid_ciphertext);
    }

    #[cfg(windows)]
    #[test]
    fn migration_keeps_pairing_when_a_protected_token_is_already_valid() {
        let app = tempdir().unwrap();
        let old = tempdir().unwrap();
        let config = old.path().join("config");
        fs::create_dir_all(&config).unwrap();
        let existing_token = "synthetic-existing-migration-token";
        let source_token = "synthetic-unselected-source-token";
        fs::write(
            config.join("data.csv"),
            format!("parameter_name;parameter_value\ntelegram_bot_token;{source_token}\n"),
        )
        .unwrap();

        let mut store = Store::open_at(app.path()).unwrap();
        save_token(app.path(), existing_token.as_bytes()).unwrap();
        store.settings_mut().telegram.token_configured = true;
        store.settings_mut().telegram.paired = true;
        store.settings_mut().telegram.status = crate::types::TelegramStatus::Connected;
        store.set_telegram_owner(100, 200);
        store.persist().unwrap();

        let mut before_install_called = false;
        let result = store
            .import_legacy_with_token_installation(old.path(), || before_install_called = true)
            .unwrap();

        assert_eq!(result.imported.settings, 0);
        assert!(!before_install_called);
        assert!(store.settings().telegram.token_configured);
        assert!(store.settings().telegram.paired);
        assert_eq!(store.telegram_owner(), Some((100, 200)));
        assert_eq!(
            load_token(app.path()).unwrap().as_deref(),
            Some(existing_token)
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

    #[test]
    fn csv_header_order_is_respected_and_legacy_chat_does_not_pair() {
        let app = tempdir().unwrap();
        let old = tempdir().unwrap();
        let config = old.path().join("config");
        fs::create_dir_all(&config).unwrap();
        fs::write(config.join("data.csv"), "parameter_value;parameter_name\nC:\\Games\\CoC.exe;coc_path\n1;auto_loop\n123;telegram_chat_id\n").unwrap();
        let mut store = Store::open_at(app.path()).unwrap();
        store.import_legacy(old.path()).unwrap();
        assert!(store.settings().loop_playback);
        assert_eq!(store.settings().coc_path, "C:\\Games\\CoC.exe");
        assert!(!store.settings().telegram.paired);
    }
}
