use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShortcutSettings {
    pub toggle: String,
    pub play: String,
    pub stop: String,
}

impl Default for ShortcutSettings {
    fn default() -> Self {
        Self {
            toggle: "F1".into(),
            play: "Ctrl+Shift+1".into(),
            stop: "Ctrl+Shift+0".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TelegramStatus {
    NotConfigured,
    Disconnected,
    WaitingPairing,
    Connected,
    Error,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TelegramSettings {
    pub token_configured: bool,
    pub paired: bool,
    pub status: TelegramStatus,
    pub pairing_code: Option<String>,
    pub pairing_expires_at: Option<String>,
}

impl Default for TelegramSettings {
    fn default() -> Self {
        Self {
            token_configured: false,
            paired: false,
            status: TelegramStatus::NotConfigured,
            pairing_code: None,
            pairing_expires_at: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    #[serde(rename = "loop")]
    pub loop_playback: bool,
    pub coc_path: String,
    pub shortcuts: ShortcutSettings,
    pub telegram: TelegramSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            loop_playback: false,
            coc_path: String::new(),
            shortcuts: ShortcutSettings::default(),
            telegram: TelegramSettings::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AppStatus {
    Idle,
    Recording {
        #[serde(rename = "macroName")]
        macro_name: String,
        phase: RecordingPhase,
        #[serde(rename = "countdownSeconds")]
        countdown_seconds: f64,
        #[serde(rename = "elapsedSeconds")]
        elapsed_seconds: f64,
    },
    Playing {
        #[serde(rename = "macroName")]
        macro_name: String,
        #[serde(rename = "elapsedSeconds")]
        elapsed_seconds: f64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordingPhase {
    Preparing,
    Capturing,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionStatus {
    pub elapsed_seconds: f64,
    pub cycles: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MacroSummary {
    pub name: String,
    pub event_count: usize,
    pub duration_seconds: f64,
    pub protected: bool,
    pub readable: bool,
    pub issue: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Snapshot {
    pub revision: u64,
    pub status: AppStatus,
    #[serde(rename = "selectedMacro")]
    pub selected_macro: Option<String>,
    pub macros: Vec<MacroSummary>,
    pub settings: Settings,
    pub session: SessionStatus,
    pub migration: MigrationStatus,
    pub onboarding_complete: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MigrationStatus {
    pub source_selected: bool,
    pub available: bool,
    pub already_imported: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MigrationMacroPreview {
    pub name: String,
    pub event_count: usize,
    pub readable: bool,
    pub issue: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MigrationPreview {
    pub available: bool,
    pub settings_found: bool,
    pub macros: Vec<MigrationMacroPreview>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct MigrationCounts {
    pub settings: usize,
    pub macros: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MigrationResult {
    pub imported: MigrationCounts,
    pub collisions: Vec<String>,
    pub preserved: Vec<String>,
    pub errors: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_result_does_not_expose_internal_source_fingerprint() {
        let result = MigrationResult {
            imported: MigrationCounts::default(),
            collisions: Vec::new(),
            preserved: Vec::new(),
            errors: Vec::new(),
        };
        let json = serde_json::to_value(result).unwrap();
        assert!(json.get("fingerprint").is_none());
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MacroFile {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub updated_at: String,
    #[serde(default)]
    pub sha1: String,
    #[serde(default)]
    pub steps: Vec<Value>,
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CreateMacroRequest {
    pub name: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RenameMacroRequest {
    pub new_name: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SelectionRequest {
    pub name: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SettingsPatch {
    #[serde(rename = "loop")]
    pub loop_playback: Option<bool>,
    pub coc_path: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ShortcutRequest {
    pub toggle: String,
    pub play: String,
    pub stop: String,
}

impl From<ShortcutRequest> for ShortcutSettings {
    fn from(request: ShortcutRequest) -> Self {
        Self {
            toggle: request.toggle,
            play: request.play,
            stop: request.stop,
        }
    }
}
