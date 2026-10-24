use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Diagnostics {
    pub app_version: String,
    pub rust_version: String,
    pub status: String,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Help {
    pub sections: Vec<HelpSection>,
}

#[derive(Clone, Debug, Serialize)]
pub struct HelpSection {
    pub title: String,
    pub body: String,
}

impl Default for Help {
    fn default() -> Self {
        Self {
            sections: vec![
                HelpSection {
                    title: "Créer une macro".into(),
                    body: "Créez une macro, sélectionnez-la puis démarrez l’enregistrement. La capture clavier/souris commence après un délai de préparation de 3 secondes. Arrêtez-la depuis l’application.".into(),
                },
                HelpSection {
                    title: "Lire et arrêter".into(),
                    body: "La lecture rejoue les entrées clavier et souris enregistrées. Gardez l’application ouverte pendant une opération et utilisez Arrêter dès qu’un replay ne se déroule pas comme prévu.".into(),
                },
                HelpSection {
                    title: "Garde du premier plan".into(),
                    body: concat!(
                        "L’enregistrement et le replay démarrent uniquement quand CoC est reconnu au premier plan. Quand la pause automatique est activée, une action lancée depuis l’atelier peut attendre jusqu’à 30 secondes que Clash of Clans passe au premier plan. Les raccourcis et Telegram ne patientent pas. Le replay envoie ses actions uniquement à la fenêtre du jeu, même si l’option est désactivée. Google Play Games peut ignorer ces actions ciblées.\n\n",
                        "Si CoC perd le premier plan pendant un replay, AUTO-COC relâche les touches et boutons maintenus. L’option activée met ensuite le replay en pause; désactivée, elle l’arrête. Si la libération échoue, le replay s’arrête avec une erreur. Pour reprendre au même point, choisissez « Reprendre », puis retournez dans le jeu. « Annuler la reprise » garde le replay en pause. Vous pouvez l’arrêter depuis l’atelier, avec le raccourci Arrêter, Telegram STOP ou en fermant l’application.\n\n",
                        "Avec la pause activée, une perte de focus pendant la capture arrête l’enregistrement et sauvegarde les actions déjà capturées. Si la sauvegarde échoue, utilisez Arrêter pour réessayer.\n\n",
                        "La capture est limitée à CoC; les actions souris sont gardées uniquement dans sa zone de jeu. Les nouvelles macros suivent les coordonnées de CoC si sa fenêtre bouge. Les anciennes macros gardent les coordonnées écran; les déplacements souris hors zone sont ignorés, mais un clic ou un défilement hors zone arrête le replay. Un crash interne du jeu peut ne pas être détecté si la fenêtre Google Play Games reste ouverte avec le même titre."
                    )
                    .into(),
                },
                HelpSection {
                    title: "Quitter AUTO-COC".into(),
                    body: "Utilisez Réglages > Système > Quitter AUTO-COC ou fermez complètement la fenêtre de l’application pour arrêter ses services. Arrêtez d’abord tout enregistrement ou replay en cours.".into(),
                },
                HelpSection {
                    title: "Données".into(),
                    body: "Les macros sont stockées dans le dossier AUTO-COC de votre profil Windows.".into(),
                },
            ],
        }
    }
}

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
    TokenRejected,
    NetworkError,
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
    #[serde(default)]
    pub require_coc_foreground: bool,
    pub coc_path: String,
    pub shortcuts: ShortcutSettings,
    pub telegram: TelegramSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            loop_playback: false,
            require_coc_foreground: true,
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
    WaitingForForeground {
        action: PendingAction,
        #[serde(rename = "macroName")]
        macro_name: String,
    },
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
    Paused {
        #[serde(rename = "macroName")]
        macro_name: String,
        #[serde(rename = "elapsedSeconds")]
        elapsed_seconds: f64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PendingAction {
    Recording,
    Playing,
    Resuming,
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
    pub onboarding_complete: bool,
    pub last_error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PairingStart {
    pub code: String,
    pub expires_at: String,
    pub snapshot: Snapshot,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShutdownPreparation {
    pub confirmation_id: String,
    pub expires_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_settings_enable_the_guard_but_legacy_settings_without_it_do_not() {
        let settings: Settings = serde_json::from_value(serde_json::json!({
            "loop": false,
            "cocPath": "",
            "shortcuts": {"toggle": "F1", "play": "Ctrl+Shift+1", "stop": "Ctrl+Shift+0"},
            "telegram": {
                "tokenConfigured": false,
                "paired": false,
                "status": "not_configured",
                "pairingCode": null,
                "pairingExpiresAt": null
            }
        }))
        .unwrap();

        assert!(!settings.require_coc_foreground);
        assert!(Settings::default().require_coc_foreground);
    }

    #[test]
    fn help_explains_foreground_guard_pause_and_resume_behavior() {
        let section = Help::default()
            .sections
            .into_iter()
            .find(|section| section.title == "Garde du premier plan")
            .unwrap();

        for phrase in [
            "depuis l’atelier peut attendre jusqu’à 30 secondes",
            "Les raccourcis et Telegram ne patientent pas",
            "L’enregistrement et le replay démarrent uniquement quand CoC est reconnu au premier plan",
            "uniquement à la fenêtre du jeu",
            "Google Play Games peut ignorer",
            "Si CoC perd le premier plan pendant un replay",
            "met ensuite le replay en pause",
            "désactivée, elle l’arrête",
            "la libération échoue",
            "Reprendre",
            "au même point",
            "Annuler la reprise",
            "Telegram STOP",
            "fermant l’application",
            "une perte de focus pendant la capture arrête l’enregistrement",
            "sauvegarde les actions déjà capturées",
            "La capture est limitée à CoC",
            "Les nouvelles macros suivent les coordonnées de CoC si sa fenêtre bouge",
            "Les anciennes macros gardent les coordonnées écran",
            "les déplacements souris hors zone sont ignorés",
            "Un crash interne du jeu peut ne pas être détecté",
            "la fenêtre Google Play Games reste ouverte avec le même titre",
        ] {
            assert!(section.body.contains(phrase), "help is missing: {phrase}");
        }
    }

    #[test]
    fn help_does_not_claim_to_import_or_copy_legacy_macros() {
        let section = Help::default()
            .sections
            .into_iter()
            .find(|section| section.title == "Données")
            .unwrap();

        assert!(section.body.contains("dossier AUTO-COC"));
        assert!(!section.body.contains("héritées"));
        assert!(!section.body.contains("copiées"));
    }

    #[test]
    fn help_does_not_show_static_shortcuts() {
        assert!(
            Help::default()
                .sections
                .iter()
                .all(|section| section.title != "Raccourcis par défaut")
        );
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
pub struct SettingsPatch {
    #[serde(rename = "loop")]
    pub loop_playback: Option<bool>,
    pub require_coc_foreground: Option<bool>,
    pub coc_path: Option<String>,
}
