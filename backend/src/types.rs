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
                    title: "Replay en arrière-plan".into(),
                    body: concat!(
                        "L’enregistrement commence uniquement quand CoC est au premier plan et s’arrête si le focus est perdu. Avec l’option activée, le replay peut démarrer dès qu’une fenêtre Clash of Clans reconnue est ouverte et continue si une autre application passe au premier plan. Désactivée, la lecture démarre seulement au premier plan; si CoC perd le focus, elle se met en pause et attend son retour. Les actions sont postées uniquement à la fenêtre ciblée du jeu, sans injection clavier ou souris globale. Google Play Games peut ignorer ces messages.\n\n",
                        "Si la fenêtre ou le processus ciblé disparaît, le replay s’arrête. AUTO-COC tente de relâcher les touches et boutons maintenus sur cette même fenêtre; si elle est déjà détruite, cette libération peut échouer. Pour reprendre une lecture mise en pause, choisissez « Reprendre », puis retournez dans le jeu. « Annuler la reprise » garde le replay en pause. Vous pouvez aussi l’arrêter depuis l’atelier, avec le raccourci Arrêter, Telegram STOP ou en fermant l’application.\n\n",
                        "Quand le mode arrière-plan est désactivé, une action lancée depuis l’atelier peut attendre jusqu’à 30 secondes que Clash of Clans passe au premier plan. Les raccourcis et Telegram ne patientent pas. La capture est limitée à CoC; les actions souris sont gardées uniquement dans sa zone de jeu. Si la sauvegarde d’un enregistrement arrêté après une perte de focus échoue, utilisez Arrêter pour réessayer.\n\n",
                        "Les nouvelles macros suivent les coordonnées de CoC si sa fenêtre bouge. Les anciennes macros gardent les coordonnées écran; les déplacements souris hors zone sont ignorés, mais un clic ou un défilement hors zone arrête le replay. Un crash interne du jeu peut ne pas être détecté si la fenêtre Google Play Games reste ouverte avec le même titre."
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
    #[serde(
        default = "default_background_playback",
        alias = "requireCocForeground"
    )]
    pub allow_coc_background_playback: bool,
    pub coc_path: String,
    pub shortcuts: ShortcutSettings,
    pub telegram: TelegramSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            loop_playback: false,
            allow_coc_background_playback: true,
            coc_path: String::new(),
            shortcuts: ShortcutSettings::default(),
            telegram: TelegramSettings::default(),
        }
    }
}

fn default_background_playback() -> bool {
    true
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
    fn background_replay_defaults_on_and_accepts_the_previous_setting_name() {
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

        assert!(settings.allow_coc_background_playback);
        assert!(Settings::default().allow_coc_background_playback);

        let legacy: Settings = serde_json::from_value(serde_json::json!({
            "loop": false,
            "requireCocForeground": false,
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
        assert!(!legacy.allow_coc_background_playback);
        assert!(
            serde_json::to_value(legacy)
                .unwrap()
                .get("allowCocBackgroundPlayback")
                .is_some()
        );
    }

    #[test]
    fn help_explains_targeted_background_replay_and_recording_guard() {
        let section = Help::default()
            .sections
            .into_iter()
            .find(|section| section.title == "Replay en arrière-plan")
            .unwrap();

        for phrase in [
            "L’enregistrement commence uniquement quand CoC est au premier plan",
            "continue si une autre application passe au premier plan",
            "uniquement à la fenêtre ciblée du jeu",
            "sans injection clavier ou souris globale",
            "Google Play Games peut ignorer",
            "Si la fenêtre ou le processus ciblé disparaît",
            "cette libération peut échouer",
            "peut attendre jusqu’à 30 secondes",
            "s’arrête si le focus est perdu",
            "Reprendre",
            "Annuler la reprise",
            "Telegram STOP",
            "fermant l’application",
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
    #[serde(alias = "requireCocForeground")]
    pub allow_coc_background_playback: Option<bool>,
    pub coc_path: Option<String>,
}
