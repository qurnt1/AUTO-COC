use auto_coc::{
    DesktopRuntimeError, Diagnostics, Help, MigrationImport, MigrationSelection,
    MigrationStatusResponse, PairingStart, PersistenceError, ServiceError, SettingsPatch,
    ShortcutSettings, ShutdownPreparation, Snapshot,
};
use serde::Serialize;
use tauri::{AppHandle, State, ipc::Response};

use crate::HostState;

#[derive(Clone, Debug, Serialize)]
pub struct CommandError {
    code: &'static str,
    message: &'static str,
}

impl From<DesktopRuntimeError> for CommandError {
    fn from(error: DesktopRuntimeError) -> Self {
        match error {
            DesktopRuntimeError::Controller(error) => error.into(),
            DesktopRuntimeError::TelegramTaskFailed => Self {
                code: "telegram_shutdown_failed",
                message: "Le service Telegram n’a pas pu s’arrêter proprement.",
            },
        }
    }
}

impl From<ServiceError> for CommandError {
    fn from(error: ServiceError) -> Self {
        match error {
            ServiceError::Store(PersistenceError::NotFound) => {
                Self::new("not_found", "Cette macro est introuvable.")
            }
            ServiceError::Store(PersistenceError::InvalidName) => Self::new(
                "invalid_name",
                "Le nom contient des caractères non autorisés.",
            ),
            ServiceError::Store(PersistenceError::Conflict) => {
                Self::new("conflict", "Une macro de ce nom existe déjà.")
            }
            ServiceError::Store(PersistenceError::Protected) => Self::new(
                "protected_macro",
                "Cette macro système ne peut pas être modifiée.",
            ),
            ServiceError::Store(PersistenceError::Unreadable) => Self::new(
                "macro_unreadable",
                "Cette macro contient des données invalides et n’a pas été modifiée.",
            ),
            ServiceError::Store(PersistenceError::InvalidData) => {
                Self::new("invalid_data", "Les données fournies sont invalides.")
            }
            ServiceError::Store(PersistenceError::Io(_)) | ServiceError::NativeInputFailed => {
                Self::internal()
            }
            ServiceError::InvalidState => Self::new(
                "invalid_state",
                "Cette opération n’est pas disponible dans l’état actuel.",
            ),
            ServiceError::NativeInputUnavailable => Self::new(
                "native_input_unavailable",
                "Le moteur clavier/souris Windows n’est pas disponible.",
            ),
            ServiceError::InvalidShortcut => Self::new(
                "invalid_shortcut",
                "Le raccourci doit contenir une combinaison de touches prise en charge.",
            ),
            ServiceError::ShortcutUnavailable => Self::new(
                "shortcut_unavailable",
                "Au moins un raccourci est déjà utilisé par Windows ou une autre application.",
            ),
            ServiceError::RecordingSavePending => Self::new(
                "recording_save_pending",
                "Les étapes sont conservées en mémoire. Relancez Arrêter pour réessayer leur sauvegarde.",
            ),
            ServiceError::RecordingEventLimit => Self::new(
                "recording_event_limit",
                "La limite d’événements a été atteinte. La partie déjà capturée est sauvegardée, mais la macro peut être incomplète.",
            ),
            ServiceError::ScreenshotUnavailable => Self::new(
                "screenshot_unavailable",
                "La capture d’écran n’est pas disponible sur ce poste.",
            ),
            ServiceError::EmptyMacro => Self::new(
                "empty_macro",
                "La macro sélectionnée ne contient aucun événement.",
            ),
            ServiceError::TelegramNotConfigured => Self::new(
                "telegram_not_configured",
                "Configurez d’abord le bot Telegram.",
            ),
            ServiceError::TelegramUnauthorized => Self::new(
                "telegram_unauthorized",
                "Cette action Telegram n’est plus autorisée.",
            ),
            ServiceError::TelegramUnavailable => Self::new(
                "telegram_unavailable",
                "Telegram n’a pas accepté le token ou ne répond pas.",
            ),
            ServiceError::Expired => Self::new(
                "expired",
                "Cette confirmation a expiré ou a déjà été utilisée.",
            ),
            ServiceError::PairingRateLimited => Self::new(
                "pairing_rate_limited",
                "Trop d’essais. Attendez avant de générer un nouveau code.",
            ),
            ServiceError::MigrationCancelled => {
                Self::new("migration_cancelled", "Sélection de dossier annulée.")
            }
        }
    }
}

impl CommandError {
    const fn new(code: &'static str, message: &'static str) -> Self {
        Self { code, message }
    }

    const fn internal() -> Self {
        Self::new("internal_error", "L’opération n’a pas pu aboutir.")
    }
}

#[tauri::command]
pub async fn get_snapshot(state: State<'_, HostState>) -> Result<Snapshot, CommandError> {
    state.runtime.get_snapshot().await.map_err(Into::into)
}

#[tauri::command]
pub async fn get_macro(
    name: String,
    state: State<'_, HostState>,
) -> Result<serde_json::Value, CommandError> {
    state.runtime.get_macro(&name).await.map_err(Into::into)
}

#[tauri::command]
pub async fn create_macro(
    name: String,
    state: State<'_, HostState>,
) -> Result<Snapshot, CommandError> {
    state.runtime.create_macro(&name).await.map_err(Into::into)
}

#[tauri::command]
pub async fn rename_macro(
    name: String,
    new_name: String,
    state: State<'_, HostState>,
) -> Result<Snapshot, CommandError> {
    state
        .runtime
        .rename_macro(&name, &new_name)
        .await
        .map_err(Into::into)
}

#[tauri::command]
pub async fn delete_macro(
    name: String,
    state: State<'_, HostState>,
) -> Result<Snapshot, CommandError> {
    state.runtime.delete_macro(&name).await.map_err(Into::into)
}

#[tauri::command]
pub async fn select_macro(
    name: String,
    state: State<'_, HostState>,
) -> Result<Snapshot, CommandError> {
    state.runtime.select_macro(&name).await.map_err(Into::into)
}

#[tauri::command]
pub async fn start_recording(state: State<'_, HostState>) -> Result<Snapshot, CommandError> {
    state.runtime.start_recording().await.map_err(Into::into)
}

#[tauri::command]
pub async fn stop_recording(state: State<'_, HostState>) -> Result<Snapshot, CommandError> {
    state.runtime.stop_recording().await.map_err(Into::into)
}

#[tauri::command]
pub async fn start_playback(state: State<'_, HostState>) -> Result<Snapshot, CommandError> {
    state.runtime.start_playback().await.map_err(Into::into)
}

#[tauri::command]
pub async fn stop_playback(state: State<'_, HostState>) -> Result<Snapshot, CommandError> {
    state.runtime.stop_playback().await.map_err(Into::into)
}

#[tauri::command]
pub async fn patch_settings(
    settings: SettingsPatch,
    state: State<'_, HostState>,
) -> Result<Snapshot, CommandError> {
    state
        .runtime
        .patch_settings(settings)
        .await
        .map_err(Into::into)
}

#[tauri::command]
pub async fn put_shortcuts(
    shortcuts: ShortcutSettings,
    state: State<'_, HostState>,
) -> Result<Snapshot, CommandError> {
    state
        .runtime
        .put_shortcuts(shortcuts)
        .await
        .map_err(Into::into)
}

#[tauri::command]
pub async fn launch_coc(state: State<'_, HostState>) -> Result<Snapshot, CommandError> {
    state.runtime.launch_coc().await.map_err(Into::into)
}

#[tauri::command]
pub async fn screenshot(state: State<'_, HostState>) -> Result<Response, CommandError> {
    state
        .runtime
        .screenshot()
        .await
        .map(Response::new)
        .map_err(Into::into)
}

#[tauri::command]
pub async fn open_macros_folder(state: State<'_, HostState>) -> Result<Snapshot, CommandError> {
    state.runtime.open_macros_folder().await.map_err(Into::into)
}

#[tauri::command]
pub async fn quit_application(
    app: AppHandle,
    state: State<'_, HostState>,
) -> Result<(), CommandError> {
    crate::shutdown(&state).await?;
    app.exit(0);
    Ok(())
}

#[tauri::command]
pub async fn telegram_config(
    token: String,
    state: State<'_, HostState>,
) -> Result<Snapshot, CommandError> {
    state
        .runtime
        .telegram_config(&token)
        .await
        .map_err(Into::into)
}

#[tauri::command]
pub async fn telegram_config_clear(state: State<'_, HostState>) -> Result<Snapshot, CommandError> {
    state
        .runtime
        .telegram_config_clear()
        .await
        .map_err(Into::into)
}

#[tauri::command]
pub async fn pairing_start(state: State<'_, HostState>) -> Result<PairingStart, CommandError> {
    state.runtime.pairing_start().await.map_err(Into::into)
}

#[tauri::command]
pub async fn pairing_cancel(state: State<'_, HostState>) -> Result<Snapshot, CommandError> {
    state.runtime.pairing_cancel().await.map_err(Into::into)
}

#[tauri::command]
pub async fn telegram_test(state: State<'_, HostState>) -> Result<Snapshot, CommandError> {
    state.runtime.telegram_test().await.map_err(Into::into)
}

#[tauri::command]
pub async fn get_migration(
    state: State<'_, HostState>,
) -> Result<MigrationStatusResponse, CommandError> {
    state.runtime.get_migration().await.map_err(Into::into)
}

#[tauri::command]
pub async fn select_migration_source(
    state: State<'_, HostState>,
) -> Result<MigrationSelection, CommandError> {
    state
        .runtime
        .select_migration_source()
        .await
        .map_err(Into::into)
}

#[tauri::command]
pub async fn import_migration(
    state: State<'_, HostState>,
) -> Result<MigrationImport, CommandError> {
    state.runtime.import_migration().await.map_err(Into::into)
}

#[tauri::command]
pub async fn shutdown_prepare(
    state: State<'_, HostState>,
) -> Result<ShutdownPreparation, CommandError> {
    state.runtime.shutdown_prepare().await.map_err(Into::into)
}

#[tauri::command]
pub async fn shutdown_confirm(
    confirmation_id: String,
    state: State<'_, HostState>,
) -> Result<Snapshot, CommandError> {
    state
        .runtime
        .shutdown_confirm(&confirmation_id)
        .await
        .map_err(Into::into)
}

#[tauri::command]
pub async fn get_help(state: State<'_, HostState>) -> Result<Help, CommandError> {
    Ok(state.runtime.get_help().await)
}

#[tauri::command]
pub async fn get_diagnostics(state: State<'_, HostState>) -> Result<Diagnostics, CommandError> {
    state.runtime.get_diagnostics().await.map_err(Into::into)
}

#[tauri::command]
pub async fn complete_onboarding(state: State<'_, HostState>) -> Result<Snapshot, CommandError> {
    state
        .runtime
        .complete_onboarding()
        .await
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use auto_coc::ServiceError;

    use super::CommandError;

    #[test]
    fn migration_cancellation_uses_frontend_error_contract() {
        let error = CommandError::from(ServiceError::MigrationCancelled);
        let payload = serde_json::to_value(error).unwrap();

        assert_eq!(payload["code"], "migration_cancelled");
        assert_eq!(payload["message"], "Sélection de dossier annulée.");
    }
}
