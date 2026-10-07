use std::{
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use serde_json::Value;
use thiserror::Error;
use tokio::{
    sync::{Mutex, mpsc, watch},
    task::JoinHandle,
};

use crate::{
    native_input::{NativeEvent, NativeInput, PlaybackFailure},
    storage::{Store, StoreError},
    types::{
        AppStatus, MigrationPreview, MigrationResult, MigrationStatus, RecordingPhase,
        SessionStatus, Snapshot, TelegramStatus,
    },
};

const PAIRING_LIFETIME: Duration = Duration::from_secs(300);
const PAIRING_COOLDOWN: Duration = Duration::from_secs(30);
const MAX_PAIRING_ATTEMPTS: u8 = 5;
const SHUTDOWN_CONFIRM_LIFETIME: Duration = Duration::from_secs(30);

#[derive(Debug, Error)]
pub enum ControllerError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("operation is not available in this state")]
    InvalidState,
    #[error("native keyboard and mouse support is not available")]
    NativeInputUnavailable,
    #[error("shortcut syntax is invalid")]
    InvalidShortcut,
    #[error("shortcut is already registered by another application")]
    ShortcutUnavailable,
    #[error("native input operation failed")]
    NativeInputFailed,
    #[error("recorded steps could not be saved and remain available for retry")]
    RecordingSavePending,
    #[error("recording reached the event limit and only its collected events were saved")]
    RecordingEventLimit,
    #[error("screen capture is unavailable")]
    ScreenshotUnavailable,
    #[error("macro is empty")]
    EmptyMacro,
    #[error("telegram integration is not configured")]
    TelegramNotConfigured,
    #[error("telegram action is no longer authorized")]
    TelegramUnauthorized,
    #[error("telegram connectivity test is not available")]
    TelegramUnavailable,
    #[error("operation has expired")]
    Expired,
    #[error("telegram pairing is temporarily rate limited")]
    PairingRateLimited,
    #[error("migration folder selection was cancelled")]
    MigrationCancelled,
}

#[derive(Clone, Debug)]
enum Operation {
    Idle,
    Recording {
        macro_name: String,
        started: Instant,
    },
    Playing {
        macro_name: String,
        started: Instant,
        playback_id: u64,
    },
}

struct Pairing {
    code: String,
    expires: Instant,
    expires_at: String,
    attempts: u8,
}

struct ShutdownConfirmation {
    id: String,
    expires: Instant,
}

struct Machine {
    store: Store,
    operation: Operation,
    closing: bool,
    shutdown_complete: bool,
    pending_recording: Option<(String, Vec<Value>, bool)>,
    coc_launched_once: bool,
    cycles: u64,
    revision: u64,
    pairing: Option<Pairing>,
    pairing_cooldown_until: Option<Instant>,
    shutdown: Option<ShutdownConfirmation>,
    last_error: Option<String>,
    migration_source: Option<PathBuf>,
    migration_preview: Option<MigrationPreview>,
    migration_already_imported: bool,
}

pub struct Controller {
    machine: Mutex<Machine>,
    native: Mutex<Option<NativeInput>>,
    native_events_tx: mpsc::UnboundedSender<NativeEvent>,
    native_events_rx: Mutex<Option<mpsc::UnboundedReceiver<NativeEvent>>>,
    native_event_loop_started: Mutex<bool>,
    native_event_loop_shutdown: watch::Sender<bool>,
    native_event_loop_task: Mutex<Option<JoinHandle<()>>>,
    revision_tx: watch::Sender<u64>,
    #[cfg(test)]
    shutdown_race_gate: Mutex<Option<Arc<ShutdownRaceGate>>>,
}

#[cfg(test)]
struct ShutdownRaceGate {
    paused: tokio::sync::Notify,
    resume: tokio::sync::Notify,
}

impl Controller {
    pub fn new(store: Store) -> Self {
        let migration_already_imported = store.migration_was_imported().unwrap_or(false);
        let (revision_tx, _) = watch::channel(1);
        let (native_event_loop_shutdown, _) = watch::channel(false);
        let (native_events_tx, native_events_rx) = mpsc::unbounded_channel();
        Self {
            machine: Mutex::new(Machine {
                store,
                operation: Operation::Idle,
                closing: false,
                shutdown_complete: false,
                pending_recording: None,
                coc_launched_once: false,
                cycles: 0,
                revision: 1,
                pairing: None,
                pairing_cooldown_until: None,
                shutdown: None,
                last_error: None,
                migration_source: None,
                migration_preview: None,
                migration_already_imported,
            }),
            native: Mutex::new(None),
            native_events_tx,
            native_events_rx: Mutex::new(Some(native_events_rx)),
            native_event_loop_started: Mutex::new(false),
            native_event_loop_shutdown,
            native_event_loop_task: Mutex::new(None),
            revision_tx,
            #[cfg(test)]
            shutdown_race_gate: Mutex::new(None),
        }
    }

    pub async fn init_native(self: &Arc<Self>) -> Result<(), ControllerError> {
        self.start_native_event_loop().await;
        if self.native.lock().await.is_some() {
            return Ok(());
        }
        let shortcuts = self.machine.lock().await.store.settings().shortcuts.clone();
        match NativeInput::start(shortcuts, self.native_events_tx.clone()) {
            Ok(native) => {
                *self.native.lock().await = Some(native);
                let mut machine = self.machine.lock().await;
                machine.last_error = None;
                self.changed(&mut machine).await?;
            }
            Err(_) => {
                let mut machine = self.machine.lock().await;
                machine.last_error = Some("Les raccourcis globaux n’ont pas pu être enregistrés. Modifiez-les dans Réglages.".into());
                self.changed(&mut machine).await?;
            }
        }
        Ok(())
    }

    async fn start_native_event_loop(self: &Arc<Self>) {
        let mut started = self.native_event_loop_started.lock().await;
        if *started {
            return;
        }
        if *self.native_event_loop_shutdown.borrow() {
            return;
        }
        let Some(mut events_rx) = self.native_events_rx.lock().await.take() else {
            return;
        };
        *started = true;
        let controller = Arc::downgrade(self);
        let mut shutdown = self.native_event_loop_shutdown.subscribe();
        let task = tokio::spawn(async move {
            loop {
                let event = tokio::select! {
                    biased;
                    changed = shutdown.changed() => {
                        if changed.is_err() || *shutdown.borrow() {
                            break;
                        }
                        continue;
                    }
                    event = events_rx.recv() => match event {
                        Some(event) => event,
                        None => break,
                    },
                };
                let Some(controller) = controller.upgrade() else {
                    break;
                };
                if controller.handle_native_event(event).await.is_err() {
                    let mut machine = controller.machine.lock().await;
                    if machine.last_error.is_none() {
                        machine.last_error =
                            Some("Une action clavier n’a pas pu être exécutée.".into());
                    }
                    let _ = controller.changed(&mut machine).await;
                }
            }
        });
        *self.native_event_loop_task.lock().await = Some(task);
    }

    async fn stop_native_event_loop(&self) -> Result<(), ControllerError> {
        self.native_event_loop_shutdown.send_replace(true);
        if let Some(task) = self.native_event_loop_task.lock().await.take() {
            task.await.map_err(|_| ControllerError::NativeInputFailed)?;
        }
        Ok(())
    }

    async fn handle_native_event(&self, event: NativeEvent) -> Result<(), ControllerError> {
        match event {
            NativeEvent::Toggle => {
                let operation = self.machine.lock().await.operation.clone();
                match operation {
                    Operation::Idle => {
                        self.start_playback().await?;
                    }
                    Operation::Playing { .. } => {
                        self.stop_playback().await?;
                    }
                    Operation::Recording { .. } => {}
                }
            }
            NativeEvent::Play => {
                if matches!(&self.machine.lock().await.operation, Operation::Idle) {
                    self.start_playback().await?;
                }
            }
            NativeEvent::Stop => {
                let operation = self.machine.lock().await.operation.clone();
                match operation {
                    Operation::Recording { .. } => {
                        self.stop_recording().await?;
                    }
                    Operation::Playing { .. } => {
                        self.stop_playback().await?;
                    }
                    Operation::Idle => {}
                }
            }
            NativeEvent::Cycle { playback_id } => {
                let mut machine = self.machine.lock().await;
                if matches!(machine.operation, Operation::Playing { playback_id: active_id, .. } if active_id == playback_id)
                {
                    machine.cycles = machine.cycles.saturating_add(1);
                    self.changed(&mut machine).await?;
                }
            }
            NativeEvent::PlaybackEnded { playback_id } => {
                let mut machine = self.machine.lock().await;
                if matches!(machine.operation, Operation::Playing { playback_id: active_id, .. } if active_id == playback_id)
                {
                    machine.operation = Operation::Idle;
                    self.changed(&mut machine).await?;
                }
            }
            NativeEvent::PlaybackFailed {
                playback_id,
                reason,
            } => {
                let mut machine = self.machine.lock().await;
                if matches!(machine.operation, Operation::Playing { playback_id: active_id, .. } if active_id == playback_id)
                {
                    machine.operation = Operation::Idle;
                    machine.last_error = Some(match reason {
                        PlaybackFailure::Input => "La lecture a été interrompue par une erreur de saisie Windows.".into(),
                        PlaybackFailure::Release => "AUTO-COC n’a pas pu relâcher une touche ou un bouton après la lecture. Vérifiez le clavier et la souris.".into(),
                        PlaybackFailure::InputAndRelease => "La lecture a échoué et AUTO-COC n’a pas pu relâcher une touche ou un bouton. Vérifiez le clavier et la souris.".into(),
                    });
                    self.changed(&mut machine).await?;
                }
            }
        }
        Ok(())
    }

    pub async fn shutdown_services(&self) -> Result<(), ControllerError> {
        let operation = {
            let mut machine = self.machine.lock().await;
            if machine.closing {
                return if machine.shutdown_complete
                    && matches!(machine.operation, Operation::Idle)
                    && machine.pending_recording.is_none()
                {
                    Ok(())
                } else {
                    Err(ControllerError::InvalidState)
                };
            }
            machine.closing = true;
            machine.operation.clone()
        };

        #[cfg(test)]
        if let Some(gate) = self.shutdown_race_gate.lock().await.take() {
            gate.paused.notify_one();
            gate.resume.notified().await;
        }

        let stopped = match operation {
            Operation::Recording { .. } => self.stop_recording().await.map(|_| ()),
            Operation::Playing { .. } => self.stop_playback().await.map(|_| ()),
            Operation::Idle => Ok(()),
        };
        if let Err(error) = stopped {
            let mut machine = self.machine.lock().await;
            let already_stopped = matches!(&error, ControllerError::InvalidState)
                && matches!(machine.operation, Operation::Idle)
                && machine.pending_recording.is_none();
            if !already_stopped {
                machine.closing = false;
                return Err(error);
            }
        }

        if let Err(error) = self.stop_native_event_loop().await {
            self.machine.lock().await.closing = false;
            return Err(error);
        }

        let mut native_slot = self.native.lock().await;
        let shutdown_failed = if let Some(native) = native_slot.as_mut() {
            native.shutdown().await.is_err()
        } else {
            false
        };
        if shutdown_failed {
            drop(native_slot);
            self.machine.lock().await.closing = false;
            return Err(ControllerError::NativeInputFailed);
        }
        *native_slot = None;
        drop(native_slot);

        let mut machine = self.machine.lock().await;
        if !matches!(machine.operation, Operation::Idle) {
            machine.closing = false;
            return Err(ControllerError::InvalidState);
        }
        machine.pairing = None;
        machine.shutdown = None;
        machine.shutdown_complete = true;
        Ok(())
    }

    pub async fn snapshot(&self) -> Result<Snapshot, ControllerError> {
        let machine = self.machine.lock().await;
        Self::snapshot_of(&machine).await
    }

    pub async fn changed_snapshot(&self, after: u64) -> Result<Snapshot, ControllerError> {
        let mut receiver = self.revision_tx.subscribe();
        let snapshot = self.snapshot().await?;
        if snapshot.revision > after {
            return Ok(snapshot);
        }
        let wait = if matches!(snapshot.status, AppStatus::Idle) {
            Duration::from_secs(20)
        } else {
            Duration::from_secs(1)
        };
        let current = *receiver.borrow();
        if current <= after {
            let _ = tokio::time::timeout(wait, async {
                while *receiver.borrow() <= after {
                    if receiver.changed().await.is_err() {
                        break;
                    }
                }
            })
            .await;
        }
        self.snapshot().await
    }

    pub async fn macro_file(&self, name: &str) -> Result<Value, ControllerError> {
        let machine = self.machine.lock().await;
        let file = machine.store.get_macro(name)?;
        Ok(serde_json::json!({"name": name, "steps": file.steps}))
    }

    pub async fn create_macro(&self, name: &str) -> Result<Snapshot, ControllerError> {
        let mut machine = self.machine.lock().await;
        ensure_idle(&machine)?;
        machine.store.create_macro(name)?;
        self.changed(&mut machine).await
    }

    pub async fn rename_macro(
        &self,
        name: &str,
        new_name: &str,
    ) -> Result<Snapshot, ControllerError> {
        let mut machine = self.machine.lock().await;
        ensure_idle(&machine)?;
        machine.store.rename_macro(name, new_name)?;
        self.changed(&mut machine).await
    }

    pub async fn delete_macro(&self, name: &str) -> Result<Snapshot, ControllerError> {
        let mut machine = self.machine.lock().await;
        ensure_idle(&machine)?;
        machine.store.delete_macro(name)?;
        self.changed(&mut machine).await
    }

    pub async fn select_macro(&self, name: &str) -> Result<Snapshot, ControllerError> {
        self.select_macro_with_authority(name, None).await
    }

    pub async fn select_macro_for_telegram(
        &self,
        name: &str,
        token: &str,
        owner: crate::telegram::TelegramOwner,
    ) -> Result<Snapshot, ControllerError> {
        self.select_macro_with_authority(name, Some((token, owner)))
            .await
    }

    async fn select_macro_with_authority(
        &self,
        name: &str,
        authority: Option<(&str, crate::telegram::TelegramOwner)>,
    ) -> Result<Snapshot, ControllerError> {
        let mut machine = self.machine.lock().await;
        if let Some((token, owner)) = authority {
            ensure_telegram_authorized(&machine, token, owner)?;
        }
        ensure_idle(&machine)?;
        machine.store.set_selected(Some(name.to_owned()))?;
        self.changed(&mut machine).await
    }

    pub async fn set_settings(
        &self,
        loop_playback: Option<bool>,
        coc_path: Option<String>,
    ) -> Result<Snapshot, ControllerError> {
        let mut machine = self.machine.lock().await;
        ensure_idle(&machine)?;
        machine.store.update_settings(loop_playback, coc_path)?;
        self.changed(&mut machine).await
    }

    pub async fn set_shortcuts(
        self: &Arc<Self>,
        shortcuts: crate::types::ShortcutSettings,
    ) -> Result<Snapshot, ControllerError> {
        self.start_native_event_loop().await;
        crate::native_input::validate_shortcuts(&shortcuts).map_err(map_shortcut_error)?;
        let mut machine = self.machine.lock().await;
        ensure_idle(&machine)?;
        let old_shortcuts = machine.store.settings().shortcuts.clone();
        let mut native_slot = self.native.lock().await;
        let mut replacement = None;
        if let Some(native) = native_slot.as_ref() {
            native
                .set_shortcuts(shortcuts.clone())
                .await
                .map_err(map_shortcut_error)?;
        } else {
            replacement = Some(
                NativeInput::start(shortcuts.clone(), self.native_events_tx.clone())
                    .map_err(map_shortcut_error)?,
            );
        }
        if let Err(error) = machine.store.set_shortcuts(shortcuts) {
            let rollback_failed = if let Some(native) = native_slot.as_ref() {
                native.set_shortcuts(old_shortcuts).await.is_err()
            } else {
                false
            };
            if rollback_failed {
                return Err(ControllerError::NativeInputFailed);
            }
            return Err(error.into());
        }
        if let Some(native) = replacement {
            *native_slot = Some(native);
        }
        machine.last_error = None;
        self.changed(&mut machine).await
    }

    pub async fn complete_onboarding(&self) -> Result<Snapshot, ControllerError> {
        let mut machine = self.machine.lock().await;
        machine.store.complete_onboarding()?;
        self.changed(&mut machine).await
    }

    pub async fn migration_status(
        &self,
    ) -> Result<(MigrationStatus, Option<MigrationPreview>), ControllerError> {
        let machine = self.machine.lock().await;
        Ok((
            MigrationStatus {
                source_selected: machine.migration_source.is_some(),
                available: machine
                    .migration_preview
                    .as_ref()
                    .is_some_and(|preview| preview.available),
                already_imported: machine.migration_already_imported,
            },
            machine.migration_preview.clone(),
        ))
    }

    pub async fn select_migration_source(
        &self,
    ) -> Result<(Snapshot, MigrationPreview), ControllerError> {
        let machine = self.machine.lock().await;
        ensure_idle(&machine)?;
        drop(machine);
        let selected = tokio::task::spawn_blocking(|| {
            rfd::FileDialog::new()
                .set_title("Choisir l’ancien dossier AUTO-COC")
                .pick_folder()
        })
        .await
        .map_err(|_| StoreError::InvalidData)?
        .ok_or(ControllerError::MigrationCancelled)?;
        let mut machine = self.machine.lock().await;
        ensure_idle(&machine)?;
        let (config, preview, already_imported) = machine.store.preview_legacy(&selected)?;
        machine.migration_source = Some(config);
        machine.migration_preview = Some(preview.clone());
        machine.migration_already_imported = already_imported;
        let snapshot = self.changed(&mut machine).await?;
        Ok((snapshot, preview))
    }

    pub async fn import_migration_source(
        &self,
    ) -> Result<(Snapshot, MigrationResult), ControllerError> {
        let mut machine = self.machine.lock().await;
        ensure_idle(&machine)?;
        let source = machine
            .migration_source
            .clone()
            .ok_or(ControllerError::MigrationCancelled)?;
        ensure_idle(&machine)?;
        let result = {
            let Machine { store, pairing, .. } = &mut *machine;
            store.import_legacy_with_token_installation(&source, || {
                *pairing = None;
            })?
        };
        machine.migration_already_imported = true;
        let (normalized_source, preview, _) = machine.store.preview_legacy(&source)?;
        machine.migration_source = Some(normalized_source);
        machine.migration_preview = Some(preview);
        let snapshot = self.changed(&mut machine).await?;
        Ok((snapshot, result))
    }

    pub async fn start_recording(&self) -> Result<Snapshot, ControllerError> {
        let mut machine = self.machine.lock().await;
        ensure_idle(&machine)?;
        let macro_name = machine
            .store
            .selected_macro()
            .ok_or(StoreError::NotFound)?
            .to_owned();
        let native = self.native.lock().await;
        native
            .as_ref()
            .ok_or(ControllerError::NativeInputUnavailable)?
            .start_recording()
            .await
            .map_err(map_native_error)?;
        machine.operation = Operation::Recording {
            macro_name,
            started: Instant::now(),
        };
        machine.last_error = None;
        self.changed(&mut machine).await
    }

    pub async fn stop_recording(&self) -> Result<Snapshot, ControllerError> {
        self.stop_recording_with_authority(None).await
    }

    async fn stop_recording_with_authority(
        &self,
        authority: Option<(&str, crate::telegram::TelegramOwner)>,
    ) -> Result<Snapshot, ControllerError> {
        let mut machine = self.machine.lock().await;
        if let Some((token, owner)) = authority {
            ensure_telegram_authorized(&machine, token, owner)?;
        }
        let Operation::Recording { macro_name, .. } = machine.operation.clone() else {
            return Err(ControllerError::InvalidState);
        };
        if machine.pending_recording.is_none() {
            let native = self.native.lock().await;
            let capture = native
                .as_ref()
                .ok_or(ControllerError::NativeInputUnavailable)?
                .stop_recording()
                .await
                .map_err(map_native_error)?;
            machine.pending_recording = Some((macro_name, capture.steps, capture.overflowed));
        }
        let overflowed = match save_pending_recording(&mut machine) {
            Ok(overflowed) => overflowed,
            Err(error) => {
                machine.last_error = Some(
                    "Enregistrement terminé, mais non sauvegardé. Relancez Arrêter pour réessayer."
                        .into(),
                );
                return Err(error);
            }
        };
        machine.operation = Operation::Idle;
        machine.last_error = overflowed.then(|| {
            "La limite d’événements a été atteinte. Les événements déjà capturés ont été sauvegardés; la macro peut être incomplète.".into()
        });
        let snapshot = self.changed(&mut machine).await?;
        if overflowed {
            Err(ControllerError::RecordingEventLimit)
        } else {
            Ok(snapshot)
        }
    }

    pub async fn start_playback(&self) -> Result<Snapshot, ControllerError> {
        self.start_macro_playback(None, true, None).await
    }

    pub async fn start_playback_for_telegram(
        &self,
        token: &str,
        owner: crate::telegram::TelegramOwner,
    ) -> Result<Snapshot, ControllerError> {
        self.start_macro_playback(None, true, Some((token, owner)))
            .await
    }

    pub async fn play_named_macro_for_telegram(
        &self,
        name: &str,
        token: &str,
        owner: crate::telegram::TelegramOwner,
    ) -> Result<Snapshot, ControllerError> {
        self.start_macro_playback(Some(name), false, Some((token, owner)))
            .await
    }

    async fn start_macro_playback(
        &self,
        requested_name: Option<&str>,
        use_saved_loop: bool,
        authority: Option<(&str, crate::telegram::TelegramOwner)>,
    ) -> Result<Snapshot, ControllerError> {
        let mut machine = self.machine.lock().await;
        if let Some((token, owner)) = authority {
            ensure_telegram_authorized(&machine, token, owner)?;
        }
        ensure_idle(&machine)?;
        let name = match requested_name {
            Some(name) => name,
            None => machine.store.selected_macro().ok_or(StoreError::NotFound)?,
        };
        let macro_file = machine.store.get_macro(name)?;
        if macro_file.steps.is_empty() {
            return Err(ControllerError::EmptyMacro);
        }
        let macro_name = name.to_owned();
        let looped = use_saved_loop && machine.store.settings().loop_playback;
        let native = self.native.lock().await;
        let playback_id = native
            .as_ref()
            .ok_or(ControllerError::NativeInputUnavailable)?
            .start_playback(macro_file.steps, looped)
            .await
            .map_err(map_native_error)?;
        machine.operation = Operation::Playing {
            macro_name,
            started: Instant::now(),
            playback_id,
        };
        machine.cycles = 0;
        machine.last_error = None;
        self.changed(&mut machine).await
    }

    pub async fn stop_active_operation(
        &self,
        token: &str,
        owner: crate::telegram::TelegramOwner,
    ) -> Result<Snapshot, ControllerError> {
        let operation = {
            let machine = self.machine.lock().await;
            ensure_telegram_authorized(&machine, token, owner)?;
            machine.operation.clone()
        };
        match operation {
            Operation::Recording { .. } => {
                self.stop_recording_with_authority(Some((token, owner)))
                    .await
            }
            Operation::Playing { .. } => {
                self.stop_playback_with_authority(Some((token, owner)))
                    .await
            }
            Operation::Idle => Ok(self.snapshot().await?),
        }
    }

    pub async fn toggle_loop_playback_for_telegram(
        &self,
        token: &str,
        owner: crate::telegram::TelegramOwner,
    ) -> Result<(bool, Snapshot), ControllerError> {
        self.toggle_loop_playback_with_authority(Some((token, owner)))
            .await
    }

    async fn toggle_loop_playback_with_authority(
        &self,
        authority: Option<(&str, crate::telegram::TelegramOwner)>,
    ) -> Result<(bool, Snapshot), ControllerError> {
        let mut machine = self.machine.lock().await;
        if let Some((token, owner)) = authority {
            ensure_telegram_authorized(&machine, token, owner)?;
        }
        ensure_idle(&machine)?;
        let enabled = !machine.store.settings().loop_playback;
        machine.store.set_loop(enabled)?;
        let snapshot = self.changed(&mut machine).await?;
        Ok((enabled, snapshot))
    }

    pub async fn telegram_macro_names(&self) -> Result<Vec<String>, ControllerError> {
        let machine = self.machine.lock().await;
        Ok(machine
            .store
            .summaries()?
            .into_iter()
            .filter(|summary| summary.readable)
            .map(|summary| summary.name)
            .collect())
    }

    pub async fn telegram_token(&self) -> Result<Option<String>, ControllerError> {
        let machine = self.machine.lock().await;
        if !machine.store.settings().telegram.token_configured {
            return Ok(None);
        }
        Ok(crate::storage::load_token(machine.store.root())?)
    }

    pub async fn telegram_token_is_current(&self, token: &str) -> bool {
        let machine = self.machine.lock().await;
        machine.store.settings().telegram.token_configured
            && crate::storage::token_matches(machine.store.root(), token.as_bytes())
    }

    pub async fn authorize_telegram_message(
        &self,
        token: &str,
        message: &crate::telegram::TelegramMessage,
    ) -> bool {
        let Some(user) = message.from.as_ref() else {
            return false;
        };
        self.authorize_telegram_identity(
            token,
            message.chat.id,
            user.id,
            message.chat.chat_type == "private",
        )
        .await
    }

    pub async fn authorize_telegram_identity(
        &self,
        token: &str,
        chat_id: i64,
        user_id: i64,
        private_chat: bool,
    ) -> bool {
        let machine = self.machine.lock().await;
        private_chat
            && telegram_is_authorized(
                &machine,
                token,
                crate::telegram::TelegramOwner { chat_id, user_id },
            )
    }

    pub async fn authorize_telegram_callback(
        &self,
        token: &str,
        callback: &crate::telegram::TelegramCallback,
    ) -> bool {
        let Some(message) = callback.message.as_ref() else {
            return false;
        };
        self.authorize_telegram_identity(
            token,
            message.chat.id,
            callback.from.id,
            message.chat.chat_type == "private",
        )
        .await
    }

    pub async fn set_telegram_connection(
        &self,
        token: &str,
        connected: bool,
    ) -> Result<(), ControllerError> {
        let mut machine = self.machine.lock().await;
        if !machine.store.settings().telegram.token_configured
            || !crate::storage::token_matches(machine.store.root(), token.as_bytes())
        {
            return Ok(());
        }
        let status = match (connected, machine.store.settings().telegram.paired) {
            (false, _) => TelegramStatus::Error,
            (true, true) => TelegramStatus::Connected,
            (true, false) => TelegramStatus::WaitingPairing,
        };
        if machine.store.settings().telegram.status == status {
            return Ok(());
        }
        machine.store.settings_mut().telegram.status = status;
        machine.store.persist()?;
        self.changed(&mut machine).await?;
        Ok(())
    }

    pub async fn stop_playback(&self) -> Result<Snapshot, ControllerError> {
        self.stop_playback_with_authority(None).await
    }

    async fn stop_playback_with_authority(
        &self,
        authority: Option<(&str, crate::telegram::TelegramOwner)>,
    ) -> Result<Snapshot, ControllerError> {
        let mut machine = self.machine.lock().await;
        if let Some((token, owner)) = authority {
            ensure_telegram_authorized(&machine, token, owner)?;
        }
        if !matches!(machine.operation, Operation::Playing { .. }) {
            return Err(ControllerError::InvalidState);
        }
        let native = self.native.lock().await;
        native
            .as_ref()
            .ok_or(ControllerError::NativeInputUnavailable)?
            .stop_playback()
            .await
            .map_err(map_native_error)?;
        machine.operation = Operation::Idle;
        self.changed(&mut machine).await
    }

    pub async fn start_pairing(&self) -> Result<(String, String, Snapshot), ControllerError> {
        let mut machine = self.machine.lock().await;
        if !machine.store.settings().telegram.token_configured {
            return Err(ControllerError::TelegramNotConfigured);
        }
        if machine
            .pairing_cooldown_until
            .is_some_and(|until| Instant::now() < until)
        {
            return Err(ControllerError::PairingRateLimited);
        }
        machine.pairing_cooldown_until = None;
        let code = random_code();
        let expires = Instant::now() + PAIRING_LIFETIME;
        let expires_at = (chrono::Utc::now()
            + chrono::Duration::seconds(PAIRING_LIFETIME.as_secs() as i64))
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        machine.pairing = Some(Pairing {
            code: code.clone(),
            expires,
            expires_at: expires_at.clone(),
            attempts: 0,
        });
        machine.store.settings_mut().telegram.status = TelegramStatus::WaitingPairing;
        machine.store.persist()?;
        let snapshot = self.changed(&mut machine).await?;
        Ok((code, expires_at, snapshot))
    }

    pub async fn attempt_telegram_pairing(
        &self,
        token: &str,
        code: &str,
        chat_id: i64,
        user_id: i64,
        private_chat: bool,
    ) -> Result<bool, ControllerError> {
        if !private_chat {
            return Ok(false);
        }
        let mut machine = self.machine.lock().await;
        if !machine.store.settings().telegram.token_configured
            || !crate::storage::token_matches(machine.store.root(), token.as_bytes())
        {
            return Err(ControllerError::TelegramUnauthorized);
        }
        if machine
            .pairing_cooldown_until
            .is_some_and(|until| Instant::now() < until)
        {
            return Err(ControllerError::PairingRateLimited);
        }
        let Some(pairing) = machine.pairing.as_mut() else {
            return Err(ControllerError::Expired);
        };
        if Instant::now() >= pairing.expires {
            machine.pairing = None;
            return Err(ControllerError::Expired);
        }
        if pairing.code == code {
            machine.store.set_telegram_owner(chat_id, user_id);
            machine.store.settings_mut().telegram.paired = true;
            machine.store.settings_mut().telegram.status = TelegramStatus::Connected;
            machine.store.persist()?;
            machine.pairing = None;
            self.changed(&mut machine).await?;
            return Ok(true);
        }
        pairing.attempts += 1;
        if pairing.attempts >= MAX_PAIRING_ATTEMPTS {
            machine.pairing = None;
            machine.pairing_cooldown_until = Some(Instant::now() + PAIRING_COOLDOWN);
            self.changed(&mut machine).await?;
        }
        Ok(false)
    }

    pub async fn cancel_pairing(&self) -> Result<Snapshot, ControllerError> {
        let mut machine = self.machine.lock().await;
        machine.pairing = None;
        machine.store.settings_mut().telegram.pairing_code = None;
        machine.store.settings_mut().telegram.pairing_expires_at = None;
        self.changed(&mut machine).await
    }

    pub async fn configure_telegram(
        &self,
        token: Option<&str>,
    ) -> Result<Snapshot, ControllerError> {
        let mut machine = self.machine.lock().await;
        if let Some(token) = token {
            if token.trim().is_empty() || token.len() > 512 || token.contains(['\r', '\n', '\0']) {
                return Err(StoreError::InvalidData.into());
            }
            let rotated = !crate::storage::token_matches(machine.store.root(), token.as_bytes());
            if rotated {
                machine.store.settings_mut().telegram.paired = false;
                machine.store.settings_mut().telegram.pairing_code = None;
                machine.store.settings_mut().telegram.pairing_expires_at = None;
                machine.pairing = None;
                machine.store.clear_telegram_owner();
                machine.store.persist()?;
            }
            crate::storage::save_token(machine.store.root(), token.as_bytes())?;
            machine.store.settings_mut().telegram.token_configured = true;
            machine.store.settings_mut().telegram.status = TelegramStatus::Disconnected;
            machine.store.persist()?;
        } else {
            crate::storage::clear_token(machine.store.root())?;
            machine.store.settings_mut().telegram = Default::default();
            machine.pairing = None;
            machine.store.clear_telegram_owner();
            machine.store.persist()?;
        }
        self.changed(&mut machine).await
    }

    pub async fn test_telegram(&self) -> Result<Snapshot, ControllerError> {
        let token = self
            .telegram_token()
            .await?
            .ok_or(ControllerError::TelegramNotConfigured)?;
        let valid = match crate::telegram::TelegramClient::new(token.clone()) {
            Ok(client) => client.get_me().await.is_ok(),
            Err(_) => false,
        };
        self.set_telegram_connection(&token, valid).await?;
        if valid {
            Ok(self.snapshot().await?)
        } else {
            Err(ControllerError::TelegramUnavailable)
        }
    }

    pub async fn prepare_shutdown(&self) -> Result<(String, String), ControllerError> {
        let id = random_secret_id();
        self.prepare_shutdown_with_id(id, None).await
    }

    pub async fn prepare_telegram_shutdown(
        &self,
        token: &str,
        owner: crate::telegram::TelegramOwner,
    ) -> Result<String, ControllerError> {
        use rand::RngCore;
        let mut bytes = [0u8; 20];
        rand::rng().fill_bytes(&mut bytes);
        let id = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
        self.prepare_shutdown_with_id(id, Some((token, owner)))
            .await
            .map(|(id, _)| id)
    }

    async fn prepare_shutdown_with_id(
        &self,
        id: String,
        authority: Option<(&str, crate::telegram::TelegramOwner)>,
    ) -> Result<(String, String), ControllerError> {
        let mut machine = self.machine.lock().await;
        if let Some((token, owner)) = authority {
            ensure_telegram_authorized(&machine, token, owner)?;
        }
        let expires = Instant::now() + SHUTDOWN_CONFIRM_LIFETIME;
        let expires_at = (chrono::Utc::now()
            + chrono::Duration::seconds(SHUTDOWN_CONFIRM_LIFETIME.as_secs() as i64))
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        machine.shutdown = Some(ShutdownConfirmation {
            id: id.clone(),
            expires,
        });
        Ok((id, expires_at))
    }

    pub async fn confirm_shutdown(
        &self,
        confirmation_id: &str,
    ) -> Result<Snapshot, ControllerError> {
        let mut machine = self.machine.lock().await;
        ensure_idle(&machine)?;
        consume_shutdown_confirmation(&mut machine, confirmation_id)?;
        #[cfg(windows)]
        crate::system::request_shutdown().map_err(|_| ControllerError::NativeInputUnavailable)?;
        #[cfg(not(windows))]
        return Err(ControllerError::NativeInputUnavailable);
        self.changed(&mut machine).await
    }

    pub async fn confirm_telegram_shutdown(
        &self,
        token: &str,
        callback: &crate::telegram::TelegramCallback,
        confirmation_id: &str,
    ) -> Result<Snapshot, ControllerError> {
        let mut machine = self.machine.lock().await;
        if !machine.store.settings().telegram.paired
            || !crate::storage::token_matches(machine.store.root(), token.as_bytes())
        {
            return Err(ControllerError::TelegramNotConfigured);
        }
        let Some((chat_id, user_id)) = machine.store.telegram_owner() else {
            return Err(ControllerError::TelegramNotConfigured);
        };
        if !crate::telegram::callback_is_authorized(
            callback,
            crate::telegram::TelegramOwner { chat_id, user_id },
        ) {
            return Err(ControllerError::TelegramNotConfigured);
        }
        ensure_idle(&machine)?;
        consume_shutdown_confirmation(&mut machine, confirmation_id)?;
        #[cfg(windows)]
        crate::system::request_shutdown().map_err(|_| ControllerError::NativeInputUnavailable)?;
        #[cfg(not(windows))]
        return Err(ControllerError::NativeInputUnavailable);
        self.changed(&mut machine).await
    }

    pub async fn cancel_shutdown(
        &self,
        token: &str,
        owner: crate::telegram::TelegramOwner,
    ) -> Result<(), ControllerError> {
        let mut machine = self.machine.lock().await;
        ensure_telegram_authorized(&machine, token, owner)?;
        machine.shutdown = None;
        Ok(())
    }

    pub async fn capture_screenshot_png(&self) -> Result<Vec<u8>, ControllerError> {
        tokio::task::spawn_blocking(crate::system::screenshot_png)
            .await
            .map_err(|_| ControllerError::ScreenshotUnavailable)?
            .map_err(|_| ControllerError::ScreenshotUnavailable)
    }

    pub async fn diagnostics(&self) -> Result<crate::types::Diagnostics, ControllerError> {
        let machine = self.machine.lock().await;
        Ok(crate::types::Diagnostics {
            app_version: env!("CARGO_PKG_VERSION").into(),
            rust_version: option_env!("RUSTC_VERSION")
                .unwrap_or("Rust toolchain")
                .into(),
            status: state_label(&machine.operation).into(),
            errors: machine.last_error.clone().into_iter().collect(),
        })
    }

    pub async fn launch_coc(&self) -> Result<Snapshot, ControllerError> {
        self.launch_coc_with_authority(None).await
    }

    pub async fn launch_coc_for_telegram(
        &self,
        token: &str,
        owner: crate::telegram::TelegramOwner,
    ) -> Result<Snapshot, ControllerError> {
        self.launch_coc_with_authority(Some((token, owner))).await
    }

    async fn launch_coc_with_authority(
        &self,
        authority: Option<(&str, crate::telegram::TelegramOwner)>,
    ) -> Result<Snapshot, ControllerError> {
        let mut machine = self.machine.lock().await;
        if let Some((token, owner)) = authority {
            ensure_telegram_authorized(&machine, token, owner)?;
        }
        ensure_idle(&machine)?;
        if machine.coc_launched_once {
            return Self::snapshot_of(&machine).await;
        }
        let path = machine.store.settings().coc_path.clone();
        crate::system::launch_configured_path(&path).map_err(|_| StoreError::InvalidData)?;
        machine.coc_launched_once = true;
        self.changed(&mut machine).await
    }

    pub async fn open_macros_folder(&self) -> Result<Snapshot, ControllerError> {
        let mut machine = self.machine.lock().await;
        crate::system::open_known_folder(&machine.store.root().join("macros"))
            .map_err(|_| StoreError::Io(std::io::Error::other("cannot open macro folder")))?;
        self.changed(&mut machine).await
    }

    async fn changed(&self, machine: &mut Machine) -> Result<Snapshot, ControllerError> {
        machine.revision = machine.revision.saturating_add(1);
        self.revision_tx.send_replace(machine.revision);
        Self::snapshot_of(machine).await
    }

    async fn snapshot_of(machine: &Machine) -> Result<Snapshot, ControllerError> {
        let (status, elapsed) = match &machine.operation {
            Operation::Idle => (AppStatus::Idle, 0.0),
            Operation::Recording {
                macro_name,
                started,
            } => {
                let elapsed = started.elapsed().as_secs_f64();
                let active_elapsed = (elapsed - 3.0).max(0.0);
                let phase = if elapsed >= 3.0 {
                    RecordingPhase::Capturing
                } else {
                    RecordingPhase::Preparing
                };
                (
                    AppStatus::Recording {
                        macro_name: macro_name.clone(),
                        phase,
                        countdown_seconds: (3.0 - elapsed).max(0.0),
                        elapsed_seconds: active_elapsed,
                    },
                    active_elapsed,
                )
            }
            Operation::Playing {
                macro_name,
                started,
                ..
            } => {
                let elapsed = started.elapsed().as_secs_f64();
                (
                    AppStatus::Playing {
                        macro_name: macro_name.clone(),
                        elapsed_seconds: elapsed,
                    },
                    elapsed,
                )
            }
        };
        let mut settings = machine.store.settings().clone();
        if let Some(pairing) = &machine.pairing {
            if Instant::now() <= pairing.expires {
                settings.telegram.pairing_code = Some(pairing.code.clone());
                settings.telegram.pairing_expires_at = Some(pairing.expires_at.clone());
            } else {
                settings.telegram.pairing_code = None;
                settings.telegram.pairing_expires_at = None;
            }
        }
        Ok(Snapshot {
            revision: machine.revision,
            status,
            selected_macro: machine.store.selected_macro().map(str::to_owned),
            macros: machine.store.summaries()?,
            settings,
            session: SessionStatus {
                elapsed_seconds: elapsed,
                cycles: machine.cycles,
            },
            migration: MigrationStatus {
                source_selected: machine.migration_source.is_some(),
                available: machine
                    .migration_preview
                    .as_ref()
                    .is_some_and(|preview| preview.available),
                already_imported: machine.migration_already_imported,
            },
            onboarding_complete: machine.store.onboarding_complete(),
            last_error: machine.last_error.clone(),
        })
    }
}

fn ensure_idle(machine: &Machine) -> Result<(), ControllerError> {
    if !machine.closing && matches!(&machine.operation, Operation::Idle) {
        Ok(())
    } else {
        Err(ControllerError::InvalidState)
    }
}

fn telegram_is_authorized(
    machine: &Machine,
    token: &str,
    owner: crate::telegram::TelegramOwner,
) -> bool {
    machine.store.settings().telegram.paired
        && machine.store.settings().telegram.token_configured
        && crate::storage::token_matches(machine.store.root(), token.as_bytes())
        && machine.store.telegram_owner() == Some((owner.chat_id, owner.user_id))
}

fn ensure_telegram_authorized(
    machine: &Machine,
    token: &str,
    owner: crate::telegram::TelegramOwner,
) -> Result<(), ControllerError> {
    if telegram_is_authorized(machine, token, owner) {
        Ok(())
    } else {
        Err(ControllerError::TelegramUnauthorized)
    }
}

fn consume_shutdown_confirmation(
    machine: &mut Machine,
    confirmation_id: &str,
) -> Result<(), ControllerError> {
    let Some(confirmation) = machine.shutdown.take() else {
        return Err(ControllerError::Expired);
    };
    if confirmation.id != confirmation_id || Instant::now() > confirmation.expires {
        return Err(ControllerError::Expired);
    }
    Ok(())
}

fn state_label(state: &Operation) -> &'static str {
    match state {
        Operation::Idle => "idle",
        Operation::Recording { .. } => "recording",
        Operation::Playing { .. } => "playing",
    }
}

fn save_pending_recording(machine: &mut Machine) -> Result<bool, ControllerError> {
    let Some((macro_name, steps, overflowed)) = machine.pending_recording.take() else {
        return Ok(false);
    };
    if machine
        .store
        .save_recorded_steps(&macro_name, &steps)
        .is_err()
    {
        machine.pending_recording = Some((macro_name, steps, overflowed));
        return Err(ControllerError::RecordingSavePending);
    }
    Ok(overflowed)
}

fn map_native_error(error: std::io::Error) -> ControllerError {
    match error.kind() {
        std::io::ErrorKind::Unsupported => ControllerError::NativeInputUnavailable,
        _ => ControllerError::NativeInputFailed,
    }
}

fn map_shortcut_error(error: std::io::Error) -> ControllerError {
    match error.kind() {
        std::io::ErrorKind::InvalidInput => ControllerError::InvalidShortcut,
        std::io::ErrorKind::AlreadyExists => ControllerError::ShortcutUnavailable,
        _ => map_native_error(error),
    }
}

fn random_code() -> String {
    use rand::Rng;
    format!("{:06}", rand::rng().random_range(0..1_000_000))
}

fn random_secret_id() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn macro_crud_updates_revision_and_selection() {
        let dir = tempdir().unwrap();
        let controller = Controller::new(Store::open_at(dir.path()).unwrap());
        let initial = controller.snapshot().await.unwrap();
        let created = controller.create_macro("Nouvelle 1").await.unwrap();
        assert!(created.revision > initial.revision);
        assert_eq!(created.selected_macro.as_deref(), Some("Nouvelle 1"));
        let renamed = controller
            .rename_macro("Nouvelle 1", "Nouvelle 2")
            .await
            .unwrap();
        assert_eq!(renamed.selected_macro.as_deref(), Some("Nouvelle 2"));
        let deleted = controller.delete_macro("Nouvelle 2").await.unwrap();
        assert_eq!(deleted.selected_macro, None);
    }

    #[tokio::test]
    async fn long_poll_returns_on_revision_change_without_poll_loop() {
        let dir = tempdir().unwrap();
        let controller = Arc::new(Controller::new(Store::open_at(dir.path()).unwrap()));
        let stale = controller.snapshot().await.unwrap().revision;
        let wake = controller.clone();
        let task = tokio::spawn(async move { wake.changed_snapshot(stale).await.unwrap() });
        tokio::task::yield_now().await;
        controller.create_macro("Wake").await.unwrap();
        let update = tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .unwrap()
            .unwrap();
        assert!(update.revision > stale);
    }

    #[tokio::test]
    async fn active_long_poll_refreshes_elapsed_time_once_per_second() {
        let dir = tempdir().unwrap();
        let controller = Controller::new(Store::open_at(dir.path()).unwrap());
        let stale = controller.snapshot().await.unwrap().revision;
        controller.machine.lock().await.operation = Operation::Recording {
            macro_name: "Recharger COC".into(),
            started: Instant::now() - Duration::from_secs(4),
        };
        let started = Instant::now();
        let snapshot = controller.changed_snapshot(stale).await.unwrap();
        assert!(started.elapsed() >= Duration::from_millis(900));
        assert!(matches!(
            snapshot.status,
            AppStatus::Recording {
                phase: RecordingPhase::Capturing,
                ..
            }
        ));
        assert!(snapshot.session.elapsed_seconds > 0.9);
    }

    #[tokio::test]
    async fn coc_launch_is_idempotent_for_the_process_lifetime() {
        let dir = tempdir().unwrap();
        let controller = Controller::new(Store::open_at(dir.path()).unwrap());
        {
            let mut machine = controller.machine.lock().await;
            machine.coc_launched_once = true;
        }
        let initial = controller.snapshot().await.unwrap();
        let returned = controller.launch_coc().await.unwrap();
        assert_eq!(returned.revision, initial.revision);
    }

    #[tokio::test]
    async fn shutdown_confirmation_cannot_run_during_playback() {
        let dir = tempdir().unwrap();
        let controller = Controller::new(Store::open_at(dir.path()).unwrap());
        let id = random_secret_id();
        {
            let mut machine = controller.machine.lock().await;
            machine.operation = Operation::Playing {
                macro_name: "Test".into(),
                started: Instant::now(),
                playback_id: 1,
            };
            machine.shutdown = Some(ShutdownConfirmation {
                id: id.clone(),
                expires: Instant::now() + SHUTDOWN_CONFIRM_LIFETIME,
            });
        }
        assert!(matches!(
            controller.confirm_shutdown(&id).await,
            Err(ControllerError::InvalidState)
        ));
        assert!(controller.machine.lock().await.shutdown.is_some());
    }

    #[tokio::test]
    async fn shutdown_retries_recording_save_before_closing() {
        let dir = tempdir().unwrap();
        let mut store = Store::open_at(dir.path()).unwrap();
        store.create_macro("Retry").unwrap();
        let macro_path = dir.path().join("macros/Retry.json");
        let original_macro = std::fs::read(&macro_path).unwrap();
        std::fs::remove_file(&macro_path).unwrap();
        std::fs::create_dir(&macro_path).unwrap();

        let controller = Controller::new(store);
        let steps = vec![serde_json::json!({"t":0.1,"type":"nop","data":{}})];
        {
            let mut machine = controller.machine.lock().await;
            machine.operation = Operation::Recording {
                macro_name: "Retry".into(),
                started: Instant::now(),
            };
            machine.pending_recording = Some(("Retry".into(), steps.clone(), false));
        }

        assert!(matches!(
            controller.shutdown_services().await,
            Err(ControllerError::RecordingSavePending)
        ));
        assert!(!controller.machine.lock().await.closing);
        assert!(matches!(
            controller.machine.lock().await.operation,
            Operation::Recording { .. }
        ));

        std::fs::remove_dir(&macro_path).unwrap();
        std::fs::write(&macro_path, original_macro).unwrap();
        controller.shutdown_services().await.unwrap();
        let machine = controller.machine.lock().await;
        assert!(machine.closing);
        assert!(matches!(machine.operation, Operation::Idle));
        assert!(machine.pending_recording.is_none());
        assert_eq!(machine.store.get_macro("Retry").unwrap().steps, steps);
        drop(machine);
        controller.shutdown_services().await.unwrap();
    }

    #[tokio::test]
    async fn shutdown_tolerates_playback_finishing_after_operation_snapshot() {
        let dir = tempdir().unwrap();
        let controller = Arc::new(Controller::new(Store::open_at(dir.path()).unwrap()));
        controller.machine.lock().await.operation = Operation::Playing {
            macro_name: "Test".into(),
            started: Instant::now(),
            playback_id: 1,
        };
        let gate = Arc::new(ShutdownRaceGate {
            paused: tokio::sync::Notify::new(),
            resume: tokio::sync::Notify::new(),
        });
        *controller.shutdown_race_gate.lock().await = Some(gate.clone());

        let closing = controller.clone();
        let shutdown = tokio::spawn(async move { closing.shutdown_services().await });
        gate.paused.notified().await;
        controller
            .handle_native_event(NativeEvent::PlaybackEnded { playback_id: 1 })
            .await
            .unwrap();
        gate.resume.notify_one();

        shutdown.await.unwrap().unwrap();
        let machine = controller.machine.lock().await;
        assert!(machine.shutdown_complete);
        assert!(matches!(machine.operation, Operation::Idle));
    }

    #[tokio::test]
    async fn playback_failure_returns_to_idle_and_is_visible_in_snapshot() {
        let dir = tempdir().unwrap();
        let controller = Controller::new(Store::open_at(dir.path()).unwrap());
        controller.machine.lock().await.operation = Operation::Playing {
            macro_name: "Test".into(),
            started: Instant::now(),
            playback_id: 7,
        };

        controller
            .handle_native_event(NativeEvent::PlaybackFailed {
                playback_id: 7,
                reason: PlaybackFailure::Release,
            })
            .await
            .unwrap();
        let snapshot = controller.snapshot().await.unwrap();

        assert!(matches!(snapshot.status, AppStatus::Idle));
        assert_eq!(
            snapshot.last_error.as_deref(),
            Some(
                "AUTO-COC n’a pas pu relâcher une touche ou un bouton après la lecture. Vérifiez le clavier et la souris."
            )
        );
    }

    #[tokio::test]
    async fn stale_playback_failure_does_not_stop_a_new_playback() {
        let dir = tempdir().unwrap();
        let controller = Controller::new(Store::open_at(dir.path()).unwrap());
        controller.machine.lock().await.operation = Operation::Playing {
            macro_name: "Test".into(),
            started: Instant::now(),
            playback_id: 8,
        };

        controller
            .handle_native_event(NativeEvent::PlaybackFailed {
                playback_id: 7,
                reason: PlaybackFailure::Input,
            })
            .await
            .unwrap();
        let snapshot = controller.snapshot().await.unwrap();

        assert!(matches!(snapshot.status, AppStatus::Playing { .. }));
        assert_eq!(snapshot.last_error, None);
    }

    #[tokio::test]
    async fn shutdown_tolerates_recording_stopped_after_operation_snapshot() {
        let dir = tempdir().unwrap();
        let mut store = Store::open_at(dir.path()).unwrap();
        store.create_macro("Race").unwrap();
        let controller = Arc::new(Controller::new(store));
        let steps = vec![serde_json::json!({"t":0.1,"type":"nop","data":{}})];
        {
            let mut machine = controller.machine.lock().await;
            machine.operation = Operation::Recording {
                macro_name: "Race".into(),
                started: Instant::now(),
            };
            machine.pending_recording = Some(("Race".into(), steps.clone(), false));
        }
        let gate = Arc::new(ShutdownRaceGate {
            paused: tokio::sync::Notify::new(),
            resume: tokio::sync::Notify::new(),
        });
        *controller.shutdown_race_gate.lock().await = Some(gate.clone());

        let closing = controller.clone();
        let shutdown = tokio::spawn(async move { closing.shutdown_services().await });
        gate.paused.notified().await;
        controller.stop_recording().await.unwrap();
        gate.resume.notify_one();

        shutdown.await.unwrap().unwrap();
        let machine = controller.machine.lock().await;
        assert!(machine.shutdown_complete);
        assert!(matches!(machine.operation, Operation::Idle));
        assert!(machine.pending_recording.is_none());
        assert_eq!(machine.store.get_macro("Race").unwrap().steps, steps);
    }

    #[tokio::test]
    async fn shutdown_does_not_treat_an_in_progress_close_as_complete() {
        let dir = tempdir().unwrap();
        let controller = Controller::new(Store::open_at(dir.path()).unwrap());
        controller.machine.lock().await.closing = true;

        assert!(matches!(
            controller.shutdown_services().await,
            Err(ControllerError::InvalidState)
        ));
    }

    #[tokio::test]
    async fn migration_import_rechecks_idle_after_the_source_was_selected() {
        let app = tempdir().unwrap();
        let old = tempdir().unwrap();
        let config = old.path().join("config");
        let macros = config.join("macros");
        std::fs::create_dir_all(&macros).unwrap();
        let source_macro = serde_json::json!({
            "name":"Recharger COC",
            "updated_at":"2025-01-01T00:00:00Z",
            "sha1":"legacy",
            "steps":[{"t":0.0,"type":"nop","data":{}}]
        });
        std::fs::write(
            macros.join("Recharger COC.json"),
            serde_json::to_vec(&source_macro).unwrap(),
        )
        .unwrap();

        let controller = Controller::new(Store::open_at(app.path()).unwrap());
        let protected_path = app.path().join("macros/Recharger COC.json");
        let placeholder = std::fs::read(&protected_path).unwrap();
        {
            let mut machine = controller.machine.lock().await;
            machine.migration_source = Some(config);
            machine.operation = Operation::Recording {
                macro_name: "Recharger COC".into(),
                started: Instant::now(),
            };
        }

        assert!(matches!(
            controller.import_migration_source().await,
            Err(ControllerError::InvalidState)
        ));
        assert_eq!(std::fs::read(&protected_path).unwrap(), placeholder);

        controller.machine.lock().await.operation = Operation::Idle;
        let (_, result) = controller.import_migration_source().await.unwrap();
        assert_eq!(result.imported.macros, 1);
        assert_eq!(
            controller
                .machine
                .lock()
                .await
                .store
                .get_macro("Recharger COC")
                .unwrap()
                .steps,
            source_macro["steps"].as_array().unwrap().clone()
        );
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn migration_token_replacement_clears_old_telegram_authorization() {
        let app = tempdir().unwrap();
        let old = tempdir().unwrap();
        let config = old.path().join("config");
        std::fs::create_dir_all(&config).unwrap();
        let old_token = "synthetic-old-migration-token";
        let new_token = "synthetic-new-migration-token";

        let mut store = Store::open_at(app.path()).unwrap();
        crate::storage::save_token(app.path(), old_token.as_bytes()).unwrap();
        store.settings_mut().telegram.token_configured = true;
        store.settings_mut().telegram.paired = true;
        store.settings_mut().telegram.status = TelegramStatus::WaitingPairing;
        store.settings_mut().telegram.pairing_code = Some("654321".into());
        store.settings_mut().telegram.pairing_expires_at = Some("synthetic-expiry".into());
        store.set_telegram_owner(100, 200);
        store.persist().unwrap();
        std::fs::write(
            app.path().join("telegram-token.bin"),
            b"synthetic-corrupt-ciphertext",
        )
        .unwrap();
        std::fs::write(
            config.join("data.csv"),
            format!("parameter_name;parameter_value\ntelegram_bot_token;{new_token}\n"),
        )
        .unwrap();

        let controller = Controller::new(store);
        {
            let mut machine = controller.machine.lock().await;
            machine.migration_source = Some(config);
            machine.pairing = Some(Pairing {
                code: "654321".into(),
                expires: Instant::now() + Duration::from_secs(30),
                expires_at: "synthetic-expiry".into(),
                attempts: 0,
            });
        }

        let (snapshot, result) = controller.import_migration_source().await.unwrap();
        assert_eq!(result.imported.settings, 1);
        assert!(snapshot.settings.telegram.token_configured);
        assert!(!snapshot.settings.telegram.paired);
        assert!(snapshot.settings.telegram.pairing_code.is_none());
        assert!(snapshot.settings.telegram.pairing_expires_at.is_none());
        assert!(controller.machine.lock().await.pairing.is_none());
        assert_eq!(controller.machine.lock().await.store.telegram_owner(), None);
        assert_eq!(
            crate::storage::load_token(app.path()).unwrap().as_deref(),
            Some(new_token)
        );
        assert!(
            !controller
                .authorize_telegram_identity(new_token, 100, 200, true)
                .await
        );

        let persisted: serde_json::Value =
            serde_json::from_slice(&std::fs::read(app.path().join("settings.json")).unwrap())
                .unwrap();
        assert_eq!(persisted["settings"]["telegram"]["tokenConfigured"], true);
        assert_eq!(persisted["settings"]["telegram"]["paired"], false);
        assert_eq!(
            persisted["settings"]["telegram"]["pairingCode"],
            serde_json::Value::Null
        );
        assert_eq!(
            persisted["settings"]["telegram"]["pairingExpiresAt"],
            serde_json::Value::Null
        );
        assert_eq!(persisted["telegramOwner"], serde_json::Value::Null);
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn migration_failure_after_token_install_still_cancels_old_pairing() {
        let app = tempdir().unwrap();
        let old = tempdir().unwrap();
        let config = old.path().join("config");
        std::fs::create_dir_all(&config).unwrap();
        let old_token = "synthetic-old-partial-migration-token";
        let new_token = "synthetic-new-partial-migration-token";
        std::fs::write(
            config.join("data.csv"),
            format!("parameter_name;parameter_value\ntelegram_bot_token;{new_token}\n"),
        )
        .unwrap();

        let mut store = Store::open_at(app.path()).unwrap();
        crate::storage::save_token(app.path(), old_token.as_bytes()).unwrap();
        store.settings_mut().telegram.token_configured = true;
        store.settings_mut().telegram.paired = true;
        store.settings_mut().telegram.status = TelegramStatus::WaitingPairing;
        store.settings_mut().telegram.pairing_code = Some("654321".into());
        store.settings_mut().telegram.pairing_expires_at = Some("synthetic-expiry".into());
        store.set_telegram_owner(100, 200);
        store.persist().unwrap();
        std::fs::write(
            app.path().join("telegram-token.bin"),
            b"synthetic-corrupt-ciphertext-for-partial-migration",
        )
        .unwrap();

        let controller = Controller::new(store);
        {
            let mut machine = controller.machine.lock().await;
            machine.pairing = Some(Pairing {
                code: "654321".into(),
                expires: Instant::now() + Duration::from_secs(30),
                expires_at: "synthetic-expiry".into(),
                attempts: 0,
            });

            let root = app.path().to_path_buf();
            let Machine { store, pairing, .. } = &mut *machine;
            let result = store.import_legacy_with_token_installation(&config, || {
                *pairing = None;
                let manifests = root.join("migration-manifests");
                std::fs::remove_dir_all(&manifests).unwrap();
                std::fs::write(manifests, b"synthetic manifest write blocker").unwrap();
            });
            assert!(result.is_err());
        }

        assert!(controller.machine.lock().await.pairing.is_none());
        let machine = controller.machine.lock().await;
        assert!(machine.store.settings().telegram.token_configured);
        assert!(!machine.store.settings().telegram.paired);
        assert!(machine.store.settings().telegram.pairing_code.is_none());
        assert!(
            machine
                .store
                .settings()
                .telegram
                .pairing_expires_at
                .is_none()
        );
        assert_eq!(machine.store.telegram_owner(), None);
        drop(machine);
        assert_eq!(
            crate::storage::load_token(app.path()).unwrap().as_deref(),
            Some(new_token)
        );
        assert!(
            !controller
                .authorize_telegram_identity(new_token, 100, 200, true)
                .await
        );

        let persisted: serde_json::Value =
            serde_json::from_slice(&std::fs::read(app.path().join("settings.json")).unwrap())
                .unwrap();
        assert_eq!(persisted["settings"]["telegram"]["tokenConfigured"], true);
        assert_eq!(persisted["settings"]["telegram"]["paired"], false);
        assert_eq!(persisted["telegramOwner"], serde_json::Value::Null);
    }

    #[test]
    fn failed_recording_save_keeps_steps_for_a_later_retry() {
        let dir = tempdir().unwrap();
        let mut store = Store::open_at(dir.path()).unwrap();
        store.create_macro("Retry").unwrap();
        let macro_path = dir.path().join("macros/Retry.json");
        std::fs::remove_file(&macro_path).unwrap();
        std::fs::create_dir(&macro_path).unwrap();
        let steps = vec![serde_json::json!({"t":0.1,"type":"nop","data":{}})];
        let mut machine = Machine {
            store,
            operation: Operation::Recording {
                macro_name: "Retry".into(),
                started: Instant::now(),
            },
            closing: false,
            shutdown_complete: false,
            pending_recording: Some(("Retry".into(), steps.clone(), true)),
            coc_launched_once: false,
            cycles: 0,
            revision: 1,
            pairing: None,
            pairing_cooldown_until: None,
            shutdown: None,
            last_error: None,
            migration_source: None,
            migration_preview: None,
            migration_already_imported: false,
        };

        assert!(matches!(
            save_pending_recording(&mut machine),
            Err(ControllerError::RecordingSavePending)
        ));
        assert_eq!(machine.pending_recording.as_ref().unwrap().1, steps);

        std::fs::remove_dir(&macro_path).unwrap();
        machine.store.create_macro("Retry").unwrap();
        assert!(save_pending_recording(&mut machine).unwrap());
        assert!(machine.pending_recording.is_none());
        assert_eq!(machine.store.get_macro("Retry").unwrap().steps, steps);
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn telegram_token_rotation_clears_pairing_but_resubmitting_same_token_keeps_it() {
        let dir = tempdir().unwrap();
        let controller = Controller::new(Store::open_at(dir.path()).unwrap());
        let original = "000000000000000000000000000000001";
        let owner = crate::telegram::TelegramOwner {
            chat_id: 100,
            user_id: 200,
        };
        controller.configure_telegram(Some(original)).await.unwrap();
        let (code, _, _) = controller.start_pairing().await.unwrap();
        assert!(
            controller
                .attempt_telegram_pairing(original, &code, 100, 200, true)
                .await
                .unwrap()
        );
        assert_eq!(
            controller.machine.lock().await.store.telegram_owner(),
            Some((100, 200))
        );

        let (pending_code, _, _) = controller.start_pairing().await.unwrap();
        controller.configure_telegram(Some(original)).await.unwrap();
        assert!(
            controller
                .snapshot()
                .await
                .unwrap()
                .settings
                .telegram
                .paired
        );
        assert!(controller.machine.lock().await.pairing.is_some());

        controller
            .configure_telegram(Some("000000000000000000000000000000002"))
            .await
            .unwrap();
        assert!(
            !controller
                .snapshot()
                .await
                .unwrap()
                .settings
                .telegram
                .paired
        );
        assert!(controller.machine.lock().await.pairing.is_none());
        assert_eq!(controller.machine.lock().await.store.telegram_owner(), None);
        assert!(matches!(
            controller
                .attempt_telegram_pairing(original, &pending_code, 100, 200, true)
                .await,
            Err(ControllerError::TelegramUnauthorized)
        ));
        assert!(matches!(
            controller
                .start_playback_for_telegram(original, owner)
                .await,
            Err(ControllerError::TelegramUnauthorized)
        ));
    }

    #[tokio::test]
    async fn failed_telegram_token_removal_keeps_configured_state_persisted() {
        let dir = tempdir().unwrap();
        let mut store = Store::open_at(dir.path()).unwrap();
        store.settings_mut().telegram.token_configured = true;
        store.settings_mut().telegram.paired = true;
        store.settings_mut().telegram.status = TelegramStatus::Connected;
        store.set_telegram_owner(100, 200);
        store.persist().unwrap();
        std::fs::create_dir(dir.path().join("telegram-token.bin")).unwrap();
        let controller = Controller::new(store);

        assert!(matches!(
            controller.configure_telegram(None).await,
            Err(ControllerError::Store(StoreError::Io(_)))
        ));
        assert!(dir.path().join("telegram-token.bin").is_dir());

        let machine = controller.machine.lock().await;
        assert!(machine.store.settings().telegram.token_configured);
        assert!(machine.store.settings().telegram.paired);
        assert_eq!(
            machine.store.settings().telegram.status,
            TelegramStatus::Connected
        );
        assert_eq!(machine.store.telegram_owner(), Some((100, 200)));
        drop(machine);

        let persisted: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("settings.json")).unwrap())
                .unwrap();
        assert_eq!(persisted["settings"]["telegram"]["tokenConfigured"], true);
        assert_eq!(persisted["settings"]["telegram"]["paired"], true);
        assert_eq!(persisted["settings"]["telegram"]["status"], "connected");
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn fifth_wrong_pairing_code_expires_code_and_starts_cooldown() {
        let dir = tempdir().unwrap();
        let controller = Controller::new(Store::open_at(dir.path()).unwrap());
        let original = "000000000000000000000000000000001";
        controller.configure_telegram(Some(original)).await.unwrap();
        let (code, _, _) = controller.start_pairing().await.unwrap();
        let wrong = format!("{:06}", (code.parse::<u32>().unwrap() + 1) % 1_000_000);
        for _ in 0..MAX_PAIRING_ATTEMPTS {
            assert!(
                !controller
                    .attempt_telegram_pairing(original, &wrong, 100, 200, true)
                    .await
                    .unwrap()
            );
        }
        assert!(controller.machine.lock().await.pairing.is_none());
        assert!(matches!(
            controller.start_pairing().await,
            Err(ControllerError::PairingRateLimited)
        ));
        controller
            .configure_telegram(Some("000000000000000000000000000000002"))
            .await
            .unwrap();
        assert!(matches!(
            controller.start_pairing().await,
            Err(ControllerError::PairingRateLimited)
        ));
        controller.cancel_pairing().await.unwrap();
        assert!(matches!(
            controller.start_pairing().await,
            Err(ControllerError::PairingRateLimited)
        ));
    }
}
