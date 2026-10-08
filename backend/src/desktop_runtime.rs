use std::sync::Arc;

use serde_json::Value;
use thiserror::Error;
use tokio::{
    sync::{Mutex, watch},
    task::JoinHandle,
};

use crate::{
    controller::{Controller, ControllerError},
    storage::Store,
    telegram,
    types::{
        Diagnostics, Help, MigrationImport, MigrationSelection, MigrationStatusResponse,
        PairingStart, SettingsPatch, ShortcutSettings, ShutdownPreparation, Snapshot,
    },
};

pub use crate::controller::ControllerError as ServiceError;
pub use crate::storage::StoreError as PersistenceError;

pub const SNAPSHOT_UPDATED_EVENT: &str = "snapshot-updated";

#[derive(Debug, Error)]
pub enum DesktopRuntimeError {
    #[error(transparent)]
    Controller(#[from] ControllerError),
    #[error("Telegram service did not shut down cleanly")]
    TelegramTaskFailed,
}

pub struct DesktopRuntime {
    controller: Arc<Controller>,
    telegram_shutdown: watch::Sender<bool>,
    telegram_task: Mutex<Option<JoinHandle<()>>>,
}

impl DesktopRuntime {
    pub async fn open() -> Result<Self, DesktopRuntimeError> {
        let store = Store::open().map_err(ControllerError::from)?;
        let controller = Arc::new(Controller::new(store));
        #[cfg(windows)]
        controller.init_native().await?;

        let (telegram_shutdown, receiver) = watch::channel(false);
        let telegram_task = tokio::spawn(telegram::run_polling(controller.clone(), receiver));
        Ok(Self {
            controller,
            telegram_shutdown,
            telegram_task: Mutex::new(Some(telegram_task)),
        })
    }

    pub async fn get_snapshot(&self) -> Result<Snapshot, DesktopRuntimeError> {
        Ok(self.controller.snapshot().await?)
    }

    pub fn subscribe_snapshots(&self) -> SnapshotSubscription {
        SnapshotSubscription {
            controller: self.controller.clone(),
            shutdown: self.telegram_shutdown.subscribe(),
            initial: true,
            last_revision: 0,
        }
    }

    pub async fn get_macro(&self, name: &str) -> Result<Value, DesktopRuntimeError> {
        Ok(self.controller.macro_file(name).await?)
    }

    pub async fn create_macro(&self, name: &str) -> Result<Snapshot, DesktopRuntimeError> {
        Ok(self.controller.create_macro(name).await?)
    }

    pub async fn rename_macro(
        &self,
        name: &str,
        new_name: &str,
    ) -> Result<Snapshot, DesktopRuntimeError> {
        Ok(self.controller.rename_macro(name, new_name).await?)
    }

    pub async fn delete_macro(&self, name: &str) -> Result<Snapshot, DesktopRuntimeError> {
        Ok(self.controller.delete_macro(name).await?)
    }

    pub async fn select_macro(&self, name: &str) -> Result<Snapshot, DesktopRuntimeError> {
        Ok(self.controller.select_macro(name).await?)
    }

    pub async fn start_recording(&self) -> Result<Snapshot, DesktopRuntimeError> {
        Ok(self.controller.start_recording_from_ui().await?)
    }

    pub async fn stop_recording(&self) -> Result<Snapshot, DesktopRuntimeError> {
        Ok(self.controller.stop_recording().await?)
    }

    pub async fn start_playback(&self) -> Result<Snapshot, DesktopRuntimeError> {
        Ok(self.controller.start_playback_from_ui().await?)
    }

    pub async fn stop_playback(&self) -> Result<Snapshot, DesktopRuntimeError> {
        Ok(self.controller.stop_playback().await?)
    }

    pub async fn cancel_resume_wait(&self) -> Result<Snapshot, DesktopRuntimeError> {
        Ok(self.controller.cancel_resume_wait().await?)
    }

    pub async fn patch_settings(
        &self,
        settings: SettingsPatch,
    ) -> Result<Snapshot, DesktopRuntimeError> {
        Ok(self
            .controller
            .set_settings(
                settings.loop_playback,
                settings.coc_path,
                settings.require_coc_foreground,
            )
            .await?)
    }

    pub async fn put_shortcuts(
        &self,
        shortcuts: ShortcutSettings,
    ) -> Result<Snapshot, DesktopRuntimeError> {
        Ok(self.controller.set_shortcuts(shortcuts).await?)
    }

    pub async fn launch_coc(&self) -> Result<Snapshot, DesktopRuntimeError> {
        Ok(self.controller.launch_coc().await?)
    }

    pub async fn screenshot(&self) -> Result<Vec<u8>, DesktopRuntimeError> {
        Ok(self.controller.capture_screenshot_png().await?)
    }

    pub async fn open_macros_folder(&self) -> Result<Snapshot, DesktopRuntimeError> {
        Ok(self.controller.open_macros_folder().await?)
    }

    pub async fn telegram_config(&self, token: &str) -> Result<Snapshot, DesktopRuntimeError> {
        Ok(self.controller.configure_telegram(Some(token)).await?)
    }

    pub async fn telegram_config_clear(&self) -> Result<Snapshot, DesktopRuntimeError> {
        Ok(self.controller.configure_telegram(None).await?)
    }

    pub async fn pairing_start(&self) -> Result<PairingStart, DesktopRuntimeError> {
        let (code, expires_at, snapshot) = self.controller.start_pairing().await?;
        Ok(PairingStart {
            code,
            expires_at,
            snapshot,
        })
    }

    pub async fn pairing_cancel(&self) -> Result<Snapshot, DesktopRuntimeError> {
        Ok(self.controller.cancel_pairing().await?)
    }

    pub async fn telegram_test(&self) -> Result<Snapshot, DesktopRuntimeError> {
        Ok(self.controller.test_telegram().await?)
    }

    pub async fn get_migration(&self) -> Result<MigrationStatusResponse, DesktopRuntimeError> {
        let (status, preview) = self.controller.migration_status().await?;
        Ok(MigrationStatusResponse { status, preview })
    }

    pub async fn select_migration_source(&self) -> Result<MigrationSelection, DesktopRuntimeError> {
        let (snapshot, preview) = self.controller.select_migration_source().await?;
        Ok(MigrationSelection { snapshot, preview })
    }

    pub async fn import_migration(&self) -> Result<MigrationImport, DesktopRuntimeError> {
        let (snapshot, result) = self.controller.import_migration_source().await?;
        Ok(MigrationImport {
            snapshot,
            imported: result.imported,
            collisions: result.collisions,
            preserved: result.preserved,
            errors: result.errors,
        })
    }

    pub async fn shutdown_prepare(&self) -> Result<ShutdownPreparation, DesktopRuntimeError> {
        let (confirmation_id, expires_at) = self.controller.prepare_shutdown().await?;
        Ok(ShutdownPreparation {
            confirmation_id,
            expires_at,
        })
    }

    pub async fn shutdown_confirm(
        &self,
        confirmation_id: &str,
    ) -> Result<Snapshot, DesktopRuntimeError> {
        Ok(self.controller.confirm_shutdown(confirmation_id).await?)
    }

    pub async fn get_help(&self) -> Help {
        Help::default()
    }

    pub async fn get_diagnostics(&self) -> Result<Diagnostics, DesktopRuntimeError> {
        Ok(self.controller.diagnostics().await?)
    }

    pub async fn complete_onboarding(&self) -> Result<Snapshot, DesktopRuntimeError> {
        Ok(self.controller.complete_onboarding().await?)
    }

    pub async fn quit_application(&self) -> Result<Snapshot, DesktopRuntimeError> {
        self.controller.shutdown_services().await?;
        self.telegram_shutdown.send_replace(true);
        if let Some(task) = self.telegram_task.lock().await.take() {
            task.await
                .map_err(|_| DesktopRuntimeError::TelegramTaskFailed)?;
        }
        Ok(self.controller.snapshot().await?)
    }
}

pub struct SnapshotSubscription {
    controller: Arc<Controller>,
    shutdown: watch::Receiver<bool>,
    initial: bool,
    last_revision: u64,
}

impl SnapshotSubscription {
    pub async fn recv(&mut self) -> Result<Option<Snapshot>, DesktopRuntimeError> {
        if self.initial {
            self.initial = false;
            let snapshot = self.controller.snapshot().await?;
            self.last_revision = snapshot.revision;
            return Ok(Some(snapshot));
        }

        loop {
            if *self.shutdown.borrow() {
                return Ok(None);
            }
            tokio::select! {
                result = self.controller.changed_snapshot(self.last_revision) => {
                    let snapshot = result?;
                    self.last_revision = self.last_revision.max(snapshot.revision);
                    return Ok(Some(snapshot));
                }
                changed = self.shutdown.changed() => {
                    if changed.is_err() || *self.shutdown.borrow() {
                        return Ok(None);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    async fn runtime_in(root: std::path::PathBuf) -> DesktopRuntime {
        let controller = Arc::new(Controller::new(Store::open_at(root).unwrap()));
        let (telegram_shutdown, receiver) = watch::channel(false);
        let telegram_task = tokio::spawn(telegram::run_polling(controller.clone(), receiver));
        DesktopRuntime {
            controller,
            telegram_shutdown,
            telegram_task: Mutex::new(Some(telegram_task)),
        }
    }

    #[tokio::test]
    async fn subscription_emits_initial_and_changed_snapshots() {
        let dir = tempdir().unwrap();
        let runtime = runtime_in(dir.path().to_path_buf()).await;
        let mut updates = runtime.subscribe_snapshots();

        let initial = updates.recv().await.unwrap().unwrap();
        let changed = runtime.create_macro("runtime-subscription").await.unwrap();
        let update = updates.recv().await.unwrap().unwrap();

        assert!(changed.revision > initial.revision);
        assert_eq!(update.revision, changed.revision);
        runtime.quit_application().await.unwrap();
        assert!(updates.recv().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn help_and_migration_dtos_keep_frontend_field_names() {
        let dir = tempdir().unwrap();
        let runtime = runtime_in(dir.path().to_path_buf()).await;

        let help = serde_json::to_value(runtime.get_help().await).unwrap();
        let migration = serde_json::to_value(runtime.get_migration().await.unwrap()).unwrap();
        assert!(help["sections"].is_array());
        assert!(migration["status"].is_object());
        assert!(migration.get("preview").is_some());
        runtime.quit_application().await.unwrap();
    }
}
