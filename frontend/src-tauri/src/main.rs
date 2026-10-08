#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use auto_coc::{DesktopRuntime, SNAPSHOT_UPDATED_EVENT};
use tauri::{
    AppHandle, Emitter, Manager, RunEvent, WindowEvent,
    async_runtime::Mutex,
    webview::{NewWindowResponse, WebviewWindowBuilder},
};

use commands::CommandError;

struct HostState {
    runtime: Arc<DesktopRuntime>,
    shutdown: ShutdownGate,
}

#[derive(Default)]
struct ShutdownGate {
    lock: Mutex<()>,
    complete: AtomicBool,
}

async fn shutdown(state: &HostState) -> Result<(), CommandError> {
    let _guard = state.shutdown.lock.lock().await;
    if state.shutdown.complete.load(Ordering::Acquire) {
        return Ok(());
    }

    state
        .runtime
        .quit_application()
        .await
        .map_err(CommandError::from)?;
    state.shutdown.complete.store(true, Ordering::Release);
    Ok(())
}

fn shutdown_is_complete(app: &AppHandle) -> bool {
    app.state::<HostState>()
        .shutdown
        .complete
        .load(Ordering::Acquire)
}

async fn shutdown_and_exit(app: AppHandle) {
    let result = {
        let state = app.state::<HostState>();
        shutdown(&state).await
    };

    match result {
        Ok(()) => app.exit(0),
        Err(error) => {
            let _ = app.emit("shutdown-error", error);
        }
    }
}

fn allows_navigation(url: &tauri::Url, is_dev: bool) -> bool {
    let production_origin =
        url.scheme() == "http" && url.host_str() == Some("tauri.localhost") && url.port().is_none();
    let dev_origin = is_dev
        && url.scheme() == "http"
        && url.host_str() == Some("127.0.0.1")
        && url.port() == Some(5173);

    production_origin || dev_origin
}

fn focus_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            focus_main_window(app);
        }))
        .setup(|app| {
            let runtime = Arc::new(tauri::async_runtime::block_on(DesktopRuntime::open())?);
            app.manage(HostState {
                runtime: runtime.clone(),
                shutdown: ShutdownGate::default(),
            });

            let mut snapshots = runtime.subscribe_snapshots();
            let app_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                while let Ok(Some(snapshot)) = snapshots.recv().await {
                    if app_handle.emit(SNAPSHOT_UPDATED_EVENT, snapshot).is_err() {
                        break;
                    }
                }
            });

            let config = app
                .config()
                .app
                .windows
                .iter()
                .find(|window| window.label == "main")
                .ok_or("missing main window configuration")?;
            let window = WebviewWindowBuilder::from_config(app.handle(), config)?
                .on_navigation(|url| allows_navigation(url, cfg!(dev)))
                .on_new_window(|_url, _features| NewWindowResponse::Deny)
                .build()?;

            let app_handle = app.handle().clone();
            window.on_window_event(move |event| {
                if let WindowEvent::CloseRequested { api, .. } = event {
                    api.prevent_close();
                    let app_handle = app_handle.clone();
                    tauri::async_runtime::spawn(shutdown_and_exit(app_handle));
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_snapshot,
            commands::get_macro,
            commands::create_macro,
            commands::rename_macro,
            commands::delete_macro,
            commands::select_macro,
            commands::start_recording,
            commands::stop_recording,
            commands::start_playback,
            commands::stop_playback,
            commands::cancel_resume_wait,
            commands::patch_settings,
            commands::put_shortcuts,
            commands::launch_coc,
            commands::screenshot,
            commands::open_macros_folder,
            commands::quit_application,
            commands::telegram_config,
            commands::telegram_config_clear,
            commands::pairing_start,
            commands::pairing_cancel,
            commands::telegram_test,
            commands::get_migration,
            commands::select_migration_source,
            commands::import_migration,
            commands::shutdown_prepare,
            commands::shutdown_confirm,
            commands::get_help,
            commands::get_diagnostics,
            commands::complete_onboarding,
        ])
        .build(tauri::generate_context!())
        .expect("failed to build AUTO-COC desktop host");

    app.run(|app_handle, event| {
        if let RunEvent::ExitRequested { api, .. } = event {
            if !shutdown_is_complete(app_handle) {
                api.prevent_exit();
                let app_handle = app_handle.clone();
                tauri::async_runtime::spawn(shutdown_and_exit(app_handle));
            }
        }
    });
}

fn main() {
    run();
}

#[cfg(test)]
mod tests {
    use tauri::Url;

    use super::allows_navigation;

    #[test]
    fn navigation_only_allows_app_and_vite_origins() {
        let app_url = Url::parse("http://tauri.localhost/macros").unwrap();
        let vite_url = Url::parse("http://127.0.0.1:5173/macros").unwrap();
        assert!(allows_navigation(&app_url, false));
        assert!(allows_navigation(&vite_url, true));
        assert!(!allows_navigation(&vite_url, false));
        assert!(!allows_navigation(
            &Url::parse("https://tauri.localhost/macros").unwrap(),
            true
        ));
        assert!(!allows_navigation(
            &Url::parse("http://127.0.0.1:5174/macros").unwrap(),
            true
        ));
        assert!(!allows_navigation(
            &Url::parse("https://example.com/").unwrap(),
            true
        ));
    }
}
