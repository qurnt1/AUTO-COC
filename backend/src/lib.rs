mod api;
mod controller;
mod native_input;
mod security;
mod storage;
mod system;
mod telegram;
mod types;

use std::sync::Arc;

use axum::{Router, middleware, routing::get};
use tokio::{net::TcpListener, sync::watch};
use tracing::{error, info};

use crate::{api::api_router, controller::Controller, security::SecurityState, storage::Store};

#[derive(rust_embed::RustEmbed)]
#[folder = "$OUT_DIR/web-dist/"]
pub(crate) struct WebAssets;

pub async fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let store = Store::open()?;
    let controller = Arc::new(Controller::new(store));
    #[cfg(windows)]
    controller.init_native().await?;

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let origin = format!("http://{addr}");
    let security = SecurityState::new(addr, origin.clone());
    let (quit_tx, mut quit_rx) = watch::channel(false);
    let telegram_task = tokio::spawn(telegram::run_polling(controller.clone(), quit_rx.clone()));

    let app_state = AppState {
        controller: controller.clone(),
        security: security.clone(),
        quit_tx: quit_tx.clone(),
    };
    let app = Router::new()
        .route("/api/session", get(api::get_session))
        .merge(api_router())
        .fallback(api::serve_asset)
        .layer(middleware::from_fn_with_state(
            security,
            api::security_middleware,
        ))
        .with_state(app_state);

    info!("AUTO-COC local site listening on {addr}");
    webbrowser::open(&origin)?;

    let shutdown_tx = quit_tx.clone();
    let shutdown_controller = controller.clone();
    let server = axum::serve(listener, app).with_graceful_shutdown(async move {
        tokio::select! {
            signal = tokio::signal::ctrl_c() => {
                if let Err(signal_error) = signal {
                    error!(%signal_error, "failed to listen for Ctrl-C");
                    let _ = quit_rx.changed().await;
                    return;
                }
                match shutdown_controller.shutdown_services().await {
                    Ok(()) => { shutdown_tx.send_replace(true); }
                    Err(shutdown_error) => {
                        error!(%shutdown_error, "Ctrl-C left AUTO-COC open so the active operation can be retried or reviewed");
                        let _ = quit_rx.changed().await;
                    }
                }
            },
            _ = quit_rx.changed() => {},
        }
    });

    server.await?;
    quit_tx.send_replace(true);
    let _ = telegram_task.await;
    Ok(())
}

#[derive(Clone)]
pub struct AppState {
    pub controller: Arc<Controller>,
    pub security: SecurityState,
    pub quit_tx: watch::Sender<bool>,
}
