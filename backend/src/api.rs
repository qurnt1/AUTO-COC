use std::{path::Path, time::Duration};

use axum::{
    Json, Router,
    body::Body,
    extract::{Path as RoutePath, Query, State},
    http::{HeaderValue, Method, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
    routing::{get, patch, post, put},
};
use rust_embed::EmbeddedFile;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::time::timeout;

use crate::{
    AppState, WebAssets,
    controller::ControllerError,
    security::SecurityState,
    types::{
        CreateMacroRequest, RenameMacroRequest, SelectionRequest, SettingsPatch, ShortcutRequest,
    },
};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Diagnostics {
    pub app_version: String,
    pub rust_version: String,
    pub status: String,
    pub errors: Vec<String>,
}

#[derive(Serialize)]
pub struct Help {
    pub sections: Vec<HelpSection>,
}

#[derive(Serialize)]
pub struct HelpSection {
    pub title: &'static str,
    pub body: &'static str,
}

pub fn api_router() -> Router<AppState> {
    Router::new()
        .route("/api/snapshot", get(get_snapshot))
        .route("/api/events", get(get_events))
        .route("/api/macros", post(create_macro))
        .route(
            "/api/macros/{name}",
            get(get_macro).patch(rename_macro).delete(delete_macro),
        )
        .route("/api/selection", post(select_macro))
        .route("/api/recording/start", post(start_recording))
        .route("/api/recording/stop", post(stop_recording))
        .route("/api/playback/start", post(start_playback))
        .route("/api/playback/stop", post(stop_playback))
        .route("/api/settings", patch(patch_settings))
        .route("/api/shortcuts", put(put_shortcuts))
        .route("/api/launch-coc", post(launch_coc))
        .route("/api/screenshot", post(screenshot))
        .route(
            "/api/telegram/config",
            put(telegram_config).delete(telegram_config_clear),
        )
        .route("/api/telegram/pairing/start", post(pairing_start))
        .route("/api/telegram/pairing/cancel", post(pairing_cancel))
        .route("/api/telegram/test", post(telegram_test))
        .route("/api/migration", get(get_migration))
        .route(
            "/api/migration/select-source",
            post(select_migration_source),
        )
        .route("/api/migration/import", post(import_migration))
        .route("/api/system/shutdown/prepare", post(shutdown_prepare))
        .route("/api/system/shutdown/confirm", post(shutdown_confirm))
        .route("/api/macros-folder/open", post(open_macros_folder))
        .route("/api/application/quit", post(quit_application))
        .route("/api/help", get(get_help))
        .route("/api/diagnostics", get(get_diagnostics))
        .route("/api/onboarding/complete", post(complete_onboarding))
        .layer(axum::extract::DefaultBodyLimit::max(64 * 1024))
}

pub async fn get_session(State(state): State<AppState>) -> Response {
    let security = &state.security;
    (
        StatusCode::OK,
        [(header::CACHE_CONTROL, "no-store")],
        Json(json!({"token": security.token()})),
    )
        .into_response()
}

pub async fn security_middleware(
    axum::extract::State(security): axum::extract::State<SecurityState>,
    request: axum::http::Request<Body>,
    next: Next,
) -> Response {
    let path = request.uri().path();
    let headers = request.headers();
    let method = request.method();
    let valid_host = security.host_is_valid(headers);
    let allowed = if !valid_host {
        false
    } else if path == "/api/session" {
        *method == Method::GET && security.bootstrap_origin_valid(headers)
    } else if path.starts_with("/api/") {
        security.browser_origin_valid(method, headers)
            && security.token_is_valid(headers)
            && json_body_is_valid(method, headers)
    } else {
        true
    };

    if !allowed {
        let code = if valid_host {
            "request_origin_or_session_invalid"
        } else {
            "host_invalid"
        };
        let body = Json(
            json!({"error":{"code":code,"message":"La requête locale n’a pas pu être vérifiée."}}),
        );
        return secured((StatusCode::FORBIDDEN, body).into_response());
    }
    secured(next.run(request).await)
}

fn json_body_is_valid(method: &Method, headers: &axum::http::HeaderMap) -> bool {
    if matches!(*method, Method::GET | Method::HEAD) {
        return true;
    }
    let content_types = headers.get_all(header::CONTENT_TYPE);
    content_types.iter().count() == 1
        && content_types
            .iter()
            .next()
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| {
                value
                    .split(';')
                    .next()
                    .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("application/json"))
            })
}

fn secured(mut response: Response) -> Response {
    let headers = response.headers_mut();
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("same-origin"),
    );
    headers.insert("x-frame-options", HeaderValue::from_static("DENY"));
    headers.insert(header::CONTENT_SECURITY_POLICY, HeaderValue::from_static("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"));
    response
}

#[derive(Deserialize)]
struct EventsQuery {
    after: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EmptyRequest {}

async fn get_snapshot(
    State(state): State<AppState>,
) -> Result<Json<crate::types::Snapshot>, ApiError> {
    Ok(Json(state.controller.snapshot().await?))
}

async fn get_events(
    Query(query): Query<EventsQuery>,
    State(state): State<AppState>,
) -> Result<Json<crate::types::Snapshot>, ApiError> {
    let snapshot = timeout(
        Duration::from_secs(21),
        state.controller.changed_snapshot(query.after),
    )
    .await
    .map_err(|_| ApiError::internal())??;
    Ok(Json(snapshot))
}

async fn get_macro(
    RoutePath(name): RoutePath<String>,
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(state.controller.macro_file(&name).await?))
}

async fn create_macro(
    State(state): State<AppState>,
    Json(request): Json<CreateMacroRequest>,
) -> Result<Json<Value>, ApiError> {
    mutation(state.controller.create_macro(&request.name).await?)
}

async fn rename_macro(
    RoutePath(name): RoutePath<String>,
    State(state): State<AppState>,
    Json(request): Json<RenameMacroRequest>,
) -> Result<Json<Value>, ApiError> {
    mutation(
        state
            .controller
            .rename_macro(&name, &request.new_name)
            .await?,
    )
}

async fn delete_macro(
    RoutePath(name): RoutePath<String>,
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    mutation(state.controller.delete_macro(&name).await?)
}

async fn select_macro(
    State(state): State<AppState>,
    Json(request): Json<SelectionRequest>,
) -> Result<Json<Value>, ApiError> {
    mutation(state.controller.select_macro(&request.name).await?)
}

async fn start_recording(
    State(state): State<AppState>,
    Json(_): Json<EmptyRequest>,
) -> Result<Json<Value>, ApiError> {
    mutation(state.controller.start_recording().await?)
}

async fn stop_recording(
    State(state): State<AppState>,
    Json(_): Json<EmptyRequest>,
) -> Result<Json<Value>, ApiError> {
    mutation(state.controller.stop_recording().await?)
}

async fn start_playback(
    State(state): State<AppState>,
    Json(_): Json<EmptyRequest>,
) -> Result<Json<Value>, ApiError> {
    mutation(state.controller.start_playback().await?)
}

async fn stop_playback(
    State(state): State<AppState>,
    Json(_): Json<EmptyRequest>,
) -> Result<Json<Value>, ApiError> {
    mutation(state.controller.stop_playback().await?)
}

async fn patch_settings(
    State(state): State<AppState>,
    Json(request): Json<SettingsPatch>,
) -> Result<Json<Value>, ApiError> {
    mutation(
        state
            .controller
            .set_settings(request.loop_playback, request.coc_path)
            .await?,
    )
}

async fn put_shortcuts(
    State(state): State<AppState>,
    Json(request): Json<ShortcutRequest>,
) -> Result<Json<Value>, ApiError> {
    mutation(state.controller.set_shortcuts(request.into()).await?)
}

async fn complete_onboarding(
    State(state): State<AppState>,
    Json(_): Json<EmptyRequest>,
) -> Result<Json<Value>, ApiError> {
    mutation(state.controller.complete_onboarding().await?)
}

async fn pairing_start(
    State(state): State<AppState>,
    Json(_): Json<EmptyRequest>,
) -> Result<Json<Value>, ApiError> {
    let (code, expires_at, snapshot) = state.controller.start_pairing().await?;
    Ok(Json(
        json!({"code":code,"expiresAt":expires_at,"snapshot":snapshot}),
    ))
}

async fn pairing_cancel(
    State(state): State<AppState>,
    Json(_): Json<EmptyRequest>,
) -> Result<Json<Value>, ApiError> {
    mutation(state.controller.cancel_pairing().await?)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TelegramTokenRequest {
    token: String,
}

async fn telegram_config(
    State(state): State<AppState>,
    Json(request): Json<TelegramTokenRequest>,
) -> Result<Json<Value>, ApiError> {
    mutation(
        state
            .controller
            .configure_telegram(Some(&request.token))
            .await?,
    )
}

async fn telegram_config_clear(
    State(state): State<AppState>,
    Json(_): Json<EmptyRequest>,
) -> Result<Json<Value>, ApiError> {
    mutation(state.controller.configure_telegram(None).await?)
}

async fn telegram_test(
    State(state): State<AppState>,
    Json(_): Json<EmptyRequest>,
) -> Result<Json<Value>, ApiError> {
    mutation(state.controller.test_telegram().await?)
}

async fn shutdown_prepare(
    State(state): State<AppState>,
    Json(_): Json<EmptyRequest>,
) -> Result<Json<Value>, ApiError> {
    let (confirmation_id, expires_at) = state.controller.prepare_shutdown().await?;
    Ok(Json(
        json!({"confirmationId":confirmation_id,"expiresAt":expires_at}),
    ))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ShutdownRequest {
    confirmation_id: String,
}

async fn shutdown_confirm(
    State(state): State<AppState>,
    Json(request): Json<ShutdownRequest>,
) -> Result<Json<Value>, ApiError> {
    mutation(
        state
            .controller
            .confirm_shutdown(&request.confirmation_id)
            .await?,
    )
}

async fn quit_application(
    State(state): State<AppState>,
    Json(_): Json<EmptyRequest>,
) -> Result<Json<Value>, ApiError> {
    state.controller.shutdown_services().await?;
    let response = match state.controller.snapshot().await {
        Ok(snapshot) => mutation(snapshot),
        Err(error) => Err(ApiError::from(error)),
    };
    let quit_tx = state.quit_tx.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        quit_tx.send_replace(true);
    });
    response
}

async fn launch_coc(
    State(state): State<AppState>,
    Json(_): Json<EmptyRequest>,
) -> Result<Json<Value>, ApiError> {
    mutation(state.controller.launch_coc().await?)
}

async fn screenshot(
    State(state): State<AppState>,
    Json(_): Json<EmptyRequest>,
) -> Result<Response, ApiError> {
    let png = state.controller.capture_screenshot_png().await?;
    let mut response = Response::new(Body::from(png));
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static("image/png"));
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    Ok(response)
}

async fn open_macros_folder(
    State(state): State<AppState>,
    Json(_): Json<EmptyRequest>,
) -> Result<Json<Value>, ApiError> {
    mutation(state.controller.open_macros_folder().await?)
}

async fn get_help() -> Json<Help> {
    Json(Help {
        sections: vec![
            HelpSection {
                title: "Créer une macro",
                body: "Créez une macro, sélectionnez-la puis démarrez l’enregistrement. La capture clavier/souris commence après un délai de préparation de 3 secondes. Arrêtez-la depuis l’application.",
            },
            HelpSection {
                title: "Lire et arrêter",
                body: "La lecture rejoue les entrées clavier et souris enregistrées. Gardez l’application ouverte pendant une opération et utilisez Arrêter dès qu’un replay ne se déroule pas comme prévu.",
            },
            HelpSection {
                title: "Raccourcis par défaut",
                body: "F1 bascule la lecture et l’arrêt, Ctrl+Shift+1 démarre une lecture, Ctrl+Shift+0 arrête une lecture ou un enregistrement. Les raccourcis personnalisés sont enregistrés seulement si Windows accepte les trois touches.",
            },
            HelpSection {
                title: "Quitter AUTO-COC",
                body: "Fermer l’onglet du navigateur ne quitte pas le service local. Utilisez « Quitter AUTO-COC » dans Réglages > Outils de cet ordinateur, ou Ctrl+C dans le terminal qui a lancé le service.",
            },
            HelpSection {
                title: "Données",
                body: "Les macros sont stockées dans le dossier local AUTO-COC de votre profil Windows. Les macros héritées sont copiées sans supprimer les fichiers d’origine.",
            },
        ],
    })
}

async fn get_diagnostics(State(state): State<AppState>) -> Result<Json<Diagnostics>, ApiError> {
    Ok(Json(state.controller.diagnostics().await?))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MigrationResponse {
    status: crate::types::MigrationStatus,
    preview: Option<crate::types::MigrationPreview>,
}

async fn get_migration(State(state): State<AppState>) -> Result<Json<MigrationResponse>, ApiError> {
    let (status, preview) = state.controller.migration_status().await?;
    Ok(Json(MigrationResponse { status, preview }))
}

async fn select_migration_source(
    State(state): State<AppState>,
    Json(_): Json<EmptyRequest>,
) -> Result<Json<Value>, ApiError> {
    let (snapshot, preview) = state.controller.select_migration_source().await?;
    Ok(Json(json!({"snapshot":snapshot,"preview":preview})))
}

async fn import_migration(
    State(state): State<AppState>,
    Json(_): Json<EmptyRequest>,
) -> Result<Json<Value>, ApiError> {
    let (snapshot, result) = state.controller.import_migration_source().await?;
    Ok(Json(
        json!({"snapshot":snapshot,"imported":result.imported,"collisions":result.collisions,"preserved":result.preserved,"errors":result.errors}),
    ))
}

pub(crate) async fn serve_asset(uri: axum::http::Uri) -> Response {
    let request_path = uri.path().trim_start_matches('/');
    if request_path == "api" || request_path.starts_with("api/") {
        return ApiError::not_found().into_response();
    }
    let normalized = if request_path.is_empty() {
        "index.html"
    } else {
        request_path
    };
    if normalized
        .split('/')
        .any(|part| part == ".." || part.is_empty())
    {
        return ApiError::not_found().into_response();
    }
    let direct = WebAssets::get(normalized);
    let asset = direct.or_else(|| {
        if normalized.contains('.') {
            None
        } else {
            WebAssets::get("index.html")
        }
    });
    match asset {
        Some(asset) => embedded_response(normalized, asset),
        None => ApiError::not_found().into_response(),
    }
}

fn embedded_response(path: &str, asset: EmbeddedFile) -> Response {
    let content_type = mime_guess::from_path(Path::new(path))
        .first_or_octet_stream()
        .to_string();
    let mut response = Response::new(Body::from(asset.data.into_owned()));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(&content_type)
            .unwrap_or_else(|_| HeaderValue::from_static("application/octet-stream")),
    );
    response
}

fn mutation(snapshot: crate::types::Snapshot) -> Result<Json<Value>, ApiError> {
    Ok(Json(json!({"snapshot":snapshot})))
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: &'static str,
}

impl ApiError {
    fn internal() -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "internal_error",
            message: "L’opération n’a pas pu aboutir.",
        }
    }
    fn not_found() -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "not_found",
            message: "Cette ressource est introuvable.",
        }
    }
}

impl From<ControllerError> for ApiError {
    fn from(error: ControllerError) -> Self {
        match error {
            ControllerError::Store(crate::storage::StoreError::NotFound) => Self {
                status: StatusCode::NOT_FOUND,
                code: "not_found",
                message: "Cette macro est introuvable.",
            },
            ControllerError::Store(crate::storage::StoreError::InvalidName) => Self {
                status: StatusCode::BAD_REQUEST,
                code: "invalid_name",
                message: "Le nom contient des caractères non autorisés.",
            },
            ControllerError::Store(crate::storage::StoreError::Conflict) => Self {
                status: StatusCode::CONFLICT,
                code: "conflict",
                message: "Une macro de ce nom existe déjà.",
            },
            ControllerError::Store(crate::storage::StoreError::Protected) => Self {
                status: StatusCode::FORBIDDEN,
                code: "protected_macro",
                message: "Cette macro système ne peut pas être modifiée.",
            },
            ControllerError::Store(crate::storage::StoreError::Unreadable) => Self {
                status: StatusCode::UNPROCESSABLE_ENTITY,
                code: "macro_unreadable",
                message: "Cette macro contient des données invalides et n’a pas été modifiée.",
            },
            ControllerError::Store(crate::storage::StoreError::InvalidData) => Self {
                status: StatusCode::BAD_REQUEST,
                code: "invalid_data",
                message: "Les données fournies sont invalides.",
            },
            ControllerError::Store(crate::storage::StoreError::Io(_)) => Self::internal(),
            ControllerError::InvalidState => Self {
                status: StatusCode::CONFLICT,
                code: "invalid_state",
                message: "Cette opération n’est pas disponible dans l’état actuel.",
            },
            ControllerError::NativeInputUnavailable => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                code: "native_input_unavailable",
                message: "Le moteur clavier/souris Windows n’est pas disponible.",
            },
            ControllerError::InvalidShortcut => Self {
                status: StatusCode::BAD_REQUEST,
                code: "invalid_shortcut",
                message: "Le raccourci doit contenir une combinaison de touches prise en charge.",
            },
            ControllerError::ShortcutUnavailable => Self {
                status: StatusCode::CONFLICT,
                code: "shortcut_unavailable",
                message: "Au moins un raccourci est déjà utilisé par Windows ou une autre application.",
            },
            ControllerError::NativeInputFailed => Self::internal(),
            ControllerError::RecordingSavePending => Self {
                status: StatusCode::INSUFFICIENT_STORAGE,
                code: "recording_save_pending",
                message: "Les étapes sont conservées en mémoire. Relancez Arrêter pour réessayer leur sauvegarde.",
            },
            ControllerError::RecordingEventLimit => Self {
                status: StatusCode::PAYLOAD_TOO_LARGE,
                code: "recording_event_limit",
                message: "La limite d’événements a été atteinte. La partie déjà capturée est sauvegardée, mais la macro peut être incomplète.",
            },
            ControllerError::ScreenshotUnavailable => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                code: "screenshot_unavailable",
                message: "La capture d’écran n’est pas disponible sur ce poste.",
            },
            ControllerError::EmptyMacro => Self {
                status: StatusCode::UNPROCESSABLE_ENTITY,
                code: "empty_macro",
                message: "La macro sélectionnée ne contient aucun événement.",
            },
            ControllerError::TelegramNotConfigured => Self {
                status: StatusCode::CONFLICT,
                code: "telegram_not_configured",
                message: "Configurez d’abord le bot Telegram.",
            },
            ControllerError::TelegramUnauthorized => Self {
                status: StatusCode::FORBIDDEN,
                code: "telegram_unauthorized",
                message: "Cette action Telegram n’est plus autorisée.",
            },
            ControllerError::TelegramUnavailable => Self {
                status: StatusCode::BAD_GATEWAY,
                code: "telegram_unavailable",
                message: "Telegram n’a pas accepté le token ou ne répond pas.",
            },
            ControllerError::Expired => Self {
                status: StatusCode::GONE,
                code: "expired",
                message: "Cette confirmation a expiré ou a déjà été utilisée.",
            },
            ControllerError::PairingRateLimited => Self {
                status: StatusCode::TOO_MANY_REQUESTS,
                code: "pairing_rate_limited",
                message: "Trop d’essais. Attendez avant de générer un nouveau code.",
            },
            ControllerError::MigrationCancelled => Self {
                status: StatusCode::CONFLICT,
                code: "migration_cancelled",
                message: "Sélection de dossier annulée.",
            },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({"error":{"code":self.code,"message":self.message}})),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;
    use crate::{controller::Controller, security::SecurityState, storage::Store};

    fn test_app() -> (Router, String) {
        let (router, token, dir, _) = test_app_with_quit_signal();
        let _ = dir.keep();
        (router, token)
    }

    fn test_app_with_quit_signal() -> (
        Router,
        String,
        tempfile::TempDir,
        tokio::sync::watch::Receiver<bool>,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let controller = Arc::new(Controller::new(Store::open_at(dir.path()).unwrap()));
        let addr = "127.0.0.1:41234".parse().unwrap();
        let security = SecurityState::new(addr, "http://127.0.0.1:41234".into());
        let token = security.token().to_owned();
        let (quit_tx, quit_rx) = tokio::sync::watch::channel(false);
        let state = AppState {
            controller,
            security: security.clone(),
            quit_tx,
        };
        let router = Router::new()
            .route("/api/session", get(get_session))
            .merge(api_router())
            .fallback(serve_asset)
            .layer(axum::middleware::from_fn_with_state(
                security,
                security_middleware,
            ))
            .with_state(state);
        (router, token, dir, quit_rx)
    }

    fn request(uri: &str, token: Option<&str>, body: Option<&str>) -> Request<Body> {
        request_method(
            if body.is_some() { "POST" } else { "GET" },
            uri,
            token,
            body,
        )
    }

    fn request_method(
        method: &str,
        uri: &str,
        token: Option<&str>,
        body: Option<&str>,
    ) -> Request<Body> {
        let mut builder = Request::builder()
            .method(method)
            .uri(uri)
            .header("host", "127.0.0.1:41234")
            .header("origin", "http://127.0.0.1:41234");
        if let Some(token) = token {
            builder = builder.header("x-auto-coc-session", token);
        }
        if body.is_some() {
            builder = builder.header("content-type", "application/json");
        }
        builder
            .body(body.map_or_else(Body::empty, |value| Body::from(value.to_owned())))
            .unwrap()
    }

    #[tokio::test]
    async fn session_bootstrap_and_mutation_require_origin_session_and_json() {
        let (app, token) = test_app();
        let session = app
            .clone()
            .oneshot(request("/api/session", None, None))
            .await
            .unwrap();
        assert_eq!(session.status(), StatusCode::OK);
        let missing = app
            .clone()
            .oneshot(request("/api/snapshot", None, None))
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::FORBIDDEN);
        let created = app
            .clone()
            .oneshot(request(
                "/api/macros",
                Some(&token),
                Some(r#"{"name":"Test"}"#),
            ))
            .await
            .unwrap();
        assert_eq!(created.status(), StatusCode::OK);
        let response: Value =
            serde_json::from_slice(&created.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(response["snapshot"]["selectedMacro"], "Test");
    }

    #[tokio::test]
    async fn mutation_rejects_cross_origin_and_wrong_host() {
        let (app, token) = test_app();
        let cross = Request::builder()
            .uri("/api/macros")
            .header("host", "127.0.0.1:41234")
            .header("origin", "http://evil.test")
            .header("x-auto-coc-session", token.as_str())
            .header("content-type", "application/json")
            .body(Body::from(r#"{"name":"X"}"#))
            .unwrap();
        assert_eq!(
            app.clone().oneshot(cross).await.unwrap().status(),
            StatusCode::FORBIDDEN
        );
        let wrong_host = Request::builder()
            .uri("/api/snapshot")
            .header("host", "localhost:41234")
            .header("origin", "http://127.0.0.1:41234")
            .header("x-auto-coc-session", token.as_str())
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.oneshot(wrong_host).await.unwrap().status(),
            StatusCode::FORBIDDEN
        );
    }

    #[tokio::test]
    async fn help_explains_how_to_quit_the_local_service() {
        let (app, token) = test_app();
        let response = app
            .oneshot(request("/api/help", Some(&token), None))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let help: Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        let section = help["sections"]
            .as_array()
            .unwrap()
            .iter()
            .find(|section| section["title"] == "Quitter AUTO-COC")
            .unwrap();
        let body = section["body"].as_str().unwrap();
        assert!(body.contains("Fermer l’onglet du navigateur ne quitte pas le service local"));
        assert!(body.contains("Réglages > Outils de cet ordinateur"));
        assert!(body.contains("Ctrl+C"));
    }

    #[tokio::test]
    async fn quit_signal_is_sent_even_when_snapshot_fails_after_shutdown() {
        let (app, token, dir, mut quit_rx) = test_app_with_quit_signal();
        std::fs::remove_dir_all(dir.path().join("macros")).unwrap();
        std::fs::write(dir.path().join("macros"), b"not a directory").unwrap();

        let response = app
            .oneshot(request_method(
                "POST",
                "/api/application/quit",
                Some(&token),
                Some("{}"),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        tokio::time::timeout(Duration::from_secs(1), quit_rx.changed())
            .await
            .unwrap()
            .unwrap();
        assert!(*quit_rx.borrow());
    }

    #[tokio::test]
    async fn playback_uses_empty_json_and_telegram_clear_uses_delete() {
        let (app, token) = test_app();
        let playback = app
            .clone()
            .oneshot(request("/api/playback/start", Some(&token), Some("{}")))
            .await
            .unwrap();
        assert_eq!(playback.status(), StatusCode::NOT_FOUND);

        let empty_token = app
            .clone()
            .oneshot(request_method(
                "PUT",
                "/api/telegram/config",
                Some(&token),
                Some(r#"{"token":""}"#),
            ))
            .await
            .unwrap();
        assert_eq!(empty_token.status(), StatusCode::BAD_REQUEST);
        let cleared = app
            .oneshot(request_method(
                "DELETE",
                "/api/telegram/config",
                Some(&token),
                Some("{}"),
            ))
            .await
            .unwrap();
        assert_eq!(cleared.status(), StatusCode::OK);
    }
}
