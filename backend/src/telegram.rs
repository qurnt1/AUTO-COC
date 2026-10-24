use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use std::{collections::HashMap, future::Future, sync::Arc, time::Duration};
use thiserror::Error;
use tokio::sync::watch;

use crate::controller::{Controller, ControllerError};

#[derive(Debug, Error)]
pub enum TelegramError {
    #[error("Telegram request failed")]
    Request,
    #[error("Telegram refused the token")]
    Unauthorized,
    #[error("Telegram rejected the request")]
    Rejected,
    #[error("Telegram message already has the requested content")]
    MessageNotModified,
    #[error("Telegram response was invalid")]
    InvalidResponse,
}

pub struct TelegramClient {
    client: reqwest::Client,
    token: String,
}

impl TelegramClient {
    pub fn new(token: String) -> Result<Self, TelegramError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(40))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|_| TelegramError::Request)?;
        Ok(Self { client, token })
    }

    pub async fn get_me(&self) -> Result<TelegramUser, TelegramError> {
        self.request("getMe", &json!({})).await
    }

    pub async fn get_updates(&self, offset: i64) -> Result<Vec<TelegramUpdate>, TelegramError> {
        self.fetch_updates(offset, 30).await
    }

    pub async fn drop_pending_updates(&self) -> Result<i64, TelegramError> {
        let pending = self.fetch_updates(-1, 0).await?;
        Ok(pending
            .last()
            .map_or(0, |update| update.update_id.saturating_add(1)))
    }

    async fn fetch_updates(
        &self,
        offset: i64,
        timeout_seconds: u8,
    ) -> Result<Vec<TelegramUpdate>, TelegramError> {
        self.request(
            "getUpdates",
            &json!({
                "offset": offset,
                "timeout": timeout_seconds,
                "limit": 100,
                "allowed_updates": ["message", "callback_query"]
            }),
        )
        .await
    }

    pub async fn send_message(
        &self,
        chat_id: i64,
        text: &str,
        keyboard: Option<Value>,
    ) -> Result<i64, TelegramError> {
        let mut body = json!({"chat_id":chat_id,"text":text});
        if let Some(keyboard) = keyboard {
            body["reply_markup"] = keyboard;
        }
        self.request::<TelegramSentMessage>("sendMessage", &body)
            .await
            .map(|message| message.message_id)
    }

    pub async fn edit_message_text(
        &self,
        chat_id: i64,
        message_id: i64,
        text: &str,
        keyboard: Option<Value>,
    ) -> Result<(), TelegramError> {
        let mut body = json!({"chat_id":chat_id,"message_id":message_id,"text":text});
        if let Some(keyboard) = keyboard {
            body["reply_markup"] = keyboard;
        }
        match self.request::<Value>("editMessageText", &body).await {
            Err(TelegramError::MessageNotModified) => Ok(()),
            result => result.map(|_| ()),
        }
    }

    pub async fn send_photo(
        &self,
        chat_id: i64,
        png: Vec<u8>,
        caption: &str,
    ) -> Result<(), TelegramError> {
        let form = reqwest::multipart::Form::new()
            .text("chat_id", chat_id.to_string())
            .text("caption", caption.to_owned())
            .part(
                "photo",
                reqwest::multipart::Part::bytes(png).file_name("capture.png"),
            );
        let response = self
            .client
            .post(self.endpoint("sendPhoto"))
            .multipart(form)
            .send()
            .await
            .map_err(|_| TelegramError::Request)?;
        if !response.status().is_success() {
            return Err(TelegramError::Rejected);
        }
        let envelope: TelegramEnvelope<Value> = response
            .json()
            .await
            .map_err(|_| TelegramError::InvalidResponse)?;
        if !envelope.ok {
            return Err(TelegramError::Rejected);
        }
        Ok(())
    }

    pub async fn answer_callback(
        &self,
        callback_id: &str,
        text: &str,
    ) -> Result<(), TelegramError> {
        self.request::<Value>(
            "answerCallbackQuery",
            &json!({"callback_query_id":callback_id,"text":text}),
        )
        .await
        .map(|_| ())
    }

    async fn request<T: DeserializeOwned>(
        &self,
        method: &str,
        body: &Value,
    ) -> Result<T, TelegramError> {
        let response = self
            .client
            .post(self.endpoint(method))
            .json(body)
            .send()
            .await
            .map_err(|_| TelegramError::Request)?;
        let status = response.status();
        if status.as_u16() == 401 {
            return Err(TelegramError::Unauthorized);
        }
        let envelope: TelegramEnvelope<T> = response.json().await.map_err(|_| {
            telegram_error_for_http_status(status).unwrap_or(TelegramError::InvalidResponse)
        })?;
        if envelope.error_code == Some(401) {
            return Err(TelegramError::Unauthorized);
        }
        if envelope
            .description
            .as_deref()
            .is_some_and(|description| description.contains("message is not modified"))
        {
            return Err(TelegramError::MessageNotModified);
        }
        if let Some(error) = telegram_error_for_http_status(status) {
            return Err(error);
        }
        if !envelope.ok {
            return Err(TelegramError::Rejected);
        }
        envelope.result.ok_or(TelegramError::InvalidResponse)
    }

    fn endpoint(&self, method: &str) -> String {
        format!("https://api.telegram.org/bot{}/{method}", self.token)
    }
}

fn telegram_error_for_http_status(status: reqwest::StatusCode) -> Option<TelegramError> {
    if status.as_u16() == 401 {
        Some(TelegramError::Unauthorized)
    } else if status.is_success() {
        None
    } else {
        Some(TelegramError::Rejected)
    }
}

#[derive(Deserialize)]
pub struct TelegramEnvelope<T> {
    ok: bool,
    result: Option<T>,
    #[serde(default)]
    error_code: Option<u16>,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Deserialize)]
struct TelegramSentMessage {
    message_id: i64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TelegramUser {
    pub id: i64,
    #[serde(default)]
    pub first_name: String,
    #[serde(default)]
    pub is_bot: bool,
    pub username: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TelegramChat {
    pub id: i64,
    #[serde(rename = "type")]
    pub chat_type: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TelegramMessage {
    pub message_id: i64,
    pub chat: TelegramChat,
    pub from: Option<TelegramUser>,
    pub text: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TelegramCallback {
    pub id: String,
    pub from: TelegramUser,
    pub message: Option<TelegramMessage>,
    pub data: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TelegramUpdate {
    pub update_id: i64,
    pub message: Option<TelegramMessage>,
    pub callback_query: Option<TelegramCallback>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TelegramOwner {
    pub chat_id: i64,
    pub user_id: i64,
}

#[cfg(test)]
fn message_is_authorized(message: &TelegramMessage, owner: TelegramOwner) -> bool {
    message.chat.chat_type == "private"
        && message.chat.id == owner.chat_id
        && message
            .from
            .as_ref()
            .is_some_and(|user| user.id == owner.user_id)
}

pub fn callback_is_authorized(callback: &TelegramCallback, owner: TelegramOwner) -> bool {
    callback.message.as_ref().is_some_and(|message| {
        message.chat.chat_type == "private"
            && message.chat.id == owner.chat_id
            && callback.from.id == owner.user_id
    })
}

#[cfg(test)]
fn pairing_code_from_start(text: &str) -> Option<&str> {
    pairing_code_from_start_for(text, None)
}

fn pairing_code_from_start_for<'a>(text: &'a str, bot_username: Option<&str>) -> Option<&'a str> {
    let mut parts = text.split_whitespace();
    let command = parts.next()?;
    let command = command.strip_prefix('/')?;
    let (command, addressed_bot) = command
        .split_once('@')
        .map_or((command, None), |(name, bot)| (name, Some(bot)));
    if !command.eq_ignore_ascii_case("start") {
        return None;
    }
    if let Some(addressed_bot) = addressed_bot
        && !bot_username.is_some_and(|name| name.eq_ignore_ascii_case(addressed_bot))
    {
        return None;
    }
    let code = parts.next()?;
    if parts.next().is_some() || code.len() != 6 || !code.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    Some(code)
}

fn is_start_without_code(text: &str, bot_username: Option<&str>) -> bool {
    let mut parts = text.split_whitespace();
    let Some(command) = parts.next().and_then(|command| command.strip_prefix('/')) else {
        return false;
    };
    if parts.next().is_some() {
        return false;
    }
    let (name, addressed_bot) = command
        .split_once('@')
        .map_or((command, None), |(name, bot)| (name, Some(bot)));
    name.eq_ignore_ascii_case("start")
        && addressed_bot
            .is_none_or(|bot| bot_username.is_some_and(|name| name.eq_ignore_ascii_case(bot)))
}

pub fn text_command(text: &str) -> Option<&'static str> {
    match text.trim().to_lowercase().as_str() {
        "stop" | "/stop" | "arreter" | "arrêter" | "pause" => Some("STOP"),
        "go" | "/go" | "start" | "lancer" | "reprendre" => Some("GO"),
        "shutdown" | "/shutdown" | "eteindre" | "éteindre" | "poweroff" => Some("SHUTDOWN_ASK"),
        "capture" | "/capture" | "screenshot" | "screen" => Some("CAPTURE"),
        "gif" => Some("CAPTURE_GIF"),
        "menu" | "/menu" => Some("MENU"),
        "relancer" | "/relancer" => Some("RELOAD_COC"),
        "launch" | "/launch" => Some("LAUNCH_COC"),
        _ => None,
    }
}

pub fn callback_command(data: &str) -> Option<&str> {
    const ALLOWED: &[&str] = &[
        "STOP",
        "GO",
        "MENU",
        "BACK",
        "SHUTDOWN_ASK",
        "SHUTDOWN_CONFIRM",
        "SHUTDOWN_CANCEL",
        "CAPTURE",
        "LAUNCH_COC",
        "RELOAD_COC",
        "TOGGLE_LOOP",
        "DUMMY_COC_STATUS",
        "SELECT_MACRO_LIST",
        "CANCEL_SELECTION",
    ];
    if data.len() <= 64 && data.starts_with("SELECT_MACRO:") && data.len() > 13 {
        Some(data)
    } else if let Some(id) = data.strip_prefix("SHUTDOWN_CONFIRM:")
        && id.len() == 40
        && id.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        Some(data)
    } else if ALLOWED.contains(&data) {
        Some(data)
    } else {
        None
    }
}

pub async fn run_polling(controller: Arc<Controller>, mut shutdown: watch::Receiver<bool>) {
    let mut active_token: Option<String> = None;
    let mut active_offset = 0;
    let mut callback_names = HashMap::new();

    'service: loop {
        if *shutdown.borrow() {
            break;
        }
        let token = match controller.telegram_token().await {
            Ok(Some(token)) => token,
            Ok(None) => {
                active_token = None;
                if wait_or_shutdown(&mut shutdown, Duration::from_secs(3)).await {
                    break;
                }
                continue;
            }
            Err(_) => {
                if wait_or_shutdown(&mut shutdown, Duration::from_secs(5)).await {
                    break;
                }
                continue;
            }
        };
        let client = match TelegramClient::new(token.clone()) {
            Ok(client) => client,
            Err(error) => {
                let _ = controller.set_telegram_error(&token, error).await;
                if wait_or_shutdown(&mut shutdown, Duration::from_secs(5)).await {
                    break;
                }
                continue;
            }
        };

        let Some(bot_result) = await_or_shutdown(&mut shutdown, client.get_me()).await else {
            break 'service;
        };
        let bot = match bot_result {
            Ok(bot) => bot,
            Err(error) => {
                let _ = controller.set_telegram_error(&token, error).await;
                if wait_or_shutdown(&mut shutdown, Duration::from_secs(5)).await {
                    break;
                }
                continue;
            }
        };
        let is_new_token = active_token.as_deref() != Some(&token);
        let mut offset = if is_new_token {
            callback_names.clear();
            let Some(drop_result) =
                await_or_shutdown(&mut shutdown, client.drop_pending_updates()).await
            else {
                break 'service;
            };
            match drop_result {
                Ok(offset) => offset,
                Err(error) => {
                    let _ = controller.set_telegram_error(&token, error).await;
                    if wait_or_shutdown(&mut shutdown, Duration::from_secs(5)).await {
                        break;
                    }
                    continue;
                }
            }
        } else {
            active_offset
        };
        if !controller.telegram_token_is_current(&token).await {
            continue;
        }
        active_token = Some(token.clone());
        active_offset = offset;
        let _ = controller.set_telegram_connected(&token).await;

        loop {
            if *shutdown.borrow() {
                break 'service;
            }
            if !controller.telegram_token_is_current(&token).await {
                break;
            }
            let updates = tokio::select! {
                result = client.get_updates(offset) => result,
                changed = shutdown.changed() => {
                    if changed.is_err() || *shutdown.borrow() { break 'service; }
                    continue;
                }
            };
            let updates = match updates {
                Ok(updates) => updates,
                Err(error) => {
                    let _ = controller.set_telegram_error(&token, error).await;
                    break;
                }
            };
            for update in updates {
                offset = update.update_id.saturating_add(1);
                active_offset = offset;
                if !controller.telegram_token_is_current(&token).await {
                    break;
                }
                let handler = handle_update(
                    &controller,
                    &client,
                    &token,
                    bot.username.as_deref(),
                    update,
                    &mut callback_names,
                );
                if await_or_shutdown(&mut shutdown, handler).await.is_none() {
                    break 'service;
                }
            }
        }
    }
}

async fn await_or_shutdown<F: Future>(
    shutdown: &mut watch::Receiver<bool>,
    future: F,
) -> Option<F::Output> {
    if *shutdown.borrow() {
        return None;
    }
    tokio::select! {
        output = future => Some(output),
        _ = shutdown.changed() => None,
    }
}

async fn wait_or_shutdown(shutdown: &mut watch::Receiver<bool>, duration: Duration) -> bool {
    tokio::select! {
        _ = tokio::time::sleep(duration) => false,
        changed = shutdown.changed() => changed.is_err() || *shutdown.borrow(),
    }
}

async fn handle_update(
    controller: &Controller,
    client: &TelegramClient,
    token: &str,
    bot_username: Option<&str>,
    update: TelegramUpdate,
    callback_names: &mut HashMap<String, String>,
) {
    if let Some(message) = update.message {
        let Some(text) = message.text.as_deref() else {
            return;
        };
        let Some(user) = message.from.as_ref() else {
            return;
        };
        if let Some(code) = pairing_code_from_start_for(text, bot_username) {
            if message.chat.chat_type != "private"
                || !controller.telegram_token_is_current(token).await
            {
                return;
            }
            let result = controller
                .attempt_telegram_pairing(token, code, message.chat.id, user.id, true)
                .await;
            match result {
                Ok(true) => {
                    let _ = client.send_message(message.chat.id, "🤖 AUTO-COC est connecté. Vous pouvez utiliser les boutons ou les commandes.", Some(controls_keyboard(false))).await;
                }
                Ok(false) | Err(ControllerError::Expired | ControllerError::PairingRateLimited) => {
                    let _ = client
                        .send_message(
                            message.chat.id,
                            "Code incorrect, expiré ou temporairement bloqué.",
                            None,
                        )
                        .await;
                }
                Err(_) => {}
            }
            return;
        }

        if !controller.authorize_telegram_message(token, &message).await {
            return;
        }
        if is_start_without_code(text, bot_username) {
            let _ = client.send_message(message.chat.id, "🤖 AUTO-COC est connecté. Utilisez les commandes ou les boutons pour contrôler l’application.", Some(controls_keyboard(false))).await;
            return;
        }
        if let Some(command) = text_command(text) {
            if !controller.authorize_telegram_message(token, &message).await {
                return;
            }
            let owner = TelegramOwner {
                chat_id: message.chat.id,
                user_id: user.id,
            };
            execute_command(
                controller,
                client,
                message.chat.id,
                command,
                None,
                None,
                callback_names,
                token,
                owner,
            )
            .await;
        }
        return;
    }

    let Some(callback) = update.callback_query else {
        return;
    };
    let data = callback.data.as_deref().unwrap_or("").trim();
    let Some(command) = callback_command(data) else {
        if controller
            .authorize_telegram_callback(token, &callback)
            .await
        {
            let _ = client
                .answer_callback(&callback.id, "Commande inconnue")
                .await;
        }
        return;
    };
    if !controller
        .authorize_telegram_callback(token, &callback)
        .await
    {
        return;
    }
    let callback_ack = match command {
        "LAUNCH_COC" => "Demande de lancement reçue.",
        "RELOAD_COC" => "Demande de relancement reçue.",
        _ => "",
    };
    let _ = client.answer_callback(&callback.id, callback_ack).await;
    if !controller
        .authorize_telegram_callback(token, &callback)
        .await
    {
        return;
    }
    let Some(message) = callback.message.as_ref() else {
        return;
    };
    let chat_id = message.chat.id;
    let resolved = callback_names
        .get(command)
        .cloned()
        .unwrap_or_else(|| command.to_owned());
    if let Some(confirmation_id) = resolved.strip_prefix("SHUTDOWN_CONFIRM:") {
        match controller
            .confirm_telegram_shutdown(token, &callback, confirmation_id)
            .await
        {
            Ok(_) => {
                respond(
                    client,
                    chat_id,
                    Some(message.message_id),
                    "Extinction demandée.",
                    Some(empty_keyboard()),
                )
                .await;
            }
            Err(_) => {
                respond(
                    client,
                    chat_id,
                    Some(message.message_id),
                    "La confirmation a expiré. Relancez la demande si nécessaire.",
                    Some(menu_keyboard(
                        controller
                            .snapshot()
                            .await
                            .is_ok_and(|snapshot| snapshot.settings.loop_playback),
                    )),
                )
                .await
            }
        }
        return;
    }
    execute_command(
        controller,
        client,
        chat_id,
        &resolved,
        Some(command),
        Some(message.message_id),
        callback_names,
        token,
        TelegramOwner {
            chat_id,
            user_id: callback.from.id,
        },
    )
    .await;
}

async fn execute_command(
    controller: &Controller,
    client: &TelegramClient,
    chat_id: i64,
    command: &str,
    callback_data: Option<&str>,
    message_id: Option<i64>,
    callback_names: &mut HashMap<String, String>,
    token: &str,
    owner: TelegramOwner,
) {
    if !controller
        .authorize_telegram_identity(token, owner.chat_id, owner.user_id, true)
        .await
    {
        return;
    }
    match command {
        "STOP" => {
            match controller.stop_active_operation(token, owner).await {
                Ok(_) => respond(client, chat_id, message_id, "Arrêté.", controls_markup(message_id)).await,
                Err(ControllerError::TelegramUnauthorized) => return,
                Err(ControllerError::RecordingSavePending) => {
                    respond(client, chat_id, message_id, "L’enregistrement n’a pas pu être sauvegardé. Renvoyez `stop` pour réessayer.", controls_markup(message_id)).await;
                }
                Err(ControllerError::RecordingEventLimit) => {
                    respond(client, chat_id, message_id, "La limite d’événements a été atteinte. Les événements capturés sont sauvegardés, mais la macro peut être incomplète.", controls_markup(message_id)).await;
                }
                Err(_) => respond(client, chat_id, message_id, "L’arrêt n’a pas pu être effectué.", controls_markup(message_id)).await,
            }
        }
        "GO" => match controller.start_playback_for_telegram(token, owner).await {
            Ok(_) => respond(client, chat_id, message_id, "Lecture démarrée.", controls_markup(message_id)).await,
            Err(ControllerError::EmptyMacro) => respond(client, chat_id, message_id, "La macro sélectionnée est vide.", controls_markup(message_id)).await,
            Err(ControllerError::CocNotForeground) => respond(client, chat_id, message_id, "Affichez Clash of Clans au premier plan sur le PC avant de démarrer la macro.", controls_markup(message_id)).await,
            Err(ControllerError::TelegramUnauthorized) => return,
            Err(_) => respond(client, chat_id, message_id, "La lecture n’a pas pu démarrer. Vérifiez l’état de l’application et la macro sélectionnée.", controls_markup(message_id)).await,
        },
        "MENU" => {
            let loop_enabled = controller.snapshot().await.map(|snapshot| snapshot.settings.loop_playback).unwrap_or(false);
            respond(client, chat_id, message_id, "Paramètres", Some(menu_keyboard(loop_enabled))).await;
        }
        "BACK" => respond(client, chat_id, message_id, "Commandes :", Some(controls_keyboard(false))).await,
        "CAPTURE" => match controller.capture_coc_screenshot_png().await {
            Ok(png) => {
                if controller
                    .authorize_telegram_identity(token, owner.chat_id, owner.user_id, true)
                    .await
                {
                    let _ = client.send_photo(chat_id, png, "📸 AUTO-COC").await;
                }
            }
            Err(ControllerError::TelegramUnauthorized) => return,
            Err(ControllerError::CocWindowUnavailable) => {
                respond(client, chat_id, message_id, "Fenêtre Clash of Clans introuvable ou ambiguë. Ouvrez CoC puis réessayez.", controls_markup(message_id)).await;
            }
            Err(ControllerError::CocNotForeground) => {
                respond(client, chat_id, message_id, "Windows n’a pas autorisé AUTO-COC à afficher Clash of Clans. Ouvrez CoC au premier plan sur le PC puis réessayez.", controls_markup(message_id)).await;
            }
            Err(_) => respond(client, chat_id, message_id, "La fenêtre CoC n’a pas fourni de capture.", controls_markup(message_id)).await,
        },
        "CAPTURE_GIF" => respond(client, chat_id, message_id, "La capture GIF n’est pas prise en charge.", controls_markup(message_id)).await,
        "TOGGLE_LOOP" => match controller.toggle_loop_playback_for_telegram(token, owner).await {
            Ok((enabled, _)) => {
                respond(client, chat_id, message_id, if enabled { "Loop activée." } else { "Loop désactivée." }, Some(menu_keyboard(enabled))).await;
            }
            Err(ControllerError::TelegramUnauthorized) => return,
            Err(_) => respond(client, chat_id, message_id, "Le réglage loop n’a pas pu être modifié pendant l’opération.", controls_markup(message_id)).await,
        },
        "SHUTDOWN_ASK" => match controller.prepare_telegram_shutdown(token, owner).await {
            Ok(id) => {
                let keyboard = json!({"inline_keyboard":[
                    [{"text":"Annuler","callback_data":"SHUTDOWN_CANCEL"}],
                    [{"text":"Confirmer l’extinction","callback_data":format!("SHUTDOWN_CONFIRM:{id}")}]
                ]});
                respond(client, chat_id, message_id, "⚠️ Confirmer l’extinction du PC ?", Some(keyboard)).await;
            }
            Err(ControllerError::TelegramUnauthorized) => return,
            Err(_) => respond(client, chat_id, message_id, "L’arrêt ne peut pas être préparé pendant une opération active.", controls_markup(message_id)).await,
        },
        "SHUTDOWN_CANCEL" => {
            if let Err(ControllerError::TelegramUnauthorized) =
                controller.cancel_shutdown(token, owner).await
            {
                return;
            }
            let loop_enabled = controller.snapshot().await.is_ok_and(|snapshot| snapshot.settings.loop_playback);
            respond(client, chat_id, message_id, "Extinction annulée.", Some(menu_keyboard(loop_enabled))).await;
        }
        "LAUNCH_COC" => {
            let message_id = respond_with_id(
                client,
                chat_id,
                message_id,
                "⏳ Demande de lancement reçue. Recherche de la fenêtre CoC…",
                Some(empty_keyboard()),
            )
            .await
            .or(message_id);
            match controller.launch_coc_for_telegram(token, owner).await {
                Ok(_) => respond(
                    client,
                    chat_id,
                    message_id,
                    "✅ CoC est ouvert, sa fenêtre a été détectée.",
                    Some(controls_keyboard(false)),
                ).await,
                Err(ControllerError::TelegramUnauthorized) => return,
                Err(ControllerError::CocLaunchUnconfirmed) => respond(
                    client,
                    chat_id,
                    message_id,
                    "Le lancement a été demandé, mais la fenêtre de CoC n’a pas été détectée dans le délai prévu.",
                    Some(controls_keyboard(false)),
                ).await,
                Err(_) => respond(
                    client,
                    chat_id,
                    message_id,
                    "CoC n’a pas pu être lancé. Vérifiez son chemin dans les paramètres.",
                    Some(controls_keyboard(false)),
                ).await,
            }
        }
        "RELOAD_COC" => {
            let message_id = respond_with_id(
                client,
                chat_id,
                message_id,
                "⏳ Demande de relancement reçue. Fermeture puis détection de CoC…",
                Some(empty_keyboard()),
            )
            .await
            .or(message_id);
            match controller.reload_coc_for_telegram(token, owner).await {
                Ok(_) => respond(
                    client,
                    chat_id,
                    message_id,
                    "✅ Opération terminée, fenêtre de Clash of Clans détectée.",
                    Some(controls_keyboard(false)),
                ).await,
                Err(ControllerError::TelegramUnauthorized) => return,
                Err(ControllerError::CocLaunchUnconfirmed) => respond(
                    client,
                    chat_id,
                    message_id,
                    "Le relancement a été demandé, mais la fenêtre de CoC n’est pas réapparue dans le délai prévu.",
                    Some(controls_keyboard(false)),
                ).await,
                Err(_) => respond(
                    client,
                    chat_id,
                    message_id,
                    "Clash of Clans n’a pas pu être relancé. Vérifiez son chemin dans les paramètres.",
                    Some(controls_keyboard(false)),
                ).await,
            }
        }
        "SELECT_MACRO_LIST" => match controller.telegram_macro_names().await {
            Ok(names) => {
                let keyboard = macro_selection_keyboard(&names, callback_names);
                respond(client, chat_id, message_id, "🗂️ Quelle macro choisir ?", Some(keyboard)).await;
            }
            Err(_) => respond(client, chat_id, message_id, "La liste des macros est indisponible.", controls_markup(message_id)).await,
        },
        name if name.starts_with("SELECT_MACRO:") => {
            let name = &name["SELECT_MACRO:".len()..];
            match controller.select_macro_for_telegram(name, token, owner).await {
                Ok(_) => respond(client, chat_id, message_id, &format!("Macro « {name} » sélectionnée."), controls_markup(message_id)).await,
                Err(ControllerError::TelegramUnauthorized) => return,
                Err(_) => respond(client, chat_id, message_id, "Cette macro est introuvable ou ne peut pas être sélectionnée.", controls_markup(message_id)).await,
            }
        }
        "CANCEL_SELECTION" => respond(client, chat_id, message_id, "Sélection annulée.", controls_markup(message_id)).await,
        "DUMMY_COC_STATUS" => respond(client, chat_id, message_id, "CoC a déjà été lancé pendant cette session.", controls_markup(message_id)).await,
        "SHUTDOWN_CONFIRM" => respond(client, chat_id, message_id, "Cette confirmation est invalide ou expirée.", controls_markup(message_id)).await,
        _ => {
            if callback_data.is_some() { respond(client, chat_id, message_id, "Action inconnue.", controls_markup(message_id)).await; }
        }
    }
}

fn controls_markup(message_id: Option<i64>) -> Option<Value> {
    message_id.map(|_| controls_keyboard(false))
}

fn empty_keyboard() -> Value {
    json!({"inline_keyboard": []})
}

async fn respond(
    client: &TelegramClient,
    chat_id: i64,
    message_id: Option<i64>,
    text: &str,
    keyboard: Option<Value>,
) {
    let _ = respond_with_id(client, chat_id, message_id, text, keyboard).await;
}

async fn respond_with_id(
    client: &TelegramClient,
    chat_id: i64,
    message_id: Option<i64>,
    text: &str,
    keyboard: Option<Value>,
) -> Option<i64> {
    if let Some(message_id) = message_id
        && client
            .edit_message_text(chat_id, message_id, text, keyboard.clone())
            .await
            .is_ok()
    {
        return Some(message_id);
    }
    client.send_message(chat_id, text, keyboard).await.ok()
}

fn controls_keyboard(coc_launched: bool) -> Value {
    let coc = if coc_launched {
        ("CoC lancé", "DUMMY_COC_STATUS")
    } else {
        ("Lancer CoC", "LAUNCH_COC")
    };
    json!({"inline_keyboard":[
        [{"text":"Paramètres ⚙️","callback_data":"MENU"},{"text":"Capture 📸","callback_data":"CAPTURE"}],
        [{"text":coc.0,"callback_data":coc.1}],
        [{"text":"Lancer ✅","callback_data":"GO"},{"text":"Stop ❌","callback_data":"STOP"}]
    ]})
}

fn menu_keyboard(loop_enabled: bool) -> Value {
    let loop_text = if loop_enabled {
        "Désactiver loop"
    } else {
        "Activer loop"
    };
    json!({"inline_keyboard":[
        [{"text":"⬅️ Retour","callback_data":"BACK"}],
        [{"text":"📴 Éteindre PC","callback_data":"SHUTDOWN_ASK"}],
        [{"text":"Choisir macro","callback_data":"SELECT_MACRO_LIST"}],
        [{"text":"🔃 Recharger COC","callback_data":"RELOAD_COC"}],
        [{"text":loop_text,"callback_data":"TOGGLE_LOOP"}]
    ]})
}

fn macro_selection_keyboard(
    names: &[String],
    callback_names: &mut HashMap<String, String>,
) -> Value {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    for name in names {
        let callback = format!("SELECT_MACRO:{name}");
        let callback = if callback.len() <= 64 {
            callback
        } else {
            use sha1::{Digest, Sha1};
            let digest = Sha1::digest(callback.as_bytes());
            let suffix = format!("_{:02x}{:02x}{:02x}", digest[0], digest[1], digest[2]);
            let max_prefix = 64 - suffix.len();
            let mut prefix = String::new();
            for character in callback.chars() {
                if prefix.len() + character.len_utf8() > max_prefix {
                    break;
                }
                prefix.push(character);
            }
            let short = format!("{prefix}{suffix}");
            callback_names.insert(short.clone(), callback);
            short
        };
        row.push(json!({"text":name,"callback_data":callback}));
        if row.len() == 2 {
            rows.push(std::mem::take(&mut row));
        }
    }
    if !row.is_empty() {
        rows.push(row);
    }
    rows.push(vec![
        json!({"text":"Annuler ↩️","callback_data":"CANCEL_SELECTION"}),
    ]);
    json!({"inline_keyboard":rows})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unauthorized_responses_are_distinct_from_other_api_failures() {
        assert!(matches!(
            telegram_error_for_http_status(reqwest::StatusCode::UNAUTHORIZED),
            Some(TelegramError::Unauthorized)
        ));
        assert!(matches!(
            telegram_error_for_http_status(reqwest::StatusCode::BAD_GATEWAY),
            Some(TelegramError::Rejected)
        ));
        assert!(telegram_error_for_http_status(reqwest::StatusCode::OK).is_none());
    }

    fn message(chat_id: i64, user_id: i64, chat_type: &str) -> TelegramMessage {
        TelegramMessage {
            message_id: 17,
            chat: TelegramChat {
                id: chat_id,
                chat_type: chat_type.into(),
            },
            from: Some(TelegramUser {
                id: user_id,
                first_name: "Test".into(),
                is_bot: false,
                username: None,
            }),
            text: Some("stop".into()),
        }
    }

    #[test]
    fn update_fixtures_require_both_paired_ids_and_private_chat() {
        let owner = TelegramOwner {
            chat_id: 111,
            user_id: 222,
        };
        let valid: TelegramUpdate = serde_json::from_str(r#"{"update_id":1,"message":{"message_id":17,"chat":{"id":111,"type":"private"},"from":{"id":222,"is_bot":false,"first_name":"Test"},"text":"stop"}}"#).unwrap();
        assert!(message_is_authorized(
            valid.message.as_ref().unwrap(),
            owner
        ));
        assert!(!message_is_authorized(&message(111, 333, "private"), owner));
        assert!(!message_is_authorized(&message(444, 222, "private"), owner));
        assert!(!message_is_authorized(&message(111, 222, "group"), owner));
    }

    #[test]
    fn callback_fixture_checks_paired_user_and_private_chat() {
        let owner = TelegramOwner {
            chat_id: 111,
            user_id: 222,
        };
        let update: TelegramUpdate = serde_json::from_str(r#"{"update_id":2,"callback_query":{"id":"cb-1","from":{"id":222,"is_bot":false,"first_name":"Test"},"message":{"message_id":7,"chat":{"id":111,"type":"private"}},"data":"STOP"}}"#).unwrap();
        let callback = update.callback_query.unwrap();
        assert!(callback_is_authorized(&callback, owner));
        assert_eq!(callback.message.as_ref().unwrap().message_id, 7);
        assert_eq!(
            callback_command(callback.data.as_deref().unwrap()),
            Some("STOP")
        );
        assert_eq!(callback_command("UNKNOWN"), None);
    }

    #[test]
    fn explicit_pairing_and_command_whitelists_reject_ambiguous_input() {
        assert_eq!(pairing_code_from_start("/start 038492"), Some("038492"));
        assert_eq!(
            pairing_code_from_start_for("/start@my_bot 038492", Some("my_bot")),
            Some("038492")
        );
        assert_eq!(
            pairing_code_from_start_for("/start@other_bot 038492", Some("my_bot")),
            None
        );
        assert_eq!(pairing_code_from_start("/start 12345"), None);
        assert_eq!(pairing_code_from_start("/start 123456 extra"), None);
        assert_eq!(text_command(" Arrêter "), Some("STOP"));
        assert_eq!(text_command("untrusted"), None);
        assert_eq!(text_command("relancer"), Some("RELOAD_COC"));
        assert_eq!(callback_command("RELOAD_COC"), Some("RELOAD_COC"));
        assert_eq!(callback_command("VALIDATE_ARRIVAL"), None);
        assert_eq!(
            callback_command("SELECT_MACRO:Macro 2"),
            Some("SELECT_MACRO:Macro 2")
        );
    }

    #[test]
    fn telegram_menu_has_native_reload_and_no_arrival_action() {
        let menu = menu_keyboard(false).to_string();
        assert!(menu.contains("Recharger COC"));
        assert!(menu.contains("RELOAD_COC"));
        assert!(!menu.contains("Valider arrivée"));
        assert!(!menu.contains("VALIDATE_ARRIVAL"));
    }
}
