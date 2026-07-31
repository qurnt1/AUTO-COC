# -*- coding: utf-8 -*-
"""
AUTO-COC Telegram service

Implémentation asynchrone avec python-telegram-bot v21+.
Remplace entièrement TelegramBridge de v2.1.
"""

from __future__ import annotations

import asyncio
import hashlib
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Coroutine, Dict, List, Optional, Tuple

# python-telegram-bot v21+
from telegram import (
    Bot,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.error import TelegramError

from utils.logger import get_logger


# =========================
#     Constants
# =========================

# Commandes texte acceptées (whitelist)
TEXT_COMMAND_MAP = {
    "stop": "STOP", "/stop": "STOP", "arreter": "STOP", "arrêter": "STOP", "pause": "STOP",
    "go": "GO", "/go": "GO", "start": "GO", "lancer": "GO", "reprendre": "GO",
    "shutdown": "SHUTDOWN_ASK", "/shutdown": "SHUTDOWN_ASK", "eteindre": "SHUTDOWN_ASK",
    "éteindre": "SHUTDOWN_ASK", "poweroff": "SHUTDOWN_ASK",
    "capture": "CAPTURE", "/capture": "CAPTURE", "screenshot": "CAPTURE", "screen": "CAPTURE",
    "menu": "MENU", "/menu": "MENU",
    "relancer": "RELOAD_COC", "/relancer": "RELOAD_COC",
    "launch": "LAUNCH_COC", "/launch": "LAUNCH_COC",
}

# Callbacks inline connus
KNOWN_CALLBACKS = frozenset([
    "STOP", "GO", "MENU", "BACK", "SHUTDOWN_ASK", "SHUTDOWN_CONFIRM",
    "SHUTDOWN_CANCEL", "CAPTURE", "LAUNCH_COC", "RELOAD_COC",
    "TOGGLE_LOOP", "DUMMY_COC_STATUS", "SELECT_MACRO_LIST",
    "CANCEL_SELECTION",
])


@dataclass
class TelegramCommand:
    """Commande reçue de Telegram, à traiter par l'application."""
    command: str
    meta: Dict[str, Any] = field(default_factory=dict)


class TelegramBotService:
    """
    Service Telegram asynchrone utilisant python-telegram-bot.
    
    Communication avec l’interface via queue.Queue.
    """
    
    def __init__(self, token: str, chat_id: Optional[int] = None):
        """
        Initialise le service Telegram.
        
        Args:
            token: Token du bot Telegram
            chat_id: ID du chat autorisé (optionnel, sera auto-assigné)
        """
        self._token = (token or "").strip()
        self._chat_id = chat_id
        self._log = get_logger()
        
        # Application python-telegram-bot
        self._app: Optional[Application] = None
        self._bot: Optional[Bot] = None
        
        # Thread et boucle asyncio dédiés
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = threading.Event()
        
        # Queue pour communiquer avec la GUI (thread-safe)
        self._command_queue: queue.Queue[TelegramCommand] = queue.Queue()
        
        # Un seul panneau Telegram actif à la fois.
        self._active_panel_id: Optional[int] = None
        self._last_screenshot_id: Optional[int] = None
        self._pending_delete_ids: set[int] = set()
        self._panel_lock: Optional[asyncio.Lock] = None
        
        # Map pour callback_data > 64 octets
        self._callback_data_map: Dict[str, str] = {}

    def _reset_message_state(self) -> None:
        """Réinitialise les références locales aux messages Telegram."""
        self._active_panel_id = None
        self._last_screenshot_id = None
        self._pending_delete_ids.clear()
        self._callback_data_map.clear()
    
    # =========================
    #     Properties
    # =========================
    
    @property
    def is_configured(self) -> bool:
        """Vérifie si le token est configuré."""
        return bool(self._token)
    
    @property
    def is_ready(self) -> bool:
        """Vérifie si le service est prêt (token + chat_id)."""
        loop = self._loop
        return bool(
            self._token
            and self._chat_id
            and self._running.is_set()
            and self._bot is not None
            and loop is not None
            and loop.is_running()
            and not loop.is_closed()
        )
    
    @property
    def is_running(self) -> bool:
        """Vérifie si le service est en cours d'exécution."""
        return self._running.is_set()
    
    @property
    def command_queue(self) -> queue.Queue[TelegramCommand]:
        """Queue des commandes reçues."""
        return self._command_queue

    def configure(self, token: str, chat_id: Optional[int] = None) -> None:
        """Met à jour les identifiants sans exposer l’état interne à l’interface."""
        if self.is_running or (self._thread and self._thread.is_alive()):
            self.stop()
        if self._thread and self._thread.is_alive():
            self._log.error("TG Service: ancienne boucle encore active, reconfiguration annulée.")
            return
        self._reset_message_state()
        self._token = (token or "").strip()
        self._chat_id = chat_id
        if self._token:
            self.start()
    
    def get_status(self) -> Tuple[str, str]:
        """
        Retourne (status_text, color_hex) pour l'UI.
        
        Returns:
            Tuple (texte de statut, code couleur hex)
        """
        if not self.is_configured:
            return "Token missing", "#FF6B6B"
        if not self._chat_id:
            return "Chat ID missing", "#F6C45D"
        if not self.is_ready:
            return "Poller stopped", "#F6C45D"
        return "Connected", "#63E6A4"
    
    # =========================
    #     Start / Stop
    # =========================
    
    def start(self):
        """Démarre le service dans un thread dédié."""
        if self._running.is_set():
            self._log.warning("TG Service: Déjà en cours d'exécution.")
            return
        
        if not self.is_configured:
            self._log.warning("TG Service: Token non configuré, démarrage annulé.")
            return

        if self._thread and self._thread.is_alive():
            self._log.error("TG Service: ancienne boucle encore active, démarrage annulé.")
            return
        
        self._reset_message_state()
        self._loop = None
        self._bot = None
        self._app = None
        self._running.set()
        self._thread = threading.Thread(
            target=self._run_async_loop,
            daemon=True,
            name="TG-AsyncLoop"
        )
        self._thread.start()
        self._log.info("TG Service: Démarré.")
    
    def stop(self):
        """Arrête le service proprement."""
        thread = self._thread
        if not self._running.is_set() and not (thread and thread.is_alive()):
            self._reset_message_state()
            return
        
        self._running.clear()
        
        # Attendre la fin du thread (le cleanup se fait dans _run_bot)
        if thread and thread.is_alive():
            thread.join(timeout=5.0)
        if thread and thread.is_alive():
            self._log.error("TG Service: arrêt expiré, boucle encore active.")
            return
        
        self._thread = None
        self._reset_message_state()
        self._log.info("TG Service: Arrêté.")
    
    def _run_async_loop(self):
        """Point d'entrée du thread asyncio."""
        loop = asyncio.new_event_loop()
        try:
            self._loop = loop
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._run_bot())
        except Exception as e:
            self._log.error(f"TG Service: Erreur boucle asyncio: {e}")
        finally:
            if not loop.is_closed():
                loop.close()
            if self._loop is loop:
                self._loop = None
            self._panel_lock = None
            self._app = None
            self._bot = None
            self._running.clear()
            self._reset_message_state()
    
    async def _run_bot(self):
        """Configure et lance le bot."""
        self._panel_lock = asyncio.Lock()
        # Construire l'application
        self._app = (
            Application.builder()
            .token(self._token)
            .build()
        )
        self._bot = self._app.bot
        
        # Ajouter les handlers
        self._app.add_handler(CommandHandler("start", self._on_start_command))
        self._app.add_handler(CallbackQueryHandler(self._on_callback_query))
        self._app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._on_text_message
        ))
        
        # Démarrer le polling
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        
        self._log.info("TG Service: Polling démarré.")
        
        # Boucle d'attente
        while self._running.is_set():
            await asyncio.sleep(0.5)
        
        # Cleanup propre (dans l'ordre correct)
        self._log.info("TG Service: Arrêt du polling...")
        try:
            if self._app.updater.running:
                await self._app.updater.stop()
        except Exception as e:
            self._log.warning(f"Erreur arrêt updater: {e}")
        
        try:
            if self._app.running:
                await self._app.stop()
        except Exception as e:
            self._log.warning(f"Erreur arrêt app: {e}")
        
        try:
            await self._app.shutdown()
        except Exception as e:
            self._log.warning(f"Erreur shutdown app: {e}")
    
    # =========================
    #     Handlers
    # =========================
    
    async def _on_start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler pour /start."""
        if not update.effective_chat:
            return
        
        chat_id = update.effective_chat.id
        
        # Auto-assignation du chat_id
        if not self._chat_id:
            self._chat_id = chat_id
            self._log.info(f"TG: Auto-assignation chat_id: {chat_id}")
        
        if chat_id != self._chat_id:
            self._log.warning(f"TG: Message ignoré (chat_id invalide: {chat_id})")
            return
        
        await update.message.reply_text(
            "AUTO-COC is connected.\n"
            "Use the buttons to control the application."
        )
    
    async def _on_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler pour les boutons inline."""
        query = update.callback_query
        if not query or not query.from_user:
            return
        
        from_id = query.from_user.id
        data = (query.data or "").strip()
        msg_id = query.message.message_id if query.message else None
        
        # Validation utilisateur
        if not self._chat_id:
            self._chat_id = from_id
            self._log.info(f"TG: Auto-assignation chat_id: {from_id}")
        elif from_id != self._chat_id:
            self._log.warning(f"TG: Callback ignoré (chat_id invalide: {from_id})")
            await query.answer("Not authorized.")
            return
        
        # Répondre immédiatement au callback
        await query.answer()
        
        # Parser la commande
        cmd = self._parse_callback_data(data)
        if not cmd:
            self._log.warning(f"TG: Commande callback inconnue: {data}")
            return
        
        # Envoyer à la queue
        self._command_queue.put(TelegramCommand(
            command=cmd,
            meta={"from": from_id, "message_id": msg_id}
        ))
        
        self._log.info(f"TG: Commande reçue: {cmd}")
    
    async def _on_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler pour les messages texte."""
        if not update.message or not update.effective_chat:
            return
        
        from_id = update.effective_chat.id
        text = (update.message.text or "").strip().lower()
        
        # Validation utilisateur
        if self._chat_id and from_id != self._chat_id:
            return
        if not self._chat_id:
            self._chat_id = from_id
            self._log.info(f"TG: Auto-assignation chat_id: {from_id}")
        
        # Parser la commande
        cmd = TEXT_COMMAND_MAP.get(text)
        if cmd:
            self._command_queue.put(TelegramCommand(
                command=cmd,
                meta={"from": from_id, "message_id": update.message.message_id}
            ))
            self._log.info(f"TG: Commande texte reçue: {cmd}")
    
    def _parse_callback_data(self, data: str) -> Optional[str]:
        """Parse un callback_data."""
        # Vérifier le mapping pour les données > 64 octets
        if data in self._callback_data_map:
            data = self._callback_data_map[data]
        
        # Commandes de sélection de macro
        if data.startswith("SELECT_MACRO:"):
            return data
        
        # Commandes connues
        if data in KNOWN_CALLBACKS:
            return data
        
        return None
    
    # =========================
    #     API (thread-safe)
    # =========================

    def _submit(
        self,
        coroutine: Coroutine[Any, Any, Any],
        operation: str,
    ) -> Optional[Future[Any]]:
        """Planifie une coroutine sur la boucle Telegram et observe ses erreurs."""
        if not self.is_ready or self._loop is None:
            coroutine.close()
            return None
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        except (RuntimeError, TypeError) as error:
            coroutine.close()
            self._log.error(f"TG: impossible de planifier {operation}: {error}")
            return None
        future.add_done_callback(
            lambda completed: self._observe_future(completed, operation)
        )
        return future

    def _observe_future(self, future: Future[Any], operation: str) -> None:
        try:
            future.result()
        except Exception as error:
            self._log.error(f"TG: échec asynchrone {operation}: {error}")

    def send_message(self, text: str) -> Optional[Future[Any]]:
        """Envoie un message texte."""
        return self._submit(self._async_send_message(text), "send_message")
    
    async def _async_send_message(self, text: str):
        """Envoie un message (async)."""
        try:
            await self._bot.send_message(chat_id=self._chat_id, text=text)
        except TelegramError as e:
            self._log.error(f"TG: Échec envoi message: {e}")
    
    async def _async_send_photo(self, png_bytes: bytes, caption: str):
        """Envoie une photo (async)."""
        try:
            message = await self._bot.send_photo(
                chat_id=self._chat_id,
                photo=BytesIO(png_bytes),
                caption=caption
            )
            return message
        except TelegramError as e:
            self._log.error(f"TG: Échec envoi photo: {e}")
    
    def send_latest_screenshot(
        self,
        png_bytes: bytes,
        caption: str,
        controls_text: str,
        coc_launched: bool,
    ) -> Optional[Future[Any]]:
        """Envoie la capture et remplace le panneau Telegram actif."""
        if not png_bytes:
            return None
        return self._submit(
            self._async_send_latest_screenshot(
                png_bytes,
                caption,
                controls_text,
                coc_launched,
            ),
            "send_latest_screenshot",
        )

    async def _async_send_latest_screenshot(
        self,
        png_bytes: bytes,
        caption: str,
        controls_text: str,
        coc_launched: bool,
    ) -> Optional[int]:
        if self._panel_lock is None:
            self._log.error("TG: verrou absent pour send_latest_screenshot.")
            return None

        async with self._panel_lock:
            previous_screenshot_id = self._last_screenshot_id
            previous_panel_id = self._active_panel_id
            photo = await self._async_send_photo(png_bytes, caption)
            if photo is None:
                return None

            photo_id = photo.message_id
            try:
                panel_id = await self._async_send_panel(
                    controls_text,
                    self._controls_keyboard(coc_launched),
                )
            except TelegramError as error:
                self._log.error(f"TG: échec remplacement après capture: {error}")
                await self._cleanup_message_ids({photo_id})
                return None

            self._last_screenshot_id = photo_id
            self._active_panel_id = panel_id
            await self._cleanup_message_ids({previous_screenshot_id, previous_panel_id})
            return photo_id

    def delete_message(self, message_id: int) -> Optional[Future[Any]]:
        """Supprime un message."""
        if not message_id:
            return None
        return self._submit(self._async_delete_message(message_id), "delete_message")
    
    async def _async_delete_message(self, message_id: int) -> bool:
        """Supprime un message (async)."""
        try:
            await self._bot.delete_message(chat_id=self._chat_id, message_id=message_id)
        except TelegramError as error:
            self._log.error(f"TG: échec suppression message {message_id}: {error}")
            return False
        return True

    async def _cleanup_message_ids(self, message_ids: set[int | None]) -> None:
        """Delete stale bot messages and retain transient failures for retry."""
        protected_ids = {self._active_panel_id, self._last_screenshot_id, None}
        candidates = self._pending_delete_ids.union(message_ids).difference(protected_ids)
        for message_id in sorted(candidates):
            if await self._async_delete_message(message_id):
                self._pending_delete_ids.discard(message_id)
            else:
                self._pending_delete_ids.add(message_id)
    
    # =========================
    #     Claviers inline
    # =========================
    
    def _controls_keyboard(self, coc_launched: bool) -> InlineKeyboardMarkup:
        coc_btn = (
            [InlineKeyboardButton("CoC running ✅", callback_data="DUMMY_COC_STATUS")]
            if coc_launched
            else [InlineKeyboardButton("Launch CoC", callback_data="LAUNCH_COC")]
        )
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("Settings ⚙️", callback_data="MENU"),
                InlineKeyboardButton("Screenshot 📸", callback_data="CAPTURE"),
            ],
            coc_btn,
            [
                InlineKeyboardButton("Run ✅", callback_data="GO"),
                InlineKeyboardButton("Stop ❌", callback_data="STOP"),
            ],
        ])

    async def _async_send_panel(
        self,
        text: str,
        keyboard: InlineKeyboardMarkup,
    ) -> Optional[int]:
        """Envoie le panneau avant de supprimer l'ancien."""
        message = await self._bot.send_message(
            chat_id=self._chat_id,
            text=text,
            reply_markup=keyboard,
        )
        return message.message_id

    async def _async_replace_panel(
        self,
        text: str,
        keyboard: InlineKeyboardMarkup,
        operation: str,
    ) -> Optional[int]:
        """Remplace le panneau actif sous un verrou de boucle unique."""
        if self._panel_lock is None:
            self._log.error(f"TG: verrou absent pour {operation}.")
            return None
        async with self._panel_lock:
            previous_id = self._active_panel_id
            try:
                new_id = await self._async_send_panel(text, keyboard)
            except TelegramError as error:
                self._log.error(f"TG: échec {operation}: {error}")
                return None
            self._active_panel_id = new_id
            await self._cleanup_message_ids({previous_id})
            return new_id

    def replace_controls(self, text: str = "Commandes :", coc_launched: bool = False) -> Optional[Future[Any]]:
        """Remplace le message de contrôles."""
        return self._submit(
            self._async_replace_controls(text, coc_launched),
            "replace_controls",
        )
    
    async def _async_replace_controls(self, text: str, coc_launched: bool):
        """Remplace le message de contrôles (async)."""
        return await self._async_replace_panel(
            text,
            self._controls_keyboard(coc_launched),
            "replace_controls",
        )
    
    def replace_menu(self, title: str = "Settings", loop_state: bool = False) -> Optional[Future[Any]]:
        """Remplace le message de menu."""
        return self._submit(self._async_replace_menu(title, loop_state), "replace_menu")
    
    async def _async_replace_menu(self, title: str, loop_state: bool):
        """Remplace le message de menu (async)."""
        loop_text = "Disable loop" if loop_state else "Enable loop"
            
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("⬅️ Back", callback_data="BACK")],
            [InlineKeyboardButton("📴 Shut down PC", callback_data="SHUTDOWN_ASK")],
            [InlineKeyboardButton("Choose macro", callback_data="SELECT_MACRO_LIST")],
            [InlineKeyboardButton("🔃 Reload CoC", callback_data="RELOAD_COC")],
            [InlineKeyboardButton(loop_text, callback_data="TOGGLE_LOOP")],
        ])
        return await self._async_replace_panel(title, keyboard, "replace_menu")

    def push_macro_selection(self, macro_names: List[str]) -> Optional[Future[Any]]:
        """Affiche la sélection de macros."""
        return self._submit(
            self._async_push_macro_selection(macro_names),
            "push_macro_selection",
        )
    
    async def _async_push_macro_selection(self, macro_names: List[str]):
        """Affiche la sélection de macros (async)."""
        buttons = []
        row = []
        for name in macro_names:
            callback_data = self._hash_callback_data(f"SELECT_MACRO:{name}")
            row.append(InlineKeyboardButton(name, callback_data=callback_data))
            if len(row) >= 2:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)
        buttons.append([InlineKeyboardButton("Cancel ↩️", callback_data="CANCEL_SELECTION")])
        return await self._async_replace_panel(
            "🗂️ Which macro should run?",
            InlineKeyboardMarkup(buttons),
            "push_macro_selection",
        )
    
    def push_shutdown_confirm(self) -> Optional[Future[Any]]:
        """Demande de confirmation d'extinction."""
        return self._submit(self._async_push_shutdown_confirm(), "push_shutdown_confirm")
    
    async def _async_push_shutdown_confirm(self):
        """Demande de confirmation d'extinction (async)."""
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("Cancel", callback_data="SHUTDOWN_CANCEL")],
            [InlineKeyboardButton("✅ Confirm shutdown", callback_data="SHUTDOWN_CONFIRM")],
        ])
        return await self._async_replace_panel(
            "⚠️ Confirm shutdown?",
            keyboard,
            "push_shutdown_confirm",
        )
    
    def _hash_callback_data(self, data: str) -> str:
        """Tronque et hashe le callback_data s'il dépasse 64 octets."""
        data_bytes = data.encode('utf-8')
        if len(data_bytes) <= 64:
            return data
        
        # Tronquer en gardant de la marge pour le hash
        prefix = data[:58]
        while len(prefix.encode('utf-8')) > 58:
            prefix = prefix[:-1]
        
        hash_suffix = hashlib.sha1(data_bytes).hexdigest()[:5]
        hashed_data = f"{prefix}_{hash_suffix}"
        
        # Stocker le mapping
        self._callback_data_map[hashed_data] = data
        self._log.warning(f"TG: Callback data > 64 octets, tronqué: '{data}' -> '{hashed_data}'")
        
        return hashed_data
    
    def set_chat_id(self, chat_id: int):
        """Configure le chat_id."""
        self._chat_id = chat_id
