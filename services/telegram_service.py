# -*- coding: utf-8 -*-
"""
Macro COC v3.0 — Services / Telegram Service

Implémentation asynchrone avec python-telegram-bot v21+.
Remplace entièrement TelegramBridge de v2.1.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import queue
import threading
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    "gif": "CAPTURE_GIF",
    "menu": "MENU", "/menu": "MENU",
    "relancer": "RELOAD_COC", "/relancer": "RELOAD_COC",
    "launch": "LAUNCH_COC", "/launch": "LAUNCH_COC",
}

# Callbacks inline connus
KNOWN_CALLBACKS = frozenset([
    "STOP", "GO", "MENU", "BACK", "SHUTDOWN_ASK", "SHUTDOWN_CONFIRM",
    "SHUTDOWN_CANCEL", "CAPTURE", "LAUNCH_COC", "RELOAD_COC",
    "TOGGLE_LOOP", "DUMMY_COC_STATUS", "SELECT_MACRO_LIST",
    "CANCEL_SELECTION", "VALIDATE_ARRIVAL",
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
        
        # IDs des messages pour gestion UI
        self._last_controls_id: Optional[int] = None
        self._last_menu_id: Optional[int] = None
        
        # Map pour callback_data > 64 octets
        self._callback_data_map: Dict[str, str] = {}
    
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
        return bool(self._token and self._chat_id)
    
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
        was_running = self.is_running
        if was_running:
            self.stop()
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
            return "Token manquant", "#FF6B6B"
        if not self._chat_id:
            return "Chat ID manquant", "#F6C45D"
        if not self.is_running:
            return "Poller arrêté", "#F6C45D"
        return "Connecté", "#63E6A4"
    
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
        if not self._running.is_set():
            return
        
        self._running.clear()
        
        # Attendre la fin du thread (le cleanup se fait dans _run_bot)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        
        self._log.info("TG Service: Arrêté.")
    
    def _run_async_loop(self):
        """Point d'entrée du thread asyncio."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._run_bot())
        except Exception as e:
            self._log.error(f"TG Service: Erreur boucle asyncio: {e}")
        finally:
            if self._loop:
                self._loop.close()
            self._running.clear()
    
    async def _run_bot(self):
        """Configure et lance le bot."""
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
            "🤖 Macro COC v3.0 — Bot connecté!\n"
            "Utilisez les commandes ou les boutons pour contrôler l'application."
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
            await query.answer("Non autorisé.")
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
    
    def send_message(self, text: str):
        """Envoie un message texte."""
        if not self.is_ready:
            return
        
        asyncio.run_coroutine_threadsafe(
            self._async_send_message(text),
            self._loop
        )
    
    async def _async_send_message(self, text: str):
        """Envoie un message (async)."""
        try:
            await self._bot.send_message(chat_id=self._chat_id, text=text)
        except TelegramError as e:
            self._log.error(f"TG: Échec envoi message: {e}")
    
    def send_photo(self, png_bytes: bytes, caption: str = ""):
        """Envoie une photo."""
        if not self.is_ready or not png_bytes:
            return
        
        asyncio.run_coroutine_threadsafe(
            self._async_send_photo(png_bytes, caption),
            self._loop
        )
    
    async def _async_send_photo(self, png_bytes: bytes, caption: str):
        """Envoie une photo (async)."""
        try:
            await self._bot.send_photo(
                chat_id=self._chat_id,
                photo=BytesIO(png_bytes),
                caption=caption
            )
            self._log.info("TG: Photo envoyée.")
        except TelegramError as e:
            self._log.error(f"TG: Échec envoi photo: {e}")
    
    def delete_message(self, message_id: int):
        """Supprime un message."""
        if not self.is_ready or not message_id:
            return
        
        asyncio.run_coroutine_threadsafe(
            self._async_delete_message(message_id),
            self._loop
        )
    
    async def _async_delete_message(self, message_id: int):
        """Supprime un message (async)."""
        try:
            await self._bot.delete_message(chat_id=self._chat_id, message_id=message_id)
        except TelegramError:
            pass  # Ignore les erreurs de suppression
    
    # =========================
    #     Claviers inline
    # =========================
    
    def replace_controls(self, text: str = "Commandes :", coc_launched: bool = False):
        """Remplace le message de contrôles."""
        if not self.is_ready:
            return
        
        asyncio.run_coroutine_threadsafe(
            self._async_replace_controls(text, coc_launched),
            self._loop
        )
    
    async def _async_replace_controls(self, text: str, coc_launched: bool):
        """Remplace le message de contrôles (async)."""
        try:
            # Supprimer l'ancien
            if self._last_controls_id:
                await self._async_delete_message(self._last_controls_id)
            
            # Créer le nouveau clavier
            coc_btn = (
                [InlineKeyboardButton("COC lancé ✅", callback_data="DUMMY_COC_STATUS")]
                if coc_launched
                else [InlineKeyboardButton("Lancer CoC", callback_data="LAUNCH_COC")]
            )
            
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("Paramètres ⚙️", callback_data="MENU"),
                    InlineKeyboardButton("Capture 📸", callback_data="CAPTURE")
                ],
                coc_btn,
                [
                    InlineKeyboardButton("Lancer ✅", callback_data="GO"),
                    InlineKeyboardButton("Stop ❌", callback_data="STOP")
                ],
            ])
            
            msg = await self._bot.send_message(
                chat_id=self._chat_id,
                text=text,
                reply_markup=keyboard
            )
            self._last_controls_id = msg.message_id
            
        except TelegramError as e:
            self._log.error(f"TG: Échec replace_controls: {e}")
    
    def replace_menu(self, title: str = "Paramètres", loop_state: bool = False):
        """Remplace le message de menu."""
        if not self.is_ready:
            return
        
        asyncio.run_coroutine_threadsafe(
            self._async_replace_menu(title, loop_state),
            self._loop
        )
    
    async def _async_replace_menu(self, title: str, loop_state: bool):
        """Remplace le message de menu (async)."""
        try:
            if self._last_menu_id:
                await self._async_delete_message(self._last_menu_id)
            
            loop_text = "Désactiver loop" if loop_state else "Activer loop"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("⬅️ Retour", callback_data="BACK")],
                [InlineKeyboardButton("📴 Éteindre PC", callback_data="SHUTDOWN_ASK")],
                [InlineKeyboardButton("Choisir macro", callback_data="SELECT_MACRO_LIST")],
                [InlineKeyboardButton("🔃 Recharger COC", callback_data="RELOAD_COC")],
                [InlineKeyboardButton("Valider arrivée 👌", callback_data="VALIDATE_ARRIVAL")],
                [InlineKeyboardButton(loop_text, callback_data="TOGGLE_LOOP")],
            ])
            
            msg = await self._bot.send_message(
                chat_id=self._chat_id,
                text=title,
                reply_markup=keyboard
            )
            self._last_menu_id = msg.message_id
            
        except TelegramError as e:
            self._log.error(f"TG: Échec replace_menu: {e}")
    
    def push_macro_selection(self, macro_names: List[str]):
        """Affiche la sélection de macros."""
        if not self.is_ready:
            return
        
        asyncio.run_coroutine_threadsafe(
            self._async_push_macro_selection(macro_names),
            self._loop
        )
    
    async def _async_push_macro_selection(self, macro_names: List[str]):
        """Affiche la sélection de macros (async)."""
        try:
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
            
            buttons.append([InlineKeyboardButton("Annuler ↩️", callback_data="CANCEL_SELECTION")])
            
            await self._bot.send_message(
                chat_id=self._chat_id,
                text="🗂️ Quelle macro lancer ?",
                reply_markup=InlineKeyboardMarkup(buttons)
            )
            
        except TelegramError as e:
            self._log.error(f"TG: Échec push_macro_selection: {e}")
    
    def push_shutdown_confirm(self):
        """Demande de confirmation d'extinction."""
        if not self.is_ready:
            return
        
        asyncio.run_coroutine_threadsafe(
            self._async_push_shutdown_confirm(),
            self._loop
        )
    
    async def _async_push_shutdown_confirm(self):
        """Demande de confirmation d'extinction (async)."""
        try:
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("Annuler", callback_data="SHUTDOWN_CANCEL")],
                [InlineKeyboardButton("✅ Confirmer l'extinction", callback_data="SHUTDOWN_CONFIRM")],
            ])
            
            await self._bot.send_message(
                chat_id=self._chat_id,
                text="⚠️ Confirmer l'extinction du PC ?",
                reply_markup=keyboard
            )
            
        except TelegramError as e:
            self._log.error(f"TG: Échec push_shutdown_confirm: {e}")
    
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
    
    def get_last_menu_id(self) -> Optional[int]:
        """Retourne l'ID du dernier message de menu."""
        return self._last_menu_id
    
    def clear_last_menu_id(self):
        """Réinitialise l'ID du dernier message de menu."""
        self._last_menu_id = None
