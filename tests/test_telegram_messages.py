"""Async checks for Telegram panel and screenshot message ownership."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from dataclasses import dataclass
import unittest

from telegram.error import TelegramError

from services.telegram_service import TelegramBotService


@dataclass
class _Message:
    message_id: int


class _FakeBot:
    def __init__(self) -> None:
        self.next_id = 100
        self.messages: dict[int, str] = {}
        self.events: list[tuple[str, int]] = []
        self.fail_photo = False
        self.fail_panel = False
        self.fail_delete_ids: set[int] = set()

    async def send_photo(self, *, chat_id: int, photo, caption: str) -> _Message:
        del chat_id, photo, caption
        self.events.append(("send_photo", self.next_id))
        if self.fail_photo:
            raise TelegramError("photo failure")
        message = _Message(self.next_id)
        self.next_id += 1
        self.messages[message.message_id] = "photo"
        return message

    async def send_message(self, *, chat_id: int, text: str, reply_markup) -> _Message:
        del chat_id, text, reply_markup
        self.events.append(("send_panel", self.next_id))
        if self.fail_panel:
            raise TelegramError("panel failure")
        message = _Message(self.next_id)
        self.next_id += 1
        self.messages[message.message_id] = "panel"
        return message

    async def delete_message(self, *, chat_id: int, message_id: int) -> None:
        del chat_id
        self.events.append(("delete", message_id))
        if message_id in self.fail_delete_ids:
            raise TelegramError("delete failure")
        self.messages.pop(message_id, None)


class TelegramMessageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.bot = _FakeBot()
        self.service = TelegramBotService("unit-test-token", 42)
        self.service._bot = self.bot
        self.service._running.set()
        self.service._loop = asyncio.get_running_loop()
        self.service._panel_lock = asyncio.Lock()

    async def test_all_panel_types_share_one_active_message(self) -> None:
        await self.service._async_replace_controls("Ready", coc_launched=False)
        await self.service._async_replace_menu("Settings", loop_state=False)
        await self.service._async_push_macro_selection(["Attack"])
        await self.service._async_push_shutdown_confirm()

        self.assertIsNotNone(self.service._active_panel_id)
        self.assertEqual(list(self.bot.messages.values()), ["panel"])
        self.assertEqual(len(self.bot.messages), 1)

    async def test_panel_replacement_keeps_old_id_when_initial_send_fails(self) -> None:
        self.bot.messages[7] = "panel"
        self.service._active_panel_id = 7
        self.bot.fail_panel = True

        result = await self.service._async_replace_controls("Ready", False)

        self.assertIsNone(result)
        self.assertEqual(self.service._active_panel_id, 7)
        self.assertEqual(self.bot.messages, {7: "panel"})
        self.assertEqual([event[0] for event in self.bot.events], ["send_panel"])

    async def test_panel_replacements_are_serialized_and_old_panel_is_deleted_after_send(self) -> None:
        self.bot.messages[7] = "panel"
        self.service._active_panel_id = 7

        await asyncio.gather(
            self.service._async_replace_controls("First", False),
            self.service._async_replace_menu("Second", False),
        )

        self.assertEqual(len(self.bot.messages), 1)
        self.assertEqual(self.bot.events[0][0], "send_panel")
        self.assertEqual(self.bot.events[1], ("delete", 7))
        self.assertEqual(self.bot.events[2][0], "send_panel")

    async def test_latest_screenshot_sends_new_content_before_deleting_old_ids(self) -> None:
        self.bot.messages.update({7: "old-photo", 8: "old-panel"})
        self.service._last_screenshot_id = 7
        self.service._active_panel_id = 8

        new_photo_id = await self.service._async_send_latest_screenshot(
            b"png",
            "Capture",
            "Ready",
            False,
        )

        self.assertEqual(new_photo_id, 100)
        self.assertEqual(self.service._last_screenshot_id, 100)
        self.assertEqual(self.service._active_panel_id, 101)
        self.assertEqual(
            self.bot.events,
            [("send_photo", 100), ("send_panel", 101), ("delete", 7), ("delete", 8)],
        )
        self.assertEqual(self.bot.messages, {100: "photo", 101: "panel"})

    async def test_public_latest_screenshot_returns_a_concurrent_future(self) -> None:
        future = self.service.send_latest_screenshot(b"png", "Capture", "Ready", False)

        self.assertIsInstance(future, Future)
        self.assertEqual(await asyncio.wrap_future(future), 100)
        self.assertEqual(self.service._last_screenshot_id, 100)
        self.assertEqual(self.service._active_panel_id, 101)

    async def test_latest_screenshot_failure_preserves_old_ids(self) -> None:
        self.bot.messages.update({7: "old-photo", 8: "old-panel"})
        self.service._last_screenshot_id = 7
        self.service._active_panel_id = 8
        self.bot.fail_photo = True

        result = await self.service._async_send_latest_screenshot(
            b"png",
            "Capture",
            "Ready",
            False,
        )

        self.assertIsNone(result)
        self.assertEqual(self.service._last_screenshot_id, 7)
        self.assertEqual(self.service._active_panel_id, 8)
        self.assertEqual(self.bot.messages, {7: "old-photo", 8: "old-panel"})

    async def test_panel_failure_after_photo_cleans_new_photo_and_preserves_old_ids(self) -> None:
        self.bot.messages.update({7: "old-photo", 8: "old-panel"})
        self.service._last_screenshot_id = 7
        self.service._active_panel_id = 8
        self.bot.fail_panel = True

        result = await self.service._async_send_latest_screenshot(
            b"png",
            "Capture",
            "Ready",
            False,
        )

        self.assertIsNone(result)
        self.assertEqual(self.service._last_screenshot_id, 7)
        self.assertEqual(self.service._active_panel_id, 8)
        self.assertEqual(self.bot.messages, {7: "old-photo", 8: "old-panel"})
        self.assertEqual(self.bot.events, [("send_photo", 100), ("send_panel", 101), ("delete", 100)])

    async def test_delete_failure_is_logged_without_blocking_replacement(self) -> None:
        self.bot.messages[7] = "panel"
        self.service._active_panel_id = 7
        self.bot.fail_delete_ids.add(7)

        new_id = await self.service._async_replace_controls("Ready", False)

        self.assertEqual(new_id, 100)
        self.assertEqual(self.service._active_panel_id, 100)
        self.assertIn(7, self.bot.messages)
        self.assertIn(100, self.bot.messages)
        self.assertEqual(self.service._pending_delete_ids, {7})

        self.bot.fail_delete_ids.clear()
        await self.service._async_replace_menu("Settings", False)

        self.assertNotIn(7, self.bot.messages)
        self.assertEqual(self.service._pending_delete_ids, set())

    def test_removed_commands_are_not_parsed(self) -> None:
        self.assertIsNone(self.service._parse_callback_data("CAPTURE_GIF"))
        self.assertIsNone(self.service._parse_callback_data("VALIDATE_ARRIVAL"))
        self.assertEqual(self.service._parse_callback_data("CANCEL_SELECTION"), "CANCEL_SELECTION")
        self.assertEqual(self.service._parse_callback_data("DUMMY_COC_STATUS"), "DUMMY_COC_STATUS")


if __name__ == "__main__":
    unittest.main()
