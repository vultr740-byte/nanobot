"""Telegram channel implementation using python-telegram-bot."""

import asyncio
import mimetypes
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger
from telegram import Update, InputFile, Message, User
from telegram.constants import ChatAction
from telegram.ext import Application, MessageHandler, filters, ContextTypes

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import TelegramConfig


def _markdown_to_telegram_html(text: str) -> str:
    """
    Convert markdown to Telegram-safe HTML.
    """
    if not text:
        return ""
    
    # 1. Extract and protect code blocks (preserve content from other processing)
    code_blocks: list[str] = []
    def save_code_block(m: re.Match) -> str:
        code_blocks.append(m.group(1))
        return f"\x00CB{len(code_blocks) - 1}\x00"
    
    text = re.sub(r'```[\w]*\n?([\s\S]*?)```', save_code_block, text)
    
    # 2. Extract and protect inline code
    inline_codes: list[str] = []
    def save_inline_code(m: re.Match) -> str:
        inline_codes.append(m.group(1))
        return f"\x00IC{len(inline_codes) - 1}\x00"
    
    text = re.sub(r'`([^`]+)`', save_inline_code, text)
    
    # 3. Headers # Title -> just the title text
    text = re.sub(r'^#{1,6}\s+(.+)$', r'\1', text, flags=re.MULTILINE)
    
    # 4. Blockquotes > text -> just the text (before HTML escaping)
    text = re.sub(r'^>\s*(.*)$', r'\1', text, flags=re.MULTILINE)
    
    # 5. Escape HTML special characters
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # 6. Links [text](url) - must be before bold/italic to handle nested cases
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    
    # 7. Bold **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
    
    # 8. Italic _text_ (avoid matching inside words like some_var_name)
    text = re.sub(r'(?<![a-zA-Z0-9])_([^_]+)_(?![a-zA-Z0-9])', r'<i>\1</i>', text)
    
    # 9. Strikethrough ~~text~~
    text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)
    
    # 10. Bullet lists - item -> • item
    text = re.sub(r'^[-*]\s+', '• ', text, flags=re.MULTILINE)
    
    # 11. Restore inline code with HTML tags
    for i, code in enumerate(inline_codes):
        # Escape HTML in code content
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00IC{i}\x00", f"<code>{escaped}</code>")
    
    # 12. Restore code blocks with HTML tags
    for i, code in enumerate(code_blocks):
        # Escape HTML in code content
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00CB{i}\x00", f"<pre><code>{escaped}</code></pre>")
    
    return text


@dataclass
class _PendingTextEntry:
    sender_id: str
    chat_id: int
    texts: list[str]
    metadata: dict[str, Any]
    last_message_id: int | None
    last_received_at: float
    timer: asyncio.Task | None = None


@dataclass
class _PendingFeedback:
    chat_id: int
    message_id: int | None
    timer: asyncio.Task | None = None


class TelegramChannel(BaseChannel):
    """
    Telegram channel using long polling.
    
    Simple and reliable - no webhook/public IP needed.
    """
    
    name = "telegram"
    
    def __init__(self, config: TelegramConfig, bus: MessageBus, groq_api_key: str = ""):
        super().__init__(config, bus)
        self.config: TelegramConfig = config
        self.groq_api_key = groq_api_key
        self._app: Application | None = None
        self._chat_ids: dict[str, int] = {}  # Map sender_id to chat_id for replies
        self._pending_lock = asyncio.Lock()
        self._feedback_lock = asyncio.Lock()
        self._debounce_entries: dict[str, _PendingTextEntry] = {}
        self._text_fragment_entries: dict[str, _PendingTextEntry] = {}
        self._pending_feedback: dict[int, _PendingFeedback] = {}
        # Debounce settings (ms) and text fragment stitching (OpenClaw-style defaults).
        self._debounce_ms = int(getattr(config, "debounce_ms", 600))
        self._text_fragment_start_threshold = int(getattr(config, "text_fragment_start_threshold", 4000))
        self._text_fragment_max_gap_ms = int(getattr(config, "text_fragment_max_gap_ms", 1500))
        self._text_fragment_max_id_gap = int(getattr(config, "text_fragment_max_id_gap", 1))
        self._text_fragment_max_parts = int(getattr(config, "text_fragment_max_parts", 12))
        self._text_fragment_max_total_chars = int(getattr(config, "text_fragment_max_total_chars", 50_000))
        # Typing feedback reaction settings.
        self._typing_feedback_delay_s = float(getattr(config, "typing_feedback_delay_s", 6.0))
        self._typing_feedback_emoji = getattr(config, "typing_feedback_emoji", "") or "👀"
    
    async def start(self) -> None:
        """Start the Telegram bot with long polling."""
        if not self.config.token:
            logger.error("Telegram bot token not configured")
            return
        
        self._running = True
        
        # Build the application
        self._app = (
            Application.builder()
            .token(self.config.token)
            .build()
        )
        
        # Add message handler for text, photos, voice, documents
        self._app.add_handler(
            MessageHandler(
                (filters.TEXT | filters.PHOTO | filters.VOICE | filters.AUDIO | filters.Document.ALL) 
                & ~filters.COMMAND, 
                self._on_message
            )
        )
        
        # Add /start command handler
        from telegram.ext import CommandHandler
        self._app.add_handler(CommandHandler("start", self._on_start))
        
        logger.info("Starting Telegram bot (polling mode)...")
        
        # Initialize and start polling
        await self._app.initialize()
        await self._app.start()
        
        # Get bot info
        bot_info = await self._app.bot.get_me()
        logger.info(f"Telegram bot @{bot_info.username} connected")
        
        # Start polling (this runs until stopped)
        await self._app.updater.start_polling(
            allowed_updates=["message"],
            drop_pending_updates=True  # Ignore old messages on startup
        )
        
        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)

    async def _send_media(self, chat_id: int, media_ref: str) -> None:
        """Send a local file or URL as a Telegram media message."""
        if not self._app:
            return

        is_url = self._is_url(media_ref)
        media_type = self._guess_media_type(media_ref)

        if is_url:
            await self._send_remote_media(chat_id, media_ref, media_type)
            return

        path = Path(media_ref)
        if not path.exists():
            logger.warning(f"Media path not found: {media_ref}")
            return

        # Send local file without loading into memory
        if media_type == "photo":
            with path.open("rb") as f:
                await self._app.bot.send_photo(chat_id=chat_id, photo=InputFile(f))
            return
        if media_type == "audio":
            with path.open("rb") as f:
                await self._app.bot.send_audio(chat_id=chat_id, audio=InputFile(f))
            return
        if media_type == "voice":
            with path.open("rb") as f:
                await self._app.bot.send_voice(chat_id=chat_id, voice=InputFile(f))
            return

        with path.open("rb") as f:
            await self._app.bot.send_document(chat_id=chat_id, document=InputFile(f))

    async def _send_remote_media(self, chat_id: int, url: str, media_type: str) -> None:
        """Send a remote URL as Telegram media without downloading."""
        if not self._app:
            return

        if media_type == "photo":
            await self._app.bot.send_photo(chat_id=chat_id, photo=url)
            return
        if media_type == "audio":
            await self._app.bot.send_audio(chat_id=chat_id, audio=url)
            return
        if media_type == "voice":
            await self._app.bot.send_voice(chat_id=chat_id, voice=url)
            return

        await self._app.bot.send_document(chat_id=chat_id, document=url)

    @staticmethod
    def _is_url(value: str) -> bool:
        try:
            parsed = urlparse(value)
            return parsed.scheme in ("http", "https") and bool(parsed.netloc)
        except Exception:
            return False

    @staticmethod
    def _guess_media_type(value: str) -> str:
        """Infer Telegram media type from URL or file path."""
        mime, _ = mimetypes.guess_type(value)
        if mime:
            if mime.startswith("image/"):
                return "photo"
            if mime.startswith("audio/"):
                # Prefer voice for ogg/opus
                if mime in {"audio/ogg", "audio/opus"} or value.lower().endswith(".ogg"):
                    return "voice"
                return "audio"
        lower = value.lower()
        if lower.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
            return "photo"
        if lower.endswith((".ogg", ".opus")):
            return "voice"
        if lower.endswith((".mp3", ".m4a", ".wav", ".flac", ".aac")):
            return "audio"
        return "document"

    @staticmethod
    def _pending_key(chat_id: int, sender_id: str) -> str:
        return f"{chat_id}:{sender_id}"

    @staticmethod
    def _is_command_text(text: str) -> bool:
        return text.lstrip().startswith("/")

    @staticmethod
    def _build_metadata(message: Message | None, user: User | None) -> dict[str, Any]:
        if not message or not user:
            return {}
        return {
            "message_id": message.message_id,
            "user_id": user.id,
            "username": user.username,
            "first_name": user.first_name,
            "is_group": message.chat.type != "private",
        }

    async def _send_typing(self, chat_id: int) -> None:
        if not self._app:
            return
        try:
            await self._app.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception as e:
            logger.debug(f"Failed to send typing action: {e}")

    async def _send_reaction(self, chat_id: int, message_id: int, emoji: str) -> None:
        if not self._app:
            return
        try:
            await self._app.bot.set_message_reaction(
                chat_id=chat_id,
                message_id=message_id,
                reaction=emoji,
            )
        except Exception as e:
            logger.debug(f"Failed to set reaction: {e}")

    async def _schedule_feedback_reaction(self, chat_id: int, message_id: int | None) -> None:
        if not message_id:
            return
        if self._typing_feedback_delay_s <= 0:
            return
        entry = _PendingFeedback(chat_id=chat_id, message_id=message_id)
        async with self._feedback_lock:
            existing = self._pending_feedback.get(chat_id)
            if existing and existing.timer:
                existing.timer.cancel()
            entry.timer = asyncio.create_task(self._feedback_after_delay(chat_id, entry))
            self._pending_feedback[chat_id] = entry

    async def _feedback_after_delay(self, chat_id: int, entry: _PendingFeedback) -> None:
        try:
            await asyncio.sleep(self._typing_feedback_delay_s)
        except asyncio.CancelledError:
            return
        async with self._feedback_lock:
            current = self._pending_feedback.get(chat_id)
            if current is not entry:
                return
            self._pending_feedback.pop(chat_id, None)
        if entry.message_id:
            await self._send_reaction(chat_id, entry.message_id, self._typing_feedback_emoji)

    async def _cancel_feedback(self, chat_id: int) -> None:
        async with self._feedback_lock:
            entry = self._pending_feedback.pop(chat_id, None)
            if entry and entry.timer:
                entry.timer.cancel()

    async def _emit_text_entry(self, entry: _PendingTextEntry, joiner: str) -> None:
        content = joiner.join(entry.texts).strip()
        if not content:
            return
        await self._send_typing(entry.chat_id)
        await self._schedule_feedback_reaction(entry.chat_id, entry.last_message_id)
        await self._handle_message(
            sender_id=entry.sender_id,
            chat_id=str(entry.chat_id),
            content=content,
            media=[],
            metadata=entry.metadata,
        )

    async def _schedule_debounce_flush(self, key: str, entry: _PendingTextEntry) -> None:
        try:
            await asyncio.sleep(self._debounce_ms / 1000)
        except asyncio.CancelledError:
            return
        async with self._pending_lock:
            current = self._debounce_entries.get(key)
            if current is not entry:
                return
            self._debounce_entries.pop(key, None)
        await self._emit_text_entry(entry, "\n")

    async def _schedule_text_fragment_flush(self, key: str, entry: _PendingTextEntry) -> None:
        try:
            await asyncio.sleep(self._text_fragment_max_gap_ms / 1000)
        except asyncio.CancelledError:
            return
        async with self._pending_lock:
            current = self._text_fragment_entries.get(key)
            if current is not entry:
                return
            self._text_fragment_entries.pop(key, None)
        await self._emit_text_entry(entry, "")

    async def _enqueue_debounced_text(self, key: str, entry: _PendingTextEntry, text: str) -> None:
        async with self._pending_lock:
            existing = self._debounce_entries.get(key)
            if existing:
                existing.texts.append(text)
                existing.metadata = entry.metadata
                existing.last_message_id = entry.last_message_id
                existing.last_received_at = entry.last_received_at
                if existing.timer:
                    existing.timer.cancel()
                existing.timer = asyncio.create_task(self._schedule_debounce_flush(key, existing))
                return
            entry.texts = [text]
            entry.timer = asyncio.create_task(self._schedule_debounce_flush(key, entry))
            self._debounce_entries[key] = entry

    async def _enqueue_text_fragment(self, key: str, entry: _PendingTextEntry, text: str) -> None:
        to_flush: _PendingTextEntry | None = None
        async with self._pending_lock:
            existing = self._text_fragment_entries.get(key)
            if existing:
                id_gap = (entry.last_message_id or 0) - (existing.last_message_id or 0)
                time_gap = entry.last_received_at - existing.last_received_at
                can_append = (
                    id_gap > 0
                    and id_gap <= self._text_fragment_max_id_gap
                    and time_gap >= 0
                    and time_gap <= (self._text_fragment_max_gap_ms / 1000)
                )
                total_chars = sum(len(t) for t in existing.texts) + len(text)
                if (
                    can_append
                    and len(existing.texts) + 1 <= self._text_fragment_max_parts
                    and total_chars <= self._text_fragment_max_total_chars
                ):
                    existing.texts.append(text)
                    existing.metadata = entry.metadata
                    existing.last_message_id = entry.last_message_id
                    existing.last_received_at = entry.last_received_at
                    if existing.timer:
                        existing.timer.cancel()
                    existing.timer = asyncio.create_task(
                        self._schedule_text_fragment_flush(key, existing)
                    )
                    return
                # Flush existing if we can't append this fragment
                if existing.timer:
                    existing.timer.cancel()
                self._text_fragment_entries.pop(key, None)
                to_flush = existing

            entry.texts = [text]
            entry.timer = asyncio.create_task(self._schedule_text_fragment_flush(key, entry))
            self._text_fragment_entries[key] = entry

        if to_flush:
            await self._emit_text_entry(to_flush, "")

    async def _flush_pending_for_key(self, key: str) -> None:
        debounce_entry: _PendingTextEntry | None = None
        fragment_entry: _PendingTextEntry | None = None
        async with self._pending_lock:
            debounce_entry = self._debounce_entries.pop(key, None)
            fragment_entry = self._text_fragment_entries.pop(key, None)
            if debounce_entry and debounce_entry.timer:
                debounce_entry.timer.cancel()
            if fragment_entry and fragment_entry.timer:
                fragment_entry.timer.cancel()
        if fragment_entry:
            await self._emit_text_entry(fragment_entry, "")
        if debounce_entry:
            await self._emit_text_entry(debounce_entry, "\n")
    
    async def stop(self) -> None:
        """Stop the Telegram bot."""
        self._running = False

        pending: list[_PendingTextEntry] = []
        async with self._pending_lock:
            pending.extend(self._debounce_entries.values())
            pending.extend(self._text_fragment_entries.values())
            self._debounce_entries.clear()
            self._text_fragment_entries.clear()
        for entry in pending:
            if entry.timer:
                entry.timer.cancel()
        async with self._feedback_lock:
            for entry in self._pending_feedback.values():
                if entry.timer:
                    entry.timer.cancel()
            self._pending_feedback.clear()
        
        if self._app:
            logger.info("Stopping Telegram bot...")
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            self._app = None
    
    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Telegram."""
        if not self._app:
            logger.warning("Telegram bot not running")
            return

        try:
            # chat_id should be the Telegram chat ID (integer)
            chat_id = int(msg.chat_id)
        except ValueError:
            logger.error(f"Invalid chat_id: {msg.chat_id}")
            return

        await self._cancel_feedback(chat_id)

        # 1) Send text content as a separate message (or edit if requested).
        if msg.content:
            edit_message_id = None
            if msg.metadata:
                edit_message_id = msg.metadata.get("edit_message_id")
            try:
                html_content = _markdown_to_telegram_html(msg.content)
                if edit_message_id:
                    try:
                        await self._app.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=int(edit_message_id),
                            text=html_content,
                            parse_mode="HTML"
                        )
                    except Exception as e:
                        logger.warning(f"Error editing Telegram message: {e}")
                        await self._app.bot.send_message(
                            chat_id=chat_id,
                            text=html_content,
                            parse_mode="HTML"
                        )
                else:
                    await self._app.bot.send_message(
                        chat_id=chat_id,
                        text=html_content,
                        parse_mode="HTML"
                    )
            except Exception as e:
                # Fallback to plain text if HTML parsing fails
                logger.warning(f"HTML parse failed, falling back to plain text: {e}")
                try:
                    if edit_message_id:
                        await self._app.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=int(edit_message_id),
                            text=msg.content
                        )
                    else:
                        await self._app.bot.send_message(
                            chat_id=chat_id,
                            text=msg.content
                        )
                except Exception as e2:
                    logger.error(f"Error sending Telegram message: {e2}")

        # 2) Send media attachments as separate messages.
        if not msg.media:
            return

        for item in msg.media:
            try:
                await self._send_media(chat_id, item)
            except Exception as e:
                logger.error(f"Error sending media {item}: {e}")
    
    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not update.message or not update.effective_user:
            return
        
        user = update.effective_user
        await update.message.reply_text(
            f"👋 Hi {user.first_name}!"
        )
    
    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming messages (text, photos, voice, documents)."""
        if not update.message or not update.effective_user:
            return
        
        message = update.message
        user = update.effective_user
        chat_id = message.chat_id
        
        # Use stable numeric ID, but keep username for allowlist compatibility
        sender_id = str(user.id)
        if user.username:
            sender_id = f"{sender_id}|{user.username}"
        
        # Store chat_id for replies
        self._chat_ids[sender_id] = chat_id

        text_content = message.text or message.caption or ""
        has_media = bool(message.photo or message.voice or message.audio or message.document)
        is_command = bool(message.text and self._is_command_text(message.text))
        key = self._pending_key(chat_id, sender_id)
        metadata = self._build_metadata(message, user)

        # Debounce and merge text-only messages (skip commands).
        if text_content and not has_media and not is_command:
            entry = _PendingTextEntry(
                sender_id=sender_id,
                chat_id=chat_id,
                texts=[text_content],
                metadata=metadata,
                last_message_id=message.message_id,
                last_received_at=time.monotonic(),
            )
            if len(text_content) >= self._text_fragment_start_threshold:
                await self._enqueue_text_fragment(key, entry, text_content)
            else:
                await self._enqueue_debounced_text(key, entry, text_content)
            return

        # Flush any pending text before handling media/commands to preserve ordering.
        await self._flush_pending_for_key(key)
        await self._send_typing(chat_id)
        await self._schedule_feedback_reaction(chat_id, message.message_id)
        
        # Build content from text and/or media
        content_parts = []
        media_paths = []
        
        # Text content
        if message.text:
            content_parts.append(message.text)
        if message.caption:
            content_parts.append(message.caption)
        
        # Handle media files
        media_file = None
        media_type = None
        
        if message.photo:
            media_file = message.photo[-1]  # Largest photo
            media_type = "image"
        elif message.voice:
            media_file = message.voice
            media_type = "voice"
        elif message.audio:
            media_file = message.audio
            media_type = "audio"
        elif message.document:
            media_file = message.document
            media_type = "file"
        
        # Download media if present
        if media_file and self._app:
            try:
                file = await self._app.bot.get_file(media_file.file_id)
                ext = self._get_extension(media_type, getattr(media_file, 'mime_type', None))
                
                # Save to workspace/media/
                media_dir = Path.home() / ".nanobot" / "media"
                media_dir.mkdir(parents=True, exist_ok=True)
                
                file_path = media_dir / f"{media_file.file_id[:16]}{ext}"
                await file.download_to_drive(str(file_path))
                
                media_paths.append(str(file_path))
                
                # Handle voice transcription
                if media_type == "voice" or media_type == "audio":
                    from nanobot.providers.transcription import GroqTranscriptionProvider
                    transcriber = GroqTranscriptionProvider(api_key=self.groq_api_key)
                    transcription = await transcriber.transcribe(file_path)
                    if transcription:
                        logger.info(f"Transcribed {media_type}: {transcription[:50]}...")
                        content_parts.append(f"[transcription: {transcription}]")
                    else:
                        content_parts.append(f"[{media_type}: {file_path}]")
                else:
                    content_parts.append(f"[{media_type}: {file_path}]")
                    
                logger.debug(f"Downloaded {media_type} to {file_path}")
            except Exception as e:
                logger.error(f"Failed to download media: {e}")
                content_parts.append(f"[{media_type}: download failed]")
        
        content = "\n".join(content_parts) if content_parts else "[empty message]"
        
        logger.debug(f"Telegram message from {sender_id}: {content[:50]}...")
        
        # Forward to the message bus
        await self._handle_message(
            sender_id=sender_id,
            chat_id=str(chat_id),
            content=content,
            media=media_paths,
            metadata=metadata,
        )
    
    def _get_extension(self, media_type: str, mime_type: str | None) -> str:
        """Get file extension based on media type."""
        if mime_type:
            ext_map = {
                "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
                "audio/ogg": ".ogg", "audio/mpeg": ".mp3", "audio/mp4": ".m4a",
            }
            if mime_type in ext_map:
                return ext_map[mime_type]
        
        type_map = {"image": ".jpg", "voice": ".ogg", "audio": ".mp3", "file": ""}
        return type_map.get(media_type, "")
