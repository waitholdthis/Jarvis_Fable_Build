"""Decoupled out-of-band notification hub (Blueprint Section 12).

Routes critical system alerts and human-in-the-loop (HITL) validation requests
as structured markdown payloads over end-to-end-encrypted or authenticated
channels, decoupled from the terminal or web UI. The hub also polls for user
replies so JARVIS can act on text or voice-memo responses while the user is
away from the keyboard.

Supported backends:
  telegram — Telegram Bot API (end-to-end encrypted in secret chats)
  webhook  — HTTP POST to any endpoint (Slack, Discord, custom receiver)

Configure in ~/.jarvis/config.toml:
    [hub]
    backend = "telegram"
    telegram_token = "123456:ABC..."
    telegram_chat_id = "987654321"

    # or for webhooks:
    backend = "webhook"
    webhook_url = "https://hooks.slack.com/services/..."
    webhook_secret = "..."        # sent as X-Jarvis-Secret header
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx


# ---- Message model ----------------------------------------------------------

@dataclass
class HubMessage:
    title: str
    body: str
    kind: str = "info"          # 'info' | 'warn' | 'alert' | 'hitl'
    payload: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


# ---- Abstract base ----------------------------------------------------------

class NotificationHub:
    """Abstract notification hub; subclasses implement send() and optionally _poll()."""

    def __init__(self) -> None:
        self._reply_handlers: list[Callable[[str], None]] = []
        self._poll_thread: threading.Thread | None = None
        self._running = False

    def send(self, msg: HubMessage) -> bool:
        raise NotImplementedError

    def on_reply(self, handler: Callable[[str], None]) -> None:
        """Register a callback fired when the user sends a reply through the hub."""
        self._reply_handlers.append(handler)

    def _dispatch_reply(self, text: str) -> None:
        for h in self._reply_handlers:
            try:
                h(text)
            except Exception:
                pass

    def start_polling(self) -> None:
        if self._poll_thread and self._poll_thread.is_alive():
            return
        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def stop_polling(self) -> None:
        self._running = False

    def _poll_loop(self) -> None:
        pass  # subclasses override


# ---- Telegram ---------------------------------------------------------------

class TelegramHub(NotificationHub):
    """Telegram Bot API: send alerts, receive replies via long-polling.

    1. Create a bot with @BotFather, copy the token.
    2. Start a chat with the bot and get your chat_id (e.g. via @userinfobot).
    3. Set telegram_token and telegram_chat_id in config.toml.
    """

    _API = "https://api.telegram.org/bot"

    _ICONS = {
        "info":  "ℹ️",
        "warn":  "⚠️",
        "alert": "🚨",
        "hitl":  "🤝",
    }

    def __init__(self, token: str, chat_id: str | int) -> None:
        super().__init__()
        self.token = token
        self.chat_id = str(chat_id)
        self._last_update_id = 0

    def _url(self, method: str) -> str:
        return f"{self._API}{self.token}/{method}"

    def _format_markdown(self, msg: HubMessage) -> str:
        icon = self._ICONS.get(msg.kind, "🔔")
        parts = [f"{icon} *{msg.title}*", "", msg.body]
        if msg.payload:
            parts.append("\n```\n" + json.dumps(msg.payload, indent=2) + "\n```")
        return "\n".join(parts)

    def send(self, msg: HubMessage) -> bool:
        text = self._format_markdown(msg)
        try:
            resp = httpx.post(
                self._url("sendMessage"),
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def send_text(self, text: str) -> bool:
        try:
            resp = httpx.post(
                self._url("sendMessage"),
                json={"chat_id": self.chat_id, "text": text},
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def _poll_loop(self) -> None:
        while self._running:
            try:
                resp = httpx.get(
                    self._url("getUpdates"),
                    params={
                        "offset": self._last_update_id + 1,
                        "timeout": 25,
                        "allowed_updates": json.dumps(["message"]),
                    },
                    timeout=30,
                )
                if resp.status_code != 200:
                    time.sleep(5)
                    continue

                updates = resp.json().get("result", [])
                for update in updates:
                    uid = update.get("update_id", 0)
                    self._last_update_id = max(self._last_update_id, uid)
                    text = (
                        (update.get("message") or {}).get("text") or ""
                    ).strip()
                    if text:
                        self._dispatch_reply(text)

            except httpx.TimeoutException:
                pass  # normal for long-poll; loop immediately
            except Exception:
                time.sleep(5)


# ---- Webhook ----------------------------------------------------------------

class WebhookHub(NotificationHub):
    """POST structured JSON payloads to any HTTP endpoint.

    Compatible with Slack incoming webhooks, Discord webhooks, n8n, Make.com,
    or a custom receiver.
    """

    def __init__(self, url: str, secret: str = "") -> None:
        super().__init__()
        self.url = url
        self.secret = secret

    def send(self, msg: HubMessage) -> bool:
        headers = {"Content-Type": "application/json"}
        if self.secret:
            headers["X-Jarvis-Secret"] = self.secret
        try:
            resp = httpx.post(
                self.url,
                json={
                    "title": msg.title,
                    "body": msg.body,
                    "kind": msg.kind,
                    "payload": msg.payload,
                    "ts": msg.ts,
                },
                headers=headers,
                timeout=10,
            )
            return resp.status_code < 300
        except Exception:
            return False

    # Webhooks are fire-and-forget; reply polling is not supported.


# ---- Factory ----------------------------------------------------------------

def hub_from_config(cfg) -> NotificationHub | None:
    """Construct a hub from the [hub] section of config; None if unconfigured."""
    hub_cfg = getattr(cfg, "hub", {}) or {}
    if not hub_cfg:
        return None

    backend = hub_cfg.get("backend", "").lower()

    if backend == "telegram":
        token = hub_cfg.get("telegram_token", "")
        chat_id = hub_cfg.get("telegram_chat_id", "")
        if token and chat_id:
            h = TelegramHub(token, str(chat_id))
            h.start_polling()
            return h

    if backend == "webhook":
        url = hub_cfg.get("webhook_url", "")
        if url:
            return WebhookHub(url, hub_cfg.get("webhook_secret", ""))

    return None


# ---- Tool registration ------------------------------------------------------

def register_hub_tools(registry, hub: NotificationHub) -> None:
    """Add hub_alert and hub_hitl tools."""
    from .tools import Tier, Tool

    def hub_alert(title: str, body: str, kind: str = "info") -> str:
        if kind not in ("info", "warn", "alert", "hitl"):
            kind = "info"
        ok = hub.send(HubMessage(title=title, body=body, kind=kind))
        return "alert delivered via hub" if ok else "ERROR: hub delivery failed"

    def hub_hitl(question: str) -> str:
        """Send a HITL validation request and surface it to the user."""
        msg = HubMessage(
            title="Validation Required",
            body=question,
            kind="hitl",
            payload={"action": "reply with approval/denial"},
        )
        ok = hub.send(msg)
        status = "sent" if ok else "delivery failed"
        return f"HITL request {status} via hub: {question}"

    registry.register(Tool(
        "hub_alert",
        "Send a structured alert to the out-of-band notification hub (Telegram / webhook).",
        {
            "title": "alert title",
            "body": "message body (markdown supported on Telegram)",
            "kind": "info | warn | alert | hitl",
        },
        hub_alert, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "hub_hitl",
        "Send a human-in-the-loop validation request via the hub and wait for user reply.",
        {"question": "the question or action that needs human approval"},
        hub_hitl, tier=Tier.CONFIRM,
    ))
