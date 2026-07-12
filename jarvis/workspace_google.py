"""Google Workspace integration: Gmail and Calendar with draft-first + undo.

Blueprint Section 5: guard all mutating operations with a two-phase validation
architecture. Every mutation creates an isolated draft or staged entity first
and returns a unique undo_operation_id. The user approves via the tool call
confirmation flow before anything is committed; any approved change can be
reversed with google_undo.

Gmail flow:
  gmail_search → gmail_draft_reply → (human review) → gmail_send_draft
  Undo available for: draft deletion; sent emails warn that recall is impossible.

Calendar flow:
  calendar_list → calendar_stage (creates "[DRAFT]" event) → calendar_confirm
  Undo available for: staged events (deleted); confirmed events warn.

Async multi-source sync: gmail_search accepts the full Gmail search syntax
(from:, to:, subject:, is:unread, has:attachment, after:YYYY/MM/DD, etc.),
so the agent can issue natural-language-derived queries via the Gmail Search API.
RFC 5545 RRULE strings are passed through directly to the Calendar API.

Requires:
    pip install google-auth google-auth-oauthlib google-api-python-client

Place the OAuth 2.0 client_secret JSON from Google Cloud Console at
~/.jarvis/google_credentials.json. The first run opens a browser to complete
the OAuth flow and caches the token at ~/.jarvis/google_token.json.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


class GoogleWorkspaceError(RuntimeError):
    """Raised when the Google API libraries are missing or a request fails."""


_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
]


def _build_service(name: str, version: str, creds_path: Path, token_path: Path):
    """Authenticate (OAuth2) and return a Google API resource object."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        raise GoogleWorkspaceError(
            "Google API libraries not installed.\n"
            "Run: pip install 'jarvis-assistant[google]'"
        )

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not creds_path.exists():
                raise GoogleWorkspaceError(
                    f"Google credentials file not found: {creds_path}\n"
                    "Download your OAuth 2.0 client_secret JSON from Google Cloud Console "
                    "and save it at that path."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), _SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())

    return build(name, version, credentials=creds)


class GoogleWorkspaceClient:
    """Unified Gmail + Calendar client with draft-first validation and undo log.

    All mutating operations record an undo_operation_id in self._undo_log.
    Call undo(op_id) to reverse any reversible operation.
    """

    def __init__(self, home: Path) -> None:
        self.home = home
        self._creds_path = home / "google_credentials.json"
        self._token_path = home / "google_token.json"
        self._undo_log: dict[str, dict] = {}
        self._gmail = None
        self._calendar = None

    def _gmail_svc(self):
        if self._gmail is None:
            self._gmail = _build_service("gmail", "v1", self._creds_path, self._token_path)
        return self._gmail

    def _calendar_svc(self):
        if self._calendar is None:
            self._calendar = _build_service("calendar", "v1", self._creds_path, self._token_path)
        return self._calendar

    def _new_op_id(self) -> str:
        return str(uuid.uuid4())[:8]

    # ---- Gmail ---------------------------------------------------------------

    def search_emails(self, query: str, max_results: int = 10) -> list[dict]:
        """Search Gmail using Gmail search syntax; return message summaries."""
        svc = self._gmail_svc()
        resp = svc.users().messages().list(
            userId="me", q=query, maxResults=max_results
        ).execute()
        msgs = resp.get("messages", [])
        results = []
        for m in msgs[:max_results]:
            detail = svc.users().messages().get(
                userId="me", id=m["id"], format="metadata",
                metadataHeaders=["Subject", "From", "To", "Date"],
            ).execute()
            hdrs = {h["name"]: h["value"]
                    for h in detail.get("payload", {}).get("headers", [])}
            results.append({
                "id": m["id"],
                "thread_id": detail.get("threadId", ""),
                "subject": hdrs.get("Subject", "(no subject)"),
                "from": hdrs.get("From", ""),
                "to": hdrs.get("To", ""),
                "date": hdrs.get("Date", ""),
                "snippet": detail.get("snippet", ""),
            })
        return results

    def create_draft_reply(self, message_id: str, body: str) -> dict:
        """Build an isolated draft reply. Nothing is sent until send_draft is called."""
        import base64
        from email.mime.text import MIMEText

        svc = self._gmail_svc()
        original = svc.users().messages().get(
            userId="me", id=message_id, format="metadata",
            metadataHeaders=["Subject", "From", "Message-ID"],
        ).execute()
        hdrs = {h["name"]: h["value"]
                for h in original.get("payload", {}).get("headers", [])}

        msg = MIMEText(body)
        msg["To"] = hdrs.get("From", "")
        msg["Subject"] = "Re: " + hdrs.get("Subject", "")
        msg["In-Reply-To"] = hdrs.get("Message-ID", "")
        msg["References"] = hdrs.get("Message-ID", "")

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        draft = svc.users().drafts().create(
            userId="me",
            body={"message": {"raw": raw, "threadId": original.get("threadId", "")}},
        ).execute()

        op_id = self._new_op_id()
        self._undo_log[op_id] = {
            "kind": "gmail_draft",
            "draft_id": draft["id"],
            "ts": time.time(),
        }
        return {"draft_id": draft["id"], "undo_operation_id": op_id}

    def send_draft(self, draft_id: str) -> dict:
        """Send a draft. This is irreversible — requires explicit user confirmation."""
        svc = self._gmail_svc()
        result = svc.users().drafts().send(
            userId="me", body={"id": draft_id}
        ).execute()
        op_id = self._new_op_id()
        self._undo_log[op_id] = {
            "kind": "gmail_sent",
            "message_id": result["id"],
            "ts": time.time(),
        }
        return {"message_id": result["id"], "undo_operation_id": op_id}

    def get_email_body(self, message_id: str) -> str:
        """Fetch the full plain-text body of a message."""
        import base64
        svc = self._gmail_svc()
        msg = svc.users().messages().get(
            userId="me", id=message_id, format="full"
        ).execute()

        def _extract(payload) -> str:
            mime = payload.get("mimeType", "")
            if mime == "text/plain":
                data = (payload.get("body") or {}).get("data", "")
                return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
            if mime.startswith("multipart/"):
                for part in payload.get("parts", []):
                    text = _extract(part)
                    if text:
                        return text
            return ""

        return _extract(msg.get("payload") or {}) or msg.get("snippet", "")

    # ---- Calendar ------------------------------------------------------------

    def list_events(
        self,
        calendar_id: str = "primary",
        time_min: str = "",
        time_max: str = "",
        max_results: int = 20,
    ) -> list[dict]:
        """List events in a calendar, ordered by start time."""
        import datetime
        svc = self._calendar_svc()
        now = datetime.datetime.utcnow().isoformat() + "Z"
        params: dict[str, Any] = {
            "calendarId": calendar_id,
            "timeMin": time_min or now,
            "maxResults": max_results,
            "singleEvents": True,
            "orderBy": "startTime",
        }
        if time_max:
            params["timeMax"] = time_max

        resp = svc.events().list(**params).execute()
        return [
            {
                "id": e["id"],
                "summary": e.get("summary", "(no title)"),
                "start": (e.get("start") or {}).get("dateTime")
                         or (e.get("start") or {}).get("date", ""),
                "end": (e.get("end") or {}).get("dateTime")
                       or (e.get("end") or {}).get("date", ""),
                "attendees": [a.get("email", "") for a in e.get("attendees", [])],
                "recurrence": e.get("recurrence", []),
                "description": e.get("description", ""),
            }
            for e in resp.get("items", [])
        ]

    def stage_event(
        self,
        summary: str,
        start: str,
        end: str,
        attendees: list[str] | None = None,
        recurrence: str = "",
        calendar_id: str = "primary",
        description: str = "",
    ) -> dict:
        """Create a [DRAFT] event — not visible to attendees until confirmed."""
        svc = self._calendar_svc()
        body: dict[str, Any] = {
            "summary": f"[DRAFT] {summary}",
            "start": {"dateTime": start},
            "end": {"dateTime": end},
            "description": "__jarvis_staged__" + (f"\n{description}" if description else ""),
        }
        if attendees:
            body["attendees"] = [{"email": a} for a in attendees]
        if recurrence:
            body["recurrence"] = [recurrence]  # RFC 5545 RRULE

        event = svc.events().insert(
            calendarId=calendar_id, body=body, sendUpdates="none"
        ).execute()
        op_id = self._new_op_id()
        self._undo_log[op_id] = {
            "kind": "calendar_staged",
            "event_id": event["id"],
            "calendar_id": calendar_id,
            "ts": time.time(),
        }
        return {"event_id": event["id"], "undo_operation_id": op_id}

    def confirm_event(self, event_id: str, calendar_id: str = "primary") -> dict:
        """Promote a staged [DRAFT] event to a real event and notify attendees."""
        svc = self._calendar_svc()
        event = svc.events().get(calendarId=calendar_id, eventId=event_id).execute()
        event["summary"] = event.get("summary", "").removeprefix("[DRAFT] ")
        desc = event.get("description", "") or ""
        event["description"] = desc.replace("__jarvis_staged__", "").strip()

        updated = svc.events().update(
            calendarId=calendar_id,
            eventId=event_id,
            body=event,
            sendUpdates="all",
        ).execute()
        op_id = self._new_op_id()
        self._undo_log[op_id] = {
            "kind": "calendar_confirmed",
            "event_id": event_id,
            "calendar_id": calendar_id,
            "ts": time.time(),
        }
        return {"event_id": updated["id"], "undo_operation_id": op_id}

    # ---- Undo ---------------------------------------------------------------

    def undo(self, op_id: str) -> str:
        """Reverse a mutation by its undo_operation_id."""
        op = self._undo_log.get(op_id)
        if op is None:
            return f"ERROR: unknown undo_operation_id '{op_id}'"
        kind = op["kind"]

        if kind == "gmail_draft":
            self._gmail_svc().users().drafts().delete(
                userId="me", id=op["draft_id"]
            ).execute()
            return f"draft {op['draft_id']} deleted"

        if kind == "gmail_sent":
            return (
                "WARNING: sent emails cannot be recalled via API. "
                "Manually ask the recipient to disregard it."
            )

        if kind == "calendar_staged":
            self._calendar_svc().events().delete(
                calendarId=op["calendar_id"],
                eventId=op["event_id"],
                sendUpdates="none",
            ).execute()
            return f"staged event {op['event_id']} deleted"

        if kind == "calendar_confirmed":
            return (
                "WARNING: confirmed events with attendees cannot be silently deleted. "
                f"Delete event {op['event_id']} from the calendar manually "
                "or cancel it with sendUpdates=all."
            )

        return f"ERROR: no undo handler for operation kind '{kind}'"


# ---- Tool registration ------------------------------------------------------

def register_google_tools(registry, client: GoogleWorkspaceClient) -> None:
    """Register all Google Workspace tools at CONFIRM tier."""
    from .tools import Tier, Tool

    def gmail_search(query: str) -> str:
        try:
            emails = client.search_emails(query)
        except GoogleWorkspaceError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"
        if not emails:
            return "no emails matched that query"
        return "\n---\n".join(
            f"id: {e['id']}\nfrom: {e['from']}\ndate: {e['date']}\n"
            f"subject: {e['subject']}\nsnippet: {e['snippet']}"
            for e in emails
        )

    def gmail_get_body(message_id: str) -> str:
        try:
            return client.get_email_body(message_id)[:8000]
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

    def gmail_draft_reply(message_id: str, body: str) -> str:
        try:
            result = client.create_draft_reply(message_id, body)
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"
        return (
            f"Draft created: {result['draft_id']}\n"
            f"undo_operation_id: {result['undo_operation_id']}\n"
            "Review the draft in Gmail. Call gmail_send_draft to send it."
        )

    def gmail_send_draft(draft_id: str) -> str:
        try:
            result = client.send_draft(draft_id)
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"
        return (
            f"Sent. message_id: {result['message_id']}  "
            f"undo_operation_id: {result['undo_operation_id']} (note: cannot recall)"
        )

    def calendar_list(time_min: str = "", time_max: str = "") -> str:
        try:
            events = client.list_events(time_min=time_min, time_max=time_max)
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"
        if not events:
            return "no events found in that range"
        lines = []
        for e in events:
            attendees = ", ".join(e["attendees"]) or "no attendees"
            lines.append(f"{e['start']} → {e['end']}\n  {e['summary']} [{attendees}]")
        return "\n".join(lines)

    def calendar_stage(
        summary: str, start: str, end: str,
        attendees: str = "", recurrence: str = "", description: str = "",
    ) -> str:
        attendee_list = [a.strip() for a in attendees.split(",") if a.strip()]
        try:
            result = client.stage_event(
                summary, start, end, attendee_list, recurrence, description=description
            )
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"
        return (
            f"Staged [DRAFT] event: {result['event_id']}\n"
            f"undo_operation_id: {result['undo_operation_id']}\n"
            "Attendees are NOT notified yet. Call calendar_confirm to publish."
        )

    def calendar_confirm(event_id: str) -> str:
        try:
            result = client.confirm_event(event_id)
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"
        return (
            f"Event confirmed and attendees notified: {result['event_id']}\n"
            f"undo_operation_id: {result['undo_operation_id']}"
        )

    def google_undo(undo_operation_id: str) -> str:
        try:
            return client.undo(undo_operation_id)
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

    registry.register(Tool(
        "gmail_search",
        "Search Gmail using Gmail query syntax (from:, to:, subject:, is:unread, after:YYYY/MM/DD, etc.).",
        {"query": "Gmail search query"},
        gmail_search, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "gmail_get_body",
        "Fetch the full plain-text body of a Gmail message by its ID.",
        {"message_id": "Gmail message ID (from gmail_search)"},
        gmail_get_body, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "gmail_draft_reply",
        "Create a draft reply to a Gmail message (draft-first: nothing sent until gmail_send_draft).",
        {"message_id": "Gmail message ID", "body": "reply body text"},
        gmail_draft_reply, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "gmail_send_draft",
        "Send a previously created Gmail draft. Irreversible — requires user confirmation.",
        {"draft_id": "draft ID returned by gmail_draft_reply"},
        gmail_send_draft, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "calendar_list",
        "List Google Calendar events. Accepts ISO 8601 timestamps (2025-01-15T09:00:00+05:00).",
        {"time_min": "start of range (ISO 8601, optional)", "time_max": "end of range (ISO 8601, optional)"},
        calendar_list, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "calendar_stage",
        "Stage a [DRAFT] calendar event (not visible to attendees until calendar_confirm is called).",
        {
            "summary": "event title",
            "start": "ISO 8601 start datetime with timezone",
            "end": "ISO 8601 end datetime with timezone",
            "attendees": "comma-separated email addresses (optional)",
            "recurrence": "RFC 5545 RRULE string e.g. RRULE:FREQ=WEEKLY (optional)",
            "description": "event description (optional)",
        },
        calendar_stage, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "calendar_confirm",
        "Promote a staged [DRAFT] calendar event to a real event and notify all attendees.",
        {"event_id": "event ID returned by calendar_stage"},
        calendar_confirm, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "google_undo",
        "Roll back a Google Workspace mutation by its undo_operation_id.",
        {"undo_operation_id": "the undo ID returned by the mutation tool"},
        google_undo, tier=Tier.CONFIRM,
    ))
