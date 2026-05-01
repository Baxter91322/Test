"""Create Gmail drafts via the Gmail API.

OAuth2 flow on first run: we need a credentials.json from Google Cloud
Console (Desktop App credentials, with the Gmail API enabled). After the
first interactive authorization the token.json is cached and refreshed
automatically — so subsequent cron runs are non-interactive.

Scope is gmail.compose only — we can create drafts but cannot send them.
This is deliberate: the user reviews and clicks send.
"""

from __future__ import annotations

import base64
import logging
from email.message import EmailMessage
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/gmail.compose"]


def _load_credentials(credentials_path: Path, token_path: Path) -> Credentials:
    creds: Credentials | None = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json())
        return creds

    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Gmail credentials file not found at {credentials_path}. "
            "Create OAuth client (Desktop App) in Google Cloud Console, "
            "enable the Gmail API, and download the credentials JSON."
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
    creds = flow.run_local_server(port=0)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json())
    return creds


class GmailDrafts:
    def __init__(self, credentials_path: str | Path, token_path: str | Path):
        creds = _load_credentials(Path(credentials_path), Path(token_path))
        self.service = build("gmail", "v1", credentials=creds, cache_discovery=False)

    def create_draft(self, *, to: str, subject: str, body: str) -> str:
        msg = EmailMessage()
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)

        encoded = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")
        draft = (
            self.service.users()
            .drafts()
            .create(userId="me", body={"message": {"raw": encoded}})
            .execute()
        )
        draft_id = draft.get("id", "")
        log.info("created draft %s -> %s", draft_id, to)
        return draft_id
