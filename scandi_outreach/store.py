"""SQLite store for contacts + outreach history.

The DB is the source of truth for "have we touched this person?" — the daily
loop reads it before generating drafts so the same person is never double-mailed.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS contacts (
    email TEXT PRIMARY KEY,
    fund_slug TEXT NOT NULL,
    name TEXT,
    role TEXT,
    source_url TEXT,
    role_based INTEGER NOT NULL DEFAULT 0,
    discovered_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS outreach (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL,
    fund_slug TEXT NOT NULL,
    draft_id TEXT,
    subject TEXT,
    confidence TEXT,
    hook TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (email) REFERENCES contacts(email)
);

CREATE INDEX IF NOT EXISTS idx_outreach_email ON outreach(email);
CREATE INDEX IF NOT EXISTS idx_outreach_fund ON outreach(fund_slug);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def upsert_contact(
        self,
        email: str,
        fund_slug: str,
        name: str | None,
        role: str | None,
        source_url: str,
        role_based: bool,
    ) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO contacts (email, fund_slug, name, role, source_url, role_based, discovered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(email) DO UPDATE SET
                    name = COALESCE(excluded.name, contacts.name),
                    role = COALESCE(excluded.role, contacts.role),
                    source_url = excluded.source_url
                """,
                (email, fund_slug, name, role, source_url, int(role_based), _now()),
            )

    def already_contacted(self, email: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM outreach WHERE email = ? LIMIT 1", (email,)
            ).fetchone()
            return row is not None

    def candidates_for_outreach(self, limit: int) -> list[sqlite3.Row]:
        """Pick contacts we haven't touched yet, prioritizing non-role-based
        addresses."""
        with self._conn() as c:
            return c.execute(
                """
                SELECT c.* FROM contacts c
                LEFT JOIN outreach o ON o.email = c.email
                WHERE o.id IS NULL
                ORDER BY c.role_based ASC, c.discovered_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    def record_outreach(
        self,
        email: str,
        fund_slug: str,
        draft_id: str | None,
        subject: str | None,
        confidence: str | None,
        hook: str | None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO outreach (email, fund_slug, draft_id, subject, confidence, hook, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (email, fund_slug, draft_id, subject, confidence, hook, _now()),
            )

    def stats(self) -> dict:
        with self._conn() as c:
            contacts = c.execute("SELECT COUNT(*) FROM contacts").fetchone()[0]
            sent = c.execute("SELECT COUNT(*) FROM outreach").fetchone()[0]
            funds = c.execute(
                "SELECT COUNT(DISTINCT fund_slug) FROM contacts"
            ).fetchone()[0]
        return {"contacts": contacts, "outreach": sent, "funds_with_contacts": funds}
