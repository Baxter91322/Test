"""CLI orchestrator. Three commands:

    scandi-outreach scrape   — refresh the contacts table from funds.yaml
    scandi-outreach daily    — generate N drafts for new contacts and push to Gmail
    scandi-outreach stats    — print DB summary

Daily flow:
    1. Pick top-N contacts not yet contacted (non-role-based first).
    2. For each: web-search the fund, generate draft via Claude.
    3. If the model returns INSUFFICIENT_SIGNAL, skip and try the next one.
    4. Create Gmail drafts (gmail.compose scope — never auto-sends).
    5. Record outreach in SQLite so we never double-touch.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv

from scandi_outreach.generator import DraftGenerator
from scandi_outreach.gmail_drafts import GmailDrafts
from scandi_outreach.scraper import scrape_all
from scandi_outreach.store import Store

log = logging.getLogger("scandi_outreach")
ROOT = Path(__file__).resolve().parent.parent


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Debug logging.")
def cli(verbose: bool) -> None:
    """Daily Scandinavian VC outreach automation."""
    load_dotenv()
    _setup_logging(verbose)


@cli.command()
@click.option(
    "--funds-file",
    default=str(ROOT / "config" / "funds.yaml"),
    show_default=True,
)
def scrape(funds_file: str) -> None:
    """Scrape fund team pages and upsert contacts into the DB."""
    funds = _load_yaml(Path(funds_file))["funds"]
    store = Store(os.environ.get("DATABASE_PATH", "data/contacts.db"))
    user_agent = os.environ.get(
        "USER_AGENT", "ScandiOutreachBot/0.1 (contact: you@example.com)"
    )

    contacts = asyncio.run(scrape_all(funds, user_agent=user_agent))
    for c in contacts:
        store.upsert_contact(
            email=c.email,
            fund_slug=c.fund_slug,
            name=c.name,
            role=c.role,
            source_url=c.source_url,
            role_based=c.role_based,
        )
    click.echo(f"Scraped {len(contacts)} contacts. {store.stats()}")


@cli.command()
@click.option(
    "--funds-file",
    default=str(ROOT / "config" / "funds.yaml"),
    show_default=True,
)
@click.option(
    "--profile-file",
    default=str(ROOT / "config" / "profile.yaml"),
    show_default=True,
)
@click.option(
    "--prompt-file",
    default=str(ROOT / "config" / "prompt.txt"),
    show_default=True,
)
@click.option(
    "--limit",
    type=int,
    default=lambda: int(os.environ.get("DAILY_DRAFT_LIMIT", "5")),
    help="Max drafts to create today.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Generate drafts but don't push to Gmail or record in DB.",
)
def daily(
    funds_file: str, profile_file: str, prompt_file: str, limit: int, dry_run: bool
) -> None:
    """Generate today's drafts and push them to Gmail."""
    funds_by_slug = {f["slug"]: f for f in _load_yaml(Path(funds_file))["funds"]}
    profile = _load_yaml(Path(profile_file))
    prompt_template = Path(prompt_file).read_text()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise click.UsageError("ANTHROPIC_API_KEY is not set.")

    store = Store(os.environ.get("DATABASE_PATH", "data/contacts.db"))
    candidates = store.candidates_for_outreach(limit=limit * 3)
    if not candidates:
        click.echo("No new contacts to outreach. Run `scrape` first.")
        return

    generator = DraftGenerator(
        api_key=api_key, prompt_template=prompt_template, profile=profile
    )

    gmail: GmailDrafts | None = None
    if not dry_run:
        gmail = GmailDrafts(
            credentials_path=os.environ.get(
                "GMAIL_CREDENTIALS_PATH", "secrets/gmail_credentials.json"
            ),
            token_path=os.environ.get(
                "GMAIL_TOKEN_PATH", "secrets/gmail_token.json"
            ),
        )

    created = 0
    for row in candidates:
        if created >= limit:
            break

        fund = funds_by_slug.get(row["fund_slug"])
        if fund is None:
            log.warning("contact %s has unknown fund %s, skipping", row["email"], row["fund_slug"])
            continue

        contact = {
            "email": row["email"],
            "name": row["name"],
            "role": row["role"],
        }

        log.info("generating draft for %s @ %s", contact["email"], fund["slug"])
        draft = generator.generate(fund=fund, contact=contact)
        if draft is None:
            log.info("  skipped (insufficient signal or API issue)")
            continue

        click.echo(f"\n── {draft.recipient_email} ── confidence={draft.confidence}")
        click.echo(f"   hook: {draft.hook}")
        click.echo(f"   subject: {draft.subject}")
        click.echo(f"   body:\n{draft.body}\n")

        if dry_run:
            created += 1
            continue

        assert gmail is not None
        try:
            draft_id = gmail.create_draft(
                to=draft.recipient_email,
                subject=draft.subject,
                body=draft.body,
            )
        except Exception as e:
            log.error("gmail draft failed for %s: %s", draft.recipient_email, e)
            continue

        store.record_outreach(
            email=draft.recipient_email,
            fund_slug=fund["slug"],
            draft_id=draft_id,
            subject=draft.subject,
            confidence=draft.confidence,
            hook=draft.hook,
        )
        created += 1

    click.echo(f"\nCreated {created} draft(s). {store.stats()}")


@cli.command()
def stats() -> None:
    """Print DB stats."""
    store = Store(os.environ.get("DATABASE_PATH", "data/contacts.db"))
    click.echo(store.stats())


if __name__ == "__main__":
    cli()
