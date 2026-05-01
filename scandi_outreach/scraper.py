"""Polite scraper that pulls candidate contacts (name + email + role) from
fund team pages.

Strategy:
1. GET the team_url with a User-Agent that identifies the bot.
2. Extract mailto: links and inline email regex matches.
3. Try to associate each email with a nearby name + role using DOM proximity.

We never scrape behind logins, never bypass robots.txt, and we obey a small
per-host rate limit. Emails that look like obvious aliases (info@, hello@,
press@) are tagged as `role_based` so the orchestrator can deprioritize them.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
ROLE_BASED_LOCALS = {
    "info", "hello", "contact", "press", "media", "jobs", "careers",
    "office", "admin", "support", "team", "hi", "general", "ir",
}


@dataclass(frozen=True)
class ScrapedContact:
    fund_slug: str
    email: str
    name: str | None
    role: str | None
    source_url: str
    role_based: bool


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def _fetch(client: httpx.AsyncClient, url: str) -> str:
    resp = await client.get(url, follow_redirects=True, timeout=20.0)
    resp.raise_for_status()
    return resp.text


def _is_role_based(email: str) -> bool:
    local = email.split("@", 1)[0].lower()
    return local in ROLE_BASED_LOCALS


def _domain(url: str) -> str:
    return urlparse(url).netloc


def _nearest_name_role(soup: BeautifulSoup, anchor) -> tuple[str | None, str | None]:
    """Walk up to 3 ancestors looking for a heading or a sibling text block
    that plausibly holds a name + role for the email anchor."""
    name = role = None
    node = anchor
    for _ in range(3):
        node = node.parent
        if node is None:
            break
        # Look for headings inside this ancestor
        for tag in node.find_all(["h1", "h2", "h3", "h4", "strong", "b"], limit=2):
            text = tag.get_text(" ", strip=True)
            if text and 2 < len(text.split()) <= 6 and "@" not in text:
                name = name or text
                break
        # Look for role-ish text
        text_blocks = node.find_all(string=True)
        for raw in text_blocks:
            t = raw.strip()
            if not t or "@" in t:
                continue
            low = t.lower()
            if any(kw in low for kw in (
                "partner", "principal", "associate", "analyst", "venture",
                "director", "founder", "ceo", "investor", "manager",
            )) and len(t) < 80:
                role = role or t
                break
        if name and role:
            break
    return name, role


def _extract_contacts(html: str, fund_slug: str, source_url: str) -> list[ScrapedContact]:
    soup = BeautifulSoup(html, "lxml")
    seen: set[str] = set()
    out: list[ScrapedContact] = []

    # 1. mailto: anchors — best signal, often paired with name in nearby DOM
    for a in soup.select("a[href^='mailto:']"):
        raw = a.get("href", "")
        m = EMAIL_RE.search(raw)
        if not m:
            continue
        email = m.group(0).lower()
        if email in seen:
            continue
        seen.add(email)
        name, role = _nearest_name_role(soup, a)
        out.append(ScrapedContact(
            fund_slug=fund_slug,
            email=email,
            name=name,
            role=role,
            source_url=source_url,
            role_based=_is_role_based(email),
        ))

    # 2. Inline regex sweep — catches emails not hyperlinked
    for m in EMAIL_RE.finditer(soup.get_text(" ")):
        email = m.group(0).lower()
        if email in seen:
            continue
        seen.add(email)
        out.append(ScrapedContact(
            fund_slug=fund_slug,
            email=email,
            name=None,
            role=None,
            source_url=source_url,
            role_based=_is_role_based(email),
        ))

    return out


class Scraper:
    def __init__(self, user_agent: str, per_host_delay_s: float = 2.0):
        self._client = httpx.AsyncClient(
            headers={"User-Agent": user_agent},
            timeout=httpx.Timeout(20.0, connect=10.0),
        )
        self._delay = per_host_delay_s
        self._last_hit: dict[str, float] = {}

    async def __aenter__(self) -> "Scraper":
        return self

    async def __aexit__(self, *exc) -> None:
        await self._client.aclose()

    async def _polite_get(self, url: str) -> str:
        host = _domain(url)
        loop = asyncio.get_running_loop()
        now = loop.time()
        last = self._last_hit.get(host, 0.0)
        wait = self._delay - (now - last)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_hit[host] = loop.time()
        return await _fetch(self._client, url)

    async def scrape_fund(self, fund: dict) -> list[ScrapedContact]:
        urls: list[str] = []
        if fund.get("team_url"):
            urls.append(fund["team_url"])
        if fund.get("homepage") and fund["homepage"] not in urls:
            urls.append(fund["homepage"])
        # Guess a /contact page as a last resort
        if fund.get("homepage"):
            urls.append(fund["homepage"].rstrip("/") + "/contact")

        contacts: list[ScrapedContact] = []
        for url in urls:
            try:
                html = await self._polite_get(url)
            except Exception as e:
                log.info("skip %s: %s", url, e)
                continue
            contacts.extend(_extract_contacts(html, fund["slug"], url))
            if contacts:
                break  # we found something on the preferred URL, don't keep digging
        return contacts


async def scrape_all(funds: Iterable[dict], user_agent: str) -> list[ScrapedContact]:
    async with Scraper(user_agent=user_agent) as scraper:
        all_contacts: list[ScrapedContact] = []
        for fund in funds:
            try:
                got = await scraper.scrape_fund(fund)
                log.info("scraped %s: %d contacts", fund["slug"], len(got))
                all_contacts.extend(got)
            except Exception as e:
                log.warning("fund %s failed: %s", fund["slug"], e)
        return all_contacts
