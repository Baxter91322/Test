"""Claude-powered draft generator.

Uses Claude Opus 4.7 with:
- Adaptive thinking (model decides how much to reason).
- Server-side web search to find recent signals about the fund.
- Structured JSON output (json_schema) so we never have to regex out
  subject/body from prose.
- Prompt caching on the system prompt + user profile (stable across the
  whole daily batch — should give ~90% read rate after the first call).

Returns a `DraftResult` with subject + body, or `None` if the model decided
the available signal was too weak to write a non-generic email (better to skip
than send slop).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import anthropic

log = logging.getLogger(__name__)

MODEL = "claude-opus-4-7"

EMAIL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subject": {"type": ["string", "null"]},
        "body": {"type": ["string", "null"]},
        "personalization_hook": {"type": ["string", "null"]},
        "reason": {"type": ["string", "null"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["subject", "body", "personalization_hook", "reason", "confidence"],
    "additionalProperties": False,
}


@dataclass
class DraftResult:
    recipient_email: str
    subject: str
    body: str
    hook: str
    confidence: str


def _profile_block(profile: dict) -> str:
    return (
        "# User profile (the sender)\n"
        f"Name: {profile.get('name')}\n"
        f"Current role: {profile.get('current_role')}\n"
        f"Location: {profile.get('location')}\n\n"
        f"## Background\n{profile.get('background', '').strip()}\n\n"
        f"## What they want\n{profile.get('what_you_want', '').strip()}\n\n"
        f"## Thesis / focus\n{profile.get('thesis_or_focus', '').strip()}\n\n"
        f"## Tone\n"
        f"- language: {profile.get('tone', {}).get('language', 'english')}\n"
        f"- register: {profile.get('tone', {}).get('register', '')}\n"
        f"- length: {profile.get('tone', {}).get('length', '')}\n\n"
        f"## Signature\n{profile.get('signature', '').strip()}\n"
    )


def _user_turn(fund: dict, contact: dict) -> str:
    """Per-recipient context — small and varies, so it sits AFTER the cached
    system prefix."""
    return (
        f"Fund: {fund['name']} ({fund.get('country')}, {fund.get('kind')})\n"
        f"Homepage: {fund.get('homepage')}\n"
        f"Stated thesis: {fund.get('thesis')}\n"
        f"Stages: {', '.join(fund.get('stages', []))}\n"
        f"Sectors: {', '.join(fund.get('sectors', []))}\n\n"
        f"Recipient:\n"
        f"  email: {contact['email']}\n"
        f"  name: {contact.get('name') or '(unknown — use a polite generic opener)'}\n"
        f"  role: {contact.get('role') or '(unknown)'}\n\n"
        f"Use the web_search tool to find ONE specific recent signal about "
        f"{fund['name']} (a recent investment, a partner's recent post or talk, "
        f"a fund announcement, a portfolio company in the news). Use that as "
        f"the hook. If nothing concrete shows up after 1-2 searches, return "
        f"INSUFFICIENT_SIGNAL.\n\n"
        f"Return JSON matching the required schema."
    )


class DraftGenerator:
    def __init__(self, api_key: str, prompt_template: str, profile: dict):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.prompt_template = prompt_template
        self.profile_text = _profile_block(profile)

    def _system_blocks(self) -> list[dict]:
        # Two stable text blocks. Cache_control on the LAST block caches the
        # whole prefix (tools + both system blocks) for ~5 minutes.
        return [
            {"type": "text", "text": self.prompt_template},
            {
                "type": "text",
                "text": self.profile_text,
                "cache_control": {"type": "ephemeral"},
            },
        ]

    def generate(self, fund: dict, contact: dict) -> DraftResult | None:
        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=4096,
                thinking={"type": "adaptive"},
                output_config={
                    "effort": "high",
                    "format": {"type": "json_schema", "schema": EMAIL_SCHEMA},
                },
                system=self._system_blocks(),
                tools=[
                    {
                        "type": "web_search_20260209",
                        "name": "web_search",
                        "max_uses": 3,
                    }
                ],
                messages=[{"role": "user", "content": _user_turn(fund, contact)}],
            )
        except anthropic.APIError as e:
            log.warning("Claude API error for %s: %s", contact["email"], e)
            return None

        log.debug(
            "usage cache_read=%s cache_create=%s in=%s out=%s",
            response.usage.cache_read_input_tokens,
            response.usage.cache_creation_input_tokens,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        text = next((b.text for b in response.content if b.type == "text"), "")
        if not text:
            log.info("empty response for %s", contact["email"])
            return None

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            log.warning("non-json response for %s: %s", contact["email"], text[:200])
            return None

        if not payload.get("subject") or not payload.get("body"):
            log.info(
                "skipped %s: %s",
                contact["email"],
                payload.get("reason") or "no content",
            )
            return None

        return DraftResult(
            recipient_email=contact["email"],
            subject=payload["subject"],
            body=payload["body"],
            hook=payload.get("personalization_hook") or "",
            confidence=payload.get("confidence", "medium"),
        )
