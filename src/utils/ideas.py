"""Idea pool utilities.

Store harvested ideas in JSONL and provide helpers to reuse them
for replies/original posts/reflection context.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Literal


IDEA_INDEX = Path("data/ideas/index.jsonl")


@dataclass
class Idea:
    id: str
    title: str
    summary: str
    link: str
    source: str
    created_at: str  # ISO
    status: Literal["pending", "posted", "skip", "expired"] = "pending"
    posted_at: str | None = None
    threads_post_id: str | None = None

    @property
    def created_dt(self) -> datetime:
        return datetime.fromisoformat(self.created_at)


def _hash(title: str, link: str) -> str:
    raw = (title or "") + (link or "")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_index(path: Path = IDEA_INDEX) -> list[Idea]:
    if not path.exists():
        return []
    ideas: list[Idea] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            ideas.append(Idea(**data))
    return ideas


def write_index(ideas: Iterable[Idea], path: Path = IDEA_INDEX) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for idea in ideas:
            f.write(json.dumps(asdict(idea), ensure_ascii=False) + "\n")


def upsert_ideas(
    items: Iterable[dict],
    source: str,
    path: Path = IDEA_INDEX,
) -> list[Idea]:
    """Insert new ideas; keep existing status/created_at if already present."""
    existing = {idea.id: idea for idea in read_index(path)}
    now_iso = datetime.now(timezone.utc).isoformat()

    for item in items:
        title = item.get("title", "").strip()
        link = item.get("link", "").strip()
        summary = item.get("summary", "").strip()
        if not title and not summary:
            continue
        idea_id = _hash(title, link)
        if idea_id in existing:
            idea = existing[idea_id]
            # Update summary/title/link if newer info exists, preserve status
            idea.title = title or idea.title
            idea.link = link or idea.link
            idea.summary = summary or idea.summary
            idea.source = source or idea.source
        else:
            existing[idea_id] = Idea(
                id=idea_id,
                title=title,
                summary=summary,
                link=link,
                source=source,
                created_at=now_iso,
                status="pending",
            )

    ideas_sorted = sorted(existing.values(), key=lambda x: x.created_at, reverse=True)
    write_index(ideas_sorted, path)
    return ideas_sorted


def mark_posted(idea_id: str, post_id: str | None = None, path: Path = IDEA_INDEX) -> None:
    ideas = read_index(path)
    changed = False
    for idea in ideas:
        if idea.id == idea_id:
            idea.status = "posted"
            idea.posted_at = datetime.now(timezone.utc).isoformat()
            idea.threads_post_id = post_id
            changed = True
            break
    if changed:
        write_index(ideas, path)


def get_recent_ideas(
    path: Path = IDEA_INDEX,
    max_items: int = 3,
    max_age_days: int = 7,
    statuses: tuple[str, ...] = ("pending", "posted"),
) -> list[Idea]:
    ideas = read_index(path)
    if not ideas:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    filtered = [
        idea
        for idea in ideas
        if idea.status in statuses and idea.created_dt >= cutoff
    ]
    filtered.sort(key=lambda x: x.created_at, reverse=True)
    return filtered[:max_items]


def format_ideas_for_context(ideas: list[Idea]) -> str:
    """Render ideas into a short context block for LLM prompts."""
    if not ideas:
        return ""
    lines = ["Recent AI ideas:"]
    for idea in ideas:
        lines.append(f"- {idea.summary}（來源：{idea.link or idea.source}）")
    return "\n".join(lines)


def expire_old_ideas(max_age_days: int = 7, path: Path = IDEA_INDEX) -> None:
    """Mark pending ideas older than max_age_days as expired."""
    ideas = read_index(path)
    if not ideas:
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    changed = False
    for idea in ideas:
        if idea.status == "pending" and idea.created_dt < cutoff:
            idea.status = "expired"
            changed = True
    if changed:
        write_index(ideas, path)
