"""Harvest AI/LLM 新聞並轉為口語化中文素材池。

用法：
    python -m src.utils.harvest_ideas --feeds default --limit 5

輸出：
    data/ideas/YYYY-MM-DD.md

設計：
- 從 RSS/Atom 抓最新文章標題/摘要
- 透過 OpenAI 將內容轉成口語中文短稿（含觀點/為何有趣）
- 去 AI 化：提示要求避免機器腔，給出可讀的貼文素材
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import feedparser
import httpx
from openai import AsyncOpenAI

from .config import get_settings, is_reasoning_model
from .ideas import upsert_ideas


# Default feeds (prefer官方/穩定來源)
DEFAULT_FEEDS = [
    # 官方 AI 公司 Blog
    "https://openai.com/blog/rss.xml",
    "https://huggingface.co/blog/feed.xml",
    "https://deepmind.google/blog/rss.xml",
    "https://blog.google/technology/ai/rss",
    # 科技媒體 AI 版
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "https://arstechnica.com/ai/feed",
    "https://www.technologyreview.com/topic/artificial-intelligence/feed/",
    # 開發者社群
    "https://github.blog/feed/",
    "https://dev.to/feed/tag/ai",
    # Hacker News AI
    "https://hnrss.org/newest?q=AI",
]


async def fetch_feed(url: str, timeout: int = 30) -> list[dict]:
    """Fetch and parse a feed, return entries with title/link/summary."""
    # feedparser 會自行抓取，非 async；用 httpx 先取內容再 parse
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.text)

    entries = []
    for entry in parsed.entries[:20]:
        entries.append(
            {
                "title": entry.get("title", "").strip(),
                "link": entry.get("link", ""),
                "summary": entry.get("summary", "").strip(),
                "published": entry.get("published", ""),
            }
        )
    return entries


def dedupe_entries(entries: Iterable[dict]) -> list[dict]:
    """Deduplicate entries by title/link."""
    seen = set()
    unique = []
    for e in entries:
        key = (e.get("title") or "", e.get("link") or "")
        if key in seen:
            continue
        seen.add(key)
        unique.append(e)
    return unique


async def summarize_entries(
    entries: list[dict],
    client: AsyncOpenAI,
    persona_name: str,
    limit: int,
    model: str,
    max_completion_tokens: int,
    reasoning_effort: str,
) -> list[dict]:
    """Use OpenAI to turn entries into Chinese, human-sounding snippets."""
    items = entries[:limit]
    summaries: list[dict] = []

    for e in items:
        prompt = f"""請將下面的 AI/科技新聞轉成口語中文短稿，避免機器腔，讓一般讀者容易理解。
請包含：1) 這是什麼 2) 有什麼重點/影響 3) 你（{persona_name}）的簡短看法或問題。
字數 80-140 字，保持自然口吻。

標題：{e.get('title','')}
摘要：{e.get('summary','')}
    連結：{e.get('link','')}
"""
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_completion_tokens,
        }
        if is_reasoning_model(model):
            kwargs["reasoning_effort"] = reasoning_effort

        resp = await client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        snippets = text.strip()
        summaries.append(
            {
                "title": e.get("title", ""),
                "link": e.get("link", ""),
                "summary": snippets,
                "source": e.get("link", "") or e.get("published", "") or "unknown",
            }
        )
    return summaries


async def main(feeds: Optional[list[str]] = None, limit: int = 8) -> int:
    # If no params provided, parse CLI args
    if feeds is None:
        parser = argparse.ArgumentParser(description="Harvest AI ideas into data/ideas.")
        parser.add_argument("--feeds", nargs="*", default=["default"], help="Feed URLs or 'default'")
        parser.add_argument("--limit", type=int, default=8, help="Max items to keep")
        args = parser.parse_args()
        feeds = DEFAULT_FEEDS if args.feeds == ["default"] else args.feeds
        limit = args.limit

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    # Resolve feeds
    feed_urls = DEFAULT_FEEDS if feeds == ["default"] else feeds

    all_entries: list[dict] = []
    for url in feed_urls:
        try:
            entries = await fetch_feed(url)
            all_entries.extend(entries)
        except Exception as e:
            print(f"[warn] fetch feed failed: {url} ({e})")

    unique_entries = dedupe_entries(all_entries)

    # Summarize
    summarized_items = await summarize_entries(
        unique_entries,
        client,
        settings.agent_name,
        limit=limit,
        model=settings.openai_model,
        max_completion_tokens=settings.max_completion_tokens,
        reasoning_effort=settings.reasoning_effort,
    )

    # Save markdown for human browsing
    ideas_dir = Path("data/ideas")
    ideas_dir.mkdir(parents=True, exist_ok=True)
    outfile = ideas_dir / f"{datetime.now(timezone.utc).date()}.md"
    header = f"# Ideas harvested on {datetime.now(timezone.utc).isoformat()}\n\n"
    md_lines = [f"- {item['summary']}（來源：{item['link']}）" for item in summarized_items]
    content = header + "\n".join(md_lines) + "\n"
    outfile.write_text(content, encoding="utf-8")

    # Save to JSONL index with status tracking
    upsert_ideas(
        summarized_items,
        source="harvest",
        path=Path("data/ideas/index.jsonl"),
    )

    # Also print a brief summary
    print(f"Saved {len(summarized_items)} items to {outfile} and index.jsonl")

    return 0


if __name__ == "__main__":
    asyncio.run(main())
