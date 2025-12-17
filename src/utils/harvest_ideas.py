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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import feedparser
import httpx
from openai import AsyncOpenAI

from .config import get_settings, is_reasoning_model
from .ideas import upsert_ideas


# Default feeds (prefer官方/穩定來源)
# 精選來源：低重疊、訊雜比高
DEFAULT_FEEDS = [
    # 官方／研究團隊（必留）
    "https://openai.com/blog/rss.xml",
    "https://huggingface.co/blog/feed.xml",
    "https://deepmind.google/blog/rss.xml",
    # 媒體（單一主來源）
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    # 社群／聚合（擇一保留社群聲量）
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
        published_ts = 0
        if entry.get("published_parsed"):
            try:
                published_ts = int(time.mktime(entry.get("published_parsed")))
            except Exception:
                published_ts = 0
        entries.append(
            {
                "title": entry.get("title", "").strip(),
                "link": entry.get("link", ""),
                "summary": entry.get("summary", "").strip(),
                "published": entry.get("published", ""),
                "published_ts": published_ts,
                "feed": url,
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


def round_robin_entries(
    entries: list[dict],
    per_source_limit: int = 2,
    global_limit: int = 10,
) -> list[dict]:
    """
    Round-robin mix to避免單一來源霸榜：
    - 每來源最多 per_source_limit
    - 依來源輪流取第 1 輪、第 2 輪…
    - 總數不超過 global_limit
    """
    by_source: dict[str, list[dict]] = {}
    for e in entries:
        key = e.get("feed") or "unknown"
        by_source.setdefault(key, []).append(e)

    # 各來源按時間排序並截斷
    for source, items in by_source.items():
        items.sort(key=lambda e: e.get("published_ts") or 0, reverse=True)
        by_source[source] = items[:per_source_limit]

    picked: list[dict] = []
    for round_idx in range(per_source_limit):
        for source, items in by_source.items():
            if len(picked) >= global_limit:
                return picked
            if len(items) > round_idx:
                picked.append(items[round_idx])
    return picked


async def main(
    feeds: Optional[list[str]] = None,
    limit: int = 10,
    since_days: int = 3,
) -> int:
    # If no params provided, parse CLI args
    if feeds is None:
        parser = argparse.ArgumentParser(description="Harvest AI ideas into data/ideas.")
        parser.add_argument("--feeds", nargs="*", default=["default"], help="Feed URLs or 'default'")
        parser.add_argument("--limit", type=int, default=8, help="Max items to keep")
        parser.add_argument("--since-days", type=int, default=3, help="Only keep items within N days")
        args = parser.parse_args()
        feeds = DEFAULT_FEEDS if args.feeds == ["default"] else args.feeds
        limit = args.limit
        since_days = args.since_days

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
    unique_entries.sort(key=lambda e: e.get("published_ts") or 0, reverse=True)
    # 只保留近 N 天的項目，避免初次抓取過多舊文
    cutoff_ts = int(time.time()) - since_days * 24 * 3600
    unique_entries = [e for e in unique_entries if (e.get("published_ts") or 0) >= cutoff_ts]
    unique_entries = round_robin_entries(
        unique_entries,
        per_source_limit=2,
        global_limit=limit,
    )

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
