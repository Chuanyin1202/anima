"""Minimal console webapp for Anima (single service with scheduler).

Features (MVP):
- Health check
- List pending ideas
- Manually post an idea (uses existing AgentBrain flow)
- Show recent response errors
- Starts AgentScheduler on startup (same process as web UI)

Design notes:
- Uses existing create_agent_brain helper; opens Threads client and keeps it alive.
- Scheduler runs in background via APScheduler (already async-friendly).
- Lightweight HTML via FastAPI responses; no template engine dependency.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from .agent.scheduler import AgentScheduler
from .main import create_agent_brain
from .threads import ThreadsClient, MockThreadsClient
from .utils.config import get_settings
from .utils.ideas import read_index, mark_posted

logger = structlog.get_logger()

app = FastAPI(title="Anima Console")

# Globals kept for app lifetime
brain = None
scheduler: Optional[AgentScheduler] = None
threads_client = None


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize brain, threads client, and scheduler."""
    global brain, scheduler, threads_client
    settings = get_settings()

    # Create brain (observation_mode False for real runs)
    brain = await create_agent_brain(settings=settings, observation_mode=False)

    # Select client class
    client_cls = MockThreadsClient if settings.use_mock_threads else ThreadsClient
    threads_client = client_cls(
        access_token=settings.threads_access_token or "mock_token",
        user_id=settings.threads_user_id or "mock_user",
    )
    await threads_client.open()
    brain.threads = threads_client

    # Ensure external clients ready (caches self profile if needed)
    try:
        await brain._ensure_clients_ready()  # noqa: SLF001 - internal but safe here
    except Exception:
        logger.warning("brain_client_init_failed", exc_info=True)

    # Start scheduler
    scheduler = AgentScheduler(brain)
    scheduler.start()
    logger.info("console_scheduler_started")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Gracefully stop scheduler and clients."""
    global brain, scheduler, threads_client
    try:
        if scheduler:
            scheduler.stop()
    except Exception:
        logger.warning("scheduler_stop_failed", exc_info=True)
    try:
        if brain:
            await brain.close()
    except Exception:
        logger.warning("brain_close_failed", exc_info=True)
    try:
        if threads_client:
            await threads_client.close()
    except Exception:
        logger.warning("threads_close_failed", exc_info=True)


@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "scheduler_running": scheduler is not None,
        "pending_ideas": len([i for i in read_index() if i.status == "pending"]),
    }


def _render_html(title: str, body: str) -> HTMLResponse:
    html = f"""
    <html>
      <head>
        <title>{title}</title>
        <style>
          body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 960px; margin: 2rem auto; }}
          table {{ width: 100%; border-collapse: collapse; }}
          th, td {{ padding: 8px 10px; border-bottom: 1px solid #ddd; text-align: left; }}
          a.button, button {{ background: #2563eb; color: white; padding: 6px 12px; border-radius: 6px; text-decoration: none; border: none; cursor: pointer; }}
          form {{ display: inline; }}
          .muted {{ color: #666; font-size: 0.9em; }}
          .badge {{ padding: 2px 6px; border-radius: 4px; font-size: 0.8em; background: #eef2ff; color: #4338ca; }}
        </style>
      </head>
      <body>
        <h1>{title}</h1>
        {body}
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/", response_class=HTMLResponse)
async def list_ideas():
    ideas = [i for i in read_index() if i.status == "pending"]
    rows = []
    for idea in ideas:
        rows.append(
            f"<tr><td><div><strong>{idea.title or '(no title)'}</strong></div>"
            f"<div class='muted'>{idea.summary[:240]}{'...' if len(idea.summary)>240 else ''}</div>"
            f"<div class='muted'>來源: {idea.link or idea.source}</div></td>"
            f"<td class='muted'>{idea.created_at}</td>"
            f"<td>"
            f"<form method='post' action='/ideas/{idea.id}/post'>"
            f"<button type='submit'>發佈</button>"
            f"</form>"
            f"</td></tr>"
        )
    body = """
    <p class="muted">顯示 pending ideas，可手動發佈；自動排程仍由後台 scheduler 負責。</p>
    <table>
      <thead><tr><th>Idea</th><th>Created</th><th>Action</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """.format(rows="".join(rows or ["<tr><td colspan='3'>沒有 pending ideas</td></tr>"]))
    return _render_html("Anima Console - Ideas", body)


@app.post("/ideas/{idea_id}/post")
async def post_idea(idea_id: str):
    """Manually post an idea and mark it posted."""
    ideas = read_index()
    idea = next((i for i in ideas if i.id == idea_id and i.status == "pending"), None)
    if not idea:
        raise HTTPException(status_code=404, detail="Idea not found or already processed")

    # Create post
    try:
        post_id = await brain.create_original_post(topic=idea.summary)
    except Exception as exc:  # noqa: BLE001
        logger.error("manual_post_failed", idea_id=idea_id, error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Posting failed: {exc}")

    if post_id:
        mark_posted(idea_id=idea.id, post_id=post_id)
        return {"status": "posted", "post_id": post_id}

    raise HTTPException(status_code=500, detail="Posting returned no post_id")


@app.get("/responses", response_class=HTMLResponse)
async def recent_responses():
    path = Path("data/real_logs/responses.jsonl")
    if not path.exists():
        return _render_html("Anima Console - Responses", "<p>尚無 responses.jsonl</p>")

    lines = path.read_text(encoding="utf-8").splitlines()[-50:]
    rows = []
    import json

    for line in lines:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        status = "posted" if rec.get("was_posted") else "failed"
        badge = f"<span class='badge'>{status}</span>"
        err = rec.get("error") or ""
        rows.append(
            f"<tr><td>{badge}</td>"
            f"<td>{rec.get('timestamp','')}</td>"
            f"<td><div class='muted'>{rec.get('original_post_text','')[:140]}</div>"
            f"<div>{rec.get('generated_response','')[:160]}</div>"
            f"<div class='muted'>{err}</div></td></tr>"
        )

    body = """
    <p class="muted">最近 50 筆 response 紀錄（包含失敗訊息）。</p>
    <table>
      <thead><tr><th>Status</th><th>Time</th><th>Content</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """.format(rows="".join(rows or ["<tr><td colspan='3'>尚無資料</td></tr>"]))
    return _render_html("Anima Console - Responses", body)


@app.get("/api/ideas/pending")
async def api_pending_ideas():
    ideas = [i for i in read_index() if i.status == "pending"]
    return JSONResponse([i.__dict__ for i in ideas])
