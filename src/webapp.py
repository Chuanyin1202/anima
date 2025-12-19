"""Anima Console Web UI.

Features:
- Dashboard with statistics
- List pending ideas with preview/edit before posting
- Interaction safety (prevent duplicate posts)
- Visual card-based layout
- Response history viewer

Design notes:
- Uses existing create_agent_brain helper; opens Threads client and keeps it alive.
- Scheduler runs in background via APScheduler (already async-friendly).
- Lightweight HTML via FastAPI responses; no template engine dependency.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
import httpx
from pydantic import BaseModel

from .adapters import ThreadsAdapter
from .agent.scheduler import AgentScheduler
from .main import create_agent_brain
from .threads import ThreadsClient, MockThreadsClient
from .utils.config import get_settings
from .utils.ideas import read_index, mark_posted, mark_skipped

logger = structlog.get_logger()

app = FastAPI(title="Anima Console")

# Globals kept for app lifetime
brain = None
scheduler: Optional[AgentScheduler] = None
threads_client = None
me_id: Optional[str] = None


class PostCustomRequest(BaseModel):
    content: str


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize brain, threads client, and scheduler."""
    global brain, scheduler, threads_client, me_id
    settings = get_settings()

    # Create brain (observation_mode False for real runs)
    brain = await create_agent_brain(settings=settings, observation_mode=False)

    # Select client class and create adapter
    client_cls = MockThreadsClient if settings.use_mock_threads else ThreadsClient
    threads_client = client_cls(
        access_token=settings.threads_access_token or "mock_token",
        user_id=settings.threads_user_id or "mock_user",
    )
    await threads_client.open()
    brain.platform = ThreadsAdapter(threads_client)

    # Ensure external clients ready (caches self profile if needed)
    try:
        await brain._ensure_clients_ready()  # noqa: SLF001 - internal but safe here
    except Exception:
        logger.warning("brain_client_init_failed", exc_info=True)

    # Resolve /me for quick mismatch diagnostics
    try:
        me_resp = await threads_client._request("GET", "me")  # noqa: SLF001
        me_id = me_resp.get("id")
        if me_id and settings.threads_user_id and str(me_id) != str(settings.threads_user_id):
            logger.warning(
                "threads_user_mismatch",
                token_me_id=me_id,
                configured_user_id=settings.threads_user_id,
            )
    except Exception:
        logger.warning("threads_me_check_failed", exc_info=True)

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


# =============================================================================
# CSS Styles
# =============================================================================

CSS_STYLES = """
:root {
  --primary: #2563eb;
  --primary-hover: #1d4ed8;
  --success: #10b981;
  --warning: #f59e0b;
  --danger: #ef4444;
  --gray-50: #f9fafb;
  --gray-100: #f3f4f6;
  --gray-200: #e5e7eb;
  --gray-300: #d1d5db;
  --gray-500: #6b7280;
  --gray-700: #374151;
  --gray-900: #111827;
}

* { box-sizing: border-box; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  max-width: 1100px;
  margin: 0 auto;
  padding: 1.5rem;
  background: var(--gray-50);
  color: var(--gray-900);
  line-height: 1.5;
}

h1 { margin-bottom: 0.5rem; font-size: 1.75rem; }
h2 { font-size: 1.25rem; margin: 1.5rem 0 1rem; color: var(--gray-700); }

.muted { color: var(--gray-500); font-size: 0.875rem; }
.nav { margin-bottom: 1.5rem; display: flex; gap: 1rem; }
.nav a { color: var(--primary); text-decoration: none; }
.nav a:hover { text-decoration: underline; }

/* Stats Grid */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}
.stat-card {
  background: white;
  border-radius: 12px;
  padding: 1.25rem;
  text-align: center;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.stat-value {
  font-size: 2rem;
  font-weight: 700;
  color: var(--primary);
  line-height: 1.2;
}
.stat-label { color: var(--gray-500); font-size: 0.875rem; margin-top: 0.25rem; }

/* Cards */
.card {
  background: white;
  border-radius: 12px;
  padding: 1.25rem;
  margin-bottom: 1rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.card-title {
  font-weight: 600;
  font-size: 1rem;
  margin-bottom: 0.5rem;
  color: var(--gray-900);
}
.card-content { color: var(--gray-700); font-size: 0.9rem; margin-bottom: 0.75rem; }
.card-meta { display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; }
.card-actions { display: flex; gap: 0.5rem; margin-top: 1rem; }

/* Buttons */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  border: none;
  transition: all 0.15s;
}
.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.btn-primary { background: var(--primary); color: white; }
.btn-primary:hover:not(:disabled) { background: var(--primary-hover); }
.btn-secondary { background: var(--gray-200); color: var(--gray-700); }
.btn-secondary:hover:not(:disabled) { background: var(--gray-300); }
.btn-danger { background: var(--danger); color: white; }
.btn-danger:hover:not(:disabled) { background: #dc2626; }

/* Badges */
.badge {
  display: inline-block;
  padding: 0.25rem 0.5rem;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 500;
}
.badge-success { background: #d1fae5; color: #065f46; }
.badge-warning { background: #fef3c7; color: #92400e; }
.badge-danger { background: #fee2e2; color: #991b1b; }
.badge-info { background: #dbeafe; color: #1e40af; }

/* Modal */
.modal-overlay {
  display: none;
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.5);
  z-index: 1000;
  align-items: center;
  justify-content: center;
}
.modal-overlay.active { display: flex; }
.modal {
  background: white;
  border-radius: 16px;
  padding: 1.5rem;
  width: 90%;
  max-width: 600px;
  max-height: 90vh;
  overflow-y: auto;
}
.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}
.modal-title { font-size: 1.25rem; font-weight: 600; }
.modal-close {
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: var(--gray-500);
}
.modal-body { margin-bottom: 1rem; }
.modal-footer { display: flex; gap: 0.5rem; justify-content: flex-end; }

/* Form elements */
textarea {
  width: 100%;
  border: 1px solid var(--gray-300);
  border-radius: 8px;
  padding: 0.75rem;
  font-family: inherit;
  font-size: 0.9rem;
  resize: vertical;
  min-height: 150px;
}
textarea:focus { outline: 2px solid var(--primary); border-color: transparent; }

.char-count {
  text-align: right;
  font-size: 0.8rem;
  color: var(--gray-500);
  margin-top: 0.25rem;
}
.char-count.warning { color: var(--warning); }
.char-count.danger { color: var(--danger); }

/* Toast */
.toast-container {
  position: fixed;
  bottom: 1.5rem;
  right: 1.5rem;
  z-index: 2000;
}
.toast {
  background: var(--gray-900);
  color: white;
  padding: 0.75rem 1rem;
  border-radius: 8px;
  margin-top: 0.5rem;
  animation: slideIn 0.3s ease;
}
.toast.success { background: var(--success); }
.toast.error { background: var(--danger); }
@keyframes slideIn {
  from { transform: translateX(100%); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
}

/* Loading spinner */
.spinner {
  display: inline-block;
  width: 1rem;
  height: 1rem;
  border: 2px solid currentColor;
  border-right-color: transparent;
  border-radius: 50%;
  animation: spin 0.75s linear infinite;
  margin-right: 0.5rem;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Empty state */
.empty-state {
  text-align: center;
  padding: 3rem;
  color: var(--gray-500);
}

/* Table (for responses page) */
table { width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; }
th, td { padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--gray-200); }
th { background: var(--gray-50); font-weight: 600; font-size: 0.875rem; color: var(--gray-700); }
"""


# =============================================================================
# JavaScript
# =============================================================================

JS_SCRIPTS = """
// Global state
let isPosting = false;
let currentPreviewIdeaId = null;

// Toast notifications
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

// Modal functions
function openModal(modalId) {
  document.getElementById(modalId).classList.add('active');
}

function closeModal(modalId) {
  document.getElementById(modalId).classList.remove('active');
}

// Character counter
function updateCharCount(textarea, counterId) {
  const count = textarea.value.length;
  const counter = document.getElementById(counterId);
  counter.textContent = `${count} / 500`;
  counter.className = 'char-count';
  if (count > 450) counter.classList.add('warning');
  if (count > 500) counter.classList.add('danger');
}

// Disable/enable all post buttons
function setAllButtonsDisabled(disabled) {
  document.querySelectorAll('.btn-post, .btn-preview').forEach(btn => {
    btn.disabled = disabled;
  });
}

// Preview idea
async function previewIdea(ideaId, summary) {
  if (isPosting) return;

  currentPreviewIdeaId = ideaId;
  const textarea = document.getElementById('preview-content');
  const charCounter = document.getElementById('preview-char-count');

  // Show loading state
  textarea.value = '生成中...';
  textarea.disabled = true;
  openModal('preview-modal');

  try {
    const resp = await fetch(`/api/ideas/${ideaId}/preview`, { method: 'POST' });
    const data = await resp.json();
    if (resp.ok) {
      textarea.value = data.content;
      textarea.disabled = false;
      updateCharCount(textarea, 'preview-char-count');
    } else {
      textarea.value = `生成失敗: ${data.detail || '未知錯誤'}`;
    }
  } catch (err) {
    textarea.value = `生成失敗: ${err}`;
  }
}

// Post from preview modal (with custom content)
async function postFromPreview() {
  if (isPosting || !currentPreviewIdeaId) return;

  const content = document.getElementById('preview-content').value.trim();
  if (!content) {
    showToast('內容不能為空', 'error');
    return;
  }
  if (content.length > 500) {
    showToast('內容超過 500 字限制', 'error');
    return;
  }

  isPosting = true;
  setAllButtonsDisabled(true);
  const postBtn = document.getElementById('preview-post-btn');
  postBtn.innerHTML = '<span class="spinner"></span>發佈中...';

  try {
    const resp = await fetch(`/api/ideas/${currentPreviewIdeaId}/post-custom`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content })
    });
    const data = await resp.json();
    if (resp.ok) {
      showToast('發佈成功！', 'success');
      closeModal('preview-modal');
      setTimeout(() => window.location.reload(), 1000);
    } else {
      showToast(`發佈失敗: ${data.detail || '未知錯誤'}`, 'error');
    }
  } catch (err) {
    showToast(`發佈失敗: ${err}`, 'error');
  } finally {
    isPosting = false;
    setAllButtonsDisabled(false);
    postBtn.innerHTML = '發佈';
  }
}

// Direct post (without preview)
async function postIdea(ideaId) {
  if (isPosting) return;

  if (!confirm('確定要直接發佈嗎？建議先預覽確認內容。')) return;

  isPosting = true;
  setAllButtonsDisabled(true);
  const btn = document.querySelector(`[data-idea-id="${ideaId}"] .btn-post`);
  if (btn) btn.innerHTML = '<span class="spinner"></span>發佈中...';

  try {
    const resp = await fetch(`/ideas/${ideaId}/post`, { method: 'POST' });
    const data = await resp.json();
    if (resp.ok) {
      showToast('發佈成功！', 'success');
      setTimeout(() => window.location.reload(), 1000);
    } else {
      showToast(`發佈失敗: ${data.detail || '未知錯誤'}`, 'error');
    }
  } catch (err) {
    showToast(`發佈失敗: ${err}`, 'error');
  } finally {
    isPosting = false;
    setAllButtonsDisabled(false);
    if (btn) btn.innerHTML = '直接發佈';
  }
}

// Skip idea
async function skipIdea(ideaId) {
  if (isPosting) return;
  if (!confirm('確定要跳過這個 idea 嗎？')) return;

  try {
    const resp = await fetch(`/api/ideas/${ideaId}/skip`, { method: 'POST' });
    if (resp.ok) {
      showToast('已跳過', 'success');
      setTimeout(() => window.location.reload(), 500);
    } else {
      const data = await resp.json();
      showToast(`跳過失敗: ${data.detail}`, 'error');
    }
  } catch (err) {
    showToast(`跳過失敗: ${err}`, 'error');
  }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  // Setup textarea char counter
  const previewTextarea = document.getElementById('preview-content');
  if (previewTextarea) {
    previewTextarea.addEventListener('input', () => {
      updateCharCount(previewTextarea, 'preview-char-count');
    });
  }

  // Close modal on overlay click
  document.querySelectorAll('.modal-overlay').forEach(overlay => {
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) closeModal(overlay.id);
    });
  });

  // Close modal on escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      document.querySelectorAll('.modal-overlay.active').forEach(m => m.classList.remove('active'));
    }
  });
});
"""


# =============================================================================
# HTML Rendering
# =============================================================================

def _render_html(title: str, body: str, include_modal: bool = False) -> HTMLResponse:
    modal_html = ""
    if include_modal:
        modal_html = """
        <div id="preview-modal" class="modal-overlay">
          <div class="modal">
            <div class="modal-header">
              <h3 class="modal-title">預覽 / 編輯貼文</h3>
              <button class="modal-close" onclick="closeModal('preview-modal')">&times;</button>
            </div>
            <div class="modal-body">
              <textarea id="preview-content" placeholder="貼文內容..."></textarea>
              <div id="preview-char-count" class="char-count">0 / 500</div>
            </div>
            <div class="modal-footer">
              <button class="btn btn-secondary" onclick="closeModal('preview-modal')">取消</button>
              <button id="preview-post-btn" class="btn btn-primary" onclick="postFromPreview()">發佈</button>
            </div>
          </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>{CSS_STYLES}</style>
</head>
<body>
  <nav class="nav">
    <a href="/">Dashboard</a>
    <a href="/responses">回應紀錄</a>
    <a href="/posts">發文紀錄</a>
    <a href="/healthz">健康檢查</a>
  </nav>
  <h1>{title}</h1>
  {body}
  {modal_html}
  <div id="toast-container" class="toast-container"></div>
  <script>{JS_SCRIPTS}</script>
</body>
</html>"""
    return HTMLResponse(content=html)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "scheduler_running": scheduler is not None,
        "pending_ideas": len([i for i in read_index() if i.status == "pending"]),
        "threads_me_id": me_id,
    }


@app.get("/api/stats")
async def api_stats():
    """Get dashboard statistics."""
    ideas = read_index()
    pending = [i for i in ideas if i.status == "pending"]
    posted = [i for i in ideas if i.status == "posted"]
    skipped = [i for i in ideas if i.status == "skip"]

    # Count posts today and this week
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=now.weekday())

    posted_today = 0
    posted_week = 0
    for idea in posted:
        try:
            # Parse created_at if it's a string
            if isinstance(idea.created_at, str):
                ts = datetime.fromisoformat(idea.created_at.replace("Z", "+00:00"))
            else:
                ts = idea.created_at
            if ts >= today_start:
                posted_today += 1
            if ts >= week_start:
                posted_week += 1
        except Exception:
            pass

    # Get memory count from brain if available
    memory_count = 0
    try:
        if brain and brain.memory:
            stats = await brain.memory.get_stats()
            memory_count = stats.get("total", 0)
    except Exception:
        pass

    return {
        "pending_count": len(pending),
        "posted_today": posted_today,
        "posted_week": posted_week,
        "total_posted": len(posted),
        "skipped_count": len(skipped),
        "memory_count": memory_count,
    }


@app.get("/api/ideas/pending")
async def api_pending_ideas():
    ideas = [i for i in read_index() if i.status == "pending"]
    return JSONResponse([i.__dict__ for i in ideas])


@app.post("/api/ideas/{idea_id}/preview")
async def api_preview_idea(idea_id: str):
    """Generate a preview of the post content without actually posting."""
    ideas = read_index()
    idea = next((i for i in ideas if i.id == idea_id and i.status == "pending"), None)
    if not idea:
        raise HTTPException(status_code=404, detail="Idea not found or already processed")

    try:
        # Generate content using brain but don't post
        content = await brain.generate_post_content(topic=idea.summary)
        return {"content": content, "idea_id": idea_id}
    except Exception as exc:
        logger.error("preview_generation_failed", idea_id=idea_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Failed to generate preview: {exc}")


@app.post("/api/ideas/{idea_id}/post-custom")
async def api_post_custom(idea_id: str, request: PostCustomRequest):
    """Post custom content for an idea."""
    ideas = read_index()
    idea = next((i for i in ideas if i.id == idea_id and i.status == "pending"), None)
    if not idea:
        raise HTTPException(status_code=404, detail="Idea not found or already processed")

    content = request.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Content cannot be empty")
    if len(content) > 500:
        raise HTTPException(status_code=400, detail="Content exceeds 500 character limit")

    try:
        # Use brain to post (handles signature and memory recording)
        post_id = await brain.post_custom_content(
            content=content,
            topic=idea.summary,
            source="console",
            idea_id=idea.id,
            raise_on_error=True,
        )
        if not post_id:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        mark_posted(idea_id=idea.id, post_id=post_id)
        logger.info("custom_post_created", idea_id=idea_id, post_id=post_id)
        return {"status": "posted", "post_id": post_id}
    except Exception as exc:
        logger.error("custom_post_failed", idea_id=idea_id, error=str(exc), exc_info=True)
        raise HTTPException(status_code=502, detail=f"Posting failed: {exc}")


@app.post("/api/ideas/{idea_id}/skip")
async def api_skip_idea(idea_id: str):
    """Skip an idea (mark as skipped)."""
    ideas = read_index()
    idea = next((i for i in ideas if i.id == idea_id and i.status == "pending"), None)
    if not idea:
        raise HTTPException(status_code=404, detail="Idea not found or already processed")

    try:
        mark_skipped(idea_id=idea.id)
        return {"status": "skipped"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to skip: {exc}")


@app.post("/ideas/{idea_id}/post")
async def post_idea(idea_id: str):
    """Manually post an idea and mark it posted."""
    ideas = read_index()
    idea = next((i for i in ideas if i.id == idea_id and i.status == "pending"), None)
    if not idea:
        raise HTTPException(status_code=404, detail="Idea not found or already processed")

    # Create post
    try:
        post_id = await brain.create_original_post(
            topic=idea.summary,
            source="console",
            idea_id=idea.id,
            raise_on_error=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("manual_post_failed", idea_id=idea_id, error=str(exc), exc_info=True)
        detail = str(exc)
        # Unwrap Threads API error for clearer UX
        try:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
                err = exc.response.json().get("error", {})
                user_msg = err.get("error_user_msg")
                user_title = err.get("error_user_title")
                code = err.get("code")
                subcode = err.get("error_subcode")
                message = err.get("message")
                detail_parts = [
                    f"Threads API error code {code}",
                    f"subcode {subcode}" if subcode else None,
                    message,
                    user_title,
                    user_msg,
                ]
                detail = " | ".join([p for p in detail_parts if p])
        except Exception:
            pass
        raise HTTPException(status_code=502, detail=f"Posting failed: {detail}")

    if post_id:
        mark_posted(idea_id=idea.id, post_id=post_id)
        return {"status": "posted", "post_id": post_id}

    raise HTTPException(status_code=500, detail="Posting returned no post_id")


# =============================================================================
# Page Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard with stats and ideas list."""
    ideas = [i for i in read_index() if i.status == "pending"]

    # Stats section
    stats_html = """
    <div class="stats-grid" id="stats-grid">
      <div class="stat-card">
        <div class="stat-value" id="stat-pending">-</div>
        <div class="stat-label">待發佈</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" id="stat-today">-</div>
        <div class="stat-label">今日發佈</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" id="stat-week">-</div>
        <div class="stat-label">本週發佈</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" id="stat-skipped">-</div>
        <div class="stat-label">已跳過</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" id="stat-memory">-</div>
        <div class="stat-label">記憶數量</div>
      </div>
    </div>
    <script>
      fetch('/api/stats')
        .then(r => r.json())
        .then(data => {
          document.getElementById('stat-pending').textContent = data.pending_count;
          document.getElementById('stat-today').textContent = data.posted_today;
          document.getElementById('stat-week').textContent = data.posted_week;
          document.getElementById('stat-skipped').textContent = data.skipped_count;
          document.getElementById('stat-memory').textContent = data.memory_count;
        })
        .catch(() => {});
    </script>
    """

    # Ideas list
    if not ideas:
        ideas_html = '<div class="empty-state">沒有待發佈的 ideas</div>'
    else:
        cards = []
        for idea in ideas:
            source_display = idea.link or idea.source or ""
            if source_display and len(source_display) > 60:
                source_display = source_display[:60] + "..."

            # Escape HTML in content
            title_safe = (idea.title or "(無標題)").replace("<", "&lt;").replace(">", "&gt;")
            summary_safe = idea.summary.replace("<", "&lt;").replace(">", "&gt;")
            summary_preview = summary_safe[:300] + ("..." if len(summary_safe) > 300 else "")

            cards.append(f"""
            <div class="card" data-idea-id="{idea.id}">
              <div class="card-title">{title_safe}</div>
              <div class="card-content">{summary_preview}</div>
              <div class="card-meta">
                <span class="muted">{idea.created_at}</span>
                <a href="{idea.link or '#'}" target="_blank" class="muted" style="text-decoration:none;">
                  {source_display}
                </a>
              </div>
              <div class="card-actions">
                <button class="btn btn-primary btn-preview" onclick="previewIdea('{idea.id}', '')">
                  預覽 / 編輯
                </button>
                <button class="btn btn-secondary btn-post" onclick="postIdea('{idea.id}')">
                  直接發佈
                </button>
                <button class="btn btn-secondary" onclick="skipIdea('{idea.id}')">
                  跳過
                </button>
              </div>
            </div>
            """)
        ideas_html = "".join(cards)

    body = f"""
    {stats_html}
    <h2>待發佈 Ideas ({len(ideas)})</h2>
    <p class="muted">點擊「預覽 / 編輯」可在發佈前檢視和修改內容。</p>
    {ideas_html}
    """
    return _render_html("Anima Console", body, include_modal=True)


@app.get("/responses", response_class=HTMLResponse)
async def recent_responses():
    """View recent response history."""
    path = Path("data/real_logs/responses.jsonl")
    if not path.exists():
        return _render_html("回應紀錄", '<div class="empty-state">尚無回應紀錄</div>')

    lines = path.read_text(encoding="utf-8").splitlines()[-50:]
    rows = []

    for line in reversed(lines):  # Most recent first
        try:
            rec = json.loads(line)
        except Exception:
            continue

        status = "posted" if rec.get("was_posted") else "failed"
        badge_class = "badge-success" if status == "posted" else "badge-danger"
        badge = f'<span class="badge {badge_class}">{status}</span>'
        err = rec.get("error") or ""

        original = (rec.get("original_post_text") or "")[:140]
        response = (rec.get("generated_response") or "")[:200]

        rows.append(f"""
        <tr>
          <td>{badge}</td>
          <td class="muted">{rec.get('timestamp', '')}</td>
          <td>
            <div class="muted">{original}{'...' if len(original) >= 140 else ''}</div>
            <div>{response}{'...' if len(response) >= 200 else ''}</div>
            {'<div class="muted" style="color:var(--danger)">' + err + '</div>' if err else ''}
          </td>
        </tr>
        """)

    rows_html = "".join(rows or ['<tr><td colspan="3" class="empty-state">尚無資料</td></tr>'])
    body = f"""
    <p class="muted">最近 50 筆回應紀錄（最新在前）</p>
    <table>
      <thead><tr><th style="width:80px">狀態</th><th style="width:180px">時間</th><th>內容</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """
    return _render_html("回應紀錄", body)


@app.get("/posts", response_class=HTMLResponse)
async def recent_posts():
    """View recent original post history."""
    path = Path("data/real_logs/posts.jsonl")
    if not path.exists():
        return _render_html("發文紀錄", '<div class="empty-state">尚無發文紀錄</div>')

    lines = path.read_text(encoding="utf-8").splitlines()[-50:]
    rows = []

    source_labels = {
        "scheduled": "排程",
        "console": "手動",
        "manual": "CLI",
    }

    for line in reversed(lines):  # Most recent first
        try:
            rec = json.loads(line)
        except Exception:
            continue

        status = "posted" if rec.get("was_posted") else "failed"
        badge_class = "badge-success" if status == "posted" else "badge-danger"
        badge = f'<span class="badge {badge_class}">{status}</span>'

        source = rec.get("source", "unknown")
        source_label = source_labels.get(source, source)
        source_badge = f'<span class="badge">{source_label}</span>'

        err = rec.get("error") or ""
        content = (rec.get("content") or "")[:200]
        topic = rec.get("topic") or ""

        rows.append(f"""
        <tr>
          <td>{badge} {source_badge}</td>
          <td class="muted">{rec.get('timestamp', '')[:19]}</td>
          <td>
            <div class="muted">主題: {topic}</div>
            <div>{content}{'...' if len(content) >= 200 else ''}</div>
            {'<div class="muted" style="color:var(--danger)">' + err + '</div>' if err else ''}
          </td>
        </tr>
        """)

    rows_html = "".join(rows or ['<tr><td colspan="3" class="empty-state">尚無資料</td></tr>'])
    body = f"""
    <p class="muted">最近 50 筆原創發文紀錄（最新在前）</p>
    <table>
      <thead><tr><th style="width:120px">狀態</th><th style="width:160px">時間</th><th>內容</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """
    return _render_html("發文紀錄", body)
