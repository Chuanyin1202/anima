"""Anima MCP Server - Main server implementation.

Provides MCP tools for interacting with Anima:
- Chat with persona
- Search and manage memories
- Access persona information
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI

from ..agent.persona import Persona, PersonaEngine
from ..memory.mem0_adapter import AgentMemory, MemoryType
from ..utils.config import get_settings, is_reasoning_model

# Initialize FastMCP server
mcp = FastMCP("Anima")
logger = logging.getLogger(__name__)

# Global instances (initialized lazily)
_persona: Optional[Persona] = None
_persona_engine: Optional[PersonaEngine] = None
_memory: Optional[AgentMemory] = None
_settings = None

# Current session participant identity
_current_participant_id: str = "participant_unknown"


def _extract_identity(message: str) -> Optional[str]:
    """Extract identity declaration from message.

    Supports: Chinese names, English names, whitespace, non-ASCII
    Excludes: URLs, @handles
    """
    # Name pattern: Chinese, alphanumeric, underscore, hyphen, 1-20 chars
    NAME = r"([\u4e00-\u9fff\w\-]{1,20})"

    patterns = [
        rf"我是\s*{NAME}",
        rf"我叫\s*{NAME}",
        rf"[Tt]his is\s+{NAME}",
        rf"[Ii]'m\s+{NAME}",
        rf"[Mm]y name is\s+{NAME}",
        rf"叫我\s*{NAME}",
        rf"改叫我\s*{NAME}",
    ]

    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            name = match.group(1)
            # Exclude URLs or @handles
            if name.startswith(("http", "@", "www")):
                continue
            return f"participant_{name}"
    return None


def _get_settings():
    """Get settings (lazy initialization)."""
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


def _get_persona() -> Persona:
    """Get or create Persona instance (no OpenAI needed)."""
    global _persona
    if _persona is None:
        settings = _get_settings()

        # Load persona
        persona_path = Path(settings.persona_file)
        if not persona_path.exists():
            persona_path = Path(__file__).parent.parent.parent / settings.persona_file

        if not persona_path.exists():
            raise FileNotFoundError(f"Persona file not found: {settings.persona_file}")

        _persona = Persona.from_file(persona_path)

    return _persona


async def _get_persona_engine() -> PersonaEngine:
    """Get or create PersonaEngine instance."""
    global _persona_engine
    if _persona_engine is None:
        settings = _get_settings()

        # Load persona
        persona_path = Path(settings.persona_file)
        if not persona_path.exists():
            persona_path = Path(__file__).parent.parent.parent / settings.persona_file

        if not persona_path.exists():
            raise FileNotFoundError(f"Persona file not found: {settings.persona_file}")

        persona = Persona.from_file(persona_path)

        # Initialize OpenAI client
        openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

        # Create PersonaEngine
        _persona_engine = PersonaEngine(
            persona=persona,
            openai_client=openai_client,
            model=settings.openai_model,
            advanced_model=settings.openai_model_advanced,
            max_completion_tokens=settings.max_completion_tokens,
            reasoning_effort=settings.reasoning_effort,
        )

    return _persona_engine


def _get_memory() -> AgentMemory:
    """Get or create AgentMemory instance."""
    global _memory
    if _memory is None:
        settings = _get_settings()
        _memory = AgentMemory(
            agent_id=settings.agent_name,
            openai_api_key=settings.openai_api_key,
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key,
            database_url=settings.database_url,
            llm_model=settings.openai_model,
        )
    return _memory


# =============================================================================
# MCP Tools
# =============================================================================

def _safe_error(message: str, exc: Exception | None = None) -> str:
    """Return user-friendly error message and log the exception."""
    if exc:
        logger.exception(message, exc_info=exc)
    else:
        logger.error(message)
    return f"[錯誤] {message}"


@mcp.tool()
async def anima_set_user(name: str) -> str:
    """設定當前對話者的名字。

    Args:
        name: 對話者的名字

    Returns:
        確認訊息
    """
    global _current_participant_id
    _current_participant_id = f"participant_{name}"
    return f"好的，我會記住你是 {name}"


@mcp.tool()
async def anima_chat(message: str, context: str = "") -> str:
    """與 Anima 對話。

    Args:
        message: 你想對 Anima 說的話
        context: 額外的對話脈絡（可選）

    Returns:
        Anima 的回應
    """
    global _current_participant_id

    try:
        engine = await _get_persona_engine()
        memory = _get_memory()

        # Try to extract identity from message
        if identity := _extract_identity(message):
            _current_participant_id = identity
            logger.info("identity_detected", participant_id=_current_participant_id)

        # Get memory context (include participant's memories if known)
        memory_context = memory.get_context_for_response(
            message,
            participant_id=_current_participant_id if _current_participant_id != "participant_unknown" else None,
        )

        # Combine contexts
        full_context = message
        if context:
            full_context = f"{context}\n\n{message}"

        # Generate response
        response = await engine.generate_response(
            context=full_context,
            memory_context=memory_context,
        )

        # Record the interaction in memory with participant identity
        memory.record_interaction(
            my_response=response,
            context=message,
            interaction_type="mcp_chat",
            participant_id=_current_participant_id,
        )

        return response
    except Exception as exc:  # noqa: BLE001
        return _safe_error("對話時發生錯誤，請稍後再試。", exc)


@mcp.tool()
async def anima_search_memory(query: str, limit: int = 5) -> str:
    """搜尋 Anima 的記憶。

    Args:
        query: 搜尋關鍵字或描述
        limit: 最多返回幾筆記憶（預設 5）

    Returns:
        相關記憶列表
    """
    try:
        memory = _get_memory()
        results = memory.search(query=query, limit=limit)

        if not results:
            return "找不到相關記憶。"

        lines = ["找到以下相關記憶：", ""]
        for i, mem in enumerate(results, 1):
            score_str = f" (相關度: {mem.relevance_score:.2f})" if mem.relevance_score else ""
            lines.append(f"{i}. [{mem.memory_type.value}]{score_str}")
            lines.append(f"   {mem.content}")
            lines.append("")

        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return _safe_error("搜尋記憶時發生錯誤。", exc)


@mcp.tool()
async def anima_add_memory(content: str, source: str = "mcp") -> str:
    """新增一則記憶給 Anima。

    Args:
        content: 要記住的內容
        source: 記憶來源（預設為 mcp）

    Returns:
        確認訊息
    """
    try:
        memory = _get_memory()
        memory_id = memory.observe(content=content, source=source)
        return f"已記錄：{content[:50]}...（ID: {memory_id}）"
    except Exception as exc:  # noqa: BLE001
        return _safe_error("新增記憶時發生錯誤。", exc)


@mcp.tool()
async def anima_get_recent_memories(limit: int = 10, memory_type: str = "") -> str:
    """取得 Anima 最近的記憶。

    Args:
        limit: 最多返回幾筆（預設 10）
        memory_type: 記憶類型篩選（observation, interaction, reflective, 留空表示全部）

    Returns:
        最近記憶列表
    """
    memory = _get_memory()

    try:
        # Parse memory type
        mem_type = None
        if memory_type:
            normalized = memory_type.lower()
            try:
                mem_type = MemoryType(normalized)
            except ValueError:
                return f"無效的記憶類型：{memory_type}。有效選項：observation, interaction, reflective, semantic, episodic"

        results = memory.get_recent(limit=limit, memory_type=mem_type)

        if not results:
            return "目前沒有記憶。"

        lines = [f"最近 {len(results)} 筆記憶：", ""]
        for i, mem in enumerate(results, 1):
            time_str = mem.created_at.strftime("%Y-%m-%d %H:%M")
            lines.append(f"{i}. [{mem.memory_type.value}] {time_str}")
            lines.append(f"   {mem.content[:100]}{'...' if len(mem.content) > 100 else ''}")
            lines.append("")

        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return _safe_error("取得最近記憶時發生錯誤。", exc)


@mcp.tool()
async def anima_reflect(topic: str = "") -> str:
    """讓 Anima 進行反思。

    Args:
        topic: 反思主題（可選，留空則自由反思）

    Returns:
        Anima 的反思內容
    """
    try:
        engine = await _get_persona_engine()
        memory = _get_memory()

        # Get recent memories for reflection
        recent = memory.get_recent(limit=20)
        if not recent:
            return "記憶不足，無法進行有意義的反思。"

        memories_text = "\n".join([f"- {m.content}" for m in recent[:10]])

        prompt = f"""Based on these recent experiences:
{memories_text}

{f'Focus on the topic: {topic}' if topic else 'Reflect on what you have learned and experienced.'}

Write a brief reflection (2-3 sentences) as {engine.persona.identity.name}."""

        kwargs = {
            "model": engine.model,
            "messages": [
                {"role": "system", "content": engine.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": engine.max_completion_tokens,
        }
        if is_reasoning_model(engine.model):
            kwargs["reasoning_effort"] = engine.reasoning_effort

        response = await engine.openai.chat.completions.create(**kwargs)

        reflection = response.choices[0].message.content or ""

        # Store the reflection
        memory.add_reflection(reflection)

        return reflection
    except Exception as exc:  # noqa: BLE001
        return _safe_error("生成反思時發生錯誤。", exc)


@mcp.tool()
async def anima_get_persona() -> str:
    """取得 Anima 的人格資訊。

    Returns:
        人格描述
    """
    try:
        persona = _get_persona()

        lines = [
            f"# {persona.identity.name}",
            "",
            f"**年齡**: {persona.identity.age or '未知'}",
            f"**職業**: {persona.identity.occupation or '未知'}",
            f"**地點**: {persona.identity.location or '未知'}",
            "",
            "## 背景",
            persona.identity.background,
            "",
            "## 性格特質",
            ", ".join(persona.personality.traits),
            "",
            "## 核心價值",
            ", ".join(persona.personality.values),
            "",
            "## 溝通風格",
            persona.personality.communication_style,
            "",
            "## 主要興趣",
            ", ".join(persona.interests.primary),
            "",
            "## 世界觀",
            persona.opinions.worldview,
        ]

        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return _safe_error("取得人格資訊時發生錯誤。", exc)


@mcp.tool()
async def anima_memory_stats() -> str:
    """取得 Anima 記憶統計。

    Returns:
        記憶統計資訊
    """
    try:
        memory = _get_memory()
        stats = memory.get_stats()

        lines = [
            "# 記憶統計",
            "",
            f"**總記憶數**: {stats['total_memories']}",
            "",
            "## 按類型分佈",
        ]

        for mem_type, count in stats.get("by_type", {}).items():
            lines.append(f"- {mem_type}: {count}")

        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return _safe_error("取得記憶統計時發生錯誤。", exc)


# =============================================================================
# MCP Resources
# =============================================================================


@mcp.resource("anima://persona")
async def get_persona_resource() -> str:
    """Anima 的人格定義（JSON 格式）。"""
    persona = _get_persona()
    return persona.model_dump_json(indent=2)


@mcp.resource("anima://system-prompt")
async def get_system_prompt_resource() -> str:
    """Anima 的系統提示詞。"""
    engine = await _get_persona_engine()
    return engine.system_prompt


# =============================================================================
# Entry Point
# =============================================================================


def run_server():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    run_server()
