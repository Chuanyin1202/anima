"""One-pager report generator for agent performance.

Takes simulation/real interaction logs (SimulationLogger format) and
produces a concise Markdown (optionally HTML) report for humans/agents.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import structlog

from ..agent.persona import Persona
from ..memory import AgentMemory
from ..observation.logger import SimulationLogger
from ..utils.config import get_settings

logger = structlog.get_logger()


def _parse_ts(value: Any) -> datetime:
    """Parse ISO timestamp to aware datetime (UTC fallback)."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)


def _filter_by_time(records: Iterable[dict], since: datetime) -> list[dict]:
    """Filter records whose timestamp >= since."""
    return [r for r in records if _parse_ts(r.get("timestamp")) >= since]


def _clean_text(text: str, max_len: int) -> str:
    """Collapse newlines/extra spaces for compact display."""
    cleaned = " ".join(text.split())
    return cleaned[:max_len]


class OnePagerReport:
    """Generate one-page Markdown/HTML reports for agent behavior."""

    def __init__(
        self,
        data_dir: str | Path,
        persona_path: Optional[str | Path] = None,
        since: Optional[datetime] = None,
        days: int = 7,
        exclude_mock: bool = True,
        recent_limit: int = 30,
    ):
        self.data_dir = Path(data_dir)
        self.persona_path = Path(persona_path) if persona_path else None
        self.since = since or (datetime.now(timezone.utc) - timedelta(days=days))
        self.logger = SimulationLogger(self.data_dir)
        self.exclude_mock = exclude_mock
        self.recent_limit = recent_limit

    def _load_persona(self) -> dict[str, Any]:
        """Load persona info for report."""
        try:
            settings = get_settings()
            persona_file = self.persona_path or Path(settings.persona_file)
            persona = Persona.from_file(persona_file)
            identity = persona.identity
            return {
                "name": identity.name,
                "summary": identity.background,
                "voice": persona.personality.communication_style,
                "interests": ", ".join(persona.interests.primary) if persona.interests.primary else "",
                "signature": identity.signature or "",
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("persona_load_failed", error=str(exc))
            return {}

    def _memory_stats(self) -> dict[str, Any]:
        """Fetch memory stats; tolerate failures."""
        try:
            settings = get_settings()
            memory = AgentMemory(
                agent_id=settings.agent_name,
                openai_api_key=settings.openai_api_key,
                qdrant_url=settings.qdrant_url,
                qdrant_api_key=settings.qdrant_api_key,
                database_url=settings.database_url,
                llm_model=settings.openai_model,
            )
            return memory.get_stats()
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory_stats_failed", error=str(exc))
            return {}

    def _label_distribution(self, labels: list[dict]) -> dict[str, int]:
        counts = Counter()
        for lbl in labels:
            label = lbl.get("label", "unknown")
            counts[label] += 1
        return counts

    def _strip_mock(self, records: list[dict]) -> list[dict]:
        """Filter out mock_* post interactions when requested."""
        if not self.exclude_mock:
            return records
        filtered = []
        for r in records:
            post_id = r.get("post_id") or r.get("post", {}).get("id")
            if post_id and str(post_id).startswith("mock_"):
                continue
            filtered.append(r)
        return filtered

    def _recent_interactions(self, responses: list[dict], labels: dict[str, dict], limit: int = 10) -> list[dict]:
        """Return recent responses with key fields."""
        sorted_resps = sorted(
            responses,
            key=lambda r: _parse_ts(r.get("timestamp")).timestamp(),
            reverse=True,
        )
        recent = []
        for r in sorted_resps[:limit]:
            label = labels.get(r["id"])
            recent.append(
                {
                    "ts": _parse_ts(r.get("timestamp")),
                    "post_id": r.get("post_id", ""),
                    "adherence": r.get("adherence_score"),
                    "adherence_reason": r.get("adherence_reason"),
                    "label": label.get("label") if label else None,
                    "reason": label.get("reason") if label else None,
                    "response": _clean_text(r.get("generated_response") or "", 180),
                    "was_posted": r.get("was_posted", False),
                    "error": r.get("error"),
                }
            )
        return recent

    def _engagement_stats(self, decisions: list[dict]) -> dict[str, Any]:
        engaged = [d for d in decisions if d.get("should_engage")]
        skipped = [d for d in decisions if not d.get("should_engage")]
        skip_reasons = Counter(d.get("reason", "unknown") for d in skipped)
        return {
            "total": len(decisions),
            "engaged": len(engaged),
            "skipped": len(skipped),
            "engagement_rate": len(engaged) / len(decisions) if decisions else 0,
            "top_skip_reasons": skip_reasons.most_common(5),
        }

    def _quality_stats(self, responses: list[dict], labels: list[dict]) -> dict[str, Any]:
        label_map = {l["response_id"]: l for l in labels}
        adherence = [r.get("adherence_score", 0) for r in responses]
        good_scores = [r.get("adherence_score", 0) for r in responses if label_map.get(r["id"], {}).get("label") == "good"]
        bad_scores = [r.get("adherence_score", 0) for r in responses if label_map.get(r["id"], {}).get("label") == "bad"]

        def avg(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        issue_counts = Counter()
        bad_reasons: list[str] = []
        for lbl in labels:
            if lbl.get("label") == "bad":
                issue_counts.update(lbl.get("issues", []))
                if lbl.get("reason"):
                    bad_reasons.append(lbl["reason"])

        return {
            "label_distribution": self._label_distribution(labels),
            "avg_adherence": round(avg(adherence), 3),
            "good_avg_adherence": round(avg(good_scores), 3),
            "bad_avg_adherence": round(avg(bad_scores), 3),
            "top_issues": issue_counts.most_common(5),
            "bad_reasons": bad_reasons[:5],
            "label_count": len(labels),
        }

    def _posting_health(self, responses: list[dict]) -> dict[str, Any]:
        posted = [r for r in responses if r.get("was_posted")]
        not_posted = [r for r in responses if not r.get("was_posted")]
        return {
            "posted": len(posted),
            "not_posted": len(not_posted),
        }

    def _low_adherence_cases(self, responses: list[dict], labels: dict[str, dict], threshold: float = 0.85, limit: int = 5) -> list[dict]:
        """Return responses whose adherence is below threshold."""
        low = [r for r in responses if r.get("adherence_score", 1) < threshold]
        low_sorted = sorted(low, key=lambda r: r.get("adherence_score", 1))
        cases = []
        for r in low_sorted[:limit]:
            lbl = labels.get(r["id"])
            cases.append(
                {
                    "post_id": r.get("post_id", ""),
                    "adh": r.get("adherence_score"),
                    "adherence_reason": r.get("adherence_reason"),
                    "label": lbl.get("label") if lbl else None,
                    "reason": lbl.get("reason") if lbl else None,
                    "response": (r.get("generated_response") or "")[:120],
                }
            )
        return cases

    def _suggestions(self, engagement: dict, quality: dict, low_cases: list[dict]) -> list[str]:
        """Generate actionable suggestions based on stats."""
        suggestions: list[str] = []
        if engagement.get("engagement_rate", 0) < 0.6:
            suggestions.append("互動率偏低，檢查 skip 門檻與興趣清單，適度放寬可提升觸達。")
        if low_cases:
            suggestions.append(f"有 {len(low_cases)} 筆 adherence < 0.85，回覆可增加具體細節或實例以提升一致性。")
        label_dist = quality.get("label_distribution", {})
        if label_dist.get("bad", 0) > 0:
            suggestions.append("存在 bad 標註，檢視 bad 常見問題與理由，更新 persona prompt 或回應模板。")
        if not label_dist:
            suggestions.append("尚未對真實資料標註，先進行少量標註以獲得品質基準。")
        return suggestions

    def generate(self, output_md: Path, output_html: bool = False) -> None:
        """Generate Markdown (and optionally HTML) report."""
        persona_info = self._load_persona()
        obs = self._strip_mock(_filter_by_time(self.logger.get_observations(), self.since))
        dec = self._strip_mock(_filter_by_time(self.logger.get_decisions(), self.since))
        resp = self._strip_mock(_filter_by_time(self.logger.get_responses(), self.since))
        labels = _filter_by_time(self.logger.get_labels(), self.since)
        # Drop labels for responses that were filtered out
        resp_ids = {r["id"] for r in resp}
        labels = [l for l in labels if l.get("response_id") in resp_ids]
        label_map = {l["response_id"]: l for l in labels}
        mem_stats = self._memory_stats()

        engagement = self._engagement_stats(dec)
        quality = self._quality_stats(resp, labels)
        recent = self._recent_interactions(resp, label_map, limit=self.recent_limit)
        low_cases = self._low_adherence_cases(resp, label_map)
        suggestions = self._suggestions(engagement, quality, low_cases)
        posting = self._posting_health(resp)

        lines: list[str] = []
        lines.append(f"# Agent 一頁報表（起始：{self.since.date()}）\n")

        # Persona
        lines.append("## Persona 摘要")
        if persona_info:
            lines.append(f"- 名稱：{persona_info.get('name','')}")
            lines.append(f"- 簡述：{persona_info.get('summary','')}")
            if persona_info.get("voice"):
                lines.append(f"- 口吻：{persona_info['voice']}")
            if persona_info.get("interests"):
                lines.append(f"- 興趣：{persona_info['interests']}")
            if persona_info.get("signature"):
                signature = str(persona_info["signature"]).lstrip("-— ").strip()
                lines.append(f"- 簽名：{signature}")
        else:
            lines.append("- Persona 載入失敗")
        lines.append("")

        # Memory
        lines.append("## 記憶庫概況")
        if mem_stats:
            lines.append(f"- 總記憶數：{mem_stats.get('total_memories','?')}")
            lines.append(f"- 跳過記錄：{mem_stats.get('skipped_records','?')}")
            by_type = mem_stats.get("by_type", {})
            if by_type:
                lines.append(f"- 類型分布：{by_type}")
            lines.append("- 期間內新增/寫入錯誤：暫無時間窗統計（需後續儀表增補）")
        else:
            lines.append("- 無法取得記憶庫統計（可能未連線或未設定）")
        lines.append("")

        # Engagement
        lines.append("## 決策與互動")
        lines.append(f"- 期間內決策：{engagement['total']}，互動率：{engagement['engagement_rate']:.1%}")
        lines.append(f"- 互動成功/生成回覆：{len(resp)} 筆")
        if engagement["top_skip_reasons"]:
            lines.append("- 最常見 skip 理由：")
            for k, v in engagement["top_skip_reasons"]:
                reason_text = str(k).lstrip("-— ").strip()
                lines.append(f"  - {reason_text} ({v})")
        lines.append("")

        # Posting health
        lines.append("## 互動健康度")
        lines.append(f"- 已發送回覆：{posting['posted']} 筆")
        lines.append(f"- 未發送/模擬或失敗：{posting['not_posted']} 筆")
        lines.append("")

        # Quality
        lines.append("## 品質標註 / Adherence")
        if quality["label_count"] == 0:
            lines.append("- 標註分布：未標註")
        else:
            lines.append(f"- 標註分布：{quality['label_distribution']}")
        lines.append(f"- 平均 adherence：{quality['avg_adherence']}")
        lines.append(f"- good/ bad 平均 adherence：{quality['good_avg_adherence']} / {quality['bad_avg_adherence']}")
        if quality["top_issues"]:
            issues = ", ".join(f"{i}({c})" for i, c in quality["top_issues"])
            lines.append(f"- Bad 常見問題：{issues}")
        if quality["bad_reasons"]:
            lines.append(f"- Bad 理由摘錄：{'; '.join(quality['bad_reasons'])}")
        lines.append("")

        # Diagnostics
        lines.append("## 問題診斷")
        if low_cases:
            lines.append("### Adherence 異常（< 0.85）")
            for c in low_cases:
                lines.append(
                    f"- post:{c['post_id']} adh:{c['adh']} label:{c['label'] or '未標註'}"
                )
                lines.append(f"  內容: {c['response']}")
                if c.get("adherence_reason"):
                    lines.append(f"  評分原因: {c['adherence_reason']}")
                if c["reason"]:
                    lines.append(f"  標註原因: {c['reason']}")
        else:
            lines.append("- 未發現 adherence < 0.85 的案例")
        lines.append("")

        # Recent interactions
        lines.append(f"## 最近互動摘要（最多 {self.recent_limit} 筆）")
        if not recent:
            lines.append("- 無互動記錄")
        else:
            for r in recent:
                label_str = f"{r['label']}" if r["label"] else "未標註"
                post_status = "[已發送]" if r["was_posted"] else "[未發送]"
                err_tag = f" err:{r['error']}" if r.get("error") else ""
                local_ts = r['ts'].astimezone()  # Convert to local timezone
                lines.append(
                    f"- [{local_ts.strftime('%Y-%m-%d %H:%M')}] post:{r['post_id']} "
                    f"adh:{r['adherence']} label:{label_str} {post_status}{err_tag}"
                )
                lines.append(f"  內容: {r['response']}")
                if r.get("adherence_reason"):
                    lines.append(f"  評分原因: {r['adherence_reason']}")
                if r["reason"]:
                    lines.append(f"  標註原因: {r['reason']}")
        lines.append("")

        # Suggestions
        lines.append("## 可操作建議")
        if suggestions:
            for s in suggestions:
                lines.append(f"- {s}")
        else:
            lines.append("- 暫無建議")
        lines.append("")

        # Save markdown
        output_md.parent.mkdir(parents=True, exist_ok=True)
        markdown = "\n".join(lines) + "\n"
        output_md.write_text(markdown, encoding="utf-8")
        logger.info("report_written", path=str(output_md))

        if output_html:
            html = "<html><body><pre>" + markdown.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + "</pre></body></html>"
            html_path = output_md.with_suffix(".html")
            html_path.write_text(html, encoding="utf-8")
            logger.info("report_html_written", path=str(html_path))
