"""Simulation Analyzer - 分析標註結果並產生 Persona 調整建議.

分析 bad 回應的共同模式，產生改善 Persona 的具體建議。
"""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog

from .logger import SimulationLogger

logger = structlog.get_logger()


class SimulationAnalyzer:
    """Analyzer for simulation data and labels.

    Analyzes labeled responses to:
    - Identify common issues
    - Generate persona adjustment suggestions
    - Export analysis reports
    """

    def __init__(self, data_dir: str | Path):
        """Initialize the analyzer.

        Args:
            data_dir: Directory containing simulation data files.
        """
        self.simulation_logger = SimulationLogger(data_dir)
        self.data_dir = Path(data_dir)

    def analyze(self) -> dict[str, Any]:
        """Run full analysis on the simulation data.

        Returns:
            Analysis results dictionary.
        """
        responses = self.simulation_logger.get_responses()
        labels = self.simulation_logger.get_labels()
        decisions = self.simulation_logger.get_decisions()

        # Create lookup
        label_lookup = {l["response_id"]: l for l in labels}

        # Separate by label type
        good_responses = []
        bad_responses = []
        neutral_responses = []
        unlabeled_responses = []

        for response in responses:
            label = label_lookup.get(response["id"])
            if not label:
                unlabeled_responses.append(response)
            elif label["label"] == "good":
                good_responses.append((response, label))
            elif label["label"] == "bad":
                bad_responses.append((response, label))
            else:
                neutral_responses.append((response, label))

        # Analyze issues
        issue_analysis = self._analyze_issues(bad_responses)

        # Analyze adherence scores
        adherence_analysis = self._analyze_adherence(good_responses, bad_responses)

        # Analyze engagement patterns
        engagement_analysis = self._analyze_engagement(decisions)

        # Generate suggestions
        suggestions = self._generate_suggestions(
            issue_analysis,
            adherence_analysis,
            bad_responses,
        )

        return {
            "summary": {
                "total_responses": len(responses),
                "labeled": len(labels),
                "unlabeled": len(unlabeled_responses),
                "good": len(good_responses),
                "bad": len(bad_responses),
                "neutral": len(neutral_responses),
                "good_rate": len(good_responses) / len(labels) if labels else 0,
            },
            "issue_analysis": issue_analysis,
            "adherence_analysis": adherence_analysis,
            "engagement_analysis": engagement_analysis,
            "suggestions": suggestions,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _analyze_issues(
        self, bad_responses: list[tuple[dict, dict]]
    ) -> dict[str, Any]:
        """Analyze common issues from bad responses."""
        if not bad_responses:
            return {"issues": [], "reasons": [], "message": "沒有 bad 標註"}

        # Count issues
        all_issues = []
        all_reasons = []

        for response, label in bad_responses:
            issues = label.get("issues", [])
            all_issues.extend(issues)

            reason = label.get("reason")
            if reason:
                all_reasons.append(reason)

        issue_counts = Counter(all_issues)
        top_issues = issue_counts.most_common(10)

        return {
            "total_bad": len(bad_responses),
            "issues": [
                {"issue": issue, "count": count, "percentage": count / len(bad_responses)}
                for issue, count in top_issues
            ],
            "reasons": all_reasons,
            "suggested_fixes": [
                label.get("suggested_fix")
                for _, label in bad_responses
                if label.get("suggested_fix")
            ],
        }

    def _analyze_adherence(
        self,
        good_responses: list[tuple[dict, dict]],
        bad_responses: list[tuple[dict, dict]],
    ) -> dict[str, Any]:
        """Analyze adherence scores for good vs bad responses."""
        good_scores = [r.get("adherence_score", 0) for r, _ in good_responses]
        bad_scores = [r.get("adherence_score", 0) for r, _ in bad_responses]

        def safe_avg(scores: list) -> float:
            return sum(scores) / len(scores) if scores else 0

        return {
            "good_avg_adherence": round(safe_avg(good_scores), 3),
            "bad_avg_adherence": round(safe_avg(bad_scores), 3),
            "good_min_adherence": min(good_scores) if good_scores else 0,
            "bad_max_adherence": max(bad_scores) if bad_scores else 0,
            "insight": self._adherence_insight(good_scores, bad_scores),
        }

    def _adherence_insight(
        self, good_scores: list[float], bad_scores: list[float]
    ) -> str:
        """Generate insight about adherence scores."""
        if not good_scores or not bad_scores:
            return "資料不足，無法分析 adherence 相關性"

        good_avg = sum(good_scores) / len(good_scores)
        bad_avg = sum(bad_scores) / len(bad_scores)

        diff = good_avg - bad_avg

        if diff > 0.1:
            return f"good 回應的 adherence 平均高 {diff:.2f}，建議提高 adherence 門檻"
        elif diff < -0.05:
            return "bad 回應的 adherence 反而較高，問題可能不在人格一致性，而是回應品質"
        else:
            return "adherence 分數與標註無明顯關聯，建議檢視其他因素"

    def _analyze_engagement(self, decisions: list[dict]) -> dict[str, Any]:
        """Analyze engagement decision patterns."""
        if not decisions:
            return {"message": "沒有決策資料"}

        engaged = [d for d in decisions if d.get("should_engage", False)]
        not_engaged = [d for d in decisions if not d.get("should_engage", False)]

        # Group reasons for not engaging
        skip_reasons = Counter(d.get("reason", "unknown") for d in not_engaged)

        return {
            "total_decisions": len(decisions),
            "engaged": len(engaged),
            "not_engaged": len(not_engaged),
            "engagement_rate": len(engaged) / len(decisions) if decisions else 0,
            "top_skip_reasons": skip_reasons.most_common(5),
        }

    def _generate_suggestions(
        self,
        issue_analysis: dict,
        adherence_analysis: dict,
        bad_responses: list[tuple[dict, dict]],
    ) -> list[dict[str, str]]:
        """Generate persona adjustment suggestions based on analysis."""
        suggestions = []

        # Issue-based suggestions
        issues = issue_analysis.get("issues", [])
        for issue_data in issues[:5]:  # Top 5 issues
            issue = issue_data["issue"]
            suggestion = self._issue_to_suggestion(issue)
            if suggestion:
                suggestions.append({
                    "type": "issue",
                    "issue": issue,
                    "count": issue_data["count"],
                    "suggestion": suggestion,
                })

        # Adherence-based suggestion
        if adherence_analysis.get("insight"):
            suggestions.append({
                "type": "adherence",
                "suggestion": adherence_analysis["insight"],
            })

        # User-provided suggestions
        user_fixes = issue_analysis.get("suggested_fixes", [])
        for fix in user_fixes[:5]:  # Top 5 user fixes
            suggestions.append({
                "type": "user_suggested",
                "suggestion": fix,
            })

        return suggestions

    def _issue_to_suggestion(self, issue: str) -> Optional[str]:
        """Convert a common issue to a persona adjustment suggestion."""
        issue_suggestions = {
            "語氣太正式": "調整 speech_patterns.vocabulary_level 為 'casual'，或在 personality.communication_style 加入「輕鬆、口語化」",
            "語氣太隨意": "調整 speech_patterns.vocabulary_level 為 'moderate' 或 'formal'",
            "缺乏個人風格": "在 personality.traits 加入更具體的特質，在 speech_patterns.typical_phrases 加入更多常用語",
            "回應太長": "降低 interaction_rules.max_response_length，或在 prompt 中強調簡潔",
            "回應太短": "在 personality 描述中加入「喜歡深入探討」，或調整生成參數",
            "內容不相關": "檢查 interests 設定是否準確，或調整決策引擎的相關性門檻",
            "人格不一致": "增加 personality.traits 和 values 的具體性，確保 opinions 有明確立場",
            "過度使用 emoji": "調整 speech_patterns.emoji_usage 為 'rare' 或 'none'",
            "缺少 emoji": "調整 speech_patterns.emoji_usage 為 'occasional' 或 'frequent'",
            "用詞不自然": "在 speech_patterns.typical_phrases 加入更多自然口語表達，或調整語言模型的 temperature",
        }
        return issue_suggestions.get(issue)

    def export_report(self, output_file: Optional[str] = None) -> str:
        """Export analysis report to a JSON file.

        Args:
            output_file: Output file path (default: analysis_{timestamp}.json)

        Returns:
            Path to the exported file.
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(self.data_dir / f"analysis_{timestamp}.json")

        analysis = self.analyze()

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

        logger.info("analysis_report_exported", output_file=output_file)
        return output_file

    def print_report(self) -> None:
        """Print a human-readable analysis report."""
        analysis = self.analyze()
        summary = analysis["summary"]

        print("=" * 60)
        print("  模擬分析報告")
        print("=" * 60)
        print()

        # Summary
        print("## 摘要")
        print(f"總回應數: {summary['total_responses']}")
        print(f"已標註: {summary['labeled']} / 未標註: {summary['unlabeled']}")
        print(f"Good: {summary['good']} / Bad: {summary['bad']} / Neutral: {summary['neutral']}")
        print(f"Good 率: {summary['good_rate']:.1%}")
        print()

        # Issues
        print("## 常見問題")
        issues = analysis["issue_analysis"].get("issues", [])
        if issues:
            for i, issue_data in enumerate(issues[:5], 1):
                print(f"  {i}. {issue_data['issue']} ({issue_data['count']} 次, {issue_data['percentage']:.1%})")
        else:
            print("  (無資料)")
        print()

        # Adherence
        print("## Adherence 分析")
        adh = analysis["adherence_analysis"]
        print(f"  Good 平均: {adh.get('good_avg_adherence', 'N/A')}")
        print(f"  Bad 平均: {adh.get('bad_avg_adherence', 'N/A')}")
        print(f"  洞見: {adh.get('insight', 'N/A')}")
        print()

        # Engagement
        print("## 互動決策分析")
        eng = analysis["engagement_analysis"]
        if "message" not in eng:
            print(f"  總決策: {eng['total_decisions']}")
            print(f"  互動: {eng['engaged']} / 跳過: {eng['not_engaged']}")
            print(f"  互動率: {eng['engagement_rate']:.1%}")
        else:
            print(f"  {eng['message']}")
        print()

        # Suggestions
        print("## 調整建議")
        suggestions = analysis["suggestions"]
        if suggestions:
            for i, s in enumerate(suggestions, 1):
                print(f"  {i}. [{s['type']}] {s['suggestion']}")
        else:
            print("  (暫無建議)")

        print()
        print("=" * 60)
