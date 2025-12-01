"""Review CLI for labeling simulation results.

提供互動式介面讓人工標註模擬回應的品質。
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

from .logger import SimulationLogger
from .models import LabelType

logger = structlog.get_logger()


class ReviewCLI:
    """Interactive CLI for reviewing and labeling simulation responses.

    Usage:
        reviewer = ReviewCLI(data_dir="data/simulations")
        reviewer.start_review()
    """

    def __init__(self, data_dir: str | Path):
        """Initialize the review CLI.

        Args:
            data_dir: Directory containing simulation data files.
        """
        self.simulation_logger = SimulationLogger(data_dir)
        self.data_dir = Path(data_dir)

    def _print_separator(self, char: str = "=", width: int = 60) -> None:
        """Print a separator line."""
        print(char * width)

    def _print_header(self, text: str) -> None:
        """Print a header."""
        self._print_separator()
        print(f"  {text}")
        self._print_separator()

    def _format_response(self, response: dict) -> str:
        """Format a response record for display."""
        lines = [
            f"Response ID: {response['id']}",
            f"Post ID: {response['post_id']}",
            f"Timestamp: {response['timestamp']}",
            f"Adherence Score: {response.get('adherence_score', 'N/A')}",
            "",
            "--- Original Post ---",
            response.get('original_post_text', '(no text)'),
            "",
            "--- Generated Response ---",
            response.get('generated_response', '(no response)'),
        ]

        if response.get('memory_context_used'):
            lines.extend([
                "",
                "--- Memory Context Used ---",
                *[f"  - {mem}" for mem in response['memory_context_used'][:3]],
            ])

        return "\n".join(lines)

    def start_review(self, limit: Optional[int] = None) -> int:
        """Start the interactive review session.

        Args:
            limit: Maximum number of responses to review (None = all unlabeled).

        Returns:
            Number of responses labeled.
        """
        unlabeled = self.simulation_logger.get_unlabeled_responses()

        if not unlabeled:
            print("沒有未標註的回應。")
            return 0

        if limit:
            unlabeled = unlabeled[:limit]

        total = len(unlabeled)
        labeled_count = 0

        self._print_header(f"開始標註 - 共 {total} 筆未標註回應")
        print("指令: [g]ood / [b]ad / [n]eutral / [s]kip / [q]uit")
        print()

        for i, response in enumerate(unlabeled, 1):
            self._print_separator("-")
            print(f"\n[{i}/{total}]")
            print()
            print(self._format_response(response))
            print()

            # Get label
            label = self._prompt_label()

            if label == "quit":
                print(f"\n結束標註。本次標註了 {labeled_count} 筆。")
                break

            if label == "skip":
                print("跳過此筆。")
                continue

            # Get reason for bad/neutral labels
            reason = None
            suggested_fix = None
            issues = []

            if label in ("bad", "neutral"):
                reason = self._prompt_reason()
                if label == "bad":
                    issues = self._prompt_issues()
                    suggested_fix = self._prompt_suggested_fix()

            # Save the label
            self.simulation_logger.log_label(
                response_id=response["id"],
                label=label,
                reason=reason,
                suggested_fix=suggested_fix,
                issues=issues,
            )

            labeled_count += 1
            print(f"✓ 已標註為 [{label}]")

        print()
        self._print_header(f"標註完成 - 共標註 {labeled_count} 筆")

        return labeled_count

    def _prompt_label(self) -> str:
        """Prompt user for a label."""
        while True:
            try:
                choice = input("標註 [g/b/n/s/q]: ").strip().lower()

                if choice in ("g", "good"):
                    return "good"
                elif choice in ("b", "bad"):
                    return "bad"
                elif choice in ("n", "neutral"):
                    return "neutral"
                elif choice in ("s", "skip"):
                    return "skip"
                elif choice in ("q", "quit"):
                    return "quit"
                else:
                    print("請輸入 g (good), b (bad), n (neutral), s (skip), 或 q (quit)")
            except (KeyboardInterrupt, EOFError):
                return "quit"

    def _prompt_reason(self) -> Optional[str]:
        """Prompt user for a reason."""
        try:
            reason = input("原因 (可選，直接 Enter 跳過): ").strip()
            return reason if reason else None
        except (KeyboardInterrupt, EOFError):
            return None

    def _prompt_issues(self) -> list[str]:
        """Prompt user to select issues."""
        common_issues = [
            "語氣太正式",
            "語氣太隨意",
            "缺乏個人風格",
            "回應太長",
            "回應太短",
            "內容不相關",
            "人格不一致",
            "過度使用 emoji",
            "缺少 emoji",
            "用詞不自然",
        ]

        print("\n常見問題 (輸入數字選擇，多個用空格分隔，Enter 跳過):")
        for i, issue in enumerate(common_issues, 1):
            print(f"  {i}. {issue}")

        try:
            choices = input("選擇問題: ").strip()
            if not choices:
                return []

            selected = []
            for choice in choices.split():
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(common_issues):
                        selected.append(common_issues[idx])
                except ValueError:
                    # User entered text, treat as custom issue
                    selected.append(choice)

            return selected
        except (KeyboardInterrupt, EOFError):
            return []

    def _prompt_suggested_fix(self) -> Optional[str]:
        """Prompt user for a suggested fix."""
        try:
            fix = input("建議改進 (可選): ").strip()
            return fix if fix else None
        except (KeyboardInterrupt, EOFError):
            return None

    def show_stats(self) -> None:
        """Show current labeling statistics."""
        stats = self.simulation_logger.get_stats()

        self._print_header("標註統計")
        print(f"總觀察數: {stats['total_observations']}")
        print(f"總決策數: {stats['total_decisions']}")
        print(f"總回應數: {stats['total_responses']}")
        print(f"總反思數: {stats['total_reflections']}")
        print()
        print(f"已標註: {stats['total_labels']}")
        print(f"未標註: {stats['unlabeled_count']}")
        print()
        print("標註分佈:")
        for label, count in stats['label_distribution'].items():
            print(f"  - {label}: {count}")
        print()
        print(f"互動率: {stats['engagement_rate']:.1%}")
        print(f"平均 adherence: {stats['average_adherence_score']:.3f}")
        self._print_separator()

    def show_response(self, response_id: str) -> None:
        """Show a specific response and its label (if any).

        Args:
            response_id: The response ID to display.
        """
        result = self.simulation_logger.get_response_with_label(response_id)

        if not result:
            print(f"找不到 response: {response_id}")
            return

        response, label = result

        self._print_header(f"Response: {response_id}")
        print(self._format_response(response))

        if label:
            print()
            print("--- Label ---")
            print(f"Label: {label['label']}")
            if label.get('reason'):
                print(f"Reason: {label['reason']}")
            if label.get('issues'):
                print(f"Issues: {', '.join(label['issues'])}")
            if label.get('suggested_fix'):
                print(f"Suggested Fix: {label['suggested_fix']}")
        else:
            print()
            print("(尚未標註)")

        self._print_separator()

    def export_labeled_data(self, output_file: Optional[str] = None) -> str:
        """Export labeled data to a JSON file for analysis.

        Args:
            output_file: Output file path (default: labeled_data_{timestamp}.json)

        Returns:
            Path to the exported file.
        """
        import json

        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(self.data_dir / f"labeled_data_{timestamp}.json")

        responses = self.simulation_logger.get_responses()
        labels = self.simulation_logger.get_labels()

        # Create lookup for labels
        label_lookup = {l["response_id"]: l for l in labels}

        # Merge responses with their labels
        merged_data = []
        for response in responses:
            label = label_lookup.get(response["id"])
            merged_data.append({
                "response": response,
                "label": label,
            })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2, default=str)

        print(f"已匯出到: {output_file}")
        return output_file
