"""Observation Mode - 模擬與資料收集模組.

此模組提供「觀察模式」功能：
- 完整模擬 Agent 行為但不實際發文
- 記錄所有決策、回應、反思
- 提供人工標註介面
- 分析標註結果並產生 Persona 調整建議
"""

from .logger import SimulationLogger
from .models import (
    DecisionRecord,
    LabelRecord,
    ObservationRecord,
    ReflectionRecord,
    ResponseRecord,
)
from .review import ReviewCLI
from .analyzer import SimulationAnalyzer

__all__ = [
    "SimulationLogger",
    "ObservationRecord",
    "DecisionRecord",
    "ResponseRecord",
    "ReflectionRecord",
    "LabelRecord",
    "ReviewCLI",
    "SimulationAnalyzer",
]
