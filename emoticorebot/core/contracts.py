from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TurnSignals:
    task_strength: float
    emotion_intensity: float
    relationship_need: float
    urgency: float
    safety_risk: float
    reason: str = ""


@dataclass(frozen=True)
class FusionPolicy:
    iq_weight: float
    eq_weight: float
    order: str  # EQ_FIRST | IQ_FIRST | INTERLEAVE
    empathy_depth: int
    fact_depth: int
    tool_budget: int
    tone: str
    confidence: float


@dataclass(frozen=True)
class FactPack:
    summary: str
    confidence: float
    actions_taken: list[str] = field(default_factory=list)
    raw_messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class EmpathyPack:
    opening: str
    closing: str = ""

