from __future__ import annotations

import re

from nanobot.core.contracts import TurnSignals

_TASK_HINTS = (
    "查", "查询", "帮我", "帮忙", "设置", "打开", "订", "写", "翻译", "总结", "执行", "run", "fix"
)
_EMOTION_HINTS = (
    "难过", "崩溃", "好烦", "烦死了", "气死了", "失恋", "焦虑", "抑郁", "绝望", "委屈", "开心", "谢谢"
)
_URGENCY_HINTS = ("马上", "立刻", "尽快", "紧急", "asap", "urgent")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class SignalExtractor:
    """Extract turn-level signals for the fusion policy."""

    def extract(self, user_input: str, emotion_state: str = "") -> TurnSignals:
        text = user_input.strip().lower()
        task_hits = sum(1 for w in _TASK_HINTS if w in text)
        emotion_hits = sum(1 for w in _EMOTION_HINTS if w in text)
        urgency_hits = sum(1 for w in _URGENCY_HINTS if w in text)

        # lightweight textual heuristics
        question_mark = 1 if "?" in text or "？" in text else 0
        long_text_bonus = 0.15 if len(text) > 80 else 0.0
        exclamation = len(re.findall(r"[!！]", text))

        task_strength = _clamp01(0.25 * task_hits + 0.15 * question_mark + long_text_bonus)
        emotion_intensity = _clamp01(0.22 * emotion_hits + 0.08 * min(exclamation, 3))
        relationship_need = _clamp01(0.6 * emotion_intensity + (0.1 if "你" in text else 0.0))
        urgency = _clamp01(0.35 * urgency_hits + 0.1 * question_mark)
        safety_risk = 0.0

        if any(x in text for x in ("自杀", "伤害自己", "kill myself")):
            safety_risk = 1.0

        reason = f"task_hits={task_hits}, emotion_hits={emotion_hits}, urgency_hits={urgency_hits}"
        if "焦虑" in emotion_state or "悲" in emotion_state:
            emotion_intensity = _clamp01(emotion_intensity + 0.08)
            relationship_need = _clamp01(relationship_need + 0.05)

        return TurnSignals(
            task_strength=task_strength,
            emotion_intensity=emotion_intensity,
            relationship_need=relationship_need,
            urgency=urgency,
            safety_risk=safety_risk,
            reason=reason,
        )

