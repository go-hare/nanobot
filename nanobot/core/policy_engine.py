from __future__ import annotations

from nanobot.core.contracts import FusionPolicy, TurnSignals


_DEFAULT_FUSION_CONFIG = {
    "weights": {
        "emotion_intensity": 0.45,
        "relationship_need": 0.35,
        "inverse_task_strength": 0.20,
    },
    "thresholds": {
        "eq_first_emotion": 0.7,
        "iq_first_urgency": 0.7,
        "iq_first_task": 0.55,
        "high_iq_weight": 0.7,
        "high_eq_weight": 0.7,
    },
    "tool_budget": {
        "high_iq": 6,
        "high_eq": 3,
        "balanced": 4,
    },
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class PolicyEngine:
    """Convert turn signals into a continuous IQ/EQ mixing policy."""

    def __init__(self, fusion_config: dict | None = None) -> None:
        self._cfg = _DEFAULT_FUSION_CONFIG.copy()
        if isinstance(fusion_config, dict):
            self._deep_update(self._cfg, fusion_config)

    @staticmethod
    def _deep_update(base: dict, override: dict) -> None:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                PolicyEngine._deep_update(base[key], value)
            else:
                base[key] = value

    def make_policy(
        self,
        signals: TurnSignals,
        runtime_adjustment: dict | None = None,
    ) -> FusionPolicy:
        weights = self._cfg["weights"]
        th = self._cfg["thresholds"]
        tool_budget_cfg = self._cfg["tool_budget"]
        eq_weight = _clamp01(
            float(weights["emotion_intensity"]) * signals.emotion_intensity
            + float(weights["relationship_need"]) * signals.relationship_need
            + float(weights["inverse_task_strength"]) * (1.0 - signals.task_strength)
        )
        if signals.safety_risk >= 0.8:
            eq_weight = max(eq_weight, 0.8)
        iq_weight = 1.0 - eq_weight
        if isinstance(runtime_adjustment, dict):
            eq_bias = float(runtime_adjustment.get("eq_bias", 0.0))
            iq_bias = float(runtime_adjustment.get("iq_bias", 0.0))
            eq_weight = _clamp01(eq_weight + eq_bias - iq_bias)
            iq_weight = 1.0 - eq_weight

        if signals.emotion_intensity > float(th["eq_first_emotion"]):
            order = "EQ_FIRST"
        elif (
            signals.urgency > float(th["iq_first_urgency"])
            and signals.task_strength > float(th["iq_first_task"])
        ):
            order = "IQ_FIRST"
        else:
            order = "INTERLEAVE"

        if iq_weight > float(th["high_iq_weight"]):
            tone = "professional"
            fact_depth = 2
            empathy_depth = 0
            tool_budget = int(tool_budget_cfg["high_iq"])
        elif eq_weight > float(th["high_eq_weight"]):
            tone = "warm"
            fact_depth = 1
            empathy_depth = 2
            tool_budget = int(tool_budget_cfg["high_eq"])
        else:
            tone = "balanced"
            fact_depth = 2
            empathy_depth = 1
            tool_budget = int(tool_budget_cfg["balanced"])

        if isinstance(runtime_adjustment, dict):
            tone_pref = runtime_adjustment.get("tone_preference")
            if isinstance(tone_pref, str) and tone_pref.strip():
                tone = tone_pref.strip()
            delta = int(runtime_adjustment.get("tool_budget_delta", 0))
            tool_budget = max(1, tool_budget + delta)

        confidence = _clamp01(0.35 + abs(iq_weight - eq_weight))
        return FusionPolicy(
            iq_weight=iq_weight,
            eq_weight=eq_weight,
            order=order,
            empathy_depth=empathy_depth,
            fact_depth=fact_depth,
            tool_budget=tool_budget,
            tone=tone,
            confidence=confidence,
        )

