from __future__ import annotations

from emoticorebot.core.contracts import EmpathyPack, FactPack, FusionPolicy


class Composer:
    """Compose a single natural response from IQ and EQ artifacts."""

    def compose(self, policy: FusionPolicy, fact_pack: FactPack, empathy_pack: EmpathyPack) -> str:
        fact = (fact_pack.summary or "").strip()
        opening = (empathy_pack.opening or "").strip()
        closing = (empathy_pack.closing or "").strip()

        if policy.order == "EQ_FIRST":
            parts = [p for p in (opening, fact, closing) if p]
        elif policy.order == "IQ_FIRST":
            parts = [p for p in (fact, opening, closing) if p]
        else:
            if opening and fact:
                parts = [opening, fact]
            else:
                parts = [fact or opening]
            if closing:
                parts.append(closing)
        return "\n\n".join(p for p in parts if p).strip()

