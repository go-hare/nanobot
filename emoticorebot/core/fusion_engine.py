from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Awaitable, Callable

from loguru import logger

from emoticorebot.core.composer import Composer
from emoticorebot.core.contracts import EmpathyPack, FactPack
from emoticorebot.core.policy_engine import PolicyEngine
from emoticorebot.core.signal_extractor import SignalExtractor
from emoticorebot.engines.eq_engine import EQEngine
from emoticorebot.engines.iq_engine import IQEngine
from emoticorebot.memory.memory_facade import MemoryFacade

if TYPE_CHECKING:
    from emoticorebot.agent.loop import AgentLoop
    from emoticorebot.bus.events import InboundMessage
    from emoticorebot.session.manager import Session


class FusionEngine:
    """Single pipeline orchestrator for IQ/EQ fusion."""

    def __init__(self, workspace, fusion_config: dict | None = None) -> None:
        self.signals = SignalExtractor()
        self.policy = PolicyEngine(fusion_config=fusion_config)
        self.iq = IQEngine()
        self.eq = EQEngine()
        self.composer = Composer()
        self.memory = MemoryFacade(workspace)

    async def run_turn(
        self,
        agent: "AgentLoop",
        msg: "InboundMessage",
        session: "Session",
        history: list[dict],
        progress_cb: Callable[..., Awaitable[None]] | None = None,
    ) -> str:
        emotion_prompt = agent.emotion_mgr.get_emotion_prompt()
        turn_signals = self.signals.extract(msg.content, emotion_state=emotion_prompt)
        runtime_adjustment = self.memory.load_policy_adjustment()
        policy = self.policy.make_policy(
            turn_signals, runtime_adjustment=runtime_adjustment
        )
        policy = self._apply_energy_expression_policy(agent, policy)
        logger.info(
            "Fusion policy: iq={:.2f} eq={:.2f} order={} conf={:.2f} ({})",
            policy.iq_weight,
            policy.eq_weight,
            policy.order,
            policy.confidence,
            turn_signals.reason,
        )

        fact_pack = FactPack(summary="", confidence=policy.confidence, actions_taken=[], raw_messages=[])
        empathy_pack = EmpathyPack(opening="", closing="")
        # 任务显著时，强制进入 IQ 执行，保证“任务必执行”。
        should_run_iq = policy.iq_weight >= 0.2 or turn_signals.task_strength >= 0.35

        async def _run_iq_if_needed() -> None:
            nonlocal fact_pack
            if not should_run_iq:
                return
            fact_pack = await self.iq.run(
                agent=agent,
                msg=msg,
                session=session,
                history=history,
                policy=policy,
                on_progress=progress_cb,
            )
            if fact_pack.raw_messages:
                agent._save_turn(session, fact_pack.raw_messages, 1 + len(history))

        async def _run_eq_pack() -> None:
            nonlocal empathy_pack
            empathy_pack = await self.eq.build_pack(
                agent=agent, msg=msg, history=history, policy=policy
            )

        if policy.order == "EQ_FIRST":
            await _run_eq_pack()
            await _run_iq_if_needed()
        elif policy.order == "IQ_FIRST":
            await _run_iq_if_needed()
            await _run_eq_pack()
        else:
            await _run_iq_if_needed()
            await _run_eq_pack()

        polished = ""
        if fact_pack.summary:
            polished = await self.eq.polish(
                agent=agent,
                user_input=msg.content,
                fact_pack=fact_pack,
                policy=policy,
            )
        final_text = self.composer.compose(
            policy=policy,
            fact_pack=FactPack(
                summary=polished or fact_pack.summary,
                confidence=fact_pack.confidence,
                actions_taken=fact_pack.actions_taken,
                raw_messages=[],
            ),
            empathy_pack=empathy_pack,
        )
        return final_text or polished or fact_pack.summary or empathy_pack.opening or ""

    @staticmethod
    def _apply_energy_expression_policy(agent: "AgentLoop", policy):
        """
        Energy only impacts expression pacing, never task execution.
        """
        energy = float(getattr(getattr(agent.emotion_mgr, "drive", None), "energy", 100.0))
        if energy > 20:
            return policy
        return replace(
            policy,
            empathy_depth=max(0, int(policy.empathy_depth) - 1),
            tone="concise",
        )
