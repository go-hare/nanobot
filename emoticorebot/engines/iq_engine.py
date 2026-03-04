from __future__ import annotations

import json
from typing import TYPE_CHECKING, Awaitable, Callable

from emoticorebot.core.contracts import FactPack, FusionPolicy

if TYPE_CHECKING:
    from emoticorebot.agent.loop import AgentLoop
    from emoticorebot.bus.events import InboundMessage
    from emoticorebot.session.manager import Session


class IQEngine:
    """Task execution engine focused on factual results."""

    async def _iq_execute(
        self,
        agent: "AgentLoop",
        msg: "InboundMessage",
        history: list[dict],
        intent_params: dict | None,
        tool_budget: int | None,
        fact_depth: int | None,
        on_progress: Callable[..., Awaitable[None]] | None,
    ) -> tuple[str, list[dict]]:
        current_message = msg.content
        depth_hint = ""
        if fact_depth is not None:
            if fact_depth <= 1:
                depth_hint = "输出要点结论即可，避免展开过多背景。"
            elif fact_depth >= 3:
                depth_hint = "输出尽量完整，含关键依据、步骤和边界说明。"
            else:
                depth_hint = "输出结论并给出必要依据，保持简洁。"
        if intent_params:
            current_message = (
                f"{msg.content}\n\n"
                "[Router Extracted Params]\n"
                f"{json.dumps(intent_params, ensure_ascii=False)}\n\n"
                "请优先使用以上参数执行任务；若参数不足再根据用户原文补全。"
            )
        if depth_hint:
            current_message = (
                f"{current_message}\n\n"
                "[Fact Depth]\n"
                f"{depth_hint}"
            )
        initial_messages = agent.context.build_messages(
            history=history,
            current_message=current_message,
            mode="iq",
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )
        raw_content, _, all_msgs = await agent._run_agent_loop(
            initial_messages,
            on_progress=on_progress,
            max_tool_calls=tool_budget,
        )
        return raw_content or "", all_msgs

    async def run(
        self,
        agent: "AgentLoop",
        msg: "InboundMessage",
        session: "Session",
        history: list[dict],
        policy: FusionPolicy,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> FactPack:
        metadata = msg.metadata or {}
        intent_params = metadata.get("intent_params")
        if not isinstance(intent_params, dict):
            intent_params = None
        result, all_msgs = await self._iq_execute(
            agent=agent,
            msg=msg,
            history=history,
            intent_params=intent_params,
            tool_budget=policy.tool_budget,
            fact_depth=policy.fact_depth,
            on_progress=on_progress,
        )
        return FactPack(
            summary=result or "",
            confidence=policy.confidence,
            actions_taken=[],
            raw_messages=all_msgs,
        )

