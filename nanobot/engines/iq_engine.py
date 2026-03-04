from __future__ import annotations

import json
from typing import TYPE_CHECKING, Awaitable, Callable

from nanobot.core.contracts import FactPack, FusionPolicy

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.events import InboundMessage
    from nanobot.session.manager import Session


class IQEngine:
    """Task execution engine focused on factual results."""

    async def _iq_execute(
        self,
        agent: "AgentLoop",
        msg: "InboundMessage",
        history: list[dict],
        intent_params: dict | None,
        on_progress: Callable[..., Awaitable[None]] | None,
    ) -> tuple[str, list[dict]]:
        current_message = msg.content
        if intent_params:
            current_message = (
                f"{msg.content}\n\n"
                "[Router Extracted Params]\n"
                f"{json.dumps(intent_params, ensure_ascii=False)}\n\n"
                "请优先使用以上参数执行任务；若参数不足再根据用户原文补全。"
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
            initial_messages, on_progress=on_progress
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
        result, all_msgs = await self._iq_execute(
            agent=agent,
            msg=msg,
            history=history,
            intent_params={},
            on_progress=on_progress,
        )
        return FactPack(
            summary=result or "",
            confidence=policy.confidence,
            actions_taken=[],
            raw_messages=all_msgs,
        )

