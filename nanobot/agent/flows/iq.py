from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from nanobot.bus.events import InboundMessage
from nanobot.session.manager import Session

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop


async def iq_execute(
    agent: "AgentLoop",
    msg: InboundMessage,
    session: Session,
    history: list[dict],
    intent_params: dict[str, Any] | None = None,
    on_progress: Callable[..., Awaitable[None]] | None = None,
) -> tuple[str, list[dict]]:
    """
    IQ 独占路径：工具调用，返回原始事实结果。
    System Prompt = build_iq_system_prompt（只含冷记忆）。
    """
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
    raw_content, _, all_msgs = await agent._run_agent_loop(initial_messages, on_progress=on_progress)
    return raw_content or "", all_msgs

