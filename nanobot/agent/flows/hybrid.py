from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable

from nanobot.agent.flows.eq import append_assistant_message, eq_empathy
from nanobot.agent.flows.iq import iq_execute
from nanobot.bus.events import InboundMessage, OutboundMessage
from loguru import logger
from nanobot.session.manager import Session

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop, RoutingDecision


async def hybrid_polish(agent: "AgentLoop", user_input: str, iq_result: str) -> str:
    """Hybrid 第三步：在冷暖混合上下文中做 EQ 融合转述。"""
    emotion_label = agent.emotion_mgr.get_emotion_label()
    pad = agent.emotion_mgr.pad
    hybrid_system = agent.context.build_hybrid_system_prompt(
        query=user_input,
        current_emotion=emotion_label,
        pad_state=(pad.pleasure, pad.arousal, pad.dominance),
    )
    polish_prompt = (
        f"用户问：{user_input}\n\n"
        "你已经先对用户情绪表达了关心。\n"
        "现在请将以下事实数据，用你充满关怀又傲娇的性格融合转述，"
        f"接着你之前的关心继续说：\n\n{iq_result}"
    )
    response = await agent.provider.chat(
        messages=[
            {"role": "system", "content": hybrid_system},
            {"role": "user", "content": polish_prompt},
        ],
        tools=None,
        model=agent.model,
        temperature=0.7,
        max_tokens=agent.max_tokens,
    )
    polished = agent._strip_think(response.content)
    return polished or f"我把结果整理好了：{iq_result}".strip()


async def execute_hybrid_path(
    agent: "AgentLoop",
    msg: InboundMessage,
    session: Session,
    history: list[dict],
    decision: "RoutingDecision",
    progress_cb: Callable[..., Awaitable[None]] | None,
) -> str:
    logger.info("Hybrid path → EQ empathy → IQ execute → EQ polish (cold+warm)")
    empathy = await eq_empathy(agent, msg.content)
    if empathy and decision.progress_enabled:
        await agent.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=empathy,
                metadata={**(msg.metadata or {}), "_progress": True},
            )
        )

    iq_result, all_msgs = await iq_execute(
        agent,
        msg,
        session,
        history,
        intent_params=decision.intent_params,
        on_progress=progress_cb,
    )
    agent._save_turn(session, all_msgs, 1 + len(history))
    final_content = await hybrid_polish(agent, msg.content, iq_result)
    if empathy and not decision.progress_enabled:
        final_content = f"{empathy}\n{final_content}".strip()
    append_assistant_message(session, final_content)
    return final_content

