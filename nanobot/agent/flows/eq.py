from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.session.manager import Session


def append_assistant_message(session: "Session", content: str) -> None:
    from datetime import datetime as _dt

    session.messages.append(
        {
            "role": "assistant",
            "content": content,
            "timestamp": _dt.now().isoformat(),
        }
    )


async def eq_chat(
    agent: "AgentLoop",
    user_input: str,
    history: list[dict],
    channel: str,
    chat_id: str,
) -> str:
    """EQ 独占路径：纯情感对话，不调任何工具。"""
    emotion_label = agent.emotion_mgr.get_emotion_label()
    pad = agent.emotion_mgr.pad
    messages = agent.context.build_messages(
        history=history,
        current_message=user_input,
        mode="eq",
        current_emotion=emotion_label,
        pad_state=(pad.pleasure, pad.arousal, pad.dominance),
        channel=channel,
        chat_id=chat_id,
    )
    response = await agent.provider.chat(
        messages=messages,
        tools=None,
        model=agent.model,
        temperature=0.7,
        max_tokens=agent.max_tokens,
    )
    return agent._strip_think(response.content) or ""


async def eq_polish(
    agent: "AgentLoop",
    user_input: str,
    iq_result: str,
    style: str = "professional",
) -> str:
    """EQ 渲染层：将 IQ 原始事实用人格润色。"""
    emotion_label = agent.emotion_mgr.get_emotion_label()
    pad = agent.emotion_mgr.pad
    system = agent.context.build_eq_system_prompt(
        query=user_input,
        current_emotion=emotion_label,
        pad_state=(pad.pleasure, pad.arousal, pad.dominance),
    )
    style_guide = {
        "professional": "专业简洁，保持你的性格特征，不要太情绪化",
        "caring": "充满关怀，语气温柔体贴，符合你傲娇心软的性格",
    }.get(style, "保持你的性格特征")

    polish_prompt = (
        "请将以下事实数据，用你的性格转述给用户。\n"
        "**禁止直接输出JSON或技术报错。禁止篡改数据内容。**\n\n"
        f"用户的原始问题：{user_input}\n\n"
        f"事实数据（IQ返回）：\n{iq_result}\n\n"
        f"转述风格：{style_guide}"
    )
    response = await agent.provider.chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": polish_prompt},
        ],
        tools=None,
        model=agent.model,
        temperature=0.7,
        max_tokens=agent.max_tokens,
    )
    polished = agent._strip_think(response.content)
    if polished:
        return polished
    return f"我帮你整理好了关键信息：{iq_result}".strip()


async def eq_empathy(agent: "AgentLoop", user_input: str) -> str:
    """Hybrid Step1：EQ 先行，只回应情绪。"""
    emotion_label = agent.emotion_mgr.get_emotion_label()
    pad = agent.emotion_mgr.pad
    system = agent.context.build_eq_system_prompt(
        query=user_input,
        current_emotion=emotion_label,
        pad_state=(pad.pleasure, pad.arousal, pad.dominance),
    )
    response = await agent.provider.chat(
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"请先对用户情绪做出共情回应（简短1-2句，不要提及任务，不要提数据）：\n{user_input}",
            },
        ],
        tools=None,
        model=agent.model,
        temperature=0.7,
        max_tokens=200,
    )
    return agent._strip_think(response.content) or ""


async def eq_refuse_task(agent: "AgentLoop", user_input: str) -> str:
    """energy < 10 时，EQ 以人格化方式拒绝复杂任务。"""
    emotion_label = agent.emotion_mgr.get_emotion_label()
    pad = agent.emotion_mgr.pad
    system = agent.context.build_eq_system_prompt(
        query=user_input,
        current_emotion=emotion_label,
        pad_state=(pad.pleasure, pad.arousal, pad.dominance),
    )
    response = await agent.provider.chat(
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "你现在精力不足（energy < 10），请用你的性格，委婉地告诉用户现在处理不了复杂任务，"
                    f"建议稍后再试：\n{user_input}"
                ),
            },
        ],
        tools=None,
        model=agent.model,
        temperature=0.7,
        max_tokens=200,
    )
    return agent._strip_think(response.content) or "我现在有点累了，能稍后再说吗..."

