from __future__ import annotations

from typing import TYPE_CHECKING

from nanobot.core.contracts import EmpathyPack, FactPack, FusionPolicy

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.events import InboundMessage


class EQEngine:
    """Emotion expression engine for empathy and tone rendering."""

    async def _eq_chat(
        self,
        agent: "AgentLoop",
        user_input: str,
        history: list[dict],
        channel: str,
        chat_id: str,
    ) -> str:
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

    async def _eq_empathy(self, agent: "AgentLoop", user_input: str) -> str:
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

    async def _eq_polish(
        self,
        agent: "AgentLoop",
        user_input: str,
        iq_result: str,
        style: str = "professional",
    ) -> str:
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
            "concise": "优先给结论，句子更短，减少寒暄，但保留基本礼貌与温度",
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

    async def build_pack(
        self,
        agent: "AgentLoop",
        msg: "InboundMessage",
        history: list[dict],
        policy: FusionPolicy,
    ) -> EmpathyPack:
        opening = ""
        closing = ""
        if policy.empathy_depth >= 1:
            opening = await self._eq_empathy(agent, msg.content)
        if policy.empathy_depth >= 2 and not opening:
            opening = await self._eq_chat(agent, msg.content, history, msg.channel, msg.chat_id)
        if policy.empathy_depth >= 1 and opening:
            closing = "我会陪你把这件事处理好。"
        return EmpathyPack(opening=opening or "", closing=closing)

    async def polish(
        self,
        agent: "AgentLoop",
        user_input: str,
        fact_pack: FactPack,
        policy: FusionPolicy,
    ) -> str:
        if policy.tone == "concise":
            style = "concise"
        else:
            style = "caring" if policy.eq_weight >= 0.6 else "professional"
        return await self._eq_polish(agent, user_input, fact_pack.summary, style=style)

