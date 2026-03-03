"""
意图裁决器：IQ / EQ / Hybrid 三路路由

设计依据：ai.md §3
- 三维度打分法：任务特征词 / 情感浓度 / 主观意愿
- 独立 LLM 调用（轻量模型，低延迟）
- 共情优先原则：情感信号强烈时优先 Hybrid 而非 Task
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


# ─────────────────────────────────────────────────────────────────────────────
# 路由 Prompt（ai.md §3.2 完整版）
# ─────────────────────────────────────────────────────────────────────────────

_ROUTER_PROMPT = """你是一个意图识别引擎。请分析用户的输入，判断其意图类型。
当前 AI 情绪状态：{emotion_state}

用户输入："{user_input}"

请以 JSON 格式返回（只返回 JSON，不要任何其他内容）：
{{"intent_type": "Task|Chat|Hybrid", "reason": "简要理由（一句话）", "extracted_params": {{}}}}

判断规则：
1. 如果用户明确要求执行动作（查天气、订票、查资料、写代码、翻译、帮我、设置、打开），判定为 Task。
2. 如果用户表达情绪、寻求陪伴、没有明确目的（难过、开心、无聊、烦、讨厌、想聊聊、你觉得、是不是），判定为 Chat。
3. 如果用户在任务中夹杂情感表达（"心情不好，帮我查个天气"、"好烦，帮我写代码"），判定为 Hybrid。

共情优先原则（关键！）：
- 当检测到强烈情绪词（难过、崩溃、好烦、气死了、失恋、焦虑）时，即使有任务，也优先选 Hybrid，不要直接 Task。
- Hybrid 的执行顺序：先共情回应情绪 → 后台执行任务 → EQ 融合润色结果。"""

_STRONG_EMOTION_WORDS = (
    "难过", "崩溃", "好烦", "烦死了", "气死了", "失恋", "焦虑", "抑郁", "绝望", "委屈",
)
_TASK_HINT_WORDS = (
    "查", "查询", "帮我", "帮忙", "设置", "打开", "订", "订票", "写", "翻译", "总结", "执行",
)


class IntentRouter:
    """
    基于独立 LLM 调用的意图裁决器。

    推荐使用轻量模型（如 gpt-4o-mini / deepseek-chat），成本低速度快。
    路由失败时降级为 Chat，不影响主流程。
    """

    def __init__(self, provider: "LLMProvider", model: str):
        self.provider = provider
        self.model    = model

    @staticmethod
    def _first_balanced_json_object(text: str) -> str | None:
        """提取首个花括号平衡的 JSON 对象字符串。"""
        start = text.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escaped = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
            start = text.find("{", start + 1)
        return None

    def _parse_router_json(self, text: str) -> dict | None:
        """
        容错解析路由 JSON：
        1) 直接全文 JSON
        2) ```json 代码块
        3) 首个平衡 JSON 对象
        """
        stripped = text.strip()
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        for block in blocks:
            candidate = self._first_balanced_json_object(block) or block.strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue

        candidate = self._first_balanced_json_object(text)
        if candidate:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return None
        return None

    @staticmethod
    def _contains_any(text: str, words: tuple[str, ...]) -> bool:
        return any(w in text for w in words)

    def _apply_empathy_priority(self, user_input: str, result: dict) -> dict:
        """
        共情优先后处理：
        - LLM 判定为 Task，但输入同时含有强情绪 + 任务信号时，强制升级为 Hybrid。
        """
        raw = user_input.lower()
        if result.get("intent_type") != "Task":
            return result
        has_strong_emotion = self._contains_any(raw, _STRONG_EMOTION_WORDS)
        has_task_hint = self._contains_any(raw, _TASK_HINT_WORDS)
        if has_strong_emotion and has_task_hint:
            result["intent_type"] = "Hybrid"
            reason = result.get("reason", "")
            extra = "触发共情优先：检测到强情绪+任务信号，Task 升级为 Hybrid"
            result["reason"] = f"{reason}；{extra}" if reason else extra
        return result

    async def route(self, user_input: str, emotion_state: str = "平静") -> dict:
        """
        对用户输入进行意图裁决。

        Returns:
            {
                "intent_type": "Task" | "Chat" | "Hybrid",
                "reason":      str,
                "extracted_params": dict,
            }
        """
        prompt = _ROUTER_PROMPT.format(
            emotion_state=emotion_state,
            user_input=user_input[:600],  # 防止超长输入
        )
        try:
            response = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=256,
                temperature=0.1,   # 路由需要确定性，低温度
            )
            text = response.content or ""
            result = self._parse_router_json(text)
            if result:
                intent = result.get("intent_type", "Chat")
                if intent not in ("Task", "Chat", "Hybrid"):
                    intent = "Chat"
                params = result.get("extracted_params", {})
                if not isinstance(params, dict):
                    params = {}
                result["intent_type"] = intent
                result["extracted_params"] = params
                result = self._apply_empathy_priority(user_input, result)
                logger.debug("Intent routed → {} ({})", result.get("intent_type"), result.get("reason", ""))
                return result
        except Exception as e:
            logger.warning("Intent routing failed ({}), fallback → Chat", e)

        return {
            "intent_type":       "Chat",
            "reason":            "路由异常，降级为闲聊",
            "extracted_params":  {},
        }
