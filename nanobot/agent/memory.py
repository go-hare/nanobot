"""Memory system for persistent agent memory.

改造说明（ai.md §4 冷记忆架构）：
- 冷记忆 = MEMORY.md（长期事实） + HISTORY.md（时序日志）
- 新增：get_memory_context(query) — 按段落关键词评分截取，防 Token 爆炸
- 新增：get_relevant_history(query, k) — 时效×关键词评分检索 HISTORY.md
- 冷记忆服务对象：IQ 系统（查事实用）、Hybrid 双系统
- 暖记忆由 warm_memory.py 独立管理（ChromaDB），EQ 系统使用
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "facts": {
                        "type": "array",
                        "description": (
                            "List of individual objective facts extracted from the conversation. "
                            "Each fact is stored as a separate vector entry for semantic retrieval. "
                            "Extract ALL meaningful facts, each as a single self-contained sentence."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "The fact as a complete sentence, e.g. '用户偏好深色主题，不喜欢弹窗'",
                                },
                                "importance": {
                                    "type": "integer",
                                    "description": (
                                        "Importance score 1-10. "
                                        "10=核心身份信息, 8=强偏好/重要项目, 6=一般偏好, 4=临时事件, 2=随口一提"
                                    ),
                                    "minimum": 1,
                                    "maximum": 10,
                                },
                                "category": {
                                    "type": "string",
                                    "enum": ["preference", "project", "habit", "event", "other"],
                                    "description": "Fact category for filtering",
                                },
                            },
                            "required": ["text", "importance"],
                        },
                    },
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph (2-5 sentences) summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]


class MemoryStore:
    """
    冷记忆双层存储：MEMORY.md（长期事实）+ HISTORY.md（时序日志）。

    对应 ai.md §4.1 冷记忆定义：
    - 客观、静态、无情绪色彩
    - 服务对象：IQ 系统（查参数用）、Hybrid 双系统
    - 示例："用户喜欢吃辣"、"用户常在北京"
    """

    def __init__(self, workspace: Path):
        self.memory_dir  = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"

    # ── 基础读写 ──────────────────────────────────────────────────────────────

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    # ── 冷记忆检索（IQ 路径专用）────────────────────────────────────────────

    def get_memory_context(self, query: str = "", max_chars: int = 2000) -> str:
        """
        按查询相关性截取冷记忆，防止 Token 爆炸。

        - 无 query：返回全文（记忆很短时）
        - 有 query：按 ## 段落关键词重叠评分，返回 Top 段落拼接
        """
        long_term = self.read_long_term()
        if not long_term:
            return ""

        # 记忆较短或无查询时直接返回全文
        if len(long_term) <= max_chars or not query:
            return f"## Long-term Memory\n{long_term}"

        # 按 ## 分段，关键词重叠评分
        paragraphs = self._split_by_section(long_term)
        query_words = set(query.lower().split())

        scored = []
        for para in paragraphs:
            overlap = len(query_words & set(para.lower().split()))
            scored.append((overlap, para))

        # 按相关性降序，累积至 max_chars
        scored.sort(key=lambda x: x[0], reverse=True)
        selected, total = [], 0
        for _, para in scored:
            if total + len(para) > max_chars:
                break
            selected.append(para)
            total += len(para)

        result = "\n\n".join(selected)
        return f"## Long-term Memory\n{result}" if result else ""

    def get_relevant_history(self, query: str = "", k: int = 5) -> str:
        """
        从 HISTORY.md 检索与当前 query 最相关的历史条目。

        评分 = 时效性×0.4 + 关键词相关度×0.6
        时效性公式：0.99^小时数（与 ai.md §1.3 一致）
        """
        if not self.history_file.exists():
            return ""

        raw     = self.history_file.read_text(encoding="utf-8")
        entries = [e.strip() for e in raw.split("\n\n") if e.strip()]
        if not entries:
            return ""

        query_words = set(query.lower().split()) if query else set()
        now         = datetime.now()
        ts_pattern  = re.compile(r'\[(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2})\]')

        scored = []
        for entry in entries:
            # 时效性
            m = ts_pattern.search(entry)
            if m:
                try:
                    ts    = datetime.fromisoformat(m.group(1).replace(" ", "T"))
                    hours = (now - ts).total_seconds() / 3600
                    recency = 0.99 ** hours
                except ValueError:
                    recency = 0.5
            else:
                recency = 0.3

            # 关键词相关度
            if query_words:
                entry_words = set(entry.lower().split())
                overlap     = len(query_words & entry_words)
                relevance   = min(overlap / max(len(query_words), 1), 1.0)
            else:
                relevance = 0.5

            final_score = 0.4 * recency + 0.6 * relevance
            scored.append((final_score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [e for _, e in scored[:k]]
        return "\n\n".join(top) if top else ""

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _split_by_section(text: str) -> list[str]:
        """按 ## 标题切段，保留标题与内容为整块。"""
        sections = re.split(r'\n(?=##)', text)
        return [s.strip() for s in sections if s.strip()]

    # ── 记忆压缩（反思机制）──────────────────────────────────────────────────

    async def consolidate(
        self,
        session: "Session",
        provider: "LLMProvider",
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
        cold_store: "Any | None" = None,
    ) -> bool:
        """
        将旧对话压缩写入 MEMORY.md + HISTORY.md（冷记忆写入），
        同时将 facts 写入 ColdMemoryStore 向量库（斯坦福全量向量化）。

        这是斯坦福小镇反思机制的冷记忆侧实现：
        - MEMORY.md：长期事实 Markdown（人类可读备份）
        - HISTORY.md：时序日志（带时间戳的事件记录，可 grep）
        - ColdMemoryStore：每条事实独立向量化，带 importance 评分

        :param cold_store: ColdMemoryStore 实例，传入时写向量库；None 时仅写文件
        Returns True on success, False on failure.
        """
        if archive_all:
            old_messages = session.messages
            keep_count   = 0
            logger.info("Memory consolidation (archive_all): {} messages", len(session.messages))
        else:
            keep_count    = memory_window // 2
            if len(session.messages) <= keep_count:
                return True
            if len(session.messages) - session.last_consolidated <= 0:
                return True
            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return True
            logger.info("Memory consolidation: {} to consolidate, {} keep",
                        len(old_messages), keep_count)

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")

        current_memory = self.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.
Focus on extracting objective facts about the user (preferences, projects, habits) for long-term memory.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{chr(10).join(lines)}"""

        try:
            response = await provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. "
                     "Call the save_memory tool with your consolidation of the conversation. "
                     "Focus on facts and objective information."},
                    {"role": "user", "content": prompt},
                ],
                tools=_SAVE_MEMORY_TOOL,
                model=model,
            )

            if not response.has_tool_calls:
                logger.warning("Memory consolidation: LLM did not call save_memory, skipping")
                return False

            args = response.tool_calls[0].arguments
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                logger.warning("Memory consolidation: unexpected args type {}", type(args).__name__)
                return False

            if entry := args.get("history_entry"):
                if not isinstance(entry, str):
                    entry = json.dumps(entry, ensure_ascii=False)
                self.append_history(entry)
            if update := args.get("memory_update"):
                if not isinstance(update, str):
                    update = json.dumps(update, ensure_ascii=False)
                if update != current_memory:
                    self.write_long_term(update)

            # ── 写入冷记忆向量库（斯坦福全量向量化）────────────────────────
            if cold_store is not None:
                facts = args.get("facts", [])
                if isinstance(facts, list):
                    for fact in facts:
                        if not isinstance(fact, dict):
                            continue
                        text = fact.get("text", "")
                        if not text:
                            continue
                        cold_store.save(
                            text       = text,
                            importance = int(fact.get("importance", 5)),
                            category   = str(fact.get("category", "other")),
                        )
                    if facts:
                        logger.info("ColdMemoryStore: {} facts written to vector DB", len(facts))

            session.last_consolidated = (
                0 if archive_all else len(session.messages) - keep_count
            )
            logger.info("Memory consolidation done: last_consolidated={}",
                        session.last_consolidated)
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return False
