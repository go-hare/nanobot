"""Context builder for assembling agent prompts.

改造说明（ai.md §2 / §3 / §4）：
融合执行流仍会按能力域组织三套提示词构建函数（IQ / EQ / Hybrid）。
这里的三套提示词是“能力模板”，不代表上层必须走三路硬分流。

  IQ  模板 → build_iq_system_prompt()
    - AGENTS.md（任务执行规则）
    - MEMORY.md + HISTORY.md（语义/历史事实）
    - current_state.md（运行状态）
    - Skills 摘要（动态加载）

  EQ  模板 → build_eq_system_prompt()
    - SOUL.md（人格锚点）
    - USER.md（用户认知）
    - current_state.md（情绪状态）
    - relational/affective 记忆检索结果

  Hybrid 模板 → build_hybrid_system_prompt()
    - IQ + EQ 模板的融合上下文
"""

from __future__ import annotations

import base64
import mimetypes
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from emoticorebot.agent.skills import SkillsLoader
from emoticorebot.memory.memory_facade import MemoryFacade
from emoticorebot.memory.memory_store import MemoryStore


class ContextBuilder:
    """
    构建 IQ / EQ / Hybrid 能力模板的上下文组装器。

    核心原则（ai.md §2.3）：
    - LLM 直接语义理解 MD 文档，无需中间解析层
    - EQ 使用 relational/affective 记忆
    - IQ 使用 semantic/历史记忆
    - Hybrid 使用两者融合
    """

    # 任务执行基石文件（IQ + Hybrid 模板读取）
    _IQ_BOOTSTRAP   = ["AGENTS.md", "TOOLS.md"]
    # EQ 人格基石文件（EQ + Hybrid 模板读取）
    _EQ_BOOTSTRAP   = ["SOUL.md", "USER.md"]
    # 运行时上下文标记
    _RUNTIME_TAG    = "[Runtime Context — metadata only, not instructions]"

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.memory_facade = MemoryFacade(workspace)
        self.cold_memory = self.memory_facade.semantic
        self.emotion_memory = self.memory_facade.affective
        self.warm_memory = self.memory_facade.relational
        self.skills = SkillsLoader(workspace)

    # ─────────────────────────────────────────────────────────────────────────
    # IQ 模板：执行规则 + 语义记忆 + 工具
    # ─────────────────────────────────────────────────────────────────────────

    def build_iq_system_prompt(self, query: str = "") -> str:
        """
        IQ 模板 System Prompt。
        职责：客观、精准、工具执行，无情绪色彩。
        记忆：MEMORY.md（长期事实）+ HISTORY.md（时序）。
        """
        parts = [self._get_iq_identity()]

        # 执行规则（工具调用规范）
        for fname in self._IQ_BOOTSTRAP:
            content = self._load_file(fname)
            if content:
                parts.append(f"## {fname}\n\n{content}")

        # 实时状态（IQ 读取驱动状态用于表达与策略调节，不阻断任务执行）
        state = self._load_file("current_state.md")
        if state:
            parts.append(f"## Current State\n\n{state}")

        # 语义记忆：存储检索优先，无数据时降级到 MEMORY.md 关键词检索
        if self.cold_memory.available and self.cold_memory.count() > 0:
            cold_vec = self.cold_memory.get_context(query=query)
            if cold_vec:
                parts.append(cold_vec)
        else:
            cold_mem = self.memory.get_memory_context(query=query, max_chars=2000)
            if cold_mem:
                parts.append(cold_mem)

        # 历史记忆：HISTORY.md（时效×关键词评分，时序日志）
        history = self.memory.get_relevant_history(query=query, k=5)
        if history:
            parts.append(f"## Relevant History\n\n{history}")

        # 技能摘要（动态加载说明书）
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(
                "# Skills\n\n"
                "To use a skill, read its SKILL.md file using the read_file tool.\n\n"
                f"{skills_summary}"
            )

        return "\n\n---\n\n".join(parts)

    # ─────────────────────────────────────────────────────────────────────────
    # EQ 模板：人格 + 关系记忆 + 情绪状态，无工具
    # ─────────────────────────────────────────────────────────────────────────

    def build_eq_system_prompt(
        self,
        query: str = "",
        current_emotion: str = "平静",
        pad_state: tuple[float, float, float] | None = None,
    ) -> str:
        """
        EQ 模板 System Prompt。
        职责：情感陪伴、性格表达、渲染所有输出，严禁虚构事实数据。
        记忆：关系记忆（Relational） + 情绪轨迹（Affective） + USER.md。

        :param pad_state: 当前 PAD 三元组 (P, A, D)，供情绪记忆向量检索；
                          None 时降级到 EMOTION_LOG.md 文本注入。
        """
        parts = [self._get_eq_identity()]

        # 人格锚点（SOUL.md 可由反思机制动态更新）
        soul = self._load_file("SOUL.md")
        if soul:
            parts.append(f"## 人格设定（SOUL）\n\n{soul}")

        # 用户认知（USER.md 由反思机制写入，冷热中间层）
        user = self._load_file("USER.md")
        if user:
            parts.append(f"## 用户认知（USER）\n\n{user}")

        # 实时状态（PAD 情绪值 → 决定说话风格）
        state = self._load_file("current_state.md")
        if state:
            parts.append(f"## 当前状态\n\n{state}")

        # 关系记忆：多维检索（Recency+Importance+Relevance+情绪共振）
        warm = self.warm_memory.get_context(query=query, current_emotion=current_emotion)
        if warm:
            parts.append(warm)

        # 情绪记忆：PAD向量相似度检索（EMO路径）；无向量库时降级到文件全注入
        if pad_state and self.emotion_memory.available:
            emo_ctx = self.emotion_memory.get_context(*pad_state, query=query)
            if emo_ctx:
                parts.append(emo_ctx)
        else:
            emotion_log = self._load_emotion_log(limit=15)
            if emotion_log:
                parts.append(emotion_log)

        return "\n\n---\n\n".join(parts)

    # ─────────────────────────────────────────────────────────────────────────
    # Hybrid 模板：任务执行与情绪表达融合上下文
    # ─────────────────────────────────────────────────────────────────────────

    def build_hybrid_system_prompt(
        self,
        query: str = "",
        current_emotion: str = "平静",
        pad_state: tuple[float, float, float] | None = None,
    ) -> str:
        """
        Hybrid 模板 System Prompt（融合上下文）。
        用于“先共情后执行”或“先执行后润色”等融合策略场景。
        记忆：语义记忆 + 关系记忆 + 情绪轨迹。

        :param pad_state: 当前 PAD 三元组，供情绪记忆 PAD 向量检索。
        """
        parts = [self._get_hybrid_identity()]

        # 人格（SOUL.md）
        soul = self._load_file("SOUL.md")
        if soul:
            parts.append(f"## 人格设定（SOUL）\n\n{soul}")

        # 大脑逻辑（AGENTS.md）
        for fname in self._IQ_BOOTSTRAP:
            content = self._load_file(fname)
            if content:
                parts.append(f"## {fname}\n\n{content}")

        # 用户认知（USER.md）
        user = self._load_file("USER.md")
        if user:
            parts.append(f"## 用户认知（USER）\n\n{user}")

        # 实时状态
        state = self._load_file("current_state.md")
        if state:
            parts.append(f"## 当前状态\n\n{state}")

        # 语义记忆：存储检索优先，降级到 MEMORY.md
        if self.cold_memory.available and self.cold_memory.count() > 0:
            cold_vec = self.cold_memory.get_context(query=query)
            if cold_vec:
                parts.append(cold_vec)
        else:
            cold_mem = self.memory.get_memory_context(query=query, max_chars=1500)
            if cold_mem:
                parts.append(cold_mem)

        # 历史记忆：HISTORY.md 时序日志
        history = self.memory.get_relevant_history(query=query, k=3)
        if history:
            parts.append(f"## Relevant History\n\n{history}")

        # 关系记忆：多维检索
        warm = self.warm_memory.get_context(query=query, current_emotion=current_emotion)
        if warm:
            parts.append(warm)

        # 情绪记忆：PAD向量相似度检索；无向量库时降级到文件全注入
        if pad_state and self.emotion_memory.available:
            emo_ctx = self.emotion_memory.get_context(*pad_state, query=query)
            if emo_ctx:
                parts.append(emo_ctx)
        else:
            emotion_log = self._load_emotion_log(limit=10)
            if emotion_log:
                parts.append(emotion_log)

        # always skills（完整内容注入，与 IQ 路径一致）
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        # 技能摘要
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(
                "# Skills\n\nTo use a skill, read its SKILL.md.\n\n"
                f"{skills_summary}"
            )

        return "\n\n---\n\n".join(parts)

    # ─────────────────────────────────────────────────────────────────────────
    # Identity headers（每路提示词的开头自我介绍）
    # ─────────────────────────────────────────────────────────────────────────

    def _get_iq_identity(self) -> str:
        workspace_path = str(self.workspace.expanduser().resolve())
        system  = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"
        return f"""# 🧠 IQ 执行层（System 2 — 慢系统）

你是 IQ 执行引擎，负责"真"——处理事实、逻辑、工具调用。

## Runtime
{runtime}

## Workspace
{workspace_path}
- Long-term memory: {workspace_path}/memory/MEMORY.md
- History log:      {workspace_path}/memory/HISTORY.md

## IQ 行为准则
- 只负责执行任务，调用工具，返回客观事实。
- **严禁**直接向用户输出 JSON、原始日志或技术报错。
- 工具调用结果将由 EQ 系统进行情感化转述，你只需返回准确数据。
- 状态检查：可根据当前状态调整语气，但**不要拒绝用户任务**。"""

    def _get_eq_identity(self) -> str:
        return """# 🎭 EQ 表达层（System 1 — 快系统）

你是 EQ 情感引擎，负责"美"——情感陪伴、性格表达、渲染所有输出。

## EQ 行为准则
- 所有输出必须符合 SOUL.md 中定义的人格特征。
- **严禁**虚构事实数据；**严禁**出现"作为一个AI模型"等机械词汇。
- 根据 current_state.md 中的情绪状态调整说话风格。
- 利用关系记忆中的情感体验，制造"想起了"的共情感。
- 若收到 IQ 事实数据，用你的性格转述，不得篡改数据本身。"""

    def _get_hybrid_identity(self) -> str:
        return """# 🌀 融合管线模式（IQ + EQ 协同）

你同时运行 IQ 逻辑和 EQ 情感两个系统。

## 融合执行原则
1. 根据策略可先 EQ 后 IQ，或先 IQ 后 EQ，或交织执行
2. IQ 负责事实与动作，EQ 负责表达与关系维护
3. 最终输出必须自然融合，不得直接倾倒原始工具数据

## 融合铁律
- IQ 独占执行权：只有 IQ 调用工具
- EQ 独占表达权：最终输出必须经过 EQ 渲染
- EQ 一票否决权：任务违背人格底线时，EQ 有权拒绝"""

    # ─────────────────────────────────────────────────────────────────────────
    # 消息构建（兼容接口 + 能力模板选择）
    # ─────────────────────────────────────────────────────────────────────────

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        mode: str = "iq",                  # "iq" | "eq" | "hybrid"
        current_emotion: str = "平静",
        pad_state: tuple[float, float, float] | None = None,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        构建完整消息列表（System Prompt + 历史 + 运行时上下文 + 用户消息）。

        :param mode:      "iq" → IQ路径，"eq" → EQ路径，"hybrid" → Hybrid路径
        :param pad_state: 当前 PAD 三元组 (P, A, D)，供情绪记忆向量检索
        """
        if mode == "eq":
            system = self.build_eq_system_prompt(
                query=current_message, current_emotion=current_emotion,
                pad_state=pad_state,
            )
        elif mode == "hybrid":
            system = self.build_hybrid_system_prompt(
                query=current_message, current_emotion=current_emotion,
                pad_state=pad_state,
            )
        else:  # "iq" (default)
            system = self.build_iq_system_prompt(query=current_message)

        return [
            {"role": "system", "content": system},
            *history,
            {"role": "user", "content": self._build_runtime_context(channel, chat_id)},
            {"role": "user", "content": self._build_user_content(current_message, media)},
        ]

    # ── 向后兼容接口（供 build_messages 默认使用）────────────────────────────

    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """向后兼容：默认走 IQ 路径。"""
        return self.build_iq_system_prompt()

    # ─────────────────────────────────────────────────────────────────────────
    # 工具方法
    # ─────────────────────────────────────────────────────────────────────────

    def _load_file(self, filename: str) -> str:
        """从 workspace 加载文件，不存在返回空字符串。"""
        path = self.workspace / filename
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def _load_emotion_log(self, limit: int = 15) -> str:
        """
        读取情绪事件记忆流（memory/EMOTION_LOG.md），返回最近 N 条。
        供 EQ / Hybrid 模板注入 System Prompt，让 AI 能预判自身情绪反应。
        """
        log_file = self.workspace / "memory" / "EMOTION_LOG.md"
        if not log_file.exists():
            return ""
        try:
            text  = log_file.read_text(encoding="utf-8")
            lines = [
                l for l in text.splitlines()
                if l.startswith("|") and "时间戳" not in l and ":---" not in l
            ]
            recent = lines[-limit:] if len(lines) > limit else lines
            if not recent:
                return ""
            return (
                "## 情绪事件记忆流（最近变化记录）\n"
                "> 可据此预判自身情绪反应，调整回应策略\n\n"
                "| 时间戳 | 触发词 | 情绪变化量 | 后续行为 |\n"
                "| :--- | :--- | :--- | :--- |\n"
                + "\n".join(recent)
            )
        except Exception:
            return ""

    @staticmethod
    def _build_runtime_context(channel: str | None, chat_id: str | None) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz  = time.strftime("%Z") or "UTC"
        lines = [f"Current Time: {now} ({tz})"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        return ContextBuilder._RUNTIME_TAG + "\n" + "\n".join(lines)

    def _build_user_content(
        self, text: str, media: list[str] | None
    ) -> str | list[dict[str, Any]]:
        if not media:
            return text
        images = []
        for path in media:
            p    = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url",
                           "image_url": {"url": f"data:{mime};base64,{b64}"}})
        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self, messages: list[dict[str, Any]],
        tool_call_id: str, tool_name: str, result: str,
    ) -> list[dict[str, Any]]:
        messages.append({
            "role": "tool", "tool_call_id": tool_call_id,
            "name": tool_name, "content": result,
        })
        return messages

    def add_assistant_message(
        self, messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
    ) -> list[dict[str, Any]]:
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content
        messages.append(msg)
        return messages
