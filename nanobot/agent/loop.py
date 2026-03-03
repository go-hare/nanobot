"""Agent loop: the core processing engine.

改造说明（ai.md §3 / §6 / §8）：双螺旋执行流
────────────────────────────────────────────────
意图路由 → Task / Chat / Hybrid

  Chat   路径：EQ独占 → _eq_chat()
               System Prompt = build_eq_system_prompt（暖记忆）
               无工具调用，temperature=0.7

  Task   路径：IQ执行 → _iq_execute()
               System Prompt = build_iq_system_prompt（冷记忆）
               带工具，执行完毕后
          EQ润色 → _eq_polish(style="professional")
               System Prompt = build_eq_system_prompt（暖记忆）

  Hybrid 路径：
    Step1 EQ共情 → _eq_empathy()（立即通过 progress 推送给用户）
    Step2 IQ执行 → _iq_execute()（build_iq_system_prompt，冷记忆+工具）
    Step3 EQ融合 → _eq_polish(style="caring")（build_hybrid_system_prompt，冷+暖）

对话结束后：
  - PAD 状态更新（emotion_mgr.update_from_conversation）
  - 冷记忆写入（memory.consolidate，异步后台）
  - 暖记忆写入（warm_memory.save，即时）
────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.flows.eq import (
    append_assistant_message as eq_append_assistant_message,
    eq_chat as eq_chat_flow,
    eq_polish as eq_polish_flow,
)
from nanobot.agent.flows.hybrid import (
    execute_hybrid_path as execute_hybrid_path_flow,
)
from nanobot.agent.flows.iq import iq_execute as iq_execute_flow
from nanobot.agent.memory import MemoryStore
from nanobot.agent.router import IntentRouter
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.agent.warm_memory import WarmMemoryStore
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.models.emotion_state import EmotionStateManager
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig
    from nanobot.cron.service import CronService


@dataclass(frozen=True)
class RoutingDecision:
    """路由层输出的统一决策对象。"""
    intent_type: str
    intent_params: dict[str, Any]
    emotion_label: str
    progress_enabled: bool


class AgentLoop:
    """
    双螺旋 Agent 主循环。

    核心流程：
    1. 从 bus 接收消息
    2. 路由意图（IntentRouter：Task / Chat / Hybrid）
    3. 按路由分流执行（IQ 工具链 / EQ 情感 / Hybrid 共情+执行+润色）
    4. 冷热双写记忆（MEMORY.md + ChromaDB）
    5. 更新 PAD 情绪状态
    6. 发回响应
    """

    _TOOL_RESULT_MAX_CHARS = 500

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        memory_window: int = 100,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: "ChannelsConfig | None" = None,
    ):
        from nanobot.config.schema import ExecToolConfig as _ETC
        self.bus              = bus
        self.channels_config  = channels_config
        self.provider         = provider
        self.workspace        = workspace
        self.model            = model or provider.get_default_model()
        self.max_iterations   = max_iterations
        self.temperature      = temperature
        self.max_tokens       = max_tokens
        self.memory_window    = memory_window
        self.brave_api_key    = brave_api_key
        self.exec_config      = exec_config or _ETC()
        self.cron_service     = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        # ── 双螺旋新增模块 ───────────────────────────────────────────────────
        self.emotion_mgr = EmotionStateManager(workspace)
        self._router     = IntentRouter(provider, self.model)
        self.warm_memory = WarmMemoryStore(workspace)
        # ────────────────────────────────────────────────────────────────────

        self.context  = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools    = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running             = False
        self._mcp_servers         = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected       = False
        self._mcp_connecting      = False
        self._consolidating: set[str] = set()
        self._consolidation_tasks: set[asyncio.Task] = set()
        self._consolidation_locks: dict[str, asyncio.Lock] = {}
        self._active_tasks: dict[str, list[asyncio.Task]] = {}
        self._processing_lock = asyncio.Lock()
        # 由启动层注入（cli.commands），用于处理潜意识系统内部指令
        self.subconscious_daemon = None
        self._register_default_tools()

    # ─────────────────────────────────────────────────────────────────────────
    # 工具注册（与原版一致）
    # ─────────────────────────────────────────────────────────────────────────

    def _register_default_tools(self) -> None:
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def _connect_mcp(self) -> None:
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error("Failed to connect MCP servers: {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        def _fmt(tc):
            val = next(iter(tc.arguments.values()), None) if tc.arguments else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    # ─────────────────────────────────────────────────────────────────────────
    # 核心 Agent 迭代循环（IQ 工具链，与原版一致）
    # ─────────────────────────────────────────────────────────────────────────

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        messages       = initial_messages
        iteration      = 0
        final_content  = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if response.has_tool_calls:
                if on_progress:
                    clean = self._strip_think(response.content)
                    if clean:
                        await on_progress(clean)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name,
                                  "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )
                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    result   = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                clean    = self._strip_think(response.content)
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations})."
            )
        return final_content, tools_used, messages

    # ─────────────────────────────────────────────────────────────────────────
    # 记忆双写（冷热并行）
    # ─────────────────────────────────────────────────────────────────────────

    def _write_warm_memory(
        self,
        user_input: str,
        ai_response: str,
        emotion_label: str,
    ) -> None:
        """
        写入暖记忆（ai.md §4.2 EQ通道）。
        格式：对话摘要 + 情绪标签 + 重要性。
        """
        summary = (
            f"用户：{user_input[:120]}"
            f"{'...' if len(user_input) > 120 else ''}"
            f" → AI：{ai_response[:120]}"
            f"{'...' if len(ai_response) > 120 else ''}"
        )
        # 重要性启发：含强情绪词时评 7，否则 5
        importance = 7 if any(
            w in user_input for w in ["失恋", "难过", "崩溃", "好烦", "开心", "谢谢"]
        ) else 5
        self.warm_memory.save(summary, emotion=emotion_label, importance=importance)
        logger.debug("WarmMemory written: emotion={}", emotion_label)

    # ─────────────────────────────────────────────────────────────────────────
    # Session 存储
    # ─────────────────────────────────────────────────────────────────────────

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        from datetime import datetime
        for m in messages[skip:]:
            entry = {k: v for k, v in m.items() if k != "reasoning_content"}
            if entry.get("role") == "tool" and isinstance(entry.get("content"), str):
                content = entry["content"]
                if len(content) > self._TOOL_RESULT_MAX_CHARS:
                    entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            if entry.get("role") == "user" and isinstance(entry.get("content"), list):
                entry["content"] = [
                    {"type": "text", "text": "[image]"} if (
                        c.get("type") == "image_url"
                        and c.get("image_url", {}).get("url", "").startswith("data:image/")
                    ) else c
                    for c in entry["content"]
                ]
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    def _save_simple_turn(
        self,
        session: Session,
        user_content: str,
        assistant_content: str,
    ) -> None:
        """Chat 路径（无工具调用）直接存 session，保持历史连贯。"""
        from datetime import datetime
        now = datetime.now().isoformat()
        session.messages.append({"role": "user",      "content": user_content,      "timestamp": now})
        session.messages.append({"role": "assistant",  "content": assistant_content, "timestamp": now})
        session.updated_at = datetime.now()

    async def _consolidate_memory(self, session: Session, archive_all: bool = False) -> bool:
        return await MemoryStore(self.workspace).consolidate(
            session, self.provider, self.model,
            archive_all=archive_all,
            memory_window=self.memory_window,
            cold_store=self.context.cold_memory,   # 全量向量化：facts 写入 ColdMemoryStore
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 主循环
    # ─────────────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started (双螺旋模式)")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if msg.content.strip().lower() == "/stop":
                await self._handle_stop(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                task.add_done_callback(
                    lambda t, k=msg.session_key: (
                        self._active_tasks.get(k, []) and
                        t in self._active_tasks[k] and
                        self._active_tasks[k].remove(t)
                    )
                )

    async def _handle_stop(self, msg: InboundMessage) -> None:
        tasks     = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total   = cancelled + sub_cancelled
        content = f"⏹ Stopped {total} task(s)." if total else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _dispatch(self, msg: InboundMessage) -> None:
        async with self._processing_lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass
            self._mcp_stack = None

    def stop(self) -> None:
        self._running = False
        logger.info("Agent loop stopping")

    # ─────────────────────────────────────────────────────────────────────────
    # 核心消息处理（双螺旋主流程）
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """处理内部 system 消息，不走意图路由。"""
        if msg.content.strip() == "__subconscious_recovery__":
            daemon = self.subconscious_daemon
            if daemon is not None:
                await daemon.handle_energy_recovery()
                return OutboundMessage(
                    channel="system",
                    chat_id=msg.chat_id,
                    content="Subconscious energy recovery complete.",
                )

        channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id))
        logger.info("Processing system message from {}", msg.sender_id)
        key = f"{channel}:{chat_id}"
        session = self.sessions.get_or_create(key)
        self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
        history = session.get_history(max_messages=self.memory_window)
        messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            mode="iq",
            channel=channel,
            chat_id=chat_id,
        )
        final_content, _, all_msgs = await self._run_agent_loop(messages)
        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)
        return OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=final_content or "Background task completed.",
        )

    async def _handle_slash_command(self, msg: InboundMessage, session: Session) -> OutboundMessage | None:
        """处理 /new /help 等斜杠命令。"""
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())
            self._consolidating.add(session.key)
            try:
                async with lock:
                    snapshot = session.messages[session.last_consolidated:]
                    if snapshot:
                        temp = Session(key=session.key)
                        temp.messages = list(snapshot)
                        if not await self._consolidate_memory(temp, archive_all=True):
                            return OutboundMessage(
                                channel=msg.channel,
                                chat_id=msg.chat_id,
                                content="Memory archival failed. Please try again.",
                            )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Memory archival failed. Please try again.",
                )
            finally:
                self._consolidating.discard(session.key)
                if not lock.locked():
                    self._consolidation_locks.pop(session.key, None)
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="New session started. ✨")

        if cmd == "/help":
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=(
                    "🐈 nanobot commands:\n"
                    "/new  — Start a new conversation\n"
                    "/stop — Stop the current task\n"
                    "/help — Show available commands"
                ),
            )
        return None

    def _maybe_schedule_consolidation(self, session: Session) -> None:
        """后台异步触发记忆压缩，不阻塞当前回复。"""
        unconsolidated = len(session.messages) - session.last_consolidated
        if unconsolidated < self.memory_window or session.key in self._consolidating:
            return
        self._consolidating.add(session.key)
        lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())

        async def _consolidate_and_unlock() -> None:
            try:
                async with lock:
                    await self._consolidate_memory(session)
            finally:
                self._consolidating.discard(session.key)
                if not lock.locked():
                    self._consolidation_locks.pop(session.key, None)
                _task = asyncio.current_task()
                if _task is not None:
                    self._consolidation_tasks.discard(_task)

        _task = asyncio.create_task(_consolidate_and_unlock())
        self._consolidation_tasks.add(_task)

    def _build_progress_callback(
        self, msg: InboundMessage
    ) -> Callable[[str], Awaitable[None]]:
        """构建当前消息的 progress 推送回调。"""

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                    metadata=meta,
                )
            )

        return _bus_progress

    async def _route_intent(self, user_input: str) -> RoutingDecision:
        """第一层决策：生成统一 RoutingDecision。"""
        emotion_prompt = self.emotion_mgr.get_emotion_prompt()
        emotion_label = self.emotion_mgr.get_emotion_label()
        intent = await self._router.route(user_input, emotion_prompt)
        intent_type = intent["intent_type"]
        intent_params = intent.get("extracted_params", {})
        if not isinstance(intent_params, dict):
            intent_params = {}
        logger.info("Intent routed → {} | {}", intent_type, intent.get("reason", ""))
        progress_enabled = not self.channels_config or self.channels_config.send_progress
        return RoutingDecision(
            intent_type=intent_type,
            intent_params=intent_params,
            emotion_label=emotion_label,
            progress_enabled=progress_enabled,
        )

    async def _execute_chat_path(
        self,
        msg: InboundMessage,
        session: Session,
        history: list[dict],
    ) -> str:
        logger.info("Chat path → EQ only (warm memory)")
        final_content = await eq_chat_flow(self, msg.content, history, msg.channel, msg.chat_id)
        self._save_simple_turn(session, msg.content, final_content)
        return final_content

    async def _execute_task_path(
        self,
        msg: InboundMessage,
        session: Session,
        history: list[dict],
        intent_params: dict[str, Any],
        progress_cb: Callable[..., Awaitable[None]] | None,
    ) -> str:
        logger.info("Task path → IQ execute (cold memory + tools) → EQ polish")
        iq_result, all_msgs = await iq_execute_flow(
            self,
            msg,
            session,
            history,
            intent_params=intent_params,
            on_progress=progress_cb,
        )
        self._save_turn(session, all_msgs, 1 + len(history))
        final_content = await eq_polish_flow(self, msg.content, iq_result, style="professional")
        eq_append_assistant_message(session, final_content)
        return final_content

    async def _execute_hybrid_path(
        self,
        msg: InboundMessage,
        session: Session,
        history: list[dict],
        decision: RoutingDecision,
        progress_cb: Callable[..., Awaitable[None]] | None,
    ) -> str:
        return await execute_hybrid_path_flow(
            self, msg, session, history, decision, progress_cb
        )

    async def _execute_intent_path(
        self,
        msg: InboundMessage,
        session: Session,
        history: list[dict],
        decision: RoutingDecision,
        progress_cb: Callable[..., Awaitable[None]] | None,
    ) -> str:
        """第二层决策：按路由结果执行 Chat/Task/Hybrid。"""
        if decision.intent_type == "Chat":
            return await self._execute_chat_path(msg, session, history)
        if decision.intent_type == "Task":
            return await self._execute_task_path(
                msg, session, history, decision.intent_params, progress_cb
            )
        if decision.intent_type == "Hybrid":
            return await self._execute_hybrid_path(msg, session, history, decision, progress_cb)
        logger.warning("Unknown intent type '{}', fallback to Chat path", decision.intent_type)
        return await self._execute_chat_path(msg, session, history)

    def _post_process_turn(self, msg: InboundMessage, final_content: str, emotion_label: str) -> None:
        """对话收尾：情绪更新 + 暖记忆 + 主动对话目标记录。"""
        if not final_content:
            return
        emotion_event = self.emotion_mgr.update_from_conversation(msg.content, final_content)
        if emotion_event:
            pad = self.emotion_mgr.pad
            delta_mag = math.sqrt(
                emotion_event.delta_pleasure ** 2
                + emotion_event.delta_arousal ** 2
                + emotion_event.delta_dominance ** 2
            )
            importance = min(delta_mag / 1.732, 1.0)
            description = (
                f"触发词：{emotion_event.trigger}，"
                f"情绪：{pad.get_emotion_label()}，"
                f"行为：{emotion_event.behavior}"
            )
            self.context.emotion_memory.save(
                description=description,
                pleasure=pad.pleasure,
                arousal=pad.arousal,
                dominance=pad.dominance,
                importance=importance,
            )
        self._write_warm_memory(msg.content, final_content, emotion_label)
        self._save_proactive_target(msg.channel, msg.chat_id)

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        if msg.channel == "system":
            return await self._handle_system_message(msg)

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        slash_resp = await self._handle_slash_command(msg, session)
        if slash_resp is not None:
            return slash_resp

        self._maybe_schedule_consolidation(session)
        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=self.memory_window)
        progress_cb = on_progress or self._build_progress_callback(msg)
        decision = await self._route_intent(msg.content)
        final_content = await self._execute_intent_path(
            msg=msg,
            session=session,
            history=history,
            decision=decision,
            progress_cb=progress_cb,
        )
        self._post_process_turn(msg, final_content, decision.emotion_label)

        self.sessions.save(session)

        # 若 MessageTool 已在本轮发送过消息，不重复返回
        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview_out = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview_out)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 工具方法
    # ─────────────────────────────────────────────────────────────────────────

    def _save_proactive_target(self, channel: str, chat_id: str) -> None:
        """记录最后活跃的频道和用户 ID，供潜意识守护进程主动发消息用。"""
        import json as _json
        target_file = self.workspace / "subconscious_target.json"
        try:
            target_file.write_text(
                _json.dumps({"channel": channel, "chat_id": chat_id}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """直接处理消息（CLI / cron 使用）。"""
        await self._connect_mcp()
        msg      = InboundMessage(channel=channel, sender_id="user",
                                  chat_id=chat_id, content=content)
        response = await self._process_message(
            msg, session_key=session_key, on_progress=on_progress
        )
        return response.content if response else ""
