"""
潜意识守护进程 (Subconscious Daemon)

设计依据：ai.md §7 守护进程
核心职责：
  1. 【情绪衰减】按 config/drive_config.yaml 周期执行：PAD/Drive 衰减
  2. 【反思机制】按 config/drive_config.yaml 周期执行：更新 SOUL.md / USER.md
  3. 【主动对话】social < 20：主动发起对话，经 EQ 路径生成心情感慨/话题
  4. 【能量恢复】每天凌晨 4 点（cron 表达式）：energy 按 recover_per_sleep 回升

集成方式（main.py / __main__.py 中）：
    from nanobot.subconscious.daemon import SubconsciousDaemon
    daemon = SubconsciousDaemon(agent_loop, workspace)
    daemon.start_background_tasks()   # asyncio 后台任务
    # 能量恢复（需要 CronService）
    daemon.register_energy_recovery(cron_service)
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.cron.service import CronService


class SubconsciousDaemon:
    """
    潜意识守护进程：定时触发情绪衰减、反思和主动对话。

    - 情绪衰减/反思/主动对话：asyncio.Task 后台循环
    - 能量恢复：通过 CronService cron 表达式（凌晨 4 点）
    """

    _REFLECT_PROMPT = """你是你自己（AI 角色），请根据最近的情感经历进行自我反思。

近期暖记忆摘要：
{warm_memories}

## 当前 SOUL.md
{current_soul}

## 当前 USER.md
{current_user}

请完成以下两项更新：
1. **更新 SOUL.md**（人格自我进化）：
   - 如果最近有反复出现的情感模式，考虑微调性格描述（保留原有锚点，只微调）
   - 保持格式与原文件一致，保留文件头部的 `>` 注释行

2. **更新 USER.md**（用户认知更新）：
   - 从最近对话中提炼出新的用户信息（习惯/偏好/近况）
   - 追加到"情感认知"区块下，保留已有内容

以 JSON 格式输出（只输出 JSON，不要其他内容）：
{{"soul_update": "更新后的完整 SOUL.md 内容", "user_update": "更新后的完整 USER.md 内容"}}
若无需更新，对应字段填 null。"""

    _PROACTIVE_PROMPT = """你是一个会主动找朋友聊天的 AI，最近没有交流，开始想念用户了。
当前情绪：{emotion_prompt}
请发一条主动消息（1-3句话）：
- 可以是感慨/发现/心情分享，或话题邀请
- 不要说"你在吗"，要有性格、有灵魂
- 符合你目前的情绪状态"""

    def __init__(self, agent_loop: "AgentLoop", workspace: Path):
        self.agent_loop  = agent_loop
        self.workspace   = workspace
        self.emotion_mgr = agent_loop.emotion_mgr
        self.warm_memory = agent_loop.warm_memory
        self._cfg = self.emotion_mgr.drive_config
        schedules = self._cfg.get("schedules", {})
        self._decay_minutes = int(schedules.get("decay_minutes", 30))
        self._reflect_hours = float(schedules.get("reflect_hours", 1))
        self._proactive_check_minutes = int(schedules.get("proactive_check_minutes", 10))
        triggers = self._cfg.get("triggers", {}).get("proactive_chat", {})
        self._proactive_probability = float(triggers.get("probability", 0.3))
        self._tasks: list[asyncio.Task] = []

    # ─────────────────────────────────────────────────────────────────────────
    # 启动 / 停止
    # ─────────────────────────────────────────────────────────────────────────

    def start_background_tasks(self) -> None:
        """启动后台 asyncio 任务（在事件循环中调用）。"""
        self._tasks = [
            asyncio.create_task(self._decay_loop(),   name="subconscious_decay"),
            asyncio.create_task(self._reflect_loop(), name="subconscious_reflect"),
            asyncio.create_task(self._proactive_loop(), name="subconscious_proactive"),
        ]
        logger.info("SubconsciousDaemon: 3 background tasks started")

    def stop(self) -> None:
        """取消所有后台任务。"""
        for t in self._tasks:
            if not t.done():
                t.cancel()
        self._tasks.clear()
        logger.info("SubconsciousDaemon: stopped")

    def register_energy_recovery(self, cron_service: "CronService") -> None:
        """
        向 CronService 注册能量恢复任务（凌晨 4 点）。
        使用系统消息 `__subconscious_recovery__`，由 AgentLoop 拦截处理。
        """
        from nanobot.cron.types import CronSchedule
        try:
            cron_service.add_job(
                name     = "subconscious_energy_recovery",
                schedule = CronSchedule(kind="cron", expr="0 4 * * *"),
                message  = "__subconscious_recovery__",
                deliver  = False,
            )
            logger.info("SubconsciousDaemon: energy recovery cron job registered")
        except Exception as e:
            logger.warning("Failed to register energy recovery cron: {}", e)

    # ─────────────────────────────────────────────────────────────────────────
    # 后台循环
    # ─────────────────────────────────────────────────────────────────────────

    async def _decay_loop(self) -> None:
        """按配置周期：情绪自然衰减。"""
        while True:
            try:
                await asyncio.sleep(self._decay_minutes * 60)
                self.emotion_mgr.decay(hours=self._decay_minutes / 60.0)
                logger.debug("Subconscious decay: social={:.0f} energy={:.0f}",
                             self.emotion_mgr.drive.social, self.emotion_mgr.drive.energy)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Decay loop error: {}", e)

    async def _reflect_loop(self) -> None:
        """按配置周期：反思机制，更新 SOUL.md + USER.md。"""
        await asyncio.sleep(10)   # 启动后稍等，确保主循环已就绪
        while True:
            try:
                await asyncio.sleep(int(self._reflect_hours * 60 * 60))
                await self._do_reflect()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Reflect loop error: {}", e)

    async def _proactive_loop(self) -> None:
        """按配置周期检查：social < threshold_low 时主动发起对话。"""
        await asyncio.sleep(60)   # 启动后 1 分钟再开始检查
        while True:
            try:
                await asyncio.sleep(self._proactive_check_minutes * 60)
                await self._do_proactive_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Proactive loop error: {}", e)

    # ─────────────────────────────────────────────────────────────────────────
    # 任务实现
    # ─────────────────────────────────────────────────────────────────────────

    async def _do_reflect(self) -> None:
        """
        反思机制：从暖记忆中提炼情感经历，更新 SOUL.md 和 USER.md。
        这是人格自我进化的核心（ai.md §7.2）。
        """
        logger.info("Subconscious reflection started")
        recent = self.warm_memory.get_recent(limit=15)
        if not recent:
            logger.debug("No warm memories to reflect on, skipping reflection")
            return

        warm_summary = "\n".join(
            f"- [{m['timestamp'][:16]}][情绪:{m['emotion']}] {m['text']}"
            for m in recent
        )

        soul_file    = self.workspace / "SOUL.md"
        user_file    = self.workspace / "USER.md"
        current_soul = soul_file.read_text(encoding="utf-8") if soul_file.exists() else ""
        current_user = user_file.read_text(encoding="utf-8") if user_file.exists() else ""

        prompt = self._REFLECT_PROMPT.format(
            warm_memories=warm_summary,
            current_soul=current_soul,
            current_user=current_user,
        )
        try:
            response = await self.agent_loop.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.agent_loop.model,
                tools=None,
                temperature=0.5,
                max_tokens=2000,
            )
            raw = response.content or ""
            m = re.search(r'\{[\s\S]*\}', raw)
            if not m:
                logger.warning("Reflection: no JSON found in LLM response")
                return
            result = json.loads(m.group())
            if soul_upd := result.get("soul_update"):
                soul_file.write_text(soul_upd, encoding="utf-8")
                logger.info("Subconscious reflection: SOUL.md updated")
            if user_upd := result.get("user_update"):
                user_file.write_text(user_upd, encoding="utf-8")
                logger.info("Subconscious reflection: USER.md updated")
        except Exception as e:
            logger.warning("Reflection failed: {}", e)

    async def _do_proactive_check(self) -> None:
        """
        社交渴望检查：social < 20 → 主动发起对话。
        向最后活跃的 channel:chat_id 推送消息（ai.md §7.3）。
        """
        if not self.emotion_mgr.drive.needs_proactive_chat():
            return
        if random.random() >= self._proactive_probability:
            logger.debug(
                "Proactive chat skipped by probability gate (p={:.2f})",
                self._proactive_probability,
            )
            return

        target = self._load_proactive_target()
        if not target:
            logger.debug("Proactive chat: no target available")
            return

        logger.info("Proactive chat triggered (social={:.0f} < 20)",
                    self.emotion_mgr.drive.social)

        emotion_prompt = self.emotion_mgr.get_emotion_prompt()
        prompt         = self._PROACTIVE_PROMPT.format(emotion_prompt=emotion_prompt)

        try:
            eq_system = self.agent_loop.context.build_eq_system_prompt()
            response  = await self.agent_loop.provider.chat(
                messages=[
                    {"role": "system", "content": eq_system},
                    {"role": "user",   "content": prompt},
                ],
                model=self.agent_loop.model,
                tools=None,
                temperature=0.8,
                max_tokens=300,
            )
            content = (response.content or "").strip()
            if not content:
                return

            from nanobot.bus.events import OutboundMessage
            await self.agent_loop.bus.publish_outbound(OutboundMessage(
                channel=target["channel"],
                chat_id=target["chat_id"],
                content=content,
                metadata={"_proactive": True},
            ))
            # 发出后按配置回升社交渴望
            self.emotion_mgr.drive.social += self.emotion_mgr.drive.recover_per_chat
            self.emotion_mgr.drive.clamp()
            self.emotion_mgr._save()
            logger.info("Proactive message sent to {}:{}", target["channel"], target["chat_id"])
        except Exception as e:
            logger.warning("Proactive chat failed: {}", e)

    async def handle_energy_recovery(self) -> None:
        """
        处理凌晨 4 点能量恢复指令（由 AgentLoop 从 system 消息中调用）。
        AgentLoop 检测到 '__subconscious_recovery__' 时转发到此方法。
        """
        with self.emotion_mgr._lock:
            self.emotion_mgr.drive.energy += self.emotion_mgr.drive.recover_per_sleep
            self.emotion_mgr.drive.clamp()
            self.emotion_mgr._save()
        logger.info("Energy recovery complete: energy={:.0f}", self.emotion_mgr.drive.energy)

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    def _load_proactive_target(self) -> dict | None:
        target_file = self.workspace / "subconscious_target.json"
        if not target_file.exists():
            return None
        try:
            return json.loads(target_file.read_text(encoding="utf-8"))
        except Exception:
            return None
