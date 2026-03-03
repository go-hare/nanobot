"""
暖记忆系统：ChromaDB 向量库 + 情绪共振加权检索

设计依据：ai.md §4.2 / §4.3 / §4.5 / §4.6
- 写入：带情绪标签、重要性元数据
- 检索：文本相似度×0.4 + 情绪共振度×衰减系数×0.6（斯坦福公式）
- 情绪衰减：0.99^小时数（与时效性公式一致）
- chromadb 未安装时优雅降级，不影响主流程
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from loguru import logger

# ─────────────────────────────────────────────────────────────────────────────
# 情绪亲和度映射（ai.md §4.6：情绪共振度计算）
# ─────────────────────────────────────────────────────────────────────────────
# 同悲共喜原则：相同/相近情绪共鸣高，相反情绪共鸣低
_EMOTION_AFFINITY: dict[tuple[str, str], float] = {
    ("开心", "开心"): 1.0,  ("开心", "兴奋"): 0.8,  ("开心", "平静"): 0.5,
    ("开心", "困倦"): 0.3,  ("开心", "悲伤"): 0.1,  ("开心", "愤怒"): 0.1,
    ("兴奋", "兴奋"): 1.0,  ("兴奋", "开心"): 0.8,  ("兴奋", "平静"): 0.4,
    ("兴奋", "悲伤"): 0.1,  ("兴奋", "愤怒"): 0.3,
    ("平静", "平静"): 1.0,  ("平静", "开心"): 0.5,  ("平静", "困倦"): 0.6,
    ("平静", "悲伤"): 0.4,  ("平静", "愤怒"): 0.3,
    ("困倦", "困倦"): 1.0,  ("困倦", "平静"): 0.6,  ("困倦", "悲伤"): 0.5,
    ("悲伤", "悲伤"): 1.0,  ("悲伤", "平静"): 0.4,  ("悲伤", "开心"): 0.1,
    ("悲伤", "愤怒"): 0.5,
    ("愤怒", "愤怒"): 1.0,  ("愤怒", "悲伤"): 0.5,  ("愤怒", "平静"): 0.3,
    ("愤怒", "开心"): 0.1,
}


def _emotion_resonance(current: str, stored: str) -> float:
    """计算两个情绪标签之间的共鸣度 [0.0, 1.0]（对称查表）。"""
    key = (current, stored)
    rev = (stored, current)
    return _EMOTION_AFFINITY.get(key, _EMOTION_AFFINITY.get(rev, 0.3))


# ─────────────────────────────────────────────────────────────────────────────
# 暖记忆存储
# ─────────────────────────────────────────────────────────────────────────────

class WarmMemoryStore:
    """
    暖记忆向量库（斯坦福小镇架构）。

    - 存储形式：ChromaDB 持久化向量库
    - 内容特点：主观、动态、带情绪标签
    - 服务对象：EQ 系统（说话时用）、Hybrid 双系统
    - chromadb 未安装时 available=False，所有操作静默跳过
    """

    def __init__(self, workspace: Path):
        self._available = False
        try:
            import chromadb  # type: ignore
            db_path = workspace / "data" / "chroma_db"
            db_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(db_path))
            self._col = self._client.get_or_create_collection(
                name="warm_memories",
                metadata={"hnsw:space": "cosine"},  # 余弦相似度
            )
            self._available = True
            logger.debug("WarmMemoryStore ready: {} entries", self._col.count())
        except ImportError:
            logger.info("chromadb not installed — warm memory disabled. "
                        "Run: pip install chromadb")
        except Exception as e:
            logger.warning("WarmMemoryStore init failed: {}", e)

    @property
    def available(self) -> bool:
        return self._available

    # ── 写入 ─────────────────────────────────────────────────────────────────

    def save(
        self,
        text: str,
        emotion: str = "平静",
        importance: int = 5,
    ) -> None:
        """
        写入一条暖记忆。

        :param text:       对话摘要，如 "用户失恋了，我安慰了他，情绪逐渐平静"
        :param emotion:    当前情绪标签（AI 侧），如 "悲伤"/"开心"/"平静"
        :param importance: 重要性 1-10（由调用方或反思机制评定）
        """
        if not self._available:
            return
        try:
            self._col.add(
                documents=[text],
                metadatas=[{
                    "timestamp":  datetime.now().isoformat(),
                    "emotion":    emotion,
                    "importance": importance,
                }],
                ids=[str(uuid.uuid4())],
            )
            logger.debug("WarmMemory saved: emotion={} importance={}", emotion, importance)
        except Exception as e:
            logger.warning("WarmMemory save failed: {}", e)

    # ── 检索 ─────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        current_emotion: str = "平静",
        k: int = 3,
        candidate_pool: int = 10,
    ) -> list[str]:
        """
        带情绪共振的暖记忆检索（ai.md §4.6 公式）。

        最终得分 = 文本相似度×0.4 + 情绪共振度×衰减系数×0.6

        :param query:           当前用户输入（作为检索 query）
        :param current_emotion: AI 当前情绪标签（影响共鸣权重）
        :param k:               返回条数
        :param candidate_pool:  先取多少条候选（再重排）
        :returns:               按得分排序的记忆文本列表
        """
        if not self._available:
            return []
        try:
            count = self._col.count()
            if count == 0:
                return []

            n        = min(candidate_pool, count)
            results  = self._col.query(query_texts=[query], n_results=n)
            docs      = results["documents"][0]
            metas     = results["metadatas"][0]
            distances = results["distances"][0]   # cosine distance ∈ [0, 2]

            now    = datetime.now()
            scored = []

            for doc, meta, dist in zip(docs, metas, distances):
                # 1. 文本相似度（余弦距离转相似度）
                text_sim = max(0.0, 1.0 - dist / 2.0)

                # 2. 时效性（独立维度，0.99^小时数）
                try:
                    ts    = datetime.fromisoformat(meta["timestamp"])
                    hours = (now - ts).total_seconds() / 3600
                    recency = 0.99 ** hours
                except Exception:
                    recency = 0.5

                # 3. 重要性（归一化到 [0, 1]）
                importance = float(meta.get("importance", 5)) / 10.0

                # 4. 情绪共振度（同悲共喜原则）
                resonance = _emotion_resonance(current_emotion, meta.get("emotion", "平静"))

                # 斯坦福4维公式（EQ路径：完整 Recency+Importance+Relevance+情绪共振）
                # score = 0.2×Recency + 0.3×Importance + 0.3×Relevance + 0.2×Resonance
                final_score = (
                    0.2 * recency
                    + 0.3 * importance
                    + 0.3 * text_sim
                    + 0.2 * resonance
                )
                scored.append((final_score, doc))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in scored[:k]]

        except Exception as e:
            logger.warning("WarmMemory retrieve failed: {}", e)
            return []

    def get_context(self, query: str, current_emotion: str = "平静") -> str:
        """
        返回格式化后的暖记忆片段，供注入 EQ / Hybrid System Prompt。
        """
        memories = self.retrieve(query, current_emotion)
        if not memories:
            return ""
        lines = "\n".join(f"- {m}" for m in memories)
        return f"## 近期情感记忆（暖记忆检索结果）\n{lines}"

    def get_recent(self, limit: int = 20) -> list[dict]:
        """
        获取最近的暖记忆条目（供反思机制使用）。

        :returns: list of {"text": str, "emotion": str, "timestamp": str}
        """
        if not self._available:
            return []
        try:
            count = self._col.count()
            if count == 0:
                return []
            n       = min(limit, count)
            results = self._col.get(limit=n, include=["documents", "metadatas"])
            docs    = results.get("documents", [])
            metas   = results.get("metadatas", [])
            entries = [
                {"text": d, "emotion": m.get("emotion", ""), "timestamp": m.get("timestamp", "")}
                for d, m in zip(docs, metas)
            ]
            # 按时间倒序
            entries.sort(key=lambda x: x["timestamp"], reverse=True)
            return entries
        except Exception as e:
            logger.warning("WarmMemory get_recent failed: {}", e)
            return []
