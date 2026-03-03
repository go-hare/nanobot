"""
情绪记忆向量库（EMO路径）：ChromaDB + PAD向量相似度检索

改造说明（ai.md §4 全量向量化，第3检索路径）：

检索公式：score = 0.3×Recency(0.99^h) + 0.3×Importance + 0.4×PAD_Similarity

- 存储：ChromaDB collection "emotion_memories"
- PAD_Similarity：当前PAD向量与历史PAD向量的余弦相似度（归一化到 [0,1]）
- Importance：情绪震荡幅度 = sqrt(δP²+δA²+δD²) / sqrt(3)，取值 [0,1]
- 文本嵌入：事件描述（触发词 + 情绪标签 + 后续行为）
- 供 EQ / Hybrid 路径检索：让 AI 预判在相似情绪状态下的反应模式
- 替代 EMOTION_LOG.md 的全文注入（降低 Token 占用，提升相关性）
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime
from pathlib import Path

from loguru import logger


def _pad_cosine(
    p1: float, a1: float, d1: float,
    p2: float, a2: float, d2: float,
) -> float:
    """
    两个 PAD 向量的余弦相似度，归一化到 [0, 1]。
    向量接近零时返回 0.5（中性，避免除零）。
    """
    dot  = p1 * p2 + a1 * a2 + d1 * d2
    mag1 = math.sqrt(p1 * p1 + a1 * a1 + d1 * d1)
    mag2 = math.sqrt(p2 * p2 + a2 * a2 + d2 * d2)
    if mag1 < 1e-9 or mag2 < 1e-9:
        return 0.5
    cosine = dot / (mag1 * mag2)       # ∈ [-1, 1]
    return (cosine + 1.0) / 2.0        # 归一化到 [0, 1]


class EmotionMemoryStore:
    """
    情绪记忆向量库（EMO 路径，第3检索层）。

    每条情绪记忆 = 一次 PAD 有效变化事件，包含：
    - 文本描述（用于语义嵌入，如 '触发词："喜欢你"，情绪：开心，行为：傲娇防御'）
    - 事件后 PAD 结果状态
    - 情绪震荡幅度（Importance）

    供 EQ / Hybrid 路径检索，让 AI 能在相似情绪状态下预判自身反应。
    替代 EMOTION_LOG.md 的逐行全文注入（更节省 Token，更精准）。
    """

    def __init__(self, workspace: Path):
        self._available = False
        try:
            import chromadb  # type: ignore
            db_path = workspace / "data" / "chroma_db"
            db_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(db_path))
            self._col = self._client.get_or_create_collection(
                name="emotion_memories",
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True
            logger.debug("EmotionMemoryStore ready: {} entries", self._col.count())
        except ImportError:
            logger.info("chromadb not installed — emotion memory vector store disabled. "
                        "Run: pip install chromadb")
        except Exception as e:
            logger.warning("EmotionMemoryStore init failed: {}", e)

    @property
    def available(self) -> bool:
        return self._available

    # ── 写入 ─────────────────────────────────────────────────────────────────

    def save(
        self,
        description: str,
        pleasure:    float,
        arousal:     float,
        dominance:   float,
        importance:  float = 0.3,
    ) -> None:
        """
        写入一条情绪记忆事件。

        :param description: 文本描述，如 '触发词："喜欢你"，情绪：开心，行为：傲娇防御'
        :param pleasure:    事件后 PAD-P 值 ∈ [-1, 1]
        :param arousal:     事件后 PAD-A 值 ∈ [-1, 1]
        :param dominance:   事件后 PAD-D 值 ∈ [-1, 1]
        :param importance:  震荡幅度 ∈ [0, 1]，由 sqrt(δP²+δA²+δD²)/sqrt(3) 计算
        """
        if not self._available:
            return
        try:
            self._col.add(
                documents=[description],
                metadatas=[{
                    "timestamp":  datetime.now().isoformat(),
                    "pleasure":   round(float(pleasure),  3),
                    "arousal":    round(float(arousal),   3),
                    "dominance":  round(float(dominance), 3),
                    "importance": round(min(max(float(importance), 0.0), 1.0), 3),
                }],
                ids=[str(uuid.uuid4())],
            )
            logger.debug(
                "EmotionMemory saved: P={:.2f} A={:.2f} D={:.2f} imp={:.2f}",
                pleasure, arousal, dominance, importance,
            )
        except Exception as e:
            logger.warning("EmotionMemory save failed: {}", e)

    # ── 检索 ─────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        current_P: float,
        current_A: float,
        current_D: float,
        query: str = "",
        k: int = 5,
        candidate_pool: int = 15,
    ) -> list[str]:
        """
        斯坦福3重检索（EMO路径）：
        score = 0.3×Recency(0.99^h) + 0.3×Importance + 0.4×PAD_Similarity

        PAD_Similarity 权重最高（0.4），确保检索到"情绪状态相似"的历史反应模式。

        :param current_P/A/D: 当前 PAD 状态（用于向量相似度）
        :param query:         文本查询（用于初始候选集检索，无则用 PAD 文本描述）
        :param k:             返回条数
        :param candidate_pool: 先取多少候选再按公式重排
        """
        if not self._available:
            return []
        try:
            count = self._col.count()
            if count == 0:
                return []

            # 若无明确 query，用当前 PAD 描述作为语义检索 query
            if not query.strip():
                query = (
                    f"情绪状态 愉悦{current_P:+.1f} "
                    f"激活{current_A:+.1f} 支配{current_D:+.1f}"
                )

            n       = min(candidate_pool, count)
            results  = self._col.query(query_texts=[query], n_results=n)
            docs    = results["documents"][0]
            metas   = results["metadatas"][0]

            now    = datetime.now()
            scored = []

            for doc, meta in zip(docs, metas):
                # 1. 时效性（0.99^小时数）
                try:
                    ts    = datetime.fromisoformat(meta["timestamp"])
                    hours = (now - ts).total_seconds() / 3600
                    recency = 0.99 ** hours
                except Exception:
                    recency = 0.5

                # 2. 重要性（震荡幅度，已归一化 [0, 1]）
                importance = float(meta.get("importance", 0.3))

                # 3. PAD 向量余弦相似度
                stored_P = float(meta.get("pleasure",  0.0))
                stored_A = float(meta.get("arousal",   0.5))
                stored_D = float(meta.get("dominance", 0.5))
                pad_sim  = _pad_cosine(
                    current_P, current_A, current_D,
                    stored_P,  stored_A,  stored_D,
                )

                # 斯坦福3重公式（EMO路径：PAD相似度权重最高）
                final_score = 0.3 * recency + 0.3 * importance + 0.4 * pad_sim
                scored.append((final_score, doc))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in scored[:k]]

        except Exception as e:
            logger.warning("EmotionMemory retrieve failed: {}", e)
            return []

    def get_context(
        self,
        current_P: float,
        current_A: float,
        current_D: float,
        query: str = "",
    ) -> str:
        """
        返回格式化后的情绪记忆片段，供注入 EQ / Hybrid System Prompt。
        替代 EMOTION_LOG.md 全文注入。
        """
        memories = self.retrieve(current_P, current_A, current_D, query)
        if not memories:
            return ""
        lines = "\n".join(f"- {m}" for m in memories)
        return (
            "## 情绪记忆（EMO路径·PAD向量检索）\n"
            "> 以下是情绪状态相似时的历史反应，可据此预判当前反应模式\n\n"
            f"{lines}"
        )
