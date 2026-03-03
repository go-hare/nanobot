"""
冷记忆向量库（IQ路径）：ChromaDB + 斯坦福3重检索

改造说明（ai.md §4.1 全量向量化）：

检索公式：score = 0.3×Recency(0.99^h) + 0.4×Importance/10 + 0.3×Relevance(余弦)

- 存储：ChromaDB collection "cold_memories"，每条事实独立存储
- Importance：由 consolidate 时 LLM 评分（1-10），权重最高（事实质量 > 时效）
- 回退：chromadb 未安装时 available=False，由 MemoryStore(MEMORY.md) 兜底
- 与 MEMORY.md / HISTORY.md 双写，保持人类可读备份
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from loguru import logger


class ColdMemoryStore:
    """
    冷记忆向量库（IQ 路径第1检索层）。

    一条冷记忆 = 一个客观事实，如：
    - "用户叫小明，偏好深色主题"
    - "项目名称为 nanobot，主语言 Python"

    由反思机制（consolidate）写入，每条附带 importance 评分。
    """

    def __init__(self, workspace: Path):
        self._available = False
        try:
            import chromadb  # type: ignore
            db_path = workspace / "data" / "chroma_db"
            db_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(db_path))
            self._col = self._client.get_or_create_collection(
                name="cold_memories",
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True
            logger.debug("ColdMemoryStore ready: {} entries", self._col.count())
        except ImportError:
            logger.info("chromadb not installed — cold memory vector store disabled. "
                        "Run: pip install chromadb")
        except Exception as e:
            logger.warning("ColdMemoryStore init failed: {}", e)

    @property
    def available(self) -> bool:
        return self._available

    def count(self) -> int:
        """返回已存储的冷记忆条数。"""
        if not self._available:
            return 0
        try:
            return self._col.count()
        except Exception:
            return 0

    # ── 写入 ─────────────────────────────────────────────────────────────────

    def save(
        self,
        text: str,
        importance: int = 5,
        category: str = "fact",
    ) -> None:
        """
        写入一条冷记忆（事实）。

        :param text:       事实文本，如 "用户偏好 Python，不喜欢 Java"
        :param importance: 重要性 1-10（由 consolidate 时 LLM 评分）
        :param category:   类别：preference / project / habit / event / other
        """
        if not self._available:
            return
        try:
            self._col.add(
                documents=[text],
                metadatas=[{
                    "timestamp":  datetime.now().isoformat(),
                    "importance": max(1, min(10, importance)),
                    "category":   category,
                }],
                ids=[str(uuid.uuid4())],
            )
            logger.debug("ColdMemory saved: importance={} category={}", importance, category)
        except Exception as e:
            logger.warning("ColdMemory save failed: {}", e)

    # ── 检索 ─────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = 8,
        candidate_pool: int = 20,
    ) -> list[str]:
        """
        斯坦福3重检索（IQ路径）：
        score = 0.3×Recency(0.99^h) + 0.4×Importance/10 + 0.3×Relevance(余弦)

        Importance 权重最高（0.4），因为事实的质量和重要性比时效更关键。

        :param query:          当前用户输入（作为向量检索 query）
        :param k:              返回条数
        :param candidate_pool: 先取多少候选再按公式重排
        """
        if not self._available:
            return []
        try:
            count = self._col.count()
            if count == 0:
                return []

            n       = min(candidate_pool, count)
            results  = self._col.query(query_texts=[query], n_results=n)
            docs      = results["documents"][0]
            metas     = results["metadatas"][0]
            distances = results["distances"][0]   # cosine distance ∈ [0, 2]

            now    = datetime.now()
            scored = []

            for doc, meta, dist in zip(docs, metas, distances):
                # 1. 相关度（余弦距离 → 相似度）
                relevance = max(0.0, 1.0 - dist / 2.0)

                # 2. 时效性（0.99^小时数）
                try:
                    ts    = datetime.fromisoformat(meta["timestamp"])
                    hours = (now - ts).total_seconds() / 3600
                    recency = 0.99 ** hours
                except Exception:
                    recency = 0.5

                # 3. 重要性（归一化到 [0, 1]）
                importance = float(meta.get("importance", 5)) / 10.0

                # 斯坦福3重公式（IQ路径：Importance权重最高）
                final_score = 0.3 * recency + 0.4 * importance + 0.3 * relevance
                scored.append((final_score, doc))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in scored[:k]]

        except Exception as e:
            logger.warning("ColdMemory retrieve failed: {}", e)
            return []

    def get_context(self, query: str, max_entries: int = 8) -> str:
        """
        返回格式化后的冷记忆片段，供注入 IQ / Hybrid System Prompt。
        """
        memories = self.retrieve(query, k=max_entries)
        if not memories:
            return ""
        lines = "\n".join(f"- {m}" for m in memories)
        return f"## Long-term Memory (Vector)\n{lines}"
