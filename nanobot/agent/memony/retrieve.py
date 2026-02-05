"""
检索模块 (Retrieve Module)

实现 Stanford GA 的三维记忆检索算法：
1. Recency（最近性）：时间衰减，越近的记忆权重越高
2. Importance（重要性）：记忆的重要性评分（1-10）
3. Relevance（相关性）：与查询的语义相似度

检索分数 = α·recency + β·importance + γ·relevance

参考：Stanford Generative Agents 论文
"""

import numpy as np
from typing import List, Optional, Tuple
from .concept_node import ConceptNode
from loguru import logger
import time

TAG = __name__

class RetrieveModule:
    """
    检索模块 - 从记忆中检索相关信息
    
    这是 Stanford GA 认知循环的第一步：Retrieve
    """
    
    def __init__(
        self,
        alpha: float = 1.0,    # recency权重
        beta: float = 1.0,     # importance权重
        gamma: float = 1.0,    # relevance权重
        decay_rate: float = 0.99,  # 时间衰减率（每小时衰减1%）
    ):
        """
        初始化检索模块
        
        Args:
            alpha: 最近性权重
            beta: 重要性权重
            gamma: 相关性权重
            decay_rate: 时间衰减率，默认0.99（每小时衰减1%）
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.decay_rate = decay_rate
        
        logger.bind(tag=TAG).info(
            f"RetrieveModule initialized with weights: "
            f"alpha={alpha}, beta={beta}, gamma={gamma}, decay_rate={decay_rate}"
        )
    
    def retrieve(
        self,
        query_embedding: List[float],
        memories: List[ConceptNode],
        k: int = 5,
        current_time: Optional[float] = None,
    ) -> List[Tuple[ConceptNode, float]]:
        """
        检索最相关的k条记忆
        
        Args:
            query_embedding: 查询的向量表示
            memories: 候选记忆列表
            k: 返回的记忆数量
            current_time: 当前时间戳，默认为当前时间
        
        Returns:
            List[Tuple[ConceptNode, float]]: (记忆节点, 综合得分) 的列表，按得分降序排列
        """
        if not memories:
            return []
        
        if current_time is None:
            current_time = time.time()
        
        # 计算每条记忆的三维得分
        scored_memories = []
        for memory in memories:
            score = self._calculate_retrieval_score(
                query_embedding=query_embedding,
                memory=memory,
                current_time=current_time
            )
            # 确保分数是 Python 原生 float（避免 numpy.float64 序列化问题）
            scored_memories.append((memory, float(score)))
            
            # 记录访问（会影响last_accessed）
            memory.record_access()
        
        # 按得分降序排序
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前k条
        return scored_memories[:k]
    
    def _calculate_retrieval_score(
        self,
        query_embedding: List[float],
        memory: ConceptNode,
        current_time: float,
    ) -> float:
        """
        计算单条记忆的检索得分
        
        Score = α·recency + β·importance + γ·relevance
        
        Args:
            query_embedding: 查询向量
            memory: 记忆节点
            current_time: 当前时间戳
        
        Returns:
            综合得分（0-1范围，已归一化）
        """
        # 1. 计算Recency（最近性）
        recency = self._calculate_recency(memory, current_time)
        
        # 2. 计算Importance（重要性）
        importance = self._calculate_importance(memory)
        
        # 3. 计算Relevance（相关性）
        relevance = self._calculate_relevance(query_embedding, memory)
        
        # 4. 加权组合
        total_weight = self.alpha + self.beta + self.gamma
        score = (
            self.alpha * recency +
            self.beta * importance +
            self.gamma * relevance
        ) / total_weight
        
        logger.bind(tag=TAG).debug(
            f"Memory '{memory.content[:30]}...': "
            f"recency={recency:.4f}, importance={importance:.4f}, relevance={relevance:.4f}, "
            f"final_score={score:.4f}"
        )
        
        # 确保返回 Python 原生 float 类型
        return float(score)
    
    def _calculate_recency(self, memory: ConceptNode, current_time: float) -> float:
        """
        计算最近性得分（时间衰减）
        
        recency = decay_rate ^ hours_since_creation
        
        Args:
            memory: 记忆节点
            current_time: 当前时间戳
        
        Returns:
            最近性得分（0-1）
        """
        return memory.get_recency_score(current_time, self.decay_rate)
    
    def _calculate_importance(self, memory: ConceptNode) -> float:
        """
        计算重要性得分（归一化到0-1）
        
        Args:
            memory: 记忆节点
        
        Returns:
            重要性得分（0-1）
        """
        # 重要性评分是1-10，归一化到0-1
        return (memory.importance - 1) / 9.0
    
    def _calculate_relevance(
        self,
        query_embedding: List[float],
        memory: ConceptNode,
    ) -> float:
        """
        计算相关性得分（余弦相似度）
        
        Args:
            query_embedding: 查询向量
            memory: 记忆节点
        
        Returns:
            相关性得分（0-1）
        """
        if memory.embedding is None or len(memory.embedding) == 0:
            # 如果记忆没有向量表示，返回中等相关性
            logger.bind(tag=TAG).warning(
                f"Memory '{memory.node_id}' has no embedding, using default relevance 0.5"
            )
            return 0.5
        
        # 计算余弦相似度
        similarity = self._cosine_similarity(query_embedding, memory.embedding)
        
        # 余弦相似度范围是[-1, 1]，转换到[0, 1]
        normalized_similarity = (similarity + 1) / 2
        
        return normalized_similarity
    
    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
        
        Returns:
            余弦相似度（-1到1）
        """
        if len(vec1) != len(vec2):
            logger.bind(tag=TAG).error(
                f"Vector dimension mismatch: {len(vec1)} vs {len(vec2)}"
            )
            return 0.0
        
        # 转换为numpy数组
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # 计算余弦相似度
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # 转换为 Python 原生 float，避免 msgpack 序列化问题
        return float(dot_product / (norm1 * norm2))
    
    def update_weights(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
    ):
        """
        动态更新检索权重
        
        Args:
            alpha: 新的recency权重
            beta: 新的importance权重
            gamma: 新的relevance权重
        """
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        
        logger.bind(tag=TAG).info(
            f"Retrieval weights updated: alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}"
        )


# 保持向后兼容的别名
MemoryRetrieval = RetrieveModule
