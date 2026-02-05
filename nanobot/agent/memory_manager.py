"""
MemoryManager - 统一的记忆管理器

整合 Stanford Generative Agents 的记忆系统：
- 记忆存储 (MemoryStream)
- 重要性评估 (ImportanceScorer)
- 向量生成 (EmbeddingGenerator)
- 3D检索 (RetrieveModule)
- 反思机制 (ReflectModule)

流程：
1. 收消息 → 存入记忆(observation) + 评估重要性
2. 检索相关记忆(3D retrieval)
3. 构建context（包含检索到的记忆）
4. 调LLM → 执行tools
5. 检查是否触发反思
6. 存入记忆(assistant response)
7. 响应
"""

import uuid
import time
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from loguru import logger

from .memony.concept_node import ConceptNode
from .memony.memory_stream import MemoryStream
from .memony.importance_scorer import ImportanceScorer
from .memony.embedding_generator import EmbeddingGenerator
from .memony.retrieve import RetrieveModule
from .memony.reflect import ReflectModule

TAG = __name__



class MemoryManager:
    """
    统一的记忆管理器
    
    整合所有记忆相关的操作，提供简洁的API供AgentLoop使用
    """
    
    def __init__(
        self,
        role_id: str,
        workspace: Path,
        embedding_config: Optional[Dict[str, Any]] = None,
        reflection_threshold: int = 150,
        reflection_enabled: bool = True,
        use_llm_for_importance: bool = False,
    ):
        """
        初始化记忆管理器
        
        Args:
            role_id: 角色/会话ID
            workspace: 工作空间路径
            embedding_config: 向量生成配置
            reflection_threshold: 反思触发阈值
            reflection_enabled: 是否启用反思
            use_llm_for_importance: 是否使用LLM评估重要性
        """
        self.role_id = role_id
        self.workspace = workspace
        
        # 初始化各个模块
        self.memory_stream = MemoryStream(
            role_id=role_id,
            save_to_file=True,
            memory_file=str(workspace / "memory" / f".memory_{role_id}.json")
        )
        
        self.importance_scorer = ImportanceScorer(use_llm=use_llm_for_importance)
        
        self.embedding_generator = EmbeddingGenerator(embedding_config or {
            "provider": "dummy",  # 默认使用dummy，生产环境应配置openai或local
            "dimension": 1536,
        })
        
        self.retriever = RetrieveModule(
            alpha=1.0,   # recency权重
            beta=1.0,    # importance权重
            gamma=1.0,   # relevance权重
            decay_rate=0.99,
        )
        
        self.reflector = ReflectModule(
            threshold=reflection_threshold,
            depth=3,
            min_time_interval=3600,  # 最小1小时间隔
            enabled=reflection_enabled,
        )
        
        logger.bind(tag=TAG).info(
            f"MemoryManager initialized for role_id={role_id}, "
            f"reflection_enabled={reflection_enabled}, threshold={reflection_threshold}"
        )
    
    async def add_observation(
        self,
        content: str,
        role: str = "user",
        llm=None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConceptNode:
        """
        添加观察记忆（用户消息或助手回复）
        
        这是流程的第2步和第8步：
        - 第2步：存入用户消息 + 评估重要性
        - 第8步：存入助手回复
        
        Args:
            content: 记忆内容
            role: 角色（user/assistant）
            llm: LLM实例（用于重要性评估，可选）
            metadata: 额外元数据
        
        Returns:
            创建的记忆节点
        """
        # 1. 评估重要性
        importance = self.importance_scorer.score(content, role, llm)
        
        # 2. 生成向量表示
        embedding = await self.embedding_generator.generate_embedding(content)
        
        # 3. 创建记忆节点
        current_time = time.time()
        node = ConceptNode(
            node_id=f"obs_{uuid.uuid4().hex[:12]}",
            content=content,
            created=current_time,
            last_accessed=current_time,
            importance=importance,
            embedding=embedding,
            memory_type="observation",
            role=role,
            metadata=metadata or {},
        )
        
        # 4. 添加到记忆流
        self.memory_stream.add_node(node, accumulate_importance=True)
        
        logger.bind(tag=TAG).debug(
            f"Added observation: role={role}, importance={importance}, "
            f"content={content[:50]}..."
        )
        
        return node
    
    async def retrieve_relevant_memories(
        self,
        query: str,
        k: int = 5,
    ) -> List[Tuple[ConceptNode, float]]:
        """
        检索相关记忆（3D retrieval）
        
        这是流程的第3步：检索相关记忆
        
        Args:
            query: 查询文本（通常是用户的新消息）
            k: 返回的记忆数量
        
        Returns:
            相关记忆列表，每项为(记忆节点, 综合得分)
        """
        # 1. 生成查询向量
        query_embedding = await self.embedding_generator.generate_embedding(query)
        
        # 2. 获取所有记忆
        all_memories = self.memory_stream.get_all_nodes(include_expired=False)
        
        if not all_memories:
            logger.bind(tag=TAG).debug("No memories to retrieve from")
            return []
        
        # 3. 使用3D检索
        relevant = self.retriever.retrieve(
            query_embedding=query_embedding,
            memories=all_memories,
            k=k,
        )
        
        logger.bind(tag=TAG).debug(
            f"Retrieved {len(relevant)} memories for query: {query[:50]}..."
        )
        
        return relevant
    
    def format_memories_for_context(
        self,
        memories: List[Tuple[ConceptNode, float]],
        max_memories: int = 10,
    ) -> str:
        """
        将检索到的记忆格式化为上下文字符串
        
        这是流程的第4步的一部分：构建context
        
        Args:
            memories: 检索到的记忆列表
            max_memories: 最大记忆数量
        
        Returns:
            格式化的记忆上下文
        """
        if not memories:
            return ""
        
        lines = ["## 相关记忆"]
        
        for i, (node, score) in enumerate(memories[:max_memories], 1):
            age_hours = node.get_age_hours()
            role_label = "用户" if node.role == "user" else "你"
            
            # 格式：[时间前] 角色: 内容 (相关度)
            lines.append(
                f"{i}. [{age_hours:.1f}小时前] {role_label}: {node.content} "
                f"(重要性:{node.importance}, 相关度:{score:.2f})"
            )
        
        return "\n".join(lines)
    
    async def check_and_reflect(
        self,
        llm,
        agent_name: str = "用户",
    ) -> List[ConceptNode]:
        """
        检查是否需要反思，如果需要则执行反思
        
        这是流程的第7步：检查是否触发反思
        
        Args:
            llm: LLM实例
            agent_name: Agent名称
        
        Returns:
            生成的反思记忆列表（如果触发了反思）
        """
        # 1. 检查是否应该反思
        accumulated = self.memory_stream.accumulated_importance
        if not self.reflector.should_reflect(accumulated):
            return []
        
        logger.bind(tag=TAG).info(
            f"Reflection triggered! accumulated_importance={accumulated}"
        )
        
        # 2. 获取最近的记忆
        recent_memories = self.memory_stream.get_recent_nodes(n=50)
        
        # 3. 生成反思问题
        questions = await self.reflector.generate_reflection_questions_async(
            agent_name=agent_name,
            recent_memories=recent_memories,
            llm=llm,
            num_questions=3,
        )
        
        # 4. 为每个问题检索相关记忆并生成洞见
        insights = []
        for question in questions:
            # 检索与问题相关的记忆
            relevant = await self.retrieve_relevant_memories(question, k=10)
            
            # 生成洞见
            question_insights = await self.reflector.generate_insights(
                questions=[question],
                relevant_memories=relevant,
                llm=llm,
                agent_name=agent_name,
            )
            insights.extend(question_insights)
        
        # 5. 将洞见存入记忆
        reflection_nodes = []
        for insight in insights:
            embedding = await self.embedding_generator.generate_embedding(insight["insight"])
            
            current_time = time.time()
            node = ConceptNode(
                node_id=f"ref_{uuid.uuid4().hex[:12]}",
                content=insight["insight"],
                created=current_time,
                last_accessed=current_time,
                importance=insight["importance"],
                embedding=embedding,
                memory_type="reflection",
                role="assistant",
                related_nodes=insight["evidence"],
                metadata={
                    "question": insight["question"],
                },
            )
            
            # 反思记忆不累积重要性（避免无限反思）
            self.memory_stream.add_node(node, accumulate_importance=False)
            reflection_nodes.append(node)
        
        # 6. 重置累积重要性
        self.memory_stream.reset_accumulated_importance()
        self.reflector.update_reflection_state()
        
        logger.bind(tag=TAG).info(
            f"Reflection completed: generated {len(reflection_nodes)} insights"
        )
        
        return reflection_nodes
    
    def get_recent_context(self, n: int = 10) -> str:
        """
        获取最近的记忆上下文（不基于查询）
        
        Args:
            n: 返回的记忆数量
        
        Returns:
            格式化的记忆上下文
        """
        recent = self.memory_stream.get_recent_nodes(n=n)
        
        if not recent:
            return ""
        
        lines = ["## 最近记忆"]
        
        for node in recent:
            age_hours = node.get_age_hours()
            role_label = "用户" if node.role == "user" else "你"
            type_label = "[反思]" if node.memory_type == "reflection" else ""
            
            lines.append(
                f"- [{age_hours:.1f}小时前] {type_label}{role_label}: {node.content[:100]}"
            )
        
        return "\n".join(lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        stream_stats = self.memory_stream.get_statistics()
        
        return {
            **stream_stats,
            "reflection_enabled": self.reflector.enabled,
            "reflection_threshold": self.reflector.threshold,
            "last_reflection_time": self.reflector.last_reflection_time,
        }
    
    def clear_expired(self) -> int:
        """清理过期记忆"""
        return self.memory_stream.clear_expired_nodes()
