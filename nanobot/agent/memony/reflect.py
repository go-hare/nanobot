"""
反思模块 (Reflect Module)

实现 Stanford GA 的反思机制：
1. 监控累积重要性，达到阈值时触发反思
2. 生成高层次反思问题
3. 检索相关记忆
4. 使用LLM生成洞见
5. 将洞见存回记忆流

参考：Stanford Generative Agents 论文
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time
from loguru import logger
from .concept_node import ConceptNode

TAG = __name__


class ReflectModule:
    """
    反思模块 - 从经历中提取高层次洞见
    
    这是 Stanford GA 认知循环的第二步：Reflect
    """
    
    def __init__(
        self,
        threshold: int = 150,  # 累积重要性阈值
        depth: int = 3,  # 每次反思生成的洞见数量
        min_time_interval: float = 3600,  # 最小反思间隔（秒），默认1小时
        enabled: bool = True,  # 是否启用反思
    ):
        """
        初始化反思模块
        
        Args:
            threshold: 累积重要性阈值，达到此值时触发反思
            depth: 每次反思生成的洞见数量
            min_time_interval: 最小反思间隔（秒）
            enabled: 是否启用反思
        """
        self.threshold = threshold
        self.depth = depth
        self.min_time_interval = min_time_interval
        self.enabled = enabled
        
        # 反思状态
        # 初始化为0，使得首次反思不受时间间隔限制
        self.last_reflection_time: float = 0.0
        self.accumulated_importance: int = 0
        
        logger.bind(tag=TAG).info(
            f"ReflectModule initialized: threshold={threshold}, depth={depth}, "
            f"min_interval={min_time_interval}s, enabled={enabled}"
        )
    
    def should_reflect(
        self,
        accumulated_importance: Optional[int] = None,
        current_time: Optional[float] = None
    ) -> bool:
        """
        判断是否应该触发反思
        
        Args:
            accumulated_importance: 当前累积重要性（可选，如果不提供则使用内部状态）
            current_time: 当前时间戳（可选）
        
        Returns:
            是否应该反思
        """
        if not self.enabled:
            return False
        
        if current_time is None:
            current_time = time.time()
        
        # 使用提供的累积重要性或内部状态
        importance = accumulated_importance if accumulated_importance is not None else self.accumulated_importance
        
        # 检查1：累积重要性是否达到阈值
        importance_threshold_met = importance >= self.threshold
        
        # 检查2：距离上次反思是否超过最小时间间隔
        time_elapsed = current_time - self.last_reflection_time
        time_interval_met = time_elapsed >= self.min_time_interval
        
        should_reflect = importance_threshold_met and time_interval_met
        
        if should_reflect:
            logger.bind(tag=TAG).info(
                f"Reflection triggered: accumulated_importance={importance}, "
                f"threshold={self.threshold}, time_elapsed={time_elapsed:.0f}s"
            )
        
        return should_reflect
    
    def generate_reflection_questions(
        self,
        agent_name: str,
        recent_memories: List[ConceptNode],
        focus_points: Optional[List[str]] = None,
        use_llm: bool = False,
        llm = None
    ) -> List[str]:
        """
        生成反思问题
        
        根据Stanford GA论文，反思问题应该是高层次的、开放性的问题，
        例如："用户最近关心什么？"、"用户的情绪如何变化？"
        
        支持两种模式：
        1. 模板模式（默认）：使用预定义模板快速生成
        2. LLM模式：使用LLM动态生成更个性化的问题
        
        Args:
            agent_name: Agent名称
            recent_memories: 最近的记忆列表
            focus_points: 关注点（可选），例如["情绪", "兴趣", "关系"]
            use_llm: 是否使用LLM动态生成问题
            llm: LLM实例（use_llm=True时需要）
        
        Returns:
            反思问题列表
        """
        if use_llm and llm:
            # 使用 LLM 动态生成问题（Stanford GA 原始设计）
            return self._generate_questions_with_llm(
                agent_name=agent_name,
                recent_memories=recent_memories,
                llm=llm,
                num_questions=self.depth
            )
        
        # 使用模板生成（快速模式）
        return self._generate_questions_from_template(
            agent_name=agent_name,
            focus_points=focus_points
        )
    
    def _generate_questions_from_template(
        self,
        agent_name: str,
        focus_points: Optional[List[str]] = None
    ) -> List[str]:
        """
        使用模板生成反思问题（快速模式）
        
        Args:
            agent_name: Agent名称
            focus_points: 关注点列表
            
        Returns:
            问题列表
        """
        if not focus_points:
            focus_points = ["核心关注", "情绪状态", "行为模式"]
        
        questions = []
        
        # 为每个关注点生成一个问题
        for focus in focus_points[:self.depth]:
            if focus == "核心关注":
                question = f"{agent_name}最近最关心的是什么？有哪些重要的话题或事件？"
            elif focus == "情绪状态":
                question = f"{agent_name}最近的情绪状态如何？有什么情绪变化吗？"
            elif focus == "行为模式":
                question = f"{agent_name}最近的行为模式有什么特点？有什么值得注意的习惯或倾向？"
            elif focus == "关系":
                question = f"{agent_name}与他人的关系如何？有什么重要的社交互动吗？"
            elif focus == "目标":
                question = f"{agent_name}最近有什么目标或计划吗？进展如何？"
            elif focus == "需求":
                question = f"{agent_name}当前有什么需求或期待吗？"
            else:
                question = f"关于{focus}，{agent_name}最近有什么值得注意的表现吗？"
            
            questions.append(question)
        
        logger.bind(tag=TAG).debug(f"Generated {len(questions)} reflection questions (template mode)")
        
        return questions
    
    def _generate_questions_with_llm(
        self,
        agent_name: str,
        recent_memories: List[ConceptNode],
        llm,
        num_questions: int = 3
    ) -> List[str]:
        """
        使用 LLM 动态生成反思问题（Stanford GA 原始设计）
        
        这是 Stanford GA 论文中的原始设计：
        基于最近的记忆，让 LLM 生成高层次的反思问题
        
        Args:
            agent_name: Agent名称
            recent_memories: 最近的记忆列表
            llm: LLM实例
            num_questions: 生成的问题数量
            
        Returns:
            问题列表
        """
        import asyncio
        
        # 格式化最近的记忆
        memories_text = self._format_memories_for_prompt(recent_memories[:20])
        
        # 构建 prompt（参考 Stanford GA 原始设计）
        prompt = f"""
{agent_name}最近有以下经历和对话：

{memories_text}

基于以上内容，请生成{num_questions}个高层次的反思问题。
这些问题应该帮助{agent_name}更好地理解：
- 关系和社交动态
- 当前的关注点和担忧
- 兴趣和期待
- 情绪模式和变化

要求：
1. 问题应该是开放性的，需要深入思考
2. 问题应该基于具体的记忆内容，而不是泛泛而谈
3. 每个问题一行，不要编号

请生成{num_questions}个问题：
"""
        
        try:
            # 同步调用（如果在异步上下文中）
            if asyncio.iscoroutinefunction(getattr(llm, 'response_no_stream', None)):
                # 需要在异步环境中运行
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 在已运行的事件循环中，使用 run_coroutine_threadsafe
                    import concurrent.futures
                    future = asyncio.run_coroutine_threadsafe(
                        llm.response_no_stream(
                            system_prompt="你是一个善于提出深刻问题的AI助手。请只返回问题，不要添加其他内容。",
                            user_prompt=prompt,
                            max_tokens=200
                        ),
                        loop
                    )
                    response = future.result(timeout=30)
                else:
                    response = loop.run_until_complete(
                        llm.response_no_stream(
                            system_prompt="你是一个善于提出深刻问题的AI助手。请只返回问题，不要添加其他内容。",
                            user_prompt=prompt,
                            max_tokens=200
                        )
                    )
            elif hasattr(llm, 'generate_sync'):
                response = llm.generate_sync(prompt)
            else:
                # 降级到模板模式
                logger.bind(tag=TAG).warning("LLM not compatible for sync call, falling back to template")
                return self._generate_questions_from_template(agent_name)
            
            # 解析响应
            questions = [q.strip() for q in response.strip().split('\n') if q.strip()]
            
            # 过滤掉太短的行（可能是编号或空行）
            questions = [q for q in questions if len(q) > 10]
            
            # 限制数量
            questions = questions[:num_questions]
            
            if questions:
                logger.bind(tag=TAG).info(f"Generated {len(questions)} reflection questions (LLM mode)")
                return questions
            else:
                # 如果 LLM 没有生成有效问题，降级到模板模式
                logger.bind(tag=TAG).warning("LLM generated no valid questions, falling back to template")
                return self._generate_questions_from_template(agent_name)
                
        except Exception as e:
            logger.bind(tag=TAG).error(f"LLM question generation failed: {e}")
            return self._generate_questions_from_template(agent_name)
    
    async def generate_reflection_questions_async(
        self,
        agent_name: str,
        recent_memories: List[ConceptNode],
        llm,
        num_questions: int = 3
    ) -> List[str]:
        """
        异步版本：使用 LLM 动态生成反思问题
        
        Args:
            agent_name: Agent名称
            recent_memories: 最近的记忆列表
            llm: LLM实例（支持 LiteLLMProvider 或有 response_no_stream 方法的实例）
            num_questions: 生成的问题数量
            
        Returns:
            问题列表
        """
        # 格式化最近的记忆
        memories_text = self._format_memories_for_prompt(recent_memories[:20])
        
        # 构建 prompt
        prompt = f"""
{agent_name}最近有以下经历和对话：

{memories_text}

基于以上内容，请生成{num_questions}个高层次的反思问题。
这些问题应该帮助{agent_name}更好地理解：
- 关系和社交动态
- 当前的关注点和担忧
- 兴趣和期待
- 情绪模式和变化

要求：
1. 问题应该是开放性的，需要深入思考
2. 问题应该基于具体的记忆内容，而不是泛泛而谈
3. 每个问题一行，不要编号

请生成{num_questions}个问题：
"""
        
        try:
            # 使用统一的 LLM 调用方法
            response = await self._call_llm(
                llm=llm,
                system_prompt="你是一个善于提出深刻问题的AI助手。请只返回问题，不要添加其他内容。",
                user_prompt=prompt,
                max_tokens=200
            )
            
            # 解析响应
            questions = [q.strip() for q in response.strip().split('\n') if q.strip()]
            questions = [q for q in questions if len(q) > 10]
            questions = questions[:num_questions]
            
            if questions:
                logger.bind(tag=TAG).info(f"Generated {len(questions)} reflection questions (async LLM mode)")
                return questions
            else:
                return self._generate_questions_from_template(agent_name)
                
        except Exception as e:
            logger.bind(tag=TAG).error(f"Async LLM question generation failed: {e}")
            return self._generate_questions_from_template(agent_name)
    
    def _format_memories_for_prompt(self, memories: List[ConceptNode]) -> str:
        """
        格式化记忆列表用于 prompt
        
        Args:
            memories: 记忆列表
            
        Returns:
            格式化的字符串
        """
        if not memories:
            return "（暂无记忆）"
        
        formatted = []
        for i, mem in enumerate(memories, 1):
            if hasattr(mem, 'content'):
                content = mem.content
                age_hours = mem.get_age_hours() if hasattr(mem, 'get_age_hours') else 0
                importance = mem.importance if hasattr(mem, 'importance') else 5
                formatted.append(f"- [{age_hours:.1f}小时前, 重要性{importance}] {content}")
            elif isinstance(mem, tuple) and len(mem) >= 2:
                # (ConceptNode, score) 格式
                node, score = mem[0], mem[1]
                content = node.content if hasattr(node, 'content') else str(node)
                age_hours = node.get_age_hours() if hasattr(node, 'get_age_hours') else 0
                importance = node.importance if hasattr(node, 'importance') else 5
                formatted.append(f"- [{age_hours:.1f}小时前, 重要性{importance}] {content}")
            else:
                formatted.append(f"- {str(mem)}")
        
        return "\n".join(formatted)
    
    async def generate_insights(
        self,
        questions: List[str],
        relevant_memories: List[Tuple[ConceptNode, float]],
        llm,
        agent_name: str = "用户"
    ) -> List[Dict[str, Any]]:
        """
        基于问题和相关记忆，使用LLM生成洞见
        
        Args:
            questions: 反思问题列表
            relevant_memories: 相关记忆列表（记忆节点，相关性分数）
            llm: LLM实例（支持 LiteLLMProvider 或有 response_no_stream 方法的实例）
            agent_name: Agent名称
        
        Returns:
            洞见列表，每个洞见包含：
            - question: 问题
            - insight: 洞见内容
            - evidence: 支持证据（记忆ID列表）
            - importance: 重要性评分
        """
        insights = []
        
        for question in questions:
            try:
                # 构造反思prompt
                prompt = self._build_reflection_prompt(
                    question=question,
                    memories=relevant_memories,
                    agent_name=agent_name
                )
                
                # 使用LLM生成洞见 - 支持多种LLM接口
                insight_text = await self._call_llm(
                    llm=llm,
                    system_prompt="你是一个善于观察和总结的AI助手，能够从对话中提取深层洞见。",
                    user_prompt=prompt,
                    max_tokens=300
                )
                
                # 提取支持证据（相关记忆的ID）
                evidence = [node.node_id for node, score in relevant_memories[:5]]
                
                # 评估洞见的重要性（反思生成的洞见默认较高重要性）
                importance = self._assess_insight_importance(insight_text, question)
                
                insight = {
                    "question": question,
                    "insight": insight_text.strip(),
                    "evidence": evidence,
                    "importance": importance
                }
                
                insights.append(insight)
                logger.bind(tag=TAG).debug(f"Generated insight for question: {question[:50]}...")
                
            except Exception as e:
                logger.bind(tag=TAG).error(f"Failed to generate insight for question '{question}': {e}")
        
        logger.bind(tag=TAG).info(f"Generated {len(insights)} insights from {len(questions)} questions")
        
        return insights
    
    async def _call_llm(
        self,
        llm,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 300
    ) -> str:
        """
        调用LLM生成文本 - 兼容多种LLM接口
        
        支持：
        1. LiteLLMProvider（使用 chat 方法）
        2. 有 response_no_stream 方法的 LLM
        3. 有 generate 方法的 LLM
        
        Args:
            llm: LLM实例
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            max_tokens: 最大token数
        
        Returns:
            生成的文本
        """
        # 优先使用 chat 方法（LiteLLMProvider 的标准接口）
        if hasattr(llm, 'chat'):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = await llm.chat(
                messages=messages,
                tools=None,
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.content or ""
        
        # 回退到 response_no_stream 方法
        if hasattr(llm, 'response_no_stream'):
            return await llm.response_no_stream(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens
            )
        
        # 回退到 generate 方法
        if hasattr(llm, 'generate'):
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            return await llm.generate(full_prompt)
        
        raise RuntimeError(f"LLM instance {type(llm).__name__} has no compatible method (chat, response_no_stream, or generate)")
    
    def _build_reflection_prompt(
        self,
        question: str,
        memories: List[Tuple[ConceptNode, float]],
        agent_name: str
    ) -> str:
        """
        构造反思提示词
        
        Args:
            question: 反思问题
            memories: 相关记忆列表
            agent_name: Agent名称
        
        Returns:
            完整的prompt
        """
        # 格式化记忆列表
        memory_texts = []
        for i, (node, score) in enumerate(memories[:20], 1):  # 限制在20条记忆内
            age_hours = node.get_age_hours()
            memory_texts.append(
                f"{i}. [{age_hours:.1f}小时前, 重要性{node.importance}] {node.content}"
            )
        
        memories_section = "\n".join(memory_texts) if memory_texts else "（暂无相关记忆）"
        
        prompt = f"""
基于以下{agent_name}的记忆，请回答这个问题：

问题：{question}

相关记忆：
{memories_section}

请综合以上记忆，给出一个深刻的洞见（2-3句话）。
洞见应该：
1. 总结关键模式或趋势
2. 基于具体记忆，而非泛泛而谈
3. 对理解{agent_name}有帮助

洞见：
"""
        
        return prompt
    
    def _assess_insight_importance(self, insight: str, question: str) -> int:
        """
        评估洞见的重要性
        
        Args:
            insight: 洞见内容
            question: 原始问题
        
        Returns:
            重要性评分（1-10）
        """
        # 反思生成的洞见默认有较高重要性（7-9）
        # 可以基于关键词进一步调整
        
        base_importance = 7
        
        # 高重要性关键词
        high_importance_keywords = [
            "重要", "关键", "核心", "主要", "显著",
            "明显", "强烈", "深刻", "持续", "频繁"
        ]
        
        # 中等重要性关键词
        medium_importance_keywords = [
            "倾向", "趋势", "可能", "似乎", "表现出"
        ]
        
        # 检查关键词
        insight_lower = insight.lower()
        
        if any(kw in insight_lower for kw in high_importance_keywords):
            return min(9, base_importance + 2)
        elif any(kw in insight_lower for kw in medium_importance_keywords):
            return min(8, base_importance + 1)
        else:
            return base_importance
    
    def update_reflection_state(
        self,
        current_time: Optional[float] = None,
        reset_accumulation: bool = True
    ):
        """
        更新反思状态（在完成一次反思后调用）
        
        Args:
            current_time: 当前时间戳
            reset_accumulation: 是否重置累积重要性
        """
        if current_time is None:
            current_time = time.time()
        
        self.last_reflection_time = current_time
        
        if reset_accumulation:
            self.accumulated_importance = 0
        
        logger.bind(tag=TAG).debug(
            f"Reflection state updated: last_time={current_time}, "
            f"accumulated_importance={'reset to 0' if reset_accumulation else self.accumulated_importance}"
        )
    
    def add_importance(self, importance: int):
        """
        添加到累积重要性
        
        Args:
            importance: 要添加的重要性值
        """
        self.accumulated_importance += importance


# 保持向后兼容的别名
ReflectionEngine = ReflectModule
