"""
ImportanceScorer - 重要性评分器

评估记忆的重要性（1-10分）
可以使用规则或LLM进行评估
"""

from typing import Optional
import re
from loguru import logger

TAG = __name__



class ImportanceScorer:
    """重要性评分器"""
    
    def __init__(self, use_llm: bool = False):
        """
        初始化评分器
        
        Args:
            use_llm: 是否使用LLM进行评分（更准确但成本高）
        """
        self.use_llm = use_llm
        
        # 关键词权重（基于规则的评分）
        self.keyword_weights = {
            # 情感相关（高重要性）
            '爱': 9, '恨': 9, '死': 9, '生': 8,
            '失恋': 9, '结婚': 9, '分手': 8, '离婚': 9,
            '开心': 7, '难过': 7, '高兴': 7, '伤心': 8,
            '焦虑': 7, '害怕': 7, '恐惧': 8, '担心': 6,
            '感动': 7, '激动': 7, '兴奋': 7,
            
            # 重要事件（高重要性）
            '工作': 7, '换工作': 8, '辞职': 8, '失业': 8,
            '家人': 8, '父母': 8, '孩子': 8, '朋友': 6,
            '生病': 8, '住院': 9, '手术': 9,
            '搬家': 7, '买房': 8, '买车': 7,
            
            # 日常（中等重要性）
            '吃饭': 3, '睡觉': 3, '天气': 2,
            '电影': 4, '游戏': 4, '运动': 5,
            '学习': 6, '考试': 7, '毕业': 8,
            
            # 低重要性
            '早安': 2, '晚安': 2, '你好': 1,
            '谢谢': 3, '不客气': 2,
        }
        
        # 特殊模式（正则表达式）
        self.special_patterns = [
            (r'我.{0,5}(想|要|打算|计划)', 6),  # 意图表达
            (r'(为什么|怎么|如何)', 5),  # 问题询问
            (r'(应该|必须|一定要)', 6),  # 强烈态度
            (r'(从来|永远|总是|一直)', 7),  # 绝对性表达
            (r'第一次', 8),  # 首次经历
            (r'(重要|关键|紧急)', 8),  # 显式重要性
        ]
    
    def score(self, content: str, role: str = "user", llm = None) -> int:
        """
        评估记忆重要性
        
        Args:
            content: 记忆内容
            role: 角色（user/assistant）
            llm: LLM实例（如果use_llm=True）
        
        Returns:
            重要性分数（1-10）
        """
        # 如果启用LLM且提供了LLM实例
        if self.use_llm and llm:
            return self._score_with_llm(content, llm)
        else:
            return self._score_with_rules(content, role)
    
    def _score_with_rules(self, content: str, role: str) -> int:
        """基于规则的评分"""
        
        # 基础分数
        base_score = 5
        
        # 助手的回复通常重要性较低
        if role == "assistant":
            base_score = 4
        
        # 检查关键词
        max_keyword_score = 0
        for keyword, weight in self.keyword_weights.items():
            if keyword in content:
                max_keyword_score = max(max_keyword_score, weight)
        
        # 检查特殊模式
        max_pattern_score = 0
        for pattern, weight in self.special_patterns:
            if re.search(pattern, content):
                max_pattern_score = max(max_pattern_score, weight)
        
        # 综合评分（取最大值）
        final_score = max(base_score, max_keyword_score, max_pattern_score)
        
        # 内容长度影响（很短的内容降低重要性）
        if len(content) < 5:
            final_score = min(final_score, 3)
        elif len(content) < 10:
            final_score = min(final_score, 5)
        
        # 确保在1-10范围内
        final_score = max(1, min(10, final_score))
        
        logger.bind(tag=TAG).debug(
            f"规则评分: {content[:30]}... -> {final_score}"
        )
        
        return final_score
    
    def _score_with_llm(self, content: str, llm) -> int:
        """基于LLM的评分（更准确但成本高）"""
        try:
            prompt = f"""
评估以下记忆的重要性（1-10分）：

记忆内容：{content}

评分标准：
- 1-3分：日常琐事（天气、问候等）
- 4-6分：普通对话（兴趣爱好、日常活动）
- 7-8分：重要事件（工作、学习、人际关系）
- 9-10分：重大事件（生死、爱恨、人生转折）

只需返回一个数字（1-10）。
"""
            
            # 调用LLM
            response = llm.response_no_stream(
                system_prompt="你是一个记忆重要性评估专家。",
                user_prompt=prompt,
                max_tokens=10,
                temperature=0.3
            )
            
            # 提取数字
            import re
            match = re.search(r'\d+', response)
            if match:
                score = int(match.group())
                score = max(1, min(10, score))  # 限制在1-10
                logger.bind(tag=TAG).debug(f"LLM评分: {content[:30]}... -> {score}")
                return score
            else:
                logger.bind(tag=TAG).warning(f"LLM评分失败，使用默认值: {response}")
                return 5
        
        except Exception as e:
            logger.bind(tag=TAG).error(f"LLM评分出错: {e}")
            # 回退到规则评分
            return self._score_with_rules(content, "user")
    
    def score_conversation_turn(self, user_msg: str, assistant_msg: str, llm = None) -> tuple:
        """
        评估一轮对话的重要性
        
        Args:
            user_msg: 用户消息
            assistant_msg: 助手消息
            llm: LLM实例（可选）
        
        Returns:
            (用户消息重要性, 助手消息重要性)
        """
        user_importance = self.score(user_msg, "user", llm)
        assistant_importance = self.score(assistant_msg, "assistant", llm)
        
        return user_importance, assistant_importance
