"""
ConceptNode - 记忆节点

每条记忆都是一个ConceptNode，包含：
- 内容（自然语言描述）
- 时间戳
- 重要性评分
- 向量表示（用于检索）
- 访问记录（用于计算时间衰减）
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import time


@dataclass
class ConceptNode:
    """单个记忆节点（一条记忆）"""
    
    # 唯一标识
    node_id: str
    
    # 记忆内容
    content: str                    # 完整描述
    
    # 时间信息
    created: float                  # 创建时间（时间戳，秒）
    last_accessed: float           # 最后访问时间（时间戳，秒）
    expiration: Optional[float] = None  # 过期时间（可选）
    
    # 重要性评分（1-10）
    importance: int = 5
    
    # 向量表示（用于相似度检索）
    embedding: Optional[List[float]] = None
    
    # 记忆类型
    memory_type: str = "observation"  # observation, reflection, plan
    
    # 角色信息（可选）
    role: str = "user"  # user, assistant
    
    # 访问历史（用于计算遗忘曲线）
    access_history: List[float] = field(default_factory=list)
    
    # 关联信息（可选）
    related_nodes: List[str] = field(default_factory=list)  # 关联的其他记忆ID
    emotion: Optional[str] = None  # 情绪标签
    location: Optional[str] = None  # 位置信息（用于空间记忆）
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后的验证和处理"""
        # 确保重要性在1-10范围内
        if self.importance < 1:
            self.importance = 1
        elif self.importance > 10:
            self.importance = 10
        
        # 记录首次访问
        if not self.access_history:
            self.access_history.append(self.created)
    
    def record_access(self):
        """记录一次访问"""
        current_time = time.time()
        self.last_accessed = current_time
        self.access_history.append(current_time)
        
        # 限制访问历史长度（最多保留最近100次）
        if len(self.access_history) > 100:
            self.access_history = self.access_history[-100:]
    
    def get_age_hours(self, current_time: Optional[float] = None) -> float:
        """获取记忆的年龄（小时）"""
        if current_time is None:
            current_time = time.time()
        return (current_time - self.created) / 3600
    
    def get_recency_score(self, current_time: Optional[float] = None, decay_rate: float = 0.99) -> float:
        """
        计算最近性分数（时间衰减）
        
        Args:
            current_time: 当前时间戳，默认为当前时间
            decay_rate: 衰减率，默认0.99（每小时衰减1%）
        
        Returns:
            最近性分数（0-1）
        """
        hours_ago = self.get_age_hours(current_time)
        return decay_rate ** hours_ago
    
    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """检查记忆是否过期"""
        if self.expiration is None:
            return False
        if current_time is None:
            current_time = time.time()
        return current_time > self.expiration
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            'node_id': self.node_id,
            'content': self.content,
            'created': self.created,
            'last_accessed': self.last_accessed,
            'expiration': self.expiration,
            'importance': self.importance,
            'embedding': self.embedding,
            'memory_type': self.memory_type,
            'role': self.role,
            'access_history': self.access_history[-10:],  # 只保存最近10次访问
            'related_nodes': self.related_nodes,
            'emotion': self.emotion,
            'location': self.location,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConceptNode':
        """从字典创建（用于反序列化）"""
        return cls(**data)
    
    def __str__(self) -> str:
        """字符串表示"""
        age_hours = self.get_age_hours()
        return f"[{self.memory_type}] {self.content[:50]}... (importance={self.importance}, age={age_hours:.1f}h)"
    
    def __repr__(self) -> str:
        return f"ConceptNode(id={self.node_id}, type={self.memory_type}, importance={self.importance})"
